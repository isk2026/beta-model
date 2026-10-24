"""2단계: 추세 점수(-1~+1)와 위험 점수(0~1) 계산.

모든 함수는 시계열 전체를 벡터화해서 계산하므로 '오늘 값'과 '과거 이력'을
같은 코드로 얻는다. 각 점수의 하위 구성요소도 함께 반환해 대시보드에서
"왜 이 점수인지"를 보여줄 수 있게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import RiskParams, TrendParams
from .data_pipeline import MarketData, realized_vol


def _clip(x, lo: float, hi: float):
    return np.clip(x, lo, hi)


def linear_map(x, x0: float, x1: float, y0: float = 0.0, y1: float = 1.0):
    """x0→y0, x1→y1 선형 변환 후 [min(y), max(y)]로 클립."""
    y = y0 + (np.asarray(x, dtype=float) - x0) * (y1 - y0) / (x1 - x0)
    return _clip(y, min(y0, y1), max(y0, y1))


def rolling_percentile(s: pd.Series, window: int = config.TRADING_DAYS) -> pd.Series:
    """최근 `window`일 중 현재 값의 백분위 (0~1). 동률은 절반만 반영."""
    def pct(a: np.ndarray) -> float:
        last = a[-1]
        return ((a < last).sum() + 0.5 * ((a == last).sum() - 1)) / (len(a) - 1)
    return s.rolling(window, min_periods=window // 2).apply(pct, raw=True)


def rolling_zscore(s: pd.Series, window: int = config.TRADING_DAYS) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


class BetaModelScoringEngine:
    """성상현 베타 모델의 추세/위험 점수 엔진.

    Args:
        data: `data_pipeline.build_market_data` 결과.
        trend_params / risk_params: 가중치·임계값 (기본값은 config).

    Example:
        >>> engine = BetaModelScoringEngine(load_market_data())
        >>> engine.latest("US")
        {'trend_score': 0.71, 'risk_score': 0.32, ...}
    """

    def __init__(
        self,
        data: MarketData,
        trend_params: TrendParams = config.TREND,
        risk_params: RiskParams = config.RISK,
        markets: dict = config.MARKETS,
    ) -> None:
        self.data = data
        self.tp = trend_params
        self.rp = risk_params
        self.markets = markets

    # ------------------------------------------------------------------ trend
    def index_trend(self, name: str) -> pd.DataFrame:
        """단일 지수의 추세 구성요소와 합성 점수."""
        f = self.data.features[name]
        tp = self.tp
        # 정배열: Price>SMA20>SMA60>SMA120 의 3개 인접 비교 + Price>SMA120 장기 확인
        pairs = [("close", "sma20"), ("sma20", "sma60"), ("sma60", "sma120"), ("close", "sma120")]
        if tp.alignment_scale > 0:
            # 이격도 연속값: 이평선 바로 위/아래에서 ±1로 튀지 않는다
            alignment = sum(np.tanh((f[a] / f[b] - 1) / tp.alignment_scale) for a, b in pairs) / len(pairs)
        else:
            alignment = sum(np.sign(f[a] - f[b]) for a, b in pairs) / len(pairs)
        drawdown = pd.Series(
            np.where(
                f["drawdown"] >= tp.dd_neutral,
                linear_map(f["drawdown"], tp.dd_neutral, 0.0, 0.0, 1.0),
                linear_map(f["drawdown"], tp.dd_floor, tp.dd_neutral, -1.0, 0.0),
            ),
            index=f.index,
        ).where(f["drawdown"].notna())
        momentum = np.tanh(f[f"roc_{tp.roc_window}"] / tp.roc_scale)
        score = tp.w_alignment * alignment + tp.w_drawdown * drawdown + tp.w_momentum * momentum
        if tp.smooth_span > 1:
            score = score.ewm(span=tp.smooth_span, min_periods=1).mean().where(score.notna())
        return pd.DataFrame({
            "alignment": alignment,
            "drawdown": drawdown,
            "momentum": momentum,
            "trend_score": _clip(score, -1.0, 1.0),
        })

    def trend_score(self, market: str) -> pd.DataFrame:
        """시장 추세 = 구성 지수 추세의 평균."""
        frames = [self.index_trend(n) for n in self.markets[market]["indices"]]
        return (sum(frames) / len(frames)).clip(-1.0, 1.0)

    # ------------------------------------------------------------------- risk
    def risk_components(self, market: str) -> pd.DataFrame:
        """위험 구성요소(각 0~1). 상승 방향 스파이크만 위험으로 본다."""
        p = self.data.prices
        rp = self.rp
        vol_name = self.markets[market]["vol_index"]
        idx = p.index

        vol = p[vol_name]
        vol_index = 0.5 * rolling_percentile(vol) + 0.5 * pd.Series(
            linear_map(vol, rp.vol_level_lo[vol_name], rp.vol_level_hi[vol_name]), index=idx
        )

        # 시장 대표 지수(첫 번째)의 실현변동성 백분위
        main = self.markets[market]["indices"][0]
        rv = realized_vol(p[main].dropna(), 20).reindex(idx).ffill()
        realized = rolling_percentile(rv)

        w = rp.spike_window
        rate_chg_bp = (p["US10Y"] - p["US10Y"].shift(w)) * 100
        rates = pd.Series(linear_map(rate_chg_bp, 0.0, rp.rate_spike_full_bp), index=idx)
        fx_chg = p["USDKRW"] / p["USDKRW"].shift(w) - 1
        fx = pd.Series(linear_map(fx_chg, 0.0, rp.fx_spike_full), index=idx)

        credit = self._spread_score(p.get(rp.credit_series), rp.credit_level_lo, rp.credit_level_hi, idx)
        if market == "KR" and rp.kr_credit_blend > 0 and rp.kr_credit_series in p:
            kr = self._spread_score(p[rp.kr_credit_series], rp.kr_credit_level_lo, rp.kr_credit_level_hi, idx)
            b = rp.kr_credit_blend
            credit = ((1 - b) * credit + b * kr).fillna(credit)

        return pd.DataFrame({
            "vol_index": vol_index, "realized_vol": realized,
            "rates": rates, "fx": fx, "credit": credit,
        })

    def _spread_score(self, spread: pd.Series | None, lo: float, hi: float, idx: pd.Index) -> pd.Series:
        """신용 스프레드 → 0~1: Z점수(상대) 50% + 절대 레벨 50%."""
        if spread is None:
            return pd.Series(np.nan, index=idx)
        z = pd.Series(linear_map(rolling_zscore(spread), 0.0, self.rp.credit_z_full), index=idx)
        level = pd.Series(linear_map(spread, lo, hi), index=idx)
        return (0.5 * z + 0.5 * level).where(spread.notna())

    def risk_score(self, market: str) -> pd.DataFrame:
        """가중 합성 위험 점수. 결측 구성요소는 남은 가중치로 재정규화."""
        comp = self.risk_components(market)
        weights = pd.Series(self.markets[market].get("weights") or self.rp.weights[market])
        w = comp.notna().mul(weights, axis=1)
        score = (comp.fillna(0) * weights).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
        out = comp.copy()
        out["risk_score"] = score.clip(0.0, 1.0)
        return out

    # ---------------------------------------------------------------- summary
    def scores(self, market: str) -> pd.DataFrame:
        t = self.trend_score(market).add_prefix("trend_").rename(columns={"trend_trend_score": "trend_score"})
        r = self.risk_score(market).add_prefix("risk_").rename(columns={"risk_risk_score": "risk_score"})
        return t.join(r).dropna(subset=["trend_score", "risk_score"])

    def latest(self, market: str) -> dict:
        row = self.scores(market).iloc[-1]
        return {"date": row.name.strftime("%Y-%m-%d"), **{k: round(float(v), 4) for k, v in row.items()}}
