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
from .data_pipeline import LONG_SMA, VKOSPI_PROXY_COL, MarketData, realized_vol


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


def index_weights(mk: dict) -> dict[str, float]:
    """시장 구성 지수 비중 (없으면 동일가중), 합 1."""
    w = mk.get("index_weights") or {n: 1.0 for n in mk["indices"]}
    tot = sum(w[n] for n in mk["indices"])
    return {n: w[n] / tot for n in mk["indices"]}


def market_price(data: MarketData, mk: dict) -> pd.Series:
    """시장의 대표 가격. index_weights가 있으면 매일 리밸런싱한 가중 바스켓(상장 전 종목은 제외·재정규화),
    없으면 첫 번째 지수(기존 미국=S&P500, 한국=코스피 기준 유지)."""
    if "index_weights" not in mk:
        return data.features[mk["indices"][0]]["close"]
    w = pd.Series(index_weights(mk))
    rets = pd.DataFrame({n: data.features[n]["close"].pct_change(fill_method=None) for n in mk["indices"]})
    avail = rets.notna().mul(w, axis=1)
    basket = rets.fillna(0).mul(w, axis=1).sum(axis=1) / avail.sum(axis=1)
    basket = basket.where(avail.sum(axis=1) > 0)
    start = basket.first_valid_index()
    return (1 + basket.loc[start:].fillna(0)).cumprod().mul(100).reindex(basket.index)


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
        comps = {"alignment": alignment, "drawdown": drawdown, "momentum": momentum}
        if tp.w_long > 0:   # 장기 추세 요소 (실험용) — 꺼져 있으면 기존 계산과 완전히 같다
            comps["long"] = self._long_trend(f)
            total = tp.w_alignment + tp.w_drawdown + tp.w_momentum + tp.w_long
            score = (score + tp.w_long * comps["long"]) / total
        if tp.smooth_span > 1:
            score = score.ewm(span=tp.smooth_span, min_periods=1).mean().where(score.notna())
        return pd.DataFrame({**comps, "trend_score": _clip(score, -1.0, 1.0)})

    def _long_trend(self, f: pd.DataFrame) -> pd.Series:
        """장기 추세 −1~+1: 12-1개월 모멘텀 또는 200일선 이격도의 tanh."""
        tp = self.tp
        if tp.long_kind == "tsmom":
            x = f["mom_12_1"]
        elif tp.long_kind == "sma200":
            x = f["close"] / f[f"sma{LONG_SMA}"] - 1
        else:
            raise ValueError(f"unknown long_kind: {tp.long_kind}")
        return np.tanh(x / tp.long_scale)

    def trend_score(self, market: str) -> pd.DataFrame:
        """시장 추세 = 구성 지수 추세의 평균.

        index_weights가 있으면 가중평균이고, 아직 상장 전인 지수는 빼고 남은 비중으로 재정규화한다
        (SOXX 상장 전 QQQ만으로 계산). 없으면 기존처럼 모든 지수가 있어야 값이 생긴다.
        """
        names = self.markets[market]["indices"]
        frames = {n: self.index_trend(n) for n in names}
        if "index_weights" not in self.markets[market]:
            return (sum(frames.values()) / len(frames)).clip(-1.0, 1.0)
        w = index_weights(self.markets[market])
        cols = frames[names[0]].columns
        out = {}
        for c in cols:
            df = pd.DataFrame({n: frames[n][c] for n in names})
            wt = df.notna().mul(pd.Series(w), axis=1)
            out[c] = (df.fillna(0).mul(pd.Series(w), axis=1).sum(axis=1) / wt.sum(axis=1)).where(wt.sum(axis=1) > 0)
        return pd.DataFrame(out).clip(-1.0, 1.0)

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
        if rp.vol_proxy_mode == "drop":
            # proxy(실현변동성 대체값)인 날은 실현변동성 요소와 같은 신호 → 빼고 재정규화(risk_score)
            vol_index = vol_index.mask(self.vol_is_proxy(vol_name))
        elif rp.vol_proxy_mode != "keep":
            raise ValueError(f"unknown vol_proxy_mode: {rp.vol_proxy_mode}")

        # 시장 대표 지수(첫 번째) — 가중 바스켓 시장은 바스켓 — 의 실현변동성 백분위
        rv = realized_vol(market_price(self.data, self.markets[market]).dropna(), 20).reindex(idx).ffill()
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
            "breadth": self.breadth()["breadth"],
        })

    def vol_is_proxy(self, vol_name: str) -> pd.Series:
        """변동성 지수가 날짜별로 proxy인지(True). VKOSPI만 해당.

        표시 열(VKOSPI_PROXY)이 있으면 그 값을, 없으면(이전 캐시) meta 의 출처로 판단한다.
        """
        idx = self.data.prices.index
        if vol_name != "VKOSPI":
            return pd.Series(False, index=idx)
        if VKOSPI_PROXY_COL in self.data.prices:
            return self.data.prices[VKOSPI_PROXY_COL].fillna(0) > 0.5
        return pd.Series(self.data.meta.get("vkospi_source") == "proxy", index=idx)

    def breadth(self) -> pd.DataFrame:
        """미국 시장 폭(모든 시장 공통): 동일가중(RSP) 대비 SPY 열위 + 200일선 아래 섹터 비율.

        데이터가 없는 구간(RSP 2003년, 섹터 ETF 1999년 이전)은 NaN → 위험 점수에서 재정규화.
        """
        p, rp = self.data.prices, self.rp
        idx = p.index
        nan = pd.Series(np.nan, index=idx)
        if {"RSP", "SPY"} <= set(p.columns):
            n = rp.breadth_ew_window
            ew_gap = (p["RSP"] / p["RSP"].shift(n)) / (p["SPY"] / p["SPY"].shift(n)) - 1
        else:
            ew_gap = nan
        secs = [c for c in config.BREADTH_SECTORS if c in p.columns]
        if secs:
            s = p[secs]
            sect_below = (s < s.rolling(200).mean()).where(s.rolling(200).count() == 200).mean(axis=1)
        else:
            sect_below = nan
        ew = pd.Series(linear_map(-ew_gap, 0.0, rp.breadth_ew_full), index=idx).where(ew_gap.notna())
        sect = pd.Series(linear_map(sect_below, rp.breadth_sect_lo, rp.breadth_sect_hi), index=idx).where(sect_below.notna())
        return pd.DataFrame({"ew_gap": ew_gap, "sect_below200": sect_below,
                             "breadth": pd.concat([ew, sect], axis=1).mean(axis=1)})

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
        """추세·위험 점수 시계열.

        끝부분은 이 시장 지수에 실제 종가가 있는 마지막 날에서 자른다. 공통 달력(미·한 거래일 합집합)의
        마지막 날이 다른 나라 거래일이면 이 시장 값은 전일 ffill 이라, 그 날짜로 기준일을 잡으면
        '어제 데이터로 계산한 오늘 신호'가 생긴다 (16:30 KST 실행의 미국, 한국 휴장일의 한국).
        """
        t = self.trend_score(market).add_prefix("trend_").rename(columns={"trend_trend_score": "trend_score"})
        r = self.risk_score(market).add_prefix("risk_").rename(columns={"risk_risk_score": "risk_score"})
        out = t.join(r).dropna(subset=["trend_score", "risk_score"])
        last = self.data.last_trade_date(self.markets[market]["indices"])
        return out.loc[:last] if last is not None else out

    def latest(self, market: str) -> dict:
        row = self.scores(market).iloc[-1]
        return {"date": row.name.strftime("%Y-%m-%d"), **{k: round(float(v), 4) for k, v in row.items()}}
