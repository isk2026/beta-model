"""주도 섹터 모니터 (참고 정보 — 배분 규칙에는 넣지 않는다).

미국 SPDR 섹터 ETF 11개를 상대강도(6개월 + 12-1개월 수익률 평균)로 줄 세우고,
각 섹터에 같은 추세 엔진(정배열·낙폭·모멘텀)을 돌려 '주도 섹터'를 표시한다.

배분에 넣지 않는 이유 (python -m beta_model.sectors --backtest, 1999~2026):
  주식 중 일부(고정 40% 또는 추세 점수 비례 0~40%)를 상위 2~4개 섹터에 월 1회 배분하는 규칙 18가지를
  SPY 단독과 비교 → 표본 내(1999~2012)는 대부분 우위였지만 표본 외(2013~)는 18개 모두 SPY보다 낮았다
  (SPY 14.8% vs 12.3~14.7%). 기술주 포트 QQQ/SOXX 상대강도 기울이기도 조회 기간에 따라 결과가 뒤집혀 기각.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from .data_pipeline import CACHE_FILE, MarketData, compute_features, fetch_yfinance
from .scoring import BetaModelScoringEngine

log = logging.getLogger(__name__)
SECTOR_CACHE = CACHE_FILE.parent / "sectors.pkl"
BENCH = "SPY"
SECTORS: dict[str, str] = {
    "XLK": "기술", "XLC": "커뮤니케이션", "XLY": "임의소비재", "XLF": "금융", "XLI": "산업재",
    "XLV": "헬스케어", "XLP": "필수소비재", "XLE": "에너지", "XLB": "소재", "XLU": "유틸리티", "XLRE": "부동산",
}
EXTRA: dict[str, str] = {"QQQ": "나스닥100", "SOXX": "반도체"}   # 기술주 포트 구성
TOP_N = 3
SAT_MAX = 0.4   # 옵트인 위성 비중: 주식의 최대 40%를 주도 섹터에 (추세 점수 비례)


def fetch_sectors(use_cache: bool = True, max_age_hours: float = 6) -> pd.DataFrame:
    import time
    if use_cache and SECTOR_CACHE.exists() and (time.time() - SECTOR_CACHE.stat().st_mtime) / 3600 < max_age_hours:
        with SECTOR_CACHE.open("rb") as f:
            return pickle.load(f)
    tickers = {t: t for t in [BENCH, *SECTORS, *EXTRA]}
    px = fetch_yfinance(tickers, datetime(1998, 1, 1))
    SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with SECTOR_CACHE.open("wb") as f:
        pickle.dump(px, f)
    return px


def momentum_score(px: pd.DataFrame) -> pd.DataFrame:
    """상대강도 점수 = (6개월 수익률 + 12개월 수익률(최근 1개월 제외)) / 2."""
    return ((px / px.shift(126) - 1) + (px.shift(21) / px.shift(252) - 1)) / 2


def sector_payload(px: pd.DataFrame, history_days: int = 252) -> dict:
    px = px.ffill()
    names = [*SECTORS, *EXTRA]
    md = MarketData(prices=px, features={n: compute_features(px[n].dropna()).reindex(px.index) for n in names})
    engine = BetaModelScoringEngine(md)
    mom = momentum_score(px[names + [BENCH]])
    last = px.index[-1]

    def row(t: str, label: str) -> dict:
        f = md.features[t].iloc[-1]
        p = px[t]
        r = lambda n: float(p.iloc[-1] / p.iloc[-1 - n] - 1)  # noqa: E731
        return {
            "ticker": t, "name": label,
            "trend": round(float(engine.index_trend(t).trend_score.iloc[-1]), 4),
            "score": round(float(mom[t].iloc[-1]), 4),
            "rel_6m": round(float((p.iloc[-1] / p.iloc[-127]) / (px[BENCH].iloc[-1] / px[BENCH].iloc[-127]) - 1), 4),
            "ret_1m": round(r(21), 4), "ret_3m": round(r(63), 4), "ret_6m": round(r(126), 4),
            "drawdown": round(float(f.drawdown), 4),
            "above_sma200": bool(p.iloc[-1] > p.rolling(200).mean().iloc[-1]),
        }

    rows = sorted((row(t, n) for t, n in SECTORS.items() if px[t].notna().sum() > 260),
                  key=lambda x: x["score"], reverse=True)
    leaders = [x["ticker"] for x in rows if x["trend"] > 0 and x["above_sma200"]][:TOP_N]
    for i, x in enumerate(rows):
        x["rank"] = i + 1
        x["leader"] = x["ticker"] in leaders
    # 순위 추이: 최근 1년 월말 기준 순위
    me = px.index[px.index >= px.index[-1] - pd.Timedelta(days=370)].to_series().groupby(
        lambda d: (d.year, d.month)).max().tolist()
    rank_hist = {
        d.strftime("%Y-%m"): {t: int(v) for t, v in mom.loc[d, list(SECTORS)].rank(ascending=False).dropna().items()}
        for d in me
    }
    return {
        "as_of": last.strftime("%Y-%m-%d"),
        "benchmark": {"ticker": BENCH, "ret_6m": round(float(px[BENCH].iloc[-1] / px[BENCH].iloc[-127] - 1), 4),
                      "trend": round(float(BetaModelScoringEngine(MarketData(
                          prices=px, features={BENCH: compute_features(px[BENCH].dropna()).reindex(px.index)}
                      )).index_trend(BENCH).trend_score.iloc[-1]), 4)},
        "sectors": rows,
        "leaders": leaders,
        "tech": [row(t, n) for t, n in EXTRA.items()],
        "rank_history": rank_hist,
    }


# ---------------------------------------------------------------------------
# 배분 규칙 백테스트 (채택하지 않은 근거)
# ---------------------------------------------------------------------------
def backtest_rotation(px: pd.DataFrame, trend: pd.Series, n: int = 3, max_sat: float = 0.4,
                      mode: str = "trend", cost_bps: float = 10) -> pd.Series:
    """주식 = (1−f)·SPY + f·상위 n개 섹터 균등. f = max_sat·clip(추세,0,1) 또는 고정. 월말 신호 → 익일 집행."""
    px = px.ffill()
    ret = px.pct_change(fill_method=None)
    secs = list(SECTORS)
    score = momentum_score(px[secs])
    ok = px[secs] > px[secs].rolling(200).mean()
    month_ends = px.index.to_series().groupby(px.index.to_period("M")).max()
    rows = {}
    for d in month_ends:
        if d < px.index[260]:
            continue
        top = score.loc[d].where(ok.loc[d]).dropna().sort_values(ascending=False).index[:n]
        f = max_sat * float(np.clip(trend.get(d, 0.0), 0, 1)) if mode == "trend" else max_sat
        w = pd.Series(0.0, index=[*secs, BENCH])
        w[list(top)] = f / n
        w[BENCH] = 1 - w.sum()
        rows[d] = w
    held = pd.DataFrame(rows).T.reindex(px.index).ffill().shift(2).dropna()
    r = (held * ret.reindex(held.index)[held.columns].fillna(0)).sum(axis=1)
    return r - held.diff().abs().sum(axis=1).fillna(0) * cost_bps / 1e4


# ---------------------------------------------------------------------------
# 옵트인 '주도 섹터 위성 비중' (대시보드 토글 — 검증상 근거 없음, 기본 꺼짐)
# ---------------------------------------------------------------------------
def satellite_split(trend: float, leaders: list[str], max_sat: float = SAT_MAX, n: int = TOP_N) -> dict:
    """주식 비중 내부 분할(합계 1). f = max_sat·clip(추세,0,1), 주도 섹터 n개에 f/n씩.

    주도 섹터가 n개 미만이면 남는 몫은 코어(SPY/지수 ETF)에 그대로 둔다. 주식/현금 비중은 바꾸지 않는다.
    """
    f = max_sat * float(np.clip(trend if trend is not None and np.isfinite(trend) else 0.0, 0, 1))
    legs = {t: f / n for t in list(leaders)[:n]}
    return {"fraction": f, "core": 1 - sum(legs.values()), "legs": legs}


def _perf(r: pd.Series) -> dict:
    """연복리·샤프(무위험 0)·최대낙폭."""
    eq = (1 + r).cumprod()
    return {
        "cagr": round(float(eq.iloc[-1] ** (252 / len(r)) - 1), 4),
        "sharpe": round(float(r.mean() / r.std() * 252 ** 0.5), 2) if r.std() > 0 else 0.0,
        "mdd": round(float((eq / eq.cummax() - 1).min()), 4),
    }


def rotation_summary(px: pd.DataFrame, trend: pd.Series, split: str = "2012") -> dict:
    """SPY vs 추세 비례(0~40%) 상위 3개 섹터 규칙 — 표본 내(~2012) / 표본 외(2013~)."""
    trend = trend.reindex(px.index).ffill()
    rule = backtest_rotation(px, trend, n=TOP_N, max_sat=SAT_MAX, mode="trend")
    spy = px[BENCH].ffill().pct_change(fill_method=None).loc[rule.index]
    nxt = str(int(split) + 1)
    return {
        "rule": f"주식 중 {SAT_MAX:.0%}×clip(미국 추세,0,1)을 상위 {TOP_N}개 섹터 균등, 월 1회, 비용 10bp",
        "in_sample": {"period": f"{rule.index[0].year}~{split}",
                      "spy": _perf(spy[:split]), "rule": _perf(rule[:split])},
        "out_of_sample": {"period": f"{nxt}~{rule.index[-1].year}",
                          "spy": _perf(spy[nxt:]), "rule": _perf(rule[nxt:])},
    }


def satellite_payload(sectors: dict, px: pd.DataFrame, us_trend_now: float,
                      us_trend_hist: pd.Series | None = None) -> dict:
    """대시보드 옵트인 토글용. 분할은 주식 비중 대비 비율(포트 비중 = 주식 비중 × 값)."""
    sp = satellite_split(us_trend_now, sectors["leaders"])
    names = {x["ticker"]: x["name"] for x in sectors["sectors"]}
    out = {
        "enabled_by_default": False,
        "validated": False,
        "max_fraction": SAT_MAX, "n": TOP_N,
        "us_trend": round(float(us_trend_now), 4),
        "fraction": round(sp["fraction"], 4),
        "core": round(sp["core"], 4),
        "legs": [{"ticker": t, "name": names.get(t, t), "weight": round(w, 4)} for t, w in sp["legs"].items()],
    }
    if us_trend_hist is not None:
        try:
            out["backtest"] = rotation_summary(px, us_trend_hist)
        except Exception as exc:  # noqa: BLE001 — 근거 수치가 없어도 분할 표시는 유지
            log.warning("섹터 위성 백테스트 실패: %s", exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="주도 섹터 모니터")
    ap.add_argument("--backtest", action="store_true", help="섹터 배분 규칙 표본 내/외 비교")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    px = fetch_sectors()
    if not args.backtest:
        p = sector_payload(px)
        for x in p["sectors"]:
            print(f"{x['rank']:2} {x['ticker']:5} {x['name']:8} 점수 {x['score']:+.1%} 추세 {x['trend']:+.2f} "
                  f"6M 대비 SPY {x['rel_6m']:+.1%} {'★' if x['leader'] else ''}")
        return
    from .data_pipeline import load_market_data
    trend = BetaModelScoringEngine(load_market_data()).scores("US").trend_score.reindex(px.index).ffill()
    spy = px[BENCH].ffill().pct_change(fill_method=None)

    def st(r):
        p = _perf(r)
        return f"CAGR {p['cagr']:6.1%} Sh {p['sharpe']:4.2f} MDD {p['mdd']:6.1%}"
    for name, r in [("SPY", None), *[(f"{m} n{n}", backtest_rotation(px, trend, n=n, mode=m))
                                     for m in ("trend", "fixed") for n in (2, 3, 4)]]:
        r = spy.loc[backtest_rotation(px, trend).index] if r is None else r
        print(f"{name:10} 표본내 {st(r[:'2012'])} | 표본외 {st(r['2013':])}")


if __name__ == "__main__":
    main()
