"""주도 섹터 모니터 (참고 정보 — 배분 규칙에는 넣지 않는다).

섹터 ETF를 상대강도(6개월 + 12-1개월 수익률 평균)로 줄 세우고,
각 섹터에 같은 추세 엔진(정배열·낙폭·모멘텀)을 돌려 '주도 섹터'를 표시한다.

유니버스
  US: SPDR 섹터 ETF 11개, 기준 SPY, 보조 QQQ·SOXX (기술주 포트)
  KR: TIGER 200 섹터 ETF 11개(KOSPI200 섹터지수 추종), 기준 KODEX 200, 보조 KODEX 반도체·2차전지.
      무료·무인증 일별 이력은 yfinance(.KS)뿐이다 — KRX 지수 원천(FDR 'KRX-INDEX', pykrx)은 로그인 필요.
      상장일: IT·금융·에너지화학·중공업·철강소재·건설·경기소비재 2011-04, 헬스케어·산업재·생활소비재
      2015-09, 커뮤니케이션서비스 2019-01. 260거래일 미만인 섹터는 순위에서 빠진다.

배분에 넣지 않는 이유 (python -m beta_model.sectors --backtest [--universe KR]):
  US 1999~2026: 주식 중 일부(고정 40% 또는 추세 점수 비례 0~40%)를 상위 2~4개 섹터에 월 1회 배분하는 규칙을
  SPY 단독과 비교 → 표본 내(1999~2012)는 대부분 우위였지만 표본 외(2013~)는 모두 SPY보다 낮았다.
  기술주 포트 QQQ/SOXX 상대강도 기울이기도 조회 기간에 따라 결과가 뒤집혀 기각.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import CACHE_FILE, MarketData, compute_features, fetch_yfinance
from .scoring import BetaModelScoringEngine

log = logging.getLogger(__name__)
TOP_N = 3
SAT_MAX = 0.4   # 옵트인 위성 비중: 주식의 최대 40%를 주도 섹터에 (추세 점수 비례)


@dataclass(frozen=True)
class Universe:
    key: str
    label: str                       # 패널 제목
    bench: str                       # 가격표 열 이름(= 표시 티커)
    bench_name: str
    sectors: dict[str, str]          # 열 이름 → 섹터 이름
    extra: dict[str, str]            # 보조 표시 (배분 무관)
    extra_title: str
    start: datetime
    cache: Path
    symbols: dict[str, str] = field(default_factory=dict)   # 열 이름 → yfinance 심볼 (없으면 동일)

    @property
    def columns(self) -> list[str]:
        return [self.bench, *self.sectors, *self.extra]

    def yf_map(self) -> dict[str, str]:
        return {c: self.symbols.get(c, c) for c in self.columns}


US = Universe(
    key="US", label="미국 섹터 ETF", bench="SPY", bench_name="S&P 500",
    sectors={"XLK": "기술", "XLC": "커뮤니케이션", "XLY": "임의소비재", "XLF": "금융", "XLI": "산업재",
             "XLV": "헬스케어", "XLP": "필수소비재", "XLE": "에너지", "XLB": "소재", "XLU": "유틸리티", "XLRE": "부동산"},
    extra={"QQQ": "나스닥100", "SOXX": "반도체"}, extra_title="기술주 포트 구성",
    start=datetime(1998, 1, 1), cache=CACHE_FILE.parent / "sectors.pkl",
)
_KR_SECTORS = {
    "139260": "IT", "139270": "금융", "139250": "에너지화학", "139230": "중공업", "139240": "철강소재",
    "139220": "건설", "139290": "경기소비재", "227560": "생활소비재", "227540": "헬스케어",
    "227550": "산업재", "315270": "커뮤니케이션",
}
_KR_EXTRA = {"091160": "반도체", "305720": "2차전지"}
KR = Universe(
    key="KR", label="KOSPI200 섹터 ETF (TIGER 200)", bench="069500", bench_name="KODEX 200",
    sectors=_KR_SECTORS, extra=_KR_EXTRA, extra_title="주요 테마 (KODEX)",
    start=datetime(2009, 6, 1), cache=CACHE_FILE.parent / "sectors_kr.pkl",
    symbols={c: f"{c}.KS" for c in ["069500", *_KR_SECTORS, *_KR_EXTRA]},
)
UNIVERSES: dict[str, Universe] = {"US": US, "KR": KR}

# 하위 호환 (기존 import 경로)
BENCH, SECTORS, EXTRA = US.bench, US.sectors, US.extra
SECTOR_CACHE = US.cache


def _universe(u: str | Universe) -> Universe:
    return u if isinstance(u, Universe) else UNIVERSES[u]


def fetch_sectors(use_cache: bool = True, max_age_hours: float = 6, universe: str | Universe = "US") -> pd.DataFrame:
    u = _universe(universe)
    if use_cache and u.cache.exists() and (time.time() - u.cache.stat().st_mtime) / 3600 < max_age_hours:
        with u.cache.open("rb") as f:
            return pickle.load(f)
    px = fetch_yfinance(u.yf_map(), u.start)
    u.cache.parent.mkdir(parents=True, exist_ok=True)
    with u.cache.open("wb") as f:
        pickle.dump(px, f)
    return px


def momentum_score(px: pd.DataFrame) -> pd.DataFrame:
    """상대강도 점수 = (6개월 수익률 + 12개월 수익률(최근 1개월 제외)) / 2."""
    return ((px / px.shift(126) - 1) + (px.shift(21) / px.shift(252) - 1)) / 2


def sector_payload(px: pd.DataFrame, universe: str | Universe = "US") -> dict:
    u = _universe(universe)
    bench = u.bench
    px = px.dropna(how="all").ffill()   # ffill은 상장 전(선행) NaN을 채우지 않는다
    names = [c for c in [*u.sectors, *u.extra] if c in px and px[c].notna().sum() > 260]
    md = MarketData(prices=px, features={n: compute_features(px[n].dropna()).reindex(px.index)
                                         for n in [*names, bench]})
    engine = BetaModelScoringEngine(md)
    mom = momentum_score(px[[*names, bench]])
    last = px.index[-1]
    b = px[bench]

    def row(t: str, label: str) -> dict:
        f = md.features[t].iloc[-1]
        p = px[t]
        r = lambda n: float(p.iloc[-1] / p.iloc[-1 - n] - 1)  # noqa: E731
        return {
            "ticker": t, "name": label,
            "trend": round(float(engine.index_trend(t).trend_score.iloc[-1]), 4),
            "score": round(float(mom[t].iloc[-1]), 4),
            "rel_6m": round(float((p.iloc[-1] / p.iloc[-127]) / (b.iloc[-1] / b.iloc[-127]) - 1), 4),
            "ret_1m": round(r(21), 4), "ret_3m": round(r(63), 4), "ret_6m": round(r(126), 4),
            "drawdown": round(float(f.drawdown), 4),
            "above_sma200": bool(p.iloc[-1] > p.rolling(200).mean().iloc[-1]),
        }

    secs = [t for t in u.sectors if t in names]
    rows = sorted((row(t, u.sectors[t]) for t in secs), key=lambda x: x["score"], reverse=True)
    leaders = [x["ticker"] for x in rows if x["trend"] > 0 and x["above_sma200"]][:TOP_N]
    for i, x in enumerate(rows):
        x["rank"] = i + 1
        x["leader"] = x["ticker"] in leaders
    # 순위 추이: 최근 1년 월말 기준 순위
    recent = px.index[px.index.searchsorted(last - pd.DateOffset(days=370)):]
    me = recent.to_series().groupby(recent.to_period("M")).max().tolist()
    rank_hist = {
        d.strftime("%Y-%m"): {t: int(v) for t, v in mom.loc[d, secs].rank(ascending=False).dropna().items()}
        for d in me
    }
    return {
        "universe": u.key,
        "label": u.label,
        "as_of": last.strftime("%Y-%m-%d"),
        "benchmark": {"ticker": bench, "name": u.bench_name,
                      "ret_6m": round(float(b.iloc[-1] / b.iloc[-127] - 1), 4),
                      "trend": round(float(engine.index_trend(bench).trend_score.iloc[-1]), 4)},
        "sectors": rows,
        "n_sectors": len(rows),
        "leaders": leaders,
        "extras_title": u.extra_title,
        "tech": [row(t, n) for t, n in u.extra.items() if t in names],
        "rank_history": rank_hist,
    }


# ---------------------------------------------------------------------------
# 배분 규칙 백테스트 (채택하지 않은 근거)
# ---------------------------------------------------------------------------
def backtest_rotation(px: pd.DataFrame, trend: pd.Series, n: int = 3, max_sat: float = 0.4,
                      mode: str = "trend", cost_bps: float = 10, universe: str | Universe = "US") -> pd.Series:
    """주식 = (1−f)·기준 + f·상위 n개 섹터 균등. f = max_sat·clip(추세,0,1) 또는 고정(mode='fixed').
    mode='full' 이면 100% 섹터 로테이션. 월말 신호 → 익일 집행. 상장 전 섹터는 후보에서 제외."""
    u = _universe(universe)
    px = px.ffill()
    ret = px.pct_change(fill_method=None)
    secs = [s for s in u.sectors if s in px]
    score = momentum_score(px[secs])
    ok = px[secs] > px[secs].rolling(200).mean()
    month_ends = px.index.to_series().groupby(px.index.to_period("M")).max()
    start = px[u.bench].first_valid_index()
    first = px.index[px.index.get_loc(start) + 260]
    rows = {}
    for d in month_ends:
        if d < first:
            continue
        top = score.loc[d].where(ok.loc[d]).dropna().sort_values(ascending=False).index[:n]
        f = {"trend": max_sat * float(np.clip(trend.get(d, 0.0), 0, 1)), "fixed": max_sat, "full": 1.0}[mode]
        w = pd.Series(0.0, index=[*secs, u.bench])
        w[list(top)] = f / n          # 후보가 n개 미만이면 남는 몫은 기준 지수
        w[u.bench] = 1 - w.sum()
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
    ap.add_argument("--universe", choices=list(UNIVERSES), default="US")
    ap.add_argument("--backtest", action="store_true", help="섹터 배분 규칙 표본 내/외 비교")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    u = UNIVERSES[args.universe]
    px = fetch_sectors(universe=u)
    if not args.backtest:
        p = sector_payload(px, u)
        for x in p["sectors"]:
            print(f"{x['rank']:2} {x['ticker']:6} {x['name']:8} 점수 {x['score']:+.1%} 추세 {x['trend']:+.2f} "
                  f"6M 대비 {u.bench} {x['rel_6m']:+.1%} {'★' if x['leader'] else ''}")
        return
    from .data_pipeline import load_market_data
    trend = BetaModelScoringEngine(load_market_data()).scores(u.key).trend_score.reindex(px.index).ffill()
    bench = px[u.bench].ffill().pct_change(fill_method=None)
    split = "2012" if u.key == "US" else "2018"

    def st(r):
        if len(r) < 60:
            return "–"
        eq = (1 + r).cumprod()
        mdd = (eq / eq.cummax() - 1).min()
        return f"CAGR {eq.iloc[-1] ** (252 / len(r)) - 1:6.1%} Sh {r.mean() / r.std() * 252 ** .5:4.2f} MDD {mdd:6.1%}"
    runs = [(u.bench, None), *[(f"{m} n{n}", backtest_rotation(px, trend, n=n, mode=m, universe=u))
                              for m in ("trend", "fixed") for n in (2, 3, 4)],
            ("full n3", backtest_rotation(px, trend, n=3, mode="full", universe=u))]
    base_idx = runs[1][1].index
    print(f"표본 내 ~{split} | 표본 외 {int(split) + 1}~  (시작 {base_idx[0]:%Y-%m-%d})")
    for name, r in runs:
        r = bench.loc[base_idx] if r is None else r
        print(f"{name:10} 표본내 {st(r[:split])} | 표본외 {st(r[str(int(split) + 1):])}")


if __name__ == "__main__":
    main()
