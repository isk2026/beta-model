"""장기 백테스트와 워크포워드 검증.

    python -m beta_model.backtest            # 리포트 출력 + dashboard/public/data/backtest.json
    python -m beta_model.backtest --grid     # 파라미터 그리드(표본 내 선택 → 표본 외 검증)

가정 (보수적으로 잡음):
- t일 종가 신호 → t+1일 종가에 집행 → t+2일 수익부터 반영 (하루 추가 지연으로 미국 데이터의 시차·체결 지연을 흡수)
- 매매비용: 비중 변화 1.0당 `cost_bps` (기본 10bp, 편도)
- 현금 수익: 미국 = 3개월 국채(DTB3), 한국 = CD 91일(ECOS, 없으면 FRED 3개월 금리 월간)
- 지수는 가격지수(배당 제외). 보유 그대로와 모델 모두 같은 지수로 비교한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import config
from .allocation import smooth_weights, target_stock_weight
from .config import AllocationParams, RiskParams, TrendParams
from .data_pipeline import ROOT, MarketData, load_market_data
from .scoring import BetaModelScoringEngine

OUT_FILE = ROOT / "dashboard" / "public" / "data" / "backtest.json"
SPLIT_DATE = "2013-01-01"           # 이전 = 표본 내(파라미터 선택), 이후 = 표본 외(검증)
CASH_SERIES = {"US": "UST3M", "KR": "KR3M"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    trend: TrendParams = config.TREND
    risk: RiskParams = config.RISK
    alloc: AllocationParams = config.ALLOCATION


# 변경 전(대시보드 최초 버전) 설정. 비교 기준선.
BASELINE = ModelSpec(
    "기존 모델",
    trend=replace(config.TREND, alignment_scale=0.0, smooth_span=1),
    alloc=replace(config.ALLOCATION, band_in_execution=False),
)


def simulate(md: MarketData, market: str, spec: ModelSpec, cost_bps: float = 10.0) -> pd.DataFrame:
    """일별 모델 수익률과 보유 그대로(Buy & Hold) 수익률."""
    engine = BetaModelScoringEngine(md, spec.trend, spec.risk)
    sc = engine.scores(market)
    target = pd.Series(
        [target_stock_weight(t, r, spec.alloc) for t, r in zip(sc.trend_score, sc.risk_score)], index=sc.index,
    )
    executed = smooth_weights(target, spec.alloc)          # t-1 신호 → t 집행
    main = config.MARKETS[market]["indices"][0]
    px = md.features[main]["close"].reindex(executed.index)
    ret = px.pct_change().fillna(0.0)
    cash_rate = md.prices[CASH_SERIES[market]]
    if market == "KR" and "CD91" in md.prices:   # ECOS 일별 CD 금리가 있으면 우선
        cash_rate = md.prices["CD91"].combine_first(cash_rate)
    cash = (cash_rate.reindex(executed.index).ffill() / 100 / config.TRADING_DAYS).fillna(0.0)
    held = executed.shift(1).bfill()                        # t 집행분은 t+1 수익부터
    trade = executed.diff().abs().fillna(0.0)
    strat = held * ret + (1 - held) * cash - trade * cost_bps / 1e4
    return pd.DataFrame({
        "price": px, "target": target.reindex(executed.index), "weight": executed,
        "trend": sc.trend_score.reindex(executed.index), "risk": sc.risk_score.reindex(executed.index),
        "ret_bh": ret, "ret_model": strat, "cash": cash, "trade": trade,
    })


def metrics(df: pd.DataFrame, col: str = "ret_model") -> dict:
    r = df[col]
    yrs = len(r) / config.TRADING_DAYS
    eq = (1 + r).cumprod()
    excess = r - df["cash"]
    mdd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    out = {
        "cagr": cagr,
        "vol": float(r.std() * np.sqrt(config.TRADING_DAYS)),
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS)) if excess.std() > 0 else 0.0,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
    }
    if col == "ret_model":
        out |= {
            "turnover": float(df["trade"].sum() / yrs),
            "trade_days": float((df["trade"] > 0.005).sum() / yrs),
            "avg_weight": float(df["weight"].mean()),
        }
    return out


def drawdown_episodes(df: pd.DataFrame, n: int = 6, min_depth: float = -0.15) -> list[dict]:
    """보유 그대로 기준 최악의 낙폭 구간(고점→저점)을 자동 추출하고 각 구간의 모델 성과를 비교.

    구간을 사람이 고르지 않으므로 '잘 된 구간만 골라 보여주기'를 피할 수 있다.
    """
    eq = (1 + df["ret_bh"]).cumprod()
    eq_m = (1 + df["ret_model"]).cumprod()
    dd = eq / eq.cummax() - 1
    episodes, in_dd, start = [], False, None
    for d, v in dd.items():
        if not in_dd and v < 0:
            in_dd, start = True, eq.loc[:d].idxmax()
        elif in_dd and v == 0:
            seg = dd.loc[start:d]
            episodes.append((start, seg.idxmin(), float(seg.min())))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), float(seg.min())))
    episodes = sorted([e for e in episodes if e[2] <= min_depth], key=lambda e: e[2])[:n]
    rows = []
    for peak, trough, depth in sorted(episodes):
        rows.append({
            "peak": peak.strftime("%Y-%m-%d"), "trough": trough.strftime("%Y-%m-%d"),
            "bh": depth, "model": float(eq_m.loc[trough] / eq_m.loc[peak] - 1),
            "avg_weight": float(df["weight"].loc[peak:trough].mean()),
        })
    return rows


def static_mix(df: pd.DataFrame) -> pd.DataFrame:
    """모델과 평균 주식 비중이 같은 고정 비중 포트폴리오 (매일 리밸런싱, 비용 0).

    모델의 성과가 '타이밍' 덕분인지, 단지 평균 비중이 낮아서인지 가르는 기준선.
    """
    w = df["weight"].mean()
    out = df.copy()
    out["weight"], out["trade"] = w, 0.0
    out["ret_model"] = w * df["ret_bh"] + (1 - w) * df["cash"]
    return out


def _split_metrics(df: pd.DataFrame) -> dict:
    split = pd.Timestamp(SPLIT_DATE)
    parts = {"full": df, "in_sample": df[df.index < split], "out_sample": df[df.index >= split]}
    return {k: {"model": metrics(v), "bh": metrics(v, "ret_bh")} for k, v in parts.items()}


def evaluate(md: MarketData, spec: ModelSpec, market: str) -> dict:
    df = simulate(md, market, spec)
    return _split_metrics(df) | {"_df": df}


# ---------------------------------------------------------------------------
# 워크포워드: 표본 내(~2012)에서 고르고 표본 외(2013~)에서 확인
# ---------------------------------------------------------------------------
GRID = {
    "alignment_scale": [0.0, 0.01, 0.02, 0.04],
    "smooth_span": [1, 5, 10],
    "band_in_execution": [False, True],
}


def grid_search(md: MarketData) -> pd.DataFrame:
    rows = []
    for a_scale, span, band in itertools.product(*GRID.values()):
        spec = ModelSpec(
            f"a{a_scale}/s{span}/b{int(band)}",
            trend=replace(config.TREND, alignment_scale=a_scale, smooth_span=span),
            alloc=replace(config.ALLOCATION, band_in_execution=band),
        )
        row = {"alignment_scale": a_scale, "smooth_span": span, "band_in_execution": band}
        for m in config.MARKETS:
            ev = evaluate(md, spec, m)
            for part in ("in_sample", "out_sample"):
                for k in ("sharpe", "calmar", "turnover", "mdd", "cagr"):
                    row[f"{m}_{part[:2]}_{k}"] = ev[part]["model"][k]
        # 선택 기준: 두 시장 표본 내 샤프 평균 (비용 차감 후라 회전율도 반영됨)
        row["is_score"] = np.mean([row[f"{m}_in_sharpe"] for m in config.MARKETS])
        row["oos_score"] = np.mean([row[f"{m}_ou_sharpe"] for m in config.MARKETS])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("is_score", ascending=False)


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def _weekly_curve(df: pd.DataFrame) -> list[dict]:
    eq_bh = (1 + df["ret_bh"]).cumprod()
    eq_m = (1 + df["ret_model"]).cumprod()
    w = pd.DataFrame({"bh": eq_bh, "model": eq_m, "weight": df["weight"]}).resample("W-FRI").last().dropna()
    return [{"date": d.strftime("%Y-%m-%d"), "bh": round(r.bh, 4), "model": round(r.model, 4), "weight": round(r.weight, 3)}
            for d, r in w.iterrows()]


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 4)
    return obj


def build_report(md: MarketData, specs: list[ModelSpec]) -> dict:
    report = {
        "split_date": SPLIT_DATE,
        "assumptions": "종가 신호 → 익일 집행 → 그다음 날 수익 반영, 매매비용 10bp, 현금 = 3개월 금리, 가격지수(배당 제외)",
        "current": specs[-1].name,
        "markets": {},
    }
    for m in config.MARKETS:
        mk = {"label": config.MARKETS[m]["label"], "index": config.MARKETS[m]["indices"][0], "specs": {}}
        for spec in specs:
            ev = evaluate(md, spec, m)
            df = ev.pop("_df")
            mk["specs"][spec.name] = ev
            if spec is specs[-1]:
                static = static_mix(df)
                mk["specs"]["고정 비중"] = _split_metrics(static)
                mk["static_weight"] = float(df["weight"].mean())
                mk["start"] = df.index[0].strftime("%Y-%m-%d")
                mk["curve"] = _weekly_curve(df)
                static_eq = (1 + static["ret_model"]).cumprod()
                mk["episodes"] = [
                    e | {"static": float(static_eq.loc[e["trough"]] / static_eq.loc[e["peak"]] - 1)}
                    for e in drawdown_episodes(df)
                ]
        report["markets"][m] = mk
    return _round(report)


def print_report(report: dict) -> None:
    pct = lambda x: f"{x:7.1%}" if x is not None else "    n/a"  # noqa: E731
    for m, mk in report["markets"].items():
        print(f"\n=== {mk['label']} ({mk['index']}, {mk['start']}~) ===")
        print(f"{'':16}{'구간':>6} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7} {'회전율':>8} {'매매일':>6}")
        for part, lab in [("full", "전체"), ("in_sample", "표본내"), ("out_sample", "표본외")]:
            bh = next(iter(mk["specs"].values()))[part]["bh"]
            print(f"{'보유 그대로':14}{lab:>6} {pct(bh['cagr'])} {pct(bh['mdd'])} {bh['sharpe']:7.2f} {bh['calmar']:7.2f}")
            for name, ev in mk["specs"].items():
                r = ev[part]["model"]
                print(f"{name:14}{lab:>6} {pct(r['cagr'])} {pct(r['mdd'])} {r['sharpe']:7.2f} {r['calmar']:7.2f} "
                      f"{r['turnover']:8.0%} {r['trade_days']:6.0f}")
        print("  최악 낙폭 구간 (보유 vs 모델):")
        for e in mk["episodes"]:
            print(f"   {e['peak']} → {e['trough']}  보유 {e['bh']:7.1%}  모델 {e['model']:7.1%}  "
                  f"고정 {e['static']:7.1%}  평균비중 {e['avg_weight']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Beta Model 백테스트")
    ap.add_argument("--grid", action="store_true", help="워크포워드 그리드 탐색 결과 출력")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    md = load_market_data(use_cache=not args.refresh)

    if args.grid:
        g = grid_search(md)
        cols = ["alignment_scale", "smooth_span", "band_in_execution", "is_score", "oos_score"] + \
            [f"{m}_{p}_{k}" for m in config.MARKETS for p in ("in", "ou") for k in ("sharpe", "turnover")]
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(g[cols].round(3).to_string(index=False))
        return

    report = build_report(md, [BASELINE, ModelSpec("현재 모델")])
    print_report(report)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
