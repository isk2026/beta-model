"""채택 전 실험: 기존 파라미터 선택 방식(표본 내 선택 → 표본 외 확인)을 그대로 따른다.

    python -m beta_model.experiments long-trend   # 장기 추세 요소(12-1개월 모멘텀 / 200일선) 추가 효과
    python -m beta_model.experiments vol-proxy    # VKOSPI proxy 처리(keep / drop) 비교
    (공통 옵션: --refresh 재수집, --split 2013-01-01)

규칙
- 선택은 표본 내(< split)만 본다: 미국·한국 평균 샤프가 최고치와 0.01 이내인 조합 중 회전율 최저 (backtest.py와 같음).
- 표본 외(≥ split)와 미국(기술)(파라미터 선택에 쓰지 않은 시장)은 확인만 한다.
- 장기 추세는 워밍업(최대 252일)이 길어, 모든 조합을 시장별 공통 시작일로 잘라 같은 기간끼리 비교한다.
- 예측력 지표 (포지션 시차와 같게 t+1일 종가부터 측정):
    · 추세 IC = 추세 점수와 향후 h일 수익률의 순위상관 (h = 20, 63)
    · 위험 AUC = 위험 점수로 '향후 63일 안에 −10% 이상 하락'을 구분하는 정도 (0.5 = 무작위)
  겹치는 구간 때문에 일별 표본 수가 실제 정보량보다 크다 — 차이가 작으면(IC ±0.02, AUC ±0.01) 같은 것으로 본다.

결과는 화면 출력 + data/experiments/<이름>.csv. 채택은 config.py 값을 직접 바꾸는 수동 결정으로 남긴다.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from . import config
from .backtest import SPLIT_DATE, ModelSpec, metrics, simulate
from .data_pipeline import ROOT, MarketData, load_market_data

OUT_DIR = ROOT / "data" / "experiments"
SELECT_TOL = 0.01          # 표본 내 샤프가 최고치와 이 차이 이내면 '같은 성능'으로 보고 회전율 최저를 고른다
AUC_HORIZON, AUC_DROP = 63, -0.10

# 장기 추세 그리드: 종류별 스케일 × 가중치. 가중치는 기존 세 요소(합 1)에 더한 뒤 재정규화되므로
# 0.25 → 장기 20%, 기존 요소 각 ×0.8 (시장 폭을 위험 점수에 넣을 때와 같은 방식).
LONG_GRID = {"tsmom": [0.15, 0.30], "sma200": [0.05, 0.10]}
LONG_WEIGHTS = [0.15, 0.25, 0.40]


# ---------------------------------------------------------------------------
# 예측력 지표
# ---------------------------------------------------------------------------
def forward_return(price: pd.Series, h: int, lag: int = 1) -> pd.Series:
    """t일 값 = t+lag일 종가 → t+lag+h일 종가 수익률 (t일 신호를 t+lag일에 집행)."""
    return price.shift(-(lag + h)) / price.shift(-lag) - 1


def forward_drop_event(price: pd.Series, h: int = AUC_HORIZON, drop: float = AUC_DROP, lag: int = 1) -> pd.Series:
    """t일 값 = t+lag일 종가 대비 이후 h거래일 안의 최저가가 `drop` 이하인지 (1/0). 끝부분은 NaN."""
    future_min = price.rolling(h).min().shift(-(lag + h))       # min(p[t+lag+1 .. t+lag+h])
    ev = (future_min / price.shift(-lag) - 1 <= drop).astype(float)
    return ev.where(future_min.notna() & price.shift(-lag).notna())


def auc(score: pd.Series, label: pd.Series) -> float:
    """순위 기반 AUC (Mann-Whitney). 동률은 평균 순위."""
    d = pd.DataFrame({"s": score, "y": label}).dropna()
    pos, neg = (d.y == 1).sum(), (d.y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = d.s.rank()
    return float((r[d.y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def rank_ic(score: pd.Series, fwd: pd.Series) -> float:
    d = pd.DataFrame({"s": score, "f": fwd}).dropna()
    return float(d.s.rank().corr(d.f.rank())) if len(d) > 30 else float("nan")


def _parts(df: pd.DataFrame, split: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {"is": df[df.index < split], "oos": df[df.index >= split], "full": df}


def evaluate_frame(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    """simulate() 결과 하나 → 구간별 성과 + 예측력."""
    fwd = {h: forward_return(df["price"], h) for h in (20, 63)}
    event = forward_drop_event(df["price"])
    out = {}
    for part, d in _parts(df, split).items():
        if len(d) < config.TRADING_DAYS:
            continue
        m = metrics(d)
        out |= {f"{part}_sharpe": m["sharpe"], f"{part}_mdd": m["mdd"], f"{part}_cagr": m["cagr"],
                f"{part}_turnover": m["turnover"],
                f"{part}_ic20": rank_ic(d["trend"], fwd[20].reindex(d.index)),
                f"{part}_ic63": rank_ic(d["trend"], fwd[63].reindex(d.index)),
                f"{part}_auc": auc(d["risk"], event.reindex(d.index))}
    return out


# ---------------------------------------------------------------------------
# 공통 실행: 조합 × 시장 → 같은 기간으로 잘라 평가 → 표본 내 선택
# ---------------------------------------------------------------------------
def run_specs(md: MarketData, specs: list[ModelSpec], split: str = SPLIT_DATE,
              markets: tuple[str, ...] | None = None) -> pd.DataFrame:
    split_ts = pd.Timestamp(split)
    markets = markets or tuple(config.MARKETS)
    sims = {s.name: {m: simulate(md, m, s) for m in markets} for s in specs}
    rows = {s.name: {"spec": s.name} for s in specs}
    for m in markets:
        start = max(sims[s.name][m]["trend"].first_valid_index() for s in specs)
        for s in specs:
            df = sims[s.name][m].loc[start:].dropna(subset=["trend", "risk"])
            rows[s.name] |= {f"{m}_{k}": v for k, v in evaluate_frame(df, split_ts).items()}
            rows[s.name][f"{m}_start"] = start.strftime("%Y-%m-%d")
    out = pd.DataFrame(rows.values())
    cal = [m for m in config.CALIBRATION_MARKETS if m in markets]
    for k in ("sharpe", "turnover", "ic63", "auc"):
        for part in ("is", "oos"):
            cols = [f"{m}_{part}_{k}" for m in cal if f"{m}_{part}_{k}" in out]
            out[f"cal_{part}_{k}"] = out[cols].mean(axis=1) if cols else np.nan
    return out


def select(table: pd.DataFrame, tol: float = SELECT_TOL) -> str:
    """표본 내 평균 샤프가 최고치와 tol 이내인 조합 중 표본 내 회전율 최저 (동률이면 표 순서 = 기준선 우선)."""
    best = table["cal_is_sharpe"].max()
    cands = table[table["cal_is_sharpe"] >= best - tol]
    return str(cands.loc[cands["cal_is_turnover"].idxmin(), "spec"])


def verdict(table: pd.DataFrame, chosen: str, baseline: str, check_market: str | None = "US_TECH") -> list[str]:
    """채택 권고: 표본 내 선택이 기준선이 아니고, 표본 외에서 두 보정 시장 모두 샤프가 떨어지지 않으며,
    맞춰보지 않은 시장(미국(기술))에서도 0.02 이상 나빠지지 않을 때만 '채택 검토'."""
    b = table.set_index("spec").loc[baseline]
    c = table.set_index("spec").loc[chosen]
    lines = [f"표본 내 선택: {chosen}" + (" (= 기준선)" if chosen == baseline else "")]
    if chosen == baseline:
        return lines + ["→ 기준선 유지. 표본 내에서 더 나은 조합이 없음."]
    oos_ok = all(c[f"{m}_oos_sharpe"] >= b[f"{m}_oos_sharpe"] - 1e-9 for m in config.CALIBRATION_MARKETS
                 if f"{m}_oos_sharpe" in c)
    lines.append("표본 외 샤프 (기준선 → 선택): " + ", ".join(
        f"{m} {b[f'{m}_oos_sharpe']:.3f} → {c[f'{m}_oos_sharpe']:.3f}" for m in config.CALIBRATION_MARKETS
        if f"{m}_oos_sharpe" in c))
    ext_ok = True
    if check_market and f"{check_market}_full_sharpe" in c:
        ext_ok = c[f"{check_market}_full_sharpe"] >= b[f"{check_market}_full_sharpe"] - 0.02
        lines.append(f"{check_market}(맞춰보지 않은 시장) 전체 샤프: "
                     f"{b[f'{check_market}_full_sharpe']:.3f} → {c[f'{check_market}_full_sharpe']:.3f}")
    if oos_ok and ext_ok:
        lines.append("→ 채택 검토: 표본 외·미사용 시장에서 같은 방향. 민감도·부트스트랩(validation)으로 한 번 더 확인 후 config 반영.")
    else:
        lines.append("→ 기각 권고: 표본 내 개선이 표본 외 또는 미사용 시장에서 유지되지 않음 (과적합 가능성).")
    return lines


# ---------------------------------------------------------------------------
# 실험 1: 장기 추세 요소
# ---------------------------------------------------------------------------
def long_trend_specs() -> list[ModelSpec]:
    specs = [ModelSpec("기준선")]
    for kind, scales in LONG_GRID.items():
        for scale in scales:
            for w in LONG_WEIGHTS:
                specs.append(ModelSpec(f"{kind}/s{scale}/w{w}",
                                       trend=replace(config.TREND, w_long=w, long_kind=kind, long_scale=scale)))
    return specs


def long_trend(md: MarketData, split: str = SPLIT_DATE) -> tuple[pd.DataFrame, list[str]]:
    table = run_specs(md, long_trend_specs(), split)
    chosen = select(table)
    return table, verdict(table, chosen, "기준선")


# ---------------------------------------------------------------------------
# 실험 2: VKOSPI proxy 처리
# ---------------------------------------------------------------------------
def vol_proxy(md: MarketData, split: str = SPLIT_DATE) -> tuple[pd.DataFrame, list[str]]:
    specs = [ModelSpec("keep"), ModelSpec("drop", risk=replace(config.RISK, vol_proxy_mode="drop"))]
    table = run_specs(md, specs, split, markets=("KR",))
    b, d = table.set_index("spec").loc["keep"], table.set_index("spec").loc["drop"]
    from .scoring import BetaModelScoringEngine
    share = float(BetaModelScoringEngine(md).vol_is_proxy("VKOSPI").mean())
    lines = [f"VKOSPI proxy 비율: 전체 기간의 {share:.0%} (출처 {md.meta.get('vkospi_source')})"]
    for part, lab in (("is", "표본 내"), ("oos", "표본 외")):
        lines.append(f"{lab}: AUC {b[f'KR_{part}_auc']:.3f} → {d[f'KR_{part}_auc']:.3f} · "
                     f"샤프 {b[f'KR_{part}_sharpe']:.3f} → {d[f'KR_{part}_sharpe']:.3f} · "
                     f"MDD {b[f'KR_{part}_mdd']:.1%} → {d[f'KR_{part}_mdd']:.1%} · "
                     f"회전율 {b[f'KR_{part}_turnover']:.0%} → {d[f'KR_{part}_turnover']:.0%}")
    is_ok = d["KR_is_auc"] >= b["KR_is_auc"] - 0.005 and d["KR_is_sharpe"] >= b["KR_is_sharpe"] - SELECT_TOL
    oos_ok = d["KR_oos_auc"] >= b["KR_oos_auc"] - 0.005 and d["KR_oos_sharpe"] >= b["KR_oos_sharpe"] - SELECT_TOL
    if share == 0:
        lines.append("→ proxy 구간 없음: 두 설정이 같다 (KRX 실제 VKOSPI 사용 중).")
    elif is_ok and oos_ok:
        lines.append("→ drop 채택 검토: 이중 계산을 없애도 예측력·성과가 유지됨 (단순한 쪽 선호). "
                     "config.RISK.vol_proxy_mode = \"drop\"")
    else:
        lines.append("→ keep 유지: proxy를 빼면 표본 내 또는 표본 외에서 예측력·성과가 떨어짐. "
                     "근본 해결은 KRX 인증키로 실제 VKOSPI 사용.")
    return table, lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SHOW = ["spec", "cal_is_sharpe", "cal_oos_sharpe", "cal_is_turnover", "cal_is_ic63", "cal_oos_ic63",
        "US_is_sharpe", "US_oos_sharpe", "KR_is_sharpe", "KR_oos_sharpe", "US_oos_mdd", "KR_oos_mdd",
        "US_TECH_full_sharpe", "US_TECH_full_mdd"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Beta Model 채택 전 실험")
    ap.add_argument("name", choices=["long-trend", "vol-proxy"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--split", default=SPLIT_DATE)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    md = load_market_data(use_cache=not args.refresh)

    table, lines = (long_trend if args.name == "long-trend" else vol_proxy)(md, args.split)
    cols = [c for c in SHOW if c in table] if args.name == "long-trend" else \
        ["spec"] + [c for c in table if c.startswith("KR_") and not c.endswith("start")]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table[cols].round(3).to_string(index=False))
    print("\n" + "\n".join(lines))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.name}.csv"
    table.to_csv(path, index=False)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
