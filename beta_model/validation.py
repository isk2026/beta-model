"""과적합 방지·견고성 검증.

    python -m beta_model.validation            # 전체 실행 (수 분) → dashboard/public/data/validation.json
    python -m beta_model.validation --quick    # 민감도 무작위 표본 축소

1. 다른 나라 지수   : 파라미터를 전혀 바꾸지 않고 일본·독일·영국·홍콩·대만에 적용 (진짜 표본 외)
2. 롤링 워크포워드  : 5년 단위로 '그 이전 데이터만으로' 파라미터를 고르고 다음 5년에 적용
3. 블록 부트스트랩  : 수익률을 63일 블록으로 재표본 → 고정 비중 대비 우위가 우연인지 신뢰구간
4. 민감도          : 파라미터를 하나씩 ×0.5/×1.5, 그리고 전부 동시에 무작위 ±30%
5. 비용·지연 스트레스: 매매비용 10/30/50bp, 집행 지연 +0/1/2일
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pickle
from dataclasses import fields, replace
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .backtest import GRID, ModelSpec, metrics, simulate, static_mix
from .data_pipeline import CACHE_FILE, ROOT, build_market_data, fetch_raw, fetch_yfinance

log = logging.getLogger(__name__)
OUT_FILE = ROOT / "dashboard" / "public" / "data" / "validation.json"
EXT_CACHE = CACHE_FILE.parent / "external.pkl"

# 위험 가중치에서 fx(원/달러)는 뺀다 — 한국 전용 입력이므로. 현금 수익은 0 (보수적).
_EXT_W = {"vol_index": 0.35, "realized_vol": 0.15, "rates": 0.15, "credit": 0.25}
EXTERNAL: dict[str, dict] = {
    "JP": {"label": "일본 닛케이225", "ticker": "^N225"},
    "DE": {"label": "독일 DAX", "ticker": "^GDAXI"},
    "UK": {"label": "영국 FTSE100", "ticker": "^FTSE"},
    "HK": {"label": "홍콩 항셍", "ticker": "^HSI"},
    "TW": {"label": "대만 가권", "ticker": "^TWII"},
}
EXT_MARKETS = {
    k: {"label": v["label"], "indices": [k], "vol_index": "VIX", "cash": "ZERO", "weights": _EXT_W}
    for k, v in EXTERNAL.items()
}

FOLD_STARTS = ["2001-01-01", "2006-01-01", "2011-01-01", "2016-01-01", "2021-01-01"]


def _r(x, n=4):
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


def _summary(df: pd.DataFrame) -> dict:
    m, s, b = metrics(df), metrics(static_mix(df)), metrics(df, "ret_bh")
    keys = ("cagr", "mdd", "sharpe", "sortino", "down_capture")
    return {
        "model": {k: m.get(k) for k in (*keys, "turnover")},
        "static": {k: s.get(k) for k in keys},
        "bh": {k: b.get(k) for k in ("cagr", "mdd", "sharpe", "sortino")},
        "avg_weight": float(df["weight"].mean()),
        "start": df.index[0].strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# 1. 다른 나라 지수
# ---------------------------------------------------------------------------
def external_markets(raw: pd.DataFrame, spec: ModelSpec) -> dict:
    if EXT_CACHE.exists():
        with EXT_CACHE.open("rb") as f:
            ext = pickle.load(f)
    else:
        ext = fetch_yfinance({k: v["ticker"] for k, v in EXTERNAL.items()}, datetime.fromisoformat(config.HISTORY_START))
        with EXT_CACHE.open("wb") as f:
            pickle.dump(ext, f)
    common = raw.drop(columns=[c for c in ext.columns if c in raw]).join(ext, how="outer")
    common["ZERO"] = 0.0
    out = {}
    for k in EXTERNAL:
        if k not in common or common[k].notna().sum() < 1000:
            continue
        md = build_market_data(common, markets={k: EXT_MARKETS[k]})
        df = simulate(md, k, spec, markets=EXT_MARKETS).dropna(subset=["trend", "risk"])
        out[k] = {"label": EXT_MARKETS[k]["label"], **_summary(df)}
    return out


# ---------------------------------------------------------------------------
# 2. 롤링 워크포워드
# ---------------------------------------------------------------------------
def _grid_specs() -> list[tuple[dict, ModelSpec]]:
    specs = []
    for a, s, b in itertools.product(*GRID.values()):
        params = {"alignment_scale": a, "smooth_span": s, "band_in_execution": b}
        specs.append((params, ModelSpec(
            f"a{a}/s{s}/b{int(b)}",
            trend=replace(config.TREND, alignment_scale=a, smooth_span=s),
            alloc=replace(config.ALLOCATION, band_in_execution=b),
        )))
    return specs


def _select(sims: list[dict[str, pd.DataFrame]], end: pd.Timestamp) -> int:
    """end 이전 데이터만으로: 샤프 평균이 최고치와 0.01 이내인 조합 중 회전율 최저."""
    rows = []
    for i, by_mkt in enumerate(sims):
        sh, to = [], []
        for m, df in by_mkt.items():
            if m not in config.CALIBRATION_MARKETS:
                continue
            d = df[df.index < end]
            if len(d) < config.TRADING_DAYS * 3:
                continue
            m = metrics(d)
            sh.append(m["sharpe"])
            to.append(m["turnover"])
        rows.append((i, np.mean(sh), np.mean(to)))
    best = max(r[1] for r in rows)
    return min((r for r in rows if r[1] >= best - 0.01), key=lambda r: r[2])[0]


def walk_forward(md) -> dict:
    grid = _grid_specs()
    sims = [{m: simulate(md, m, spec) for m in config.MARKETS} for _, spec in grid]
    folds, stitched = [], {m: [] for m in config.MARKETS}
    bounds = [pd.Timestamp(d) for d in FOLD_STARTS] + [pd.Timestamp.max]
    for start, end in zip(bounds[:-1], bounds[1:]):
        i = _select(sims, start)
        folds.append({"test_from": start.strftime("%Y"), "chosen": grid[i][0]})
        for m in config.MARKETS:
            stitched[m].append(sims[i][m][(sims[i][m].index >= start) & (sims[i][m].index < end)])
    out = {"folds": folds, "markets": {}}
    for m, parts in stitched.items():
        df = pd.concat(parts)
        out["markets"][m] = {"label": config.MARKETS[m]["label"], **_summary(df)}
    return out


# ---------------------------------------------------------------------------
# 3. 블록 부트스트랩
# ---------------------------------------------------------------------------
def _mdd(paths: np.ndarray) -> np.ndarray:
    eq = np.cumprod(1 + paths, axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)


def bootstrap(df: pd.DataFrame, n: int = 1000, block: int = 63, seed: int = 7) -> dict:
    """고정 블록 부트스트랩. 모델과 고정 비중을 같은 날짜 표본으로 짝지어 비교."""
    rng = np.random.default_rng(seed)
    st = static_mix(df)
    a, b, rf = df["ret_model"].to_numpy(), st["ret_model"].to_numpy(), df["rf"].to_numpy()
    T = len(a)
    starts = rng.integers(0, T - block, size=(n, T // block + 1))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    ra, rb, rr = a[idx], b[idx], rf[idx]
    d_mdd = _mdd(ra) - _mdd(rb)                  # 양수 = 모델 낙폭이 더 작음
    sh = lambda r: (r - rr).mean(1) / (r - rr).std(1) * np.sqrt(config.TRADING_DAYS)  # noqa: E731
    d_sh = sh(ra) - sh(rb)
    return {
        "n": n, "block_days": block,
        "p_mdd_better": float((d_mdd > 0).mean()),
        "mdd_diff_ci": [float(np.percentile(d_mdd, 5)), float(np.median(d_mdd)), float(np.percentile(d_mdd, 95))],
        "p_sharpe_better": float((d_sh > 0).mean()),
        "sharpe_diff_ci": [float(np.percentile(d_sh, 5)), float(np.median(d_sh)), float(np.percentile(d_sh, 95))],
    }


# ---------------------------------------------------------------------------
# 4. 민감도
# ---------------------------------------------------------------------------
PERTURB: list[tuple[str, str]] = [
    ("trend", "w_alignment"), ("trend", "w_drawdown"), ("trend", "w_momentum"),
    ("trend", "dd_neutral"), ("trend", "dd_floor"), ("trend", "roc_scale"),
    ("trend", "alignment_scale"), ("trend", "smooth_span"),
    ("risk", "fx_spike_full"), ("risk", "rate_spike_full_bp"),
    ("risk", "credit_level_lo"), ("risk", "credit_level_hi"), ("risk", "weights"), ("risk", "vol_level_hi"),
    ("alloc", "trend_span"), ("alloc", "risk_span"), ("alloc", "rebalance_band"), ("alloc", "max_daily_increase"),
]
LABELS = {
    "w_alignment": "추세: 정배열 가중", "w_drawdown": "추세: 낙폭 가중", "w_momentum": "추세: 모멘텀 가중",
    "dd_neutral": "추세: 낙폭 중립점", "dd_floor": "추세: 낙폭 하한", "roc_scale": "추세: 모멘텀 스케일",
    "alignment_scale": "추세: 이격도 스케일", "smooth_span": "추세: 평활 기간",
    "fx_spike_full": "위험: 환율 급등 기준", "rate_spike_full_bp": "위험: 금리 급등 기준",
    "credit_level_lo": "위험: 신용 하단", "credit_level_hi": "위험: 신용 상단",
    "weights": "위험: 구성 가중치(무작위)", "vol_level_hi": "위험: 변동성 상단",
    "trend_span": "배분: 추세 민감도", "risk_span": "배분: 위험 차감폭",
    "rebalance_band": "배분: 무시 구간", "max_daily_increase": "배분: 일일 증가 한도",
}


def _scaled(spec: ModelSpec, group: str, name: str, k: float, rng: np.random.Generator | None = None) -> ModelSpec:
    obj = getattr(spec, group)
    val = getattr(obj, name)
    if name == "weights":  # 구성요소마다 독립적으로 흔든 뒤 합 1로 정규화
        rng = rng or np.random.default_rng(0)
        new = {}
        for m, w in val.items():
            raw = {c: v * rng.uniform(1 - abs(1 - k), 1 + abs(1 - k)) for c, v in w.items()}
            tot = sum(raw.values())
            new[m] = {c: v / tot for c, v in raw.items()}
        val = new
    elif isinstance(val, dict):
        val = {kk: vv * k for kk, vv in val.items()}
    elif isinstance(val, bool):
        return spec
    elif isinstance(val, int):
        val = max(1, int(round(val * k)))
    else:
        val = val * k
    return replace(spec, **{group: replace(obj, **{name: val})})


def _score(md, spec: ModelSpec) -> dict:
    out = {}
    for m in config.MARKETS:
        df = simulate(md, m, spec)
        r, s = metrics(df), metrics(static_mix(df))
        out[m] = {"sharpe": r["sharpe"], "mdd": r["mdd"], "turnover": r["turnover"], "mdd_vs_static": r["mdd"] - s["mdd"]}
    return out


def sensitivity(md, n_random: int = 60) -> dict:
    base = ModelSpec("현재")
    one_at_a_time = []
    for group, name in PERTURB:
        res = {k: _score(md, _scaled(base, group, name, k)) for k in (0.5, 1.5)}
        one_at_a_time.append({"param": LABELS[name], "lo": res[0.5], "hi": res[1.5]})
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(n_random):
        spec = base
        for group, name in PERTURB:
            spec = _scaled(spec, group, name, rng.uniform(0.7, 1.3), rng)
        draws.append(_score(md, spec))
    joint = {}
    for m in config.MARKETS:
        arr = {k: np.array([d[m][k] for d in draws]) for k in ("sharpe", "mdd", "turnover", "mdd_vs_static")}
        joint[m] = {k: [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))] for k, v in arr.items()}
        joint[m]["p_mdd_better_than_static"] = float((arr["mdd_vs_static"] > 0).mean())
    return {"base": _score(md, base), "one_at_a_time": one_at_a_time, "joint": joint, "n_random": n_random}


# ---------------------------------------------------------------------------
# 5. 비용·지연
# ---------------------------------------------------------------------------
def cost_delay(md) -> dict:
    out = {}
    for m in config.MARKETS:
        rows = []
        for cost, delay in [(10, 0), (30, 0), (50, 0), (10, 1), (10, 2), (30, 1)]:
            df = simulate(md, m, ModelSpec("x"), cost_bps=cost, delay=delay)
            r, s = metrics(df), metrics(static_mix(df))
            rows.append({"cost_bps": cost, "delay": delay, "cagr": r["cagr"], "sharpe": r["sharpe"],
                         "mdd": r["mdd"], "static_mdd": s["mdd"], "static_sharpe": s["sharpe"]})
        out[m] = rows
    return out


def run(quick: bool = False) -> dict:
    raw, meta = fetch_raw()
    md = build_market_data(raw, meta)
    spec = ModelSpec("현재 모델")
    log.warning("1/5 다른 나라 지수")
    ext = external_markets(raw, spec)
    log.warning("2/5 롤링 워크포워드")
    wf = walk_forward(md)
    log.warning("3/5 부트스트랩")
    boot = {m: bootstrap(simulate(md, m, spec)) for m in config.MARKETS}
    log.warning("4/5 민감도")
    sens = sensitivity(md, n_random=15 if quick else 60)
    log.warning("5/5 비용·지연")
    cd = cost_delay(md)
    return _r({
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "home": {m: {"label": config.MARKETS[m]["label"], **_summary(simulate(md, m, spec))} for m in config.MARKETS},
        "external": ext, "walk_forward": wf, "bootstrap": boot, "sensitivity": sens, "cost_delay": cd,
    })


def print_summary(v: dict) -> None:
    p = lambda x: f"{x:6.1%}" if x is not None else "   n/a"  # noqa: E731
    print("\n[1] 다른 나라 지수 (파라미터 그대로, 현금 0%)")
    print(f"  {'시장':14} {'시작':>10} {'보유MDD':>7} {'고정MDD':>7} {'모델MDD':>7} {'보유Sh':>6} {'고정Sh':>6} {'모델Sh':>6} {'하락포착':>6} {'회전율':>6}")
    for src in (v["home"], v["external"]):
        for k, e in src.items():
            print(f"  {e['label']:14} {e['start']:>10} {p(e['bh']['mdd'])} {p(e['static']['mdd'])} {p(e['model']['mdd'])} "
                  f"{e['bh']['sharpe']:6.2f} {e['static']['sharpe']:6.2f} {e['model']['sharpe']:6.2f} "
                  f"{p(e['model']['down_capture'])} {p(e['model']['turnover'])}")
    print("\n[2] 롤링 워크포워드 (이전 데이터만으로 선택)")
    for f in v["walk_forward"]["folds"]:
        print(f"  {f['test_from']}~: {f['chosen']}")
    for k, e in v["walk_forward"]["markets"].items():
        print(f"  {e['label']}: 모델 Sh {e['model']['sharpe']:.2f} MDD {p(e['model']['mdd'])} | 고정 Sh {e['static']['sharpe']:.2f} "
              f"MDD {p(e['static']['mdd'])} | 보유 Sh {e['bh']['sharpe']:.2f} MDD {p(e['bh']['mdd'])}")
    print("\n[3] 블록 부트스트랩 (모델 − 고정 비중)")
    for k, b in v["bootstrap"].items():
        print(f"  {k}: MDD 우위 확률 {b['p_mdd_better']:.0%}, ΔMDD 90%구간 {[f'{x:+.1%}' for x in b['mdd_diff_ci']]} | "
              f"샤프 우위 확률 {b['p_sharpe_better']:.0%}, Δ샤프 {[f'{x:+.2f}' for x in b['sharpe_diff_ci']]}")
    print("\n[4] 민감도 — 동시 무작위 ±30% (5/50/95 분위)")
    for k, j in v["sensitivity"]["joint"].items():
        print(f"  {k}: 샤프 {[round(x, 2) for x in j['sharpe']]}, MDD {[f'{x:.1%}' for x in j['mdd']]}, "
              f"회전율 {[f'{x:.0%}' for x in j['turnover']]}, 고정 대비 MDD 우위 {j['p_mdd_better_than_static']:.0%}")
    print("  하나씩 ×0.5 / ×1.5 (샤프 US, KR):")
    for row in v["sensitivity"]["one_at_a_time"]:
        print(f"   {row['param']:18} US {row['lo']['US']['sharpe']:.2f}/{row['hi']['US']['sharpe']:.2f}  "
              f"KR {row['lo']['KR']['sharpe']:.2f}/{row['hi']['KR']['sharpe']:.2f}  "
              f"MDD US {p(row['lo']['US']['mdd'])}/{p(row['hi']['US']['mdd'])}")
    print("\n[5] 비용·지연")
    for k, rows in v["cost_delay"].items():
        for r in rows:
            print(f"  {k} 비용 {r['cost_bps']:>2}bp 지연 +{r['delay']}: CAGR {p(r['cagr'])} Sh {r['sharpe']:.2f} "
                  f"MDD {p(r['mdd'])} (고정 Sh {r['static_sharpe']:.2f} MDD {p(r['static_mdd'])})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Beta Model 견고성 검증")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    v = run(quick=args.quick)
    print_summary(v)
    OUT_FILE.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
