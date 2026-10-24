"""파이프라인 실행 → 대시보드용 JSON 생성.

    python -m beta_model.export            # 캐시(6시간) 사용
    python -m beta_model.export --refresh  # 강제 재수집
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import config
from .allocation import REGIME_INFO, Regime, allocate, regime_series, smooth_weights, target_stock_weight
from .data_pipeline import ROOT, MarketData, build_market_data, fetch_raw
from .scenarios import run_scenarios
from .signals import build_alert, live_payload, next_weight, update_log
from .scoring import BetaModelScoringEngine

OUT_FILE = ROOT / "dashboard" / "public" / "data" / "beta_model.json"
HISTORY_DAYS = 252
MACRO_COLS = ["VIX", "VKOSPI", "US10Y", "USDKRW", "DXY", "BAA_SPREAD", "KR_CREDIT", "HY_SPREAD"]


def _r(v, n: int = 4):
    return None if pd.isna(v) else round(float(v), n)


def market_payload(engine: BetaModelScoringEngine, md: MarketData, market: str) -> dict:
    sc = engine.scores(market)
    target = sc.apply(lambda r: target_stock_weight(r.trend_score, r.risk_score), axis=1)
    executed = smooth_weights(target)
    latest = sc.iloc[-1]
    alloc = allocate(latest.trend_score, latest.risk_score).to_dict()
    # 표시 국면은 히스테리시스 적용 (경계 근처에서 매일 뒤바뀌지 않도록)
    regimes = regime_series(sc.trend_score, sc.risk_score)
    alloc["regime_raw"] = alloc["regime"]
    alloc["regime"] = int(regimes.iloc[-1])
    alloc["regime_name"] = REGIME_INFO[Regime(alloc["regime"])]["name"]

    indices = {}
    for name in config.MARKETS[market]["indices"]:
        f = md.features[name].iloc[-1]
        t = engine.index_trend(name).iloc[-1]
        indices[name] = {
            "close": _r(f.close, 2),
            **{k: _r(f[k], 2) for k in ("sma20", "sma60", "sma120")},
            "drawdown": _r(f.drawdown), "vol_20d": _r(f.vol_20d),
            "roc_3m": _r(f[f"roc_{config.TREND.roc_window}"]),
            "trend_score": _r(t.trend_score),
        }

    main = config.MARKETS[market]["indices"][0]
    hist = sc.tail(HISTORY_DAYS)
    history = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "index": _r(md.features[main].close.get(d), 2),
            "trend": _r(row.trend_score), "risk": _r(row.risk_score),
            "target": _r(target.get(d)), "executed": _r(executed.get(d)),
            "regime": int(regimes[d]) if pd.notna(regimes.get(d)) else None,
        }
        for d, row in hist.iterrows()
    ]
    return {
        "label": config.MARKETS[market]["label"],
        "index_weights": config.MARKETS[market].get("index_weights"),
        "as_of": latest.name.strftime("%Y-%m-%d"),
        "allocation": alloc,
        "executed_weight": _r(executed.iloc[-1]),
        "next_weight": _r(next_weight(target)),
        "trend_components": {k.removeprefix("trend_"): _r(latest[k]) for k in sc.columns if k.startswith("trend_") and k != "trend_score"},
        "risk_components": {k.removeprefix("risk_"): _r(latest[k]) for k in sc.columns if k.startswith("risk_") and k != "risk_score"},
        "risk_weights": config.RISK.weights[market],
        "indices": indices,
        "history": history,
    }


def macro_payload(md: MarketData) -> dict:
    p = md.prices
    out = {}
    for c in MACRO_COLS:
        if c not in p:
            continue
        s = p[c].dropna()
        out[c] = {"value": _r(s.iloc[-1]), "chg_10d": _r(s.iloc[-1] - s.iloc[-11]) if len(s) > 11 else None}
    return out


def build_payload(md: MarketData) -> dict:
    engine = BetaModelScoringEngine(md)
    return {
        "meta": md.meta | {"kr_credit_blend": config.RISK.kr_credit_blend if "KR_CREDIT" in md.prices else 0.0},
        "params": asdict(config.ALLOCATION),
        "regimes": {int(r): REGIME_INFO[r] for r in Regime},
        "macro": macro_payload(md),
        "markets": {m: market_payload(engine, md, m) for m in config.MARKETS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Beta Model → dashboard JSON")
    ap.add_argument("--refresh", action="store_true", help="캐시 무시하고 재수집")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw, meta = fetch_raw(use_cache=not args.refresh)
    md = build_market_data(raw, meta)
    payload = build_payload(md)
    payload["scenarios"] = run_scenarios(raw, meta)
    log, fresh = update_log(payload, run_at=meta.get("fetched_at", ""))
    payload["signal_log"] = live_payload(log)
    alert = build_alert(log, {m: d["label"] for m, d in payload["markets"].items()}, fresh)
    if alert:
        print("알림:\n" + alert)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    for m, d in payload["markets"].items():
        a = d["allocation"]
        print(f"[{m}] {d['as_of']}  추세 {a['trend_score']:+.2f}  위험 {a['risk_score']:.2f}  "
              f"→ 주식 {a['stock_weight']:.0%} / 현금 {a['cash_weight']:.0%}  (국면 {a['regime']}: {a['regime_name']})")
    print(f"VKOSPI 출처: {md.meta.get('vkospi_source')}  |  저장: {args.out}")


if __name__ == "__main__":
    main()
