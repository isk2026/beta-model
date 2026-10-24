"""가상 충격 시나리오: '오늘 이후 10거래일 동안 이런 일이 벌어지면 모델은 비중을 얼마로 제시하나?'

실제 엔진(점수·히스테리시스·속도 조절)을 그대로 통과시키기 위해, 원시 데이터 끝에
10영업일을 이어 붙여 각 지표를 목표 수준까지 선형/기하 경로로 움직인다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .allocation import REGIME_INFO, Regime, regime_series, smooth_weights, target_stock_weight
from .data_pipeline import build_market_data
from .scoring import BetaModelScoringEngine

DAYS = 10
US_EQ, KR_EQ = ["SPX", "NDX"], ["KOSPI", "KOSDAQ", "KOSPI200"]
# 기술주 ETF는 미국 시나리오 하락폭에 베타를 곱한다. S&P500 대비 일간 베타는 1.1/1.4지만
# 급락장에서는 더 커서(2022년 하락폭 비율 약 1.4/1.8) 그 사이 값을 쓴다.
TECH_BETA = {"QQQ": 1.2, "SOXX": 1.6}

# 지수: 누적 수익률, 변동성 지수: 도달 레벨, 금리: bp 변화, 환율: % 변화, 신용: %p 변화
SCENARIOS: list[dict] = [
    {"id": "correction", "label": "완만한 조정", "desc": "미·한 −8%, VIX 24",
     "us": -0.08, "kr": -0.08, "VIX": 24, "VKOSPI": 26, "US10Y": 0, "USDKRW": 0.01, "BAA_SPREAD": 0.1},
    {"id": "rate", "label": "2022형 금리 쇼크", "desc": "미 10년물 +60bp, 미·한 −10%, 원/달러 +4%",
     "us": -0.10, "kr": -0.10, "VIX": 32, "VKOSPI": 30, "US10Y": 60, "USDKRW": 0.04, "BAA_SPREAD": 0.4},
    {"id": "krw", "label": "원화 급락", "desc": "원/달러 +12%, 코스피 −15%, 미국 −3%",
     "us": -0.03, "kr": -0.15, "VIX": 25, "VKOSPI": 40, "US10Y": 10, "USDKRW": 0.12, "BAA_SPREAD": 0.3},
    {"id": "credit", "label": "2008형 신용위기", "desc": "신용 +1.5%p, VIX 55, 미 −20%, 한 −25%",
     "us": -0.20, "kr": -0.25, "VIX": 55, "VKOSPI": 55, "US10Y": -50, "USDKRW": 0.10, "BAA_SPREAD": 1.5},
    {"id": "crash", "label": "2020형 급락", "desc": "10일 만에 −30%, VIX 75",
     "us": -0.30, "kr": -0.30, "VIX": 75, "VKOSPI": 65, "US10Y": -80, "USDKRW": 0.05, "BAA_SPREAD": 1.5},
]


def shock_raw(raw: pd.DataFrame, sc: dict, days: int = DAYS) -> pd.DataFrame:
    """raw 끝에 `days` 영업일을 붙이고 시나리오 경로를 적용한 사본."""
    last = raw.index.max()
    new_idx = pd.bdate_range(last + pd.offsets.BDay(1), periods=days)
    base = raw.ffill().iloc[-1]
    ext = pd.DataFrame([base.to_numpy()] * days, index=new_idx, columns=raw.columns, dtype=float)
    t = np.arange(1, days + 1) / days                       # 0→1 진행률
    for cols, total in ((US_EQ, sc["us"]), (KR_EQ, sc["kr"])):
        for c in cols:
            if c in ext:
                ext[c] = base[c] * (1 + total) ** t       # 기하 경로
    for c, beta in TECH_BETA.items():
        if c in ext:
            ext[c] = base[c] * (1 + max(sc["us"] * beta, -0.6)) ** t
    for c in ("VIX", "VKOSPI"):
        if c in ext:  # 충격은 변동성을 낮추지 않는다: 오늘 수준과 시나리오 수준 중 높은 쪽
            ext[c] = base[c] + (max(sc[c], base[c]) - base[c]) * t
    if "US10Y" in ext:
        ext["US10Y"] = base["US10Y"] + sc["US10Y"] / 100 * t
    if "USDKRW" in ext:
        ext["USDKRW"] = base["USDKRW"] * (1 + sc["USDKRW"]) ** t
    if "BAA_SPREAD" in ext:
        ext["BAA_SPREAD"] = base["BAA_SPREAD"] + sc["BAA_SPREAD"] * t
    return pd.concat([raw, ext])


def _state(raw: pd.DataFrame, meta: dict) -> dict:
    md = build_market_data(raw, meta)
    engine = BetaModelScoringEngine(md)
    out = {}
    for m in config.MARKETS:
        s = engine.scores(m)
        target = pd.Series([target_stock_weight(a, b) for a, b in zip(s.trend_score, s.risk_score)], index=s.index)
        executed = smooth_weights(target)
        reg = int(regime_series(s.trend_score, s.risk_score).iloc[-1])
        out[m] = {
            "trend": float(s.trend_score.iloc[-1]), "risk": float(s.risk_score.iloc[-1]),
            "target": float(target.iloc[-1]), "executed": float(executed.iloc[-1]),
            "regime": reg, "regime_name": REGIME_INFO[Regime(reg)]["name"],
        }
    return out


def run_scenarios(raw: pd.DataFrame, meta: dict) -> dict:
    today = _state(raw, meta)
    rows = []
    for sc in SCENARIOS:
        rows.append({k: sc[k] for k in ("id", "label", "desc")} | {"markets": _state(shock_raw(raw, sc), meta)})
    rnd = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    return {
        "days": DAYS,
        "today": {m: rnd(v) for m, v in today.items()},
        "scenarios": [r | {"markets": {m: rnd(v) for m, v in r["markets"].items()}} for r in rows],
    }
