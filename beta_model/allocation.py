"""3단계: 추세/위험 점수 → 주식/현금(MMF) 비중, 국면 판정, 리밸런싱.

원 프롬프트(Base=0.80+0.20*trend, 차감=0.20*risk)는 추세 -1이면 위험과 무관하게
이미 60%(최대 방어)가 되어 원문의 '하락 방어(과도한 현금화 지양)'와
'최대 방어(현금 최대 40%)'가 구분되지 않는다. 그래서 기본값을
Base=0.85+0.15*trend(70~100%), 차감=0.20*risk 로 조정했다 (config.AllocationParams).

원문 계좌 운용 원칙 반영:
- 주식 60~100%, 레버리지 금지 → clamp
- 줄일 때 빠르게, 채울 때 천천히 → `smooth_weights`
- 신호는 종가, 매매는 다음 거래일 → `smooth_weights`가 하루 지연(shift)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from . import config
from .config import AllocationParams


class Regime(IntEnum):
    BULL_FOLLOW = 1     # 상승세 + 낮은 불안감
    PREEMPTIVE_CUT = 2  # 상승세 + 높은 불안감
    DOWN_DEFENSE = 3    # 하락세 + 낮은 불안감
    MAX_DEFENSE = 4     # 하락세 + 높은 불안감 (Buy the Dip 기회)


REGIME_INFO: dict[Regime, dict] = {
    Regime.BULL_FOLLOW: {
        "name": "상승 따라가기",
        "summary": "추세 상승 + 불안감 낮음 → 주식 비중을 채우는 구간",
        "actions": ["목표 비중까지 주식을 천천히(분할) 채운다", "추세 이탈(SMA20 하향) 여부만 점검", "현금은 MMF에서 대기"],
    },
    Regime.PREEMPTIVE_CUT: {
        "name": "미리 줄이기",
        "summary": "추세는 살아있지만 불안감 상승 → 하락 전환 전 선제적 일부 축소",
        "actions": ["위험 점수만큼 주식 비중을 즉시 덜어낸다", "금리·환율·신용 스프레드 중 무엇이 위험을 올렸는지 확인", "추세가 유지되면 위험 하락 시 천천히 복원"],
    },
    Regime.DOWN_DEFENSE: {
        "name": "하락 방어",
        "summary": "추세 하락 + 불안감 낮음 → 방향에 맞춰 줄이되 과도한 현금화 지양",
        "actions": ["추세 점수에 맞춰 비중 축소 (70%대 유지)", "공포가 아닌 추세 조정이므로 투매 금지", "추세 반전(SMA20 회복) 시 바로 채울 준비"],
    },
    Regime.MAX_DEFENSE: {
        "name": "최대 방어 / 공포",
        "summary": "추세 하락 + 불안감 높음 → 현금 최대 40%, 동시에 Buy the Dip 후보 탐색",
        "actions": ["현금(MMF·단기채) 비중을 최대 40%까지 확대", "깊게 빠진 우량 자산 매수 리스트 작성", "위험 점수 꺾이는 시점부터 분할 매수 개시"],
    },
}


def classify_regime(trend: float, risk: float, p: AllocationParams = config.ALLOCATION) -> Regime:
    """추세/위험 점수로 4개 국면 중 하나를 반환."""
    up = trend >= p.trend_threshold
    calm = risk < p.risk_threshold
    if up:
        return Regime.BULL_FOLLOW if calm else Regime.PREEMPTIVE_CUT
    return Regime.DOWN_DEFENSE if calm else Regime.MAX_DEFENSE


@dataclass(frozen=True)
class Allocation:
    trend_score: float
    risk_score: float
    base_weight: float
    risk_deduction: float
    stock_weight: float
    cash_weight: float
    regime: int
    regime_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def target_stock_weight(trend: float, risk: float, p: AllocationParams = config.ALLOCATION) -> float:
    """Final = clamp(base_mid + trend_span*trend - risk_span*risk, floor, cap)."""
    trend = float(np.clip(trend, -1.0, 1.0))
    risk = float(np.clip(risk, 0.0, 1.0))
    return float(np.clip(p.base_mid + p.trend_span * trend - p.risk_span * risk, p.floor, p.cap))


def allocate(trend: float, risk: float, p: AllocationParams = config.ALLOCATION) -> Allocation:
    """단일 시점 배분 결과.

    Raises:
        ValueError: 점수가 NaN 인 경우.
    """
    if np.isnan(trend) or np.isnan(risk):
        raise ValueError("trend/risk score must not be NaN")
    trend_c = float(np.clip(trend, -1.0, 1.0))
    risk_c = float(np.clip(risk, 0.0, 1.0))
    stock = target_stock_weight(trend_c, risk_c, p)
    regime = classify_regime(trend_c, risk_c, p)
    return Allocation(
        trend_score=trend_c,
        risk_score=risk_c,
        base_weight=p.base_mid + p.trend_span * trend_c,
        risk_deduction=p.risk_span * risk_c,
        stock_weight=stock,
        cash_weight=1.0 - stock,
        regime=int(regime),
        regime_name=REGIME_INFO[regime]["name"],
    )


def smooth_weights(target: pd.Series, p: AllocationParams = config.ALLOCATION) -> pd.Series:
    """실제 집행 비중 시계열.

    - t일 종가 신호 → t+1일 집행 (shift 1)
    - 목표가 현재보다 낮으면 즉시 하향, 높으면 하루 최대 `max_daily_increase` 만큼만 상향
    """
    sig = target.shift(1).dropna()
    out = np.empty(len(sig))
    cur = sig.iloc[0] if len(sig) else np.nan
    for i, tgt in enumerate(sig.to_numpy()):
        cur = tgt if tgt < cur else min(tgt, cur + p.max_daily_increase)
        out[i] = cur
    return pd.Series(out, index=sig.index, name="executed_weight")


def rebalance_orders(
    total_value: float,
    current_stock_value: float,
    target_weight: float,
    band: float = config.ALLOCATION.rebalance_band,
) -> dict:
    """현재 보유액 대비 목표 비중까지의 매수/매도 금액.

    Returns:
        dict(action='BUY'|'SELL'|'HOLD', amount, target_stock_value, ...)
    """
    if total_value <= 0:
        raise ValueError("total_value must be positive")
    current_stock_value = float(np.clip(current_stock_value, 0, total_value))
    target_value = total_value * target_weight
    diff = target_value - current_stock_value
    current_weight = current_stock_value / total_value
    if abs(target_weight - current_weight) < band:
        action = "HOLD"
    else:
        action = "BUY" if diff > 0 else "SELL"
    return {
        "action": action,
        "amount": abs(diff) if action != "HOLD" else 0.0,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "target_stock_value": target_value,
        "target_cash_value": total_value - target_value,
    }
