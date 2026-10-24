"""네트워크 없이 합성 데이터로 점수·배분 로직을 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beta_model import config
from beta_model.allocation import (
    Regime, allocate, classify_regime, rebalance_orders, smooth_weights, target_stock_weight,
)
from beta_model.data_pipeline import build_market_data, compute_features
from beta_model.scoring import BetaModelScoringEngine, linear_map, rolling_percentile

N = 600


def _raw(trend: float, vol_level: float, hy_jump: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """trend: 일간 drift, vol_level: VIX/VKOSPI 수준."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=N)
    def path(d):
        return 100 * np.exp(np.cumsum(d + rng.normal(0, 0.008, N)))
    hy = np.r_[np.full(N - 20, 3.5), np.full(20, 3.5 + hy_jump)] + rng.normal(0, 0.02, N)
    return pd.DataFrame({
        "SPX": path(trend), "NDX": path(trend), "KOSPI": path(trend), "KOSDAQ": path(trend),
        "VIX": np.r_[np.full(N - 10, 15.0), np.full(10, vol_level)] + rng.normal(0, 0.3, N),
        "VKOSPI": np.r_[np.full(N - 10, 18.0), np.full(10, vol_level)] + rng.normal(0, 0.3, N),
        "US10Y": np.full(N, 4.0), "USDKRW": np.full(N, 1350.0), "DXY": np.full(N, 100.0),
        "HY_SPREAD": hy,
    }, index=idx)


def _engine(**kw) -> BetaModelScoringEngine:
    return BetaModelScoringEngine(build_market_data(_raw(**kw)))


# ------------------------------------------------------------------ pipeline
def test_features_columns_and_drawdown_bounds():
    f = compute_features(pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2024-01-01", periods=300)))
    assert {"sma20", "sma60", "sma120", "high_52w", "drawdown", "vol_20d", "roc_63"} <= set(f.columns)
    assert (f["drawdown"].dropna() <= 0).all()
    assert f["drawdown"].iloc[-1] == pytest.approx(0.0)  # 단조 상승 → 낙폭 0


def test_market_data_is_forward_filled():
    raw = _raw(0.0, 15)
    raw.iloc[100:103, raw.columns.get_loc("KOSPI")] = np.nan  # 한국 휴장
    md = build_market_data(raw)
    assert md.prices["KOSPI"].iloc[100:103].notna().all()
    assert "SPX" in md.to_payload()["features"]


# ------------------------------------------------------------------- scoring
def test_helpers():
    assert linear_map(0.5, 0, 1) == pytest.approx(0.5)
    assert linear_map(5, 0, 1) == 1.0 and linear_map(-5, 0, 1) == 0.0
    s = pd.Series(np.arange(300, dtype=float))
    assert rolling_percentile(s).iloc[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("market", ["US", "KR"])
def test_scores_within_bounds(market):
    sc = _engine(trend=0.0, vol_level=30).scores(market)
    assert sc["trend_score"].between(-1, 1).all()
    assert sc["risk_score"].between(0, 1).all()


def test_uptrend_positive_downtrend_negative():
    assert _engine(trend=0.002, vol_level=15).latest("US")["trend_score"] > 0.5
    assert _engine(trend=-0.002, vol_level=15).latest("US")["trend_score"] < -0.5


def test_vol_and_credit_spike_raise_risk():
    calm = _engine(trend=0.0, vol_level=15).latest("US")["risk_score"]
    panic = _engine(trend=0.0, vol_level=45, hy_jump=3.0).latest("US")["risk_score"]
    assert panic > calm + 0.3


# ---------------------------------------------------------------- allocation
@pytest.mark.parametrize("trend,risk,expected", [
    (1.0, 0.0, 1.00),   # 상승 따라가기: 풀 투자
    (1.0, 1.0, 0.80),   # 미리 줄이기
    (-1.0, 0.0, 0.70),  # 하락 방어: 과도한 현금화 지양
    (-1.0, 1.0, 0.60),  # 최대 방어: 현금 40%
    (0.0, 0.0, 0.85),
])
def test_target_weight_corners(trend, risk, expected):
    assert target_stock_weight(trend, risk) == pytest.approx(expected)


def test_weight_always_in_band_and_sums_to_one():
    for t in np.linspace(-1.5, 1.5, 31):
        for r in np.linspace(-0.5, 1.5, 21):
            a = allocate(t, r)
            assert config.ALLOCATION.floor <= a.stock_weight <= config.ALLOCATION.cap
            assert a.stock_weight + a.cash_weight == pytest.approx(1.0)


def test_allocate_rejects_nan():
    with pytest.raises(ValueError):
        allocate(float("nan"), 0.2)


@pytest.mark.parametrize("trend,risk,regime", [
    (0.5, 0.2, Regime.BULL_FOLLOW), (0.5, 0.7, Regime.PREEMPTIVE_CUT),
    (-0.5, 0.2, Regime.DOWN_DEFENSE), (-0.5, 0.7, Regime.MAX_DEFENSE),
])
def test_regimes(trend, risk, regime):
    assert classify_regime(trend, risk) == regime


def test_smoothing_is_asymmetric_and_lagged():
    tgt = pd.Series([1.0, 1.0, 0.6, 0.6, 1.0, 1.0, 1.0], index=pd.bdate_range("2024-01-01", periods=7))
    ex = smooth_weights(tgt)
    assert ex.index[0] == tgt.index[1]                # 다음 거래일 집행
    assert ex.iloc[2] == pytest.approx(0.6)           # 하향은 즉시 (t=2 신호 → t=3)
    assert ex.iloc[4] == pytest.approx(0.62)          # 상향은 하루 +2%p
    assert ex.iloc[5] == pytest.approx(0.64)


def test_rebalance_orders():
    o = rebalance_orders(100_000_000, 90_000_000, 0.70)
    assert o["action"] == "SELL" and o["amount"] == pytest.approx(20_000_000)
    assert rebalance_orders(100, 69, 0.70)["action"] == "HOLD"  # 밴드 이내
    assert rebalance_orders(100, 50, 0.80)["action"] == "BUY"
    with pytest.raises(ValueError):
        rebalance_orders(0, 0, 0.7)
