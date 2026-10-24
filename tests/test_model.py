"""네트워크 없이 합성 데이터로 점수·배분 로직을 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataclasses import replace

from beta_model import config
from beta_model.backtest import ModelSpec, drawdown_episodes, metrics, simulate, static_mix
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
    hy = np.r_[np.full(N - 20, 1.8), np.full(20, 1.8 + hy_jump)] + rng.normal(0, 0.02, N)
    return pd.DataFrame({
        "SPX": path(trend), "NDX": path(trend), "KOSPI": path(trend), "KOSDAQ": path(trend),
        "VIX": np.r_[np.full(N - 10, 15.0), np.full(10, vol_level)] + rng.normal(0, 0.3, N),
        "VKOSPI": np.r_[np.full(N - 10, 18.0), np.full(10, vol_level)] + rng.normal(0, 0.3, N),
        "US10Y": np.full(N, 4.0), "USDKRW": np.full(N, 1350.0), "DXY": np.full(N, 100.0),
        "BAA_SPREAD": hy, "UST3M": np.full(N, 3.0), "KR3M": np.full(N, 3.0),
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


NO_BAND = replace(config.ALLOCATION, band_in_execution=False)


def test_smoothing_is_asymmetric_and_lagged():
    tgt = pd.Series([1.0, 1.0, 0.6, 0.6, 1.0, 1.0, 1.0], index=pd.bdate_range("2024-01-01", periods=7))
    ex = smooth_weights(tgt, NO_BAND)
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


def test_band_skips_small_moves_but_completes_fills():
    p = replace(config.ALLOCATION, band_in_execution=True, rebalance_band=0.03, max_daily_increase=0.02)
    tgt = pd.Series([0.80, 0.82, 0.78, 0.80, 0.90, 0.90, 0.90, 0.90, 0.89, 0.89],
                    index=pd.bdate_range("2024-01-01", periods=10))
    ex = smooth_weights(tgt, p).round(4).tolist()  # ex[i] = tgt[i] 신호의 익일 집행
    assert ex[:4] == [0.80, 0.80, 0.80, 0.80]      # ±2%p 흔들림은 무시
    assert ex[4:8] == [0.82, 0.84, 0.86, 0.88]     # 3%p 넘으면 채우기 시작, 하루 +2%p
    assert ex[8] == 0.89                           # 목표에 닿을 때까지 계속 (밴드 안이어도)


def test_continuous_alignment_bounded_and_less_jumpy():
    md = build_market_data(_raw(trend=0.0, vol_level=15, seed=3))
    sign = BetaModelScoringEngine(md, replace(config.TREND, alignment_scale=0.0, smooth_span=1)).index_trend("SPX")
    cont = BetaModelScoringEngine(md, replace(config.TREND, alignment_scale=0.04, smooth_span=10)).index_trend("SPX")
    assert cont["trend_score"].dropna().between(-1, 1).all()
    assert cont["trend_score"].diff().abs().mean() < sign["trend_score"].diff().abs().mean()


def test_backtest_sanity():
    md = build_market_data(_raw(trend=0.0005, vol_level=15))
    df = simulate(md, "US", ModelSpec("t"))
    assert df["weight"].between(config.ALLOCATION.floor, config.ALLOCATION.cap).all()
    m = metrics(df)
    assert m["mdd"] <= 0 and m["turnover"] >= 0
    # 고정 비중 기준선은 모델과 평균 비중이 같다
    assert static_mix(df)["weight"].iloc[0] == pytest.approx(df["weight"].mean())
    for e in drawdown_episodes(df, min_depth=-0.01):
        assert e["bh"] <= -0.01 and e["peak"] < e["trough"]


# ------------------------------------------------------------- KR sources
def test_parse_ecos_and_errors():
    from beta_model.kr_sources import parse_ecos
    ok = {"StatisticSearch": {"row": [
        {"TIME": "20261021", "DATA_VALUE": "4.515"}, {"TIME": "20261020", "DATA_VALUE": "4.534"}]}}
    s = parse_ecos(ok)
    assert list(s.index.strftime("%Y%m%d")) == ["20261020", "20261021"] and s.iloc[-1] == pytest.approx(4.515)
    assert parse_ecos({"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}).empty


def test_parse_vkospi():
    from beta_model.kr_sources import parse_vkospi
    payload = {"OutBlock_1": [
        {"IDX_NM": "코스피 200 선물지수", "CLSPRC_IDX": "1,234.5"},
        {"IDX_NM": "코스피 200 변동성지수", "CLSPRC_IDX": "23.41"},
    ]}
    assert parse_vkospi(payload) == pytest.approx(23.41)
    assert parse_vkospi({"OutBlock_1": []}) is None
    with pytest.raises(PermissionError):
        parse_vkospi({"respMsg": "Unauthorized Key", "respCode": "401"})


def test_kr_credit_blends_only_for_kr():
    raw = _raw(trend=0.0, vol_level=15)
    base_us = BetaModelScoringEngine(build_market_data(raw)).risk_components("US")["credit"]
    base_kr = BetaModelScoringEngine(build_market_data(raw)).risk_components("KR")["credit"]
    raw["KR_CREDIT"] = np.r_[np.full(N - 20, 0.6), np.full(20, 2.5)]  # 국내 신용 경색
    e = BetaModelScoringEngine(build_market_data(raw), risk_params=replace(config.RISK, kr_credit_blend=0.5))
    off = BetaModelScoringEngine(build_market_data(raw), risk_params=replace(config.RISK, kr_credit_blend=0.0))
    assert off.risk_components("KR")["credit"].iloc[-1] == pytest.approx(base_kr.iloc[-1])
    assert e.risk_components("US")["credit"].iloc[-1] == pytest.approx(base_us.iloc[-1])
    assert e.risk_components("KR")["credit"].iloc[-1] > base_kr.iloc[-1] + 0.3


def test_vkospi_csv_spliced_with_scaled_proxy(tmp_path, monkeypatch):
    from beta_model import data_pipeline
    idx = pd.bdate_range("2020-01-01", periods=800)
    k200 = pd.Series(300 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 800))), index=idx)
    rv = data_pipeline.realized_vol(k200, 20) * 100
    csv = tmp_path / "vkospi.csv"
    pd.DataFrame({"Date": idx[400:], "Close": (rv * 1.3).iloc[400:].values}).to_csv(csv, index=False)
    monkeypatch.setattr(data_pipeline, "VKOSPI_CSV", csv)
    s, src = data_pipeline.load_vkospi(k200)
    assert src == "krx+proxy"
    assert s.loc[idx[399]] == pytest.approx(rv.loc[idx[399]] * 1.3, rel=1e-6)  # 실제값 비율로 보정된 proxy


# ---------------------------------------------------- step 1: metrics/regime
def test_regime_hysteresis_holds_near_threshold():
    from beta_model.allocation import regime_series
    idx = pd.bdate_range("2024-01-01", periods=6)
    trend = pd.Series([0.3, -0.05, 0.05, -0.08, -0.15, 0.05], index=idx)   # 0 근처 흔들림 후 이탈
    risk = pd.Series([0.2] * 6, index=idx)
    reg = regime_series(trend, risk).tolist()
    assert reg[:4] == [1, 1, 1, 1]          # −0.1 이하로 안 내려가면 상승 국면 유지
    assert reg[4] == 3                      # −0.15 → 하락 전환
    assert reg[5] == 3                      # +0.05 로는 복귀 안 함 (+0.1 필요)


def test_capture_and_sortino():
    md = build_market_data(_raw(trend=0.0003, vol_level=15))
    df = simulate(md, "US", ModelSpec("t"))
    m = metrics(df)
    assert 0 < m["down_capture"] < 1.2 and 0 < m["up_capture"] < 1.2
    assert m["cagr_gross"] >= m["cagr"]                 # 비용 차감 전 ≥ 후
    assert np.isfinite(m["sortino"]) and m["underwater_years"] >= 0


# ---------------------------------------------------- step 4: signal log
def test_next_weight_and_action():
    from beta_model.signals import action_of, next_weight
    idx = pd.bdate_range("2024-01-01", periods=5)
    assert next_weight(pd.Series([1.0, 1.0, 1.0, 1.0, 0.7], index=idx)) == pytest.approx(0.7)   # 줄이기: 즉시
    up = next_weight(pd.Series([0.7, 0.7, 0.7, 0.7, 1.0], index=idx))
    assert 0.7 < up < 1.0                                                                       # 늘리기: 천천히
    assert action_of(0.8, 0.7) == "SELL" and action_of(0.7, 0.72) == "BUY" and action_of(0.7, 0.702) == "HOLD"


def test_signal_log_alert_once(tmp_path, monkeypatch):
    from beta_model import signals
    monkeypatch.setattr(signals, "LOG_FILE", tmp_path / "log.csv")
    monkeypatch.setattr(signals, "ALERT_FILE", tmp_path / "alert.md")

    def payload(as_of, regime, ex, nxt):
        return {"markets": {"US": {"as_of": as_of, "label": "미국", "allocation": {"regime": regime},
                                   "executed_weight": ex, "next_weight": nxt,
                                   "history": [{"index": 100.0, "trend": 0.1, "risk": 0.2, "target": nxt}]}}}

    log, fresh = signals.update_log(payload("2024-01-02", 1, 0.9, 0.9), "t")
    assert fresh == {"US"} and signals.build_alert(log, {}, fresh) is None      # 첫 기록: 비교 대상 없음, 매매 없음
    log, fresh = signals.update_log(payload("2024-01-03", 3, 0.9, 0.75), "t")
    text = signals.build_alert(log, {}, fresh)
    assert "국면 전환" in text and "75%" in text
    log, fresh = signals.update_log(payload("2024-01-03", 3, 0.9, 0.75), "t2")   # 같은 기준일 재실행
    assert fresh == set() and signals.build_alert(log, {}, fresh) is None and len(log) == 2


# ---------------------------------------------------- weighted market (US_TECH)
def test_weighted_trend_and_basket():
    from beta_model.scoring import market_price
    raw = _raw(0.0, 15)
    rng = np.random.default_rng(1)
    raw["QQQ"] = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.01, N)))      # 상승
    soxx = 100 * np.exp(np.cumsum(-0.002 + rng.normal(0, 0.01, N)))           # 하락, 뒤늦게 상장
    soxx[:100] = np.nan
    raw["SOXX"] = soxx
    mk = {"T": {"label": "t", "indices": ["QQQ", "SOXX"], "index_weights": {"QQQ": 0.75, "SOXX": 0.25},
                "vol_index": "VIX", "cash": "UST3M"}}
    md = build_market_data(raw, markets=mk)
    eng = BetaModelScoringEngine(md, markets=mk, risk_params=replace(config.RISK, weights={"T": config.RISK.weights["US"]}))
    q, s = eng.index_trend("QQQ").trend_score, eng.index_trend("SOXX").trend_score
    t = eng.trend_score("T").trend_score
    i = t.index[-1]
    assert t[i] == pytest.approx(0.75 * q[i] + 0.25 * s[i])              # 가중평균
    j = t.index[150]
    assert t[j] == pytest.approx(q[j])                                    # 상장 전: QQQ만
    px = market_price(md, mk["T"])
    r = px.pct_change().iloc[-1]
    qr, sr = (md.features[n].close.pct_change().iloc[-1] for n in ("QQQ", "SOXX"))
    assert r == pytest.approx(0.75 * qr + 0.25 * sr)                      # 매일 리밸런싱 바스켓


# ---------------------------------------------------- sector monitor
def test_sector_payload_ranks_and_leaders():
    from beta_model.sectors import BENCH, EXTRA, SECTORS, sector_payload
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2022-01-03", periods=600)
    drift = {t: (i - 5) * 0.0004 for i, t in enumerate(SECTORS)}          # 뒤 섹터일수록 강함
    px = pd.DataFrame({t: 100 * np.exp(np.cumsum(drift.get(t, 0.0002) + rng.normal(0, 0.005, 600)))
                       for t in [BENCH, *SECTORS, *EXTRA]}, index=idx)
    p = sector_payload(px)
    scores = [x["score"] for x in p["sectors"]]
    assert scores == sorted(scores, reverse=True) and [x["rank"] for x in p["sectors"]] == list(range(1, 12))
    assert len(p["leaders"]) <= 3
    assert all(x["trend"] > 0 and x["above_sma200"] for x in p["sectors"] if x["leader"])
    assert p["sectors"][0]["ticker"] == list(SECTORS)[-1]                  # 가장 강한 drift가 1위
