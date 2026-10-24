"""성상현 베타 모델의 모든 파라미터를 한 곳에 모은 설정 모듈.

수치를 바꾸고 싶으면 이 파일만 수정하면 된다. 대시보드(JS)도 export 시
`ALLOCATION` 값을 JSON으로 받아 같은 공식을 쓰므로 Python/JS 결과가 일치한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. 수집 대상
# ---------------------------------------------------------------------------
YF_TICKERS: dict[str, str] = {
    "SPX": "^GSPC",        # S&P 500
    "NDX": "^IXIC",        # NASDAQ Composite
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "VIX": "^VIX",
    "US10Y": "^TNX",       # % 단위 (예: 4.25)
    "USDKRW": "KRW=X",
    "DXY": "DX-Y.NYB",     # 원문에 언급된 달러 인덱스 (보조 지표)
}
# yfinance에 이력이 없는 시리즈는 FinanceDataReader 로 받는다
FDR_SYMBOLS: dict[str, str] = {
    "KOSPI200": "KS200",   # VKOSPI 대체 계산용
}
FRED_SERIES: dict[str, str] = {
    "HY_SPREAD": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS (%)
}

# 시장별 구성: 추세는 지수 평균, 위험은 시장별 변동성 지수 + 공통 매크로
MARKETS: dict[str, dict] = {
    "US": {"label": "미국", "indices": ["SPX", "NDX"], "vol_index": "VIX"},
    "KR": {"label": "한국", "indices": ["KOSPI", "KOSDAQ"], "vol_index": "VKOSPI"},
}

HISTORY_YEARS = 3          # 252일 백분위·Z점수 워밍업을 위해 2년보다 넉넉히 받는다
SMA_WINDOWS = (20, 60, 120)
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 2. 점수 엔진
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendParams:
    w_alignment: float = 0.50   # 이동평균 정배열
    w_drawdown: float = 0.25    # 52주 고점 대비 낙폭
    w_momentum: float = 0.25    # 3개월 ROC
    dd_neutral: float = -0.10   # 이 낙폭에서 0점
    dd_floor: float = -0.20     # 이 낙폭 이하에서 -1점
    roc_window: int = 63        # 3개월
    roc_scale: float = 0.10     # tanh(roc / scale)


@dataclass(frozen=True)
class RiskParams:
    # 변동성 지수: 백분위(상대)와 절대 레벨을 반반 섞는다.
    # (조용한 1년 뒤에는 VIX 16도 백분위 90%가 될 수 있어 레벨로 보정)
    vol_level_lo: dict = field(default_factory=lambda: {"VIX": 13.0, "VKOSPI": 15.0})
    vol_level_hi: dict = field(default_factory=lambda: {"VIX": 35.0, "VKOSPI": 40.0})
    spike_window: int = 10
    fx_spike_full: float = 0.03       # USD/KRW 10일 +3% → 1.0
    rate_spike_full_bp: float = 40.0  # 10Y 10일 +40bp → 1.0
    hy_z_full: float = 2.0            # HY 스프레드 Z=2 → 1.0 (상대)
    hy_level_lo: float = 3.0          # OAS 3% 이하 → 0 (절대)
    hy_level_hi: float = 6.0          # OAS 6% 이상 → 1 (2008·2020·2022 스트레스 구간)
    # 시장별 가중치 (합 1.0)
    weights: dict = field(default_factory=lambda: {
        "US": {"vol_index": 0.35, "realized_vol": 0.15, "rates": 0.15, "fx": 0.10, "credit": 0.25},
        "KR": {"vol_index": 0.30, "realized_vol": 0.15, "rates": 0.10, "fx": 0.20, "credit": 0.25},
    })


# ---------------------------------------------------------------------------
# 3. 자산배분
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AllocationParams:
    """Final = clamp(base_mid + trend_span*trend - risk_span*risk, floor, cap)

    기본값: 추세 +1/위험 0 → 100%, 추세 -1/위험 0 → 70% (하락 방어, 과도한 현금화 지양),
    추세 -1/위험 0.5 이상 → 60% (최대 방어, 현금 40%), 추세 +1/위험 1 → 80% (미리 줄이기).
    """
    base_mid: float = 0.85
    trend_span: float = 0.15
    risk_span: float = 0.20
    floor: float = 0.60
    cap: float = 1.00
    # 국면 판정 경계
    trend_threshold: float = 0.0
    risk_threshold: float = 0.5
    # 속도 조절: 줄일 때는 즉시, 채울 때는 하루 최대 +N%p
    max_daily_increase: float = 0.02
    # 리밸런싱 무시 구간 (목표와 현재 차이가 이보다 작으면 매매 안 함)
    rebalance_band: float = 0.03


TREND = TrendParams()
RISK = RiskParams()
ALLOCATION = AllocationParams()
