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
    "QQQ": "QQQ",          # 나스닥100 ETF (1999~, 배당 재투자 수정주가)
    "SOXX": "SOXX",        # 반도체 ETF (2001~)
}
# 시장 폭(breadth) 위험 요소용: 동일가중 S&P500(RSP) vs SPY, 섹터 ETF 11개의 200일선 위치
BREADTH_SECTORS: tuple[str, ...] = ("XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC")
BREADTH_TICKERS: dict[str, str] = {"SPY": "SPY", "RSP": "RSP"} | {t: t for t in BREADTH_SECTORS}
# yfinance에 이력이 없는 시리즈는 FinanceDataReader 로 받는다
FDR_SYMBOLS: dict[str, str] = {
    "KOSPI200": "KS200",   # VKOSPI 대체 계산용
}
FRED_SERIES: dict[str, str] = {
    # 신용 위험은 1986년부터 이력이 있는 Baa−10Y 스프레드로 계산한다 (백테스트와 운영 일치).
    # HY OAS는 FRED가 최근 3년만 제공(ICE 라이선스)하므로 참고 지표로만 표시.
    "BAA_SPREAD": "BAA10Y",          # Moody's Baa 회사채 − 미 10년물 (%p)
    "HY_SPREAD": "BAMLH0A0HYM2",     # ICE BofA US High Yield OAS (%) — 참고용
    "CCC_OAS": "BAMLH0A3HYC",        # ICE BofA CCC 이하 OAS — 참고용 (FRED 3년치뿐이라 검증 불가)
    "BB_OAS": "BAMLH0A1HYBB",        # ICE BofA BB OAS — 참고용 (CCC−BB = 최하위 신용 스트레스)
    "USDKRW_FRED": "DEXKOUS",        # 2003년 이전 원/달러 보완 (주간 지연 공시)
    "UST3M": "DTB3",                 # 미 3개월 국채 — 현금(MMF) 수익률 근사
    "KR3M": "IR3TIB01KRM156N",       # 한국 3개월 금리(월간) — 원화 현금 수익률 근사
}

# 시장별 구성: 추세는 지수 평균(index_weights가 있으면 가중평균), 위험은 시장별 변동성 지수 + 공통 매크로.
# index_weights가 있는 시장은 수익률·실현변동성도 그 비중의 바스켓(매일 리밸런싱)으로 계산한다.
MARKETS: dict[str, dict] = {
    "US": {"label": "미국", "indices": ["SPX", "NDX"], "vol_index": "VIX", "cash": "UST3M"},
    # 기술주 중심 포트폴리오용. 비중은 실제 보유 비율에 맞게 수정 (대시보드에서도 조정 가능)
    "US_TECH": {"label": "미국(기술)", "indices": ["QQQ", "SOXX"], "index_weights": {"QQQ": 0.4, "SOXX": 0.6},
                "vol_index": "VIX", "cash": "UST3M"},
    "KR": {"label": "한국", "indices": ["KOSPI", "KOSDAQ"], "vol_index": "VKOSPI", "cash": "KR3M"},
}
# 파라미터 선택(그리드·워크포워드)에 쓰는 시장. 미국(기술)은 일부러 빼서 맞춰보지 않은 표본으로 남긴다.
CALIBRATION_MARKETS: tuple[str, ...] = ("US", "KR")

HISTORY_START = "1990-01-01"  # 백테스트용 장기 이력 (VIX 시작점)
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
    # 정배열: 0이면 부호(±1)만, >0이면 tanh(이격도/scale) 연속값 → 이평선 근처 휩쏘 완화
    # 기본값은 backtest.py 워크포워드(표본 내 1990~2012 선택 → 2013~ 검증)로 정함:
    # 표본 내 샤프가 최고치와 0.01 이내인 조합 중 회전율이 가장 낮은 것
    alignment_scale: float = 0.04
    smooth_span: int = 10       # 합성 추세 점수 EMA 기간 (1 = 평활화 없음)


@dataclass(frozen=True)
class RiskParams:
    # 변동성 지수: 백분위(상대)와 절대 레벨을 반반 섞는다.
    # (조용한 1년 뒤에는 VIX 16도 백분위 90%가 될 수 있어 레벨로 보정)
    vol_level_lo: dict = field(default_factory=lambda: {"VIX": 13.0, "VKOSPI": 15.0})
    vol_level_hi: dict = field(default_factory=lambda: {"VIX": 35.0, "VKOSPI": 40.0})
    spike_window: int = 10
    fx_spike_full: float = 0.03       # USD/KRW 10일 +3% → 1.0
    rate_spike_full_bp: float = 40.0  # 10Y 10일 +40bp → 1.0
    credit_series: str = "BAA_SPREAD"
    credit_z_full: float = 2.0        # 스프레드 Z=2 → 1.0 (상대)
    credit_level_lo: float = 2.5      # Baa−10Y 2.5%p 이하 → 0 (절대, 1990~ 상위 25% 수준까지는 평시)
    credit_level_hi: float = 4.5      # 4.5%p 이상 → 1 (2008 6.2, 2020 4.3, 2001 3.5)
    # 시장 폭(breadth) = 동일가중 열위 50% + 섹터 200일선 이탈 비율 50%. 지수는 오르는데 속은 약해지는
    # '좁은 장세'를 잡는다. 후보 지표 20여 개(유가·엔화·금리커브·MOVE·SKEW·NFCI·신용 추세 등) 중
    # 향후 63일 −10% 낙폭 예측력(AUC)이 미국·한국, 표본 내·외 모두에서 일관되게 오른 유일한 지표.
    breadth_ew_window: int = 63       # RSP/SPY 상대수익 기간
    breadth_ew_full: float = 0.05     # 동일가중이 63일간 5% 이상 뒤처지면 → 1
    breadth_sect_lo: float = 0.2      # 200일선 아래 섹터 비율 20% 이하 → 0
    breadth_sect_hi: float = 0.8      # 80% 이상 → 1
    # 국내 신용(회사채 AA− 3년 − 국고채 3년, ECOS)을 한국 시장 신용 위험에 섞는 비율.
    # 백테스트(2026-10): 50% 혼합 시 표본 내 샤프 0.456→0.451, 위험-낙폭 순위상관 −0.126→−0.115로
    # 개선이 없어 0(미국 Baa만)으로 둔다. 국내 신용은 매크로 패널에 참고 지표로 표시.
    kr_credit_series: str = "KR_CREDIT"
    kr_credit_blend: float = 0.0
    kr_credit_level_lo: float = 0.9   # %p, 1998~ 75분위 0.94 (Baa와 같은 원칙), 평시 2014~19 0.2~0.7
    kr_credit_level_hi: float = 1.8   # %p, 2022 레고랜드 고점 1.77 (2020 1.41, 2008 4.65)
    # 시장별 가중치 (합 1.0)
    weights: dict = field(default_factory=lambda: {
        # breadth 0.20 추가 (기존 가중치는 ×0.8). 0.10~0.25 모두 결과가 비슷해 중간값 — beta_model 검증 README 참고
        "US": {"vol_index": 0.28, "realized_vol": 0.12, "rates": 0.12, "fx": 0.08, "credit": 0.20, "breadth": 0.20},
        "US_TECH": {"vol_index": 0.28, "realized_vol": 0.12, "rates": 0.12, "fx": 0.08, "credit": 0.20, "breadth": 0.20},
        "KR": {"vol_index": 0.24, "realized_vol": 0.12, "rates": 0.08, "fx": 0.16, "credit": 0.20, "breadth": 0.20},
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
    # 국면 판정 경계 + 히스테리시스(이중 임계값): 상승 국면은 +h 이상에서 진입, −h 이하에서 이탈.
    # 경계 근처에서 국면 라벨·행동 지침이 매일 뒤바뀌는 것을 막는다 (비중 공식에는 영향 없음).
    trend_threshold: float = 0.0
    risk_threshold: float = 0.5
    trend_hysteresis: float = 0.10
    risk_hysteresis: float = 0.05
    # 속도 조절: 줄일 때는 즉시, 채울 때는 하루 최대 +N%p
    max_daily_increase: float = 0.02
    # 리밸런싱 무시 구간 (목표와 현재 차이가 이보다 작으면 매매 안 함)
    rebalance_band: float = 0.03
    # 집행 시뮬레이션에도 무시 구간 적용 여부 (False = 매일 목표 추종)
    band_in_execution: bool = True
    # 현금 중 달러(미 단기국채·달러 MMF) 비율 — 한국 시장(원화 투자자)에만 적용.
    # 코스피 급락 때 원/달러가 오르는 경향(일간 상관 −0.21)으로 위기 방어에 기여.
    # 백테스트: 표본 내 MDD가 50%에서 −40.1%→−35.0%로 포화(그 이상은 환 노출만 증가) → 50%.
    cash_usd_share: float = 0.5


TREND = TrendParams()
RISK = RiskParams()
ALLOCATION = AllocationParams()
