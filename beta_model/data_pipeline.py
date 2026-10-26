"""1단계: 매크로/시장 데이터 수집 및 기술 지표 계산.

- yfinance: 지수, VIX, 미 10년물, USD/KRW, 달러 인덱스
- FinanceDataReader: FRED 하이일드 스프레드 (실패 시 FRED CSV 직접 호출)
- VKOSPI: 무료 API가 없어 `data/vkospi.csv`(Date,Close)가 있으면 사용하고,
  없으면 KOSPI200 20일 실현변동성으로 대체(proxy)한다.

지표는 각 시리즈의 **자체 거래일 기준**으로 계산한 뒤 공통 달력에 정렬·ffill 한다.
(미국/한국 휴장일을 ffill 후 계산하면 수익률 0인 날이 끼어 변동성이 과소평가됨)
"""

from __future__ import annotations

import io
import logging
import pickle
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .kr_sources import VKOSPI_CSV, fetch_ecos_all

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "cache" / "raw.pkl"
VKOSPI_PROXY_RATIO = 1.15  # 내재변동성은 통상 실현변동성보다 10~20% 높다
# raw/prices 에 함께 싣는 표시 열: 1 = 그날 VKOSPI 값이 실현변동성 대체값(proxy), 0 = KRX 실제값
VKOSPI_PROXY_COL = "VKOSPI_PROXY"
# 장기 추세 요소(실험용, TrendParams.w_long) 지표 기간
LONG_SMA = 200          # 200일선
MOM_LOOKBACK = 252      # 12개월
MOM_SKIP = 21           # 최근 1개월 제외 (단기 반전 회피)


@dataclass
class MarketData:
    """스코어링 엔진에 넘기는 정제된 데이터 묶음."""

    prices: pd.DataFrame                       # 공통 달력, ffill 된 종가 (wide)
    features: dict[str, pd.DataFrame]          # 지수별 SMA/MDD/변동성/ROC
    meta: dict = field(default_factory=dict)   # 출처, proxy 여부 등
    # 지수별 '실제로 종가가 있는' 거래일 (공통 달력 ffill 전). 시장별 기준일(as_of) 판정에 쓴다.
    # 비어 있으면(직접 만든 MarketData 등) 기준일을 자르지 않는다.
    trade_dates: dict[str, pd.DatetimeIndex] = field(default_factory=dict)

    def last_trade_date(self, names: list[str]) -> pd.Timestamp | None:
        """`names` 중 하나라도 실제 종가가 있는 마지막 날. 기록이 없으면 None."""
        last = [self.trade_dates[n].max() for n in names if n in self.trade_dates and len(self.trade_dates[n])]
        return max(last) if last else None

    def to_payload(self, tail: int = 1) -> dict:
        """최근 `tail`일 데이터를 JSON 직렬화 가능한 dict로 변환."""
        out = {"meta": self.meta, "prices": {}, "features": {}}
        for col in self.prices.columns:
            s = self.prices[col].dropna().tail(tail)
            out["prices"][col] = {d.strftime("%Y-%m-%d"): round(float(v), 4) for d, v in s.items()}
        for name, df in self.features.items():
            last = df.dropna(how="all").tail(tail)
            out["features"][name] = {
                d.strftime("%Y-%m-%d"): {k: _round(v) for k, v in row.items()} for d, row in last.iterrows()
            }
        return out


def _round(v: float) -> float | None:
    return None if pd.isna(v) else round(float(v), 6)


# ---------------------------------------------------------------------------
# 수집
# ---------------------------------------------------------------------------
def fetch_yfinance(tickers: dict[str, str], start: datetime, retries: int = 3) -> pd.DataFrame:
    """종가 테이블. 일부 심볼이 비면 그 심볼만 최대 `retries`회 재시도한다."""
    import yfinance as yf

    def download(symbols: list[str]) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                symbols, start=start.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False, threads=True,
            )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(symbols, axis=1)
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close.dropna(axis=1, how="all")

    close = download(list(tickers.values()))
    for attempt in range(1, retries + 1):
        missing = [t for t in tickers.values() if t not in close.columns]
        if not missing:
            break
        log.warning("yfinance 누락 %s, 재시도 %d/%d", missing, attempt, retries)
        time.sleep(5 * attempt)
        close = close.join(download(missing), how="outer")
    missing = [k for k, t in tickers.items() if t not in close.columns]
    if missing:
        log.warning("yfinance 수집 실패: %s", sorted(missing))
    return close.rename(columns={v: k for k, v in tickers.items()})


def fetch_fred(series_id: str, start: datetime) -> pd.Series:
    """FinanceDataReader → 실패 시 FRED 공개 CSV."""
    try:
        import FinanceDataReader as fdr

        df = fdr.DataReader(f"FRED:{series_id}", start.strftime("%Y-%m-%d"))
        s = df.iloc[:, 0]
    except Exception as exc:  # noqa: BLE001 - 외부 라이브러리 예외가 다양함
        log.info("FDR FRED 실패(%s), CSV 직접 호출로 전환", exc)
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start:%Y-%m-%d}"
        with urllib.request.urlopen(url, timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode()), index_col=0, parse_dates=True, na_values=".")
        s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index).normalize()
    return pd.to_numeric(s, errors="coerce").rename(series_id)


def fetch_fdr(symbol: str, start: datetime) -> pd.Series:
    import FinanceDataReader as fdr

    df = fdr.DataReader(symbol, start.strftime("%Y-%m-%d"))
    s = df["Close"]
    s.index = pd.to_datetime(s.index).normalize()
    return s.rename(symbol)


def despike(s: pd.Series, k: float = 1.8) -> pd.Series:
    """하루짜리 오류값 제거: 전·후일 모두의 k배 이상(또는 1/k 이하)이고 전·후일끼리는 비슷하면 두 값의 평균으로.

    KRX VKOSPI 원자료에 실제로 있다 (2019-04-10 30.40, 전후 12.67·12.29). 진짜 급등은 다음날에도 높게
    남으므로(2020-03, 2024-08-05 다음날도 고점권) 전·후일이 서로 비슷할 때만 고친다.
    """
    prev, nxt = s.shift(1), s.shift(-1)
    hi = (s > k * prev) & (s > k * nxt)
    lo = (s < prev / k) & (s < nxt / k)
    calm = (prev / nxt).between(1 / 1.3, 1.3)
    bad = (hi | lo) & calm
    if bad.any():
        log.warning("VKOSPI 하루 오류값 보정: %s", ", ".join(f"{d:%Y-%m-%d} {v:.2f}" for d, v in s[bad].items()))
    return s.mask(bad, (prev + nxt) / 2)


def load_vkospi(kospi200: pd.Series) -> tuple[pd.Series, str]:
    """VKOSPI 시리즈와 출처를 반환.

    - 'krx': data/vkospi.csv (KRX 실제값). 그 이전 구간은 proxy를 실제값과의
      겹치는 기간 중앙값 비율로 보정해 이어 붙인다 → 'krx+proxy'
    - 'proxy': KOSPI200 20일 실현변동성 × 1.15
    """
    rv = realized_vol(kospi200.dropna(), 20) * 100
    proxy = (rv * VKOSPI_PROXY_RATIO).rename("VKOSPI")
    if not VKOSPI_CSV.exists():
        return proxy, "proxy"
    df = pd.read_csv(VKOSPI_CSV, parse_dates=["Date"], index_col="Date")
    real = despike(pd.to_numeric(df["Close"], errors="coerce").dropna())
    if len(real) <= config.TRADING_DAYS:
        log.warning("vkospi.csv 데이터가 252일 미만이라 proxy 사용")
        return proxy, "proxy"
    overlap = pd.concat([real, rv], axis=1, join="inner").dropna()
    ratio = float((overlap.iloc[:, 0] / overlap.iloc[:, 1]).median()) if len(overlap) > 60 else VKOSPI_PROXY_RATIO
    early = (rv * ratio)[rv.index < real.index[0]]
    combined = pd.concat([early, real]).rename("VKOSPI")
    return combined, ("krx+proxy" if len(early) else "krx")


def vkospi_proxy_flag(vk: pd.Series, source: str) -> pd.Series:
    """VKOSPI 각 날짜가 proxy인지(1) 실제값인지(0). load_vkospi 결과와 출처로 만든다.

    'krx+proxy'는 실제값이 시작되기 전 구간만 proxy다. load_vkospi 가 실제 CSV 앞에 proxy를 붙이므로
    CSV의 첫 날짜를 경계로 쓴다(CSV를 다시 읽지 않도록 값이 바뀌는 지점 대신 파일 기준).
    """
    flag = pd.Series(1.0 if source == "proxy" else 0.0, index=vk.index, name=VKOSPI_PROXY_COL)
    if source == "krx+proxy" and VKOSPI_CSV.exists():
        first_real = pd.read_csv(VKOSPI_CSV, parse_dates=["Date"], index_col="Date").index.min()
        flag[flag.index < first_real] = 1.0
    return flag


# 장중 실행(수동 갱신 등)에서 yfinance는 아직 끝나지 않은 오늘 봉을 준다 → 종가 신호 모델에 미완성 가격이 섞인다.
# 시장이 닫히고 여유(데이터 확정 지연)를 둔 뒤에만 그날 값을 쓴다.
KR_CLOSE = ("Asia/Seoul", 15, 50)          # 15:30 마감 + 20분
US_CLOSE = ("America/New_York", 16, 15)    # 16:00 마감 + 15분 (환율·금리·FRED 포함 나머지 전부)


def is_kr_column(name: str) -> bool:
    """한국 시장 시각을 따르는 열: 한국 지수·VKOSPI·ECOS 시리즈·국내 상장 ETF(.KS/.KQ)."""
    from .kr_sources import ECOS_SERIES
    return (name in {"KOSPI", "KOSDAQ", "KOSPI200", "VKOSPI", "KR_CREDIT", *ECOS_SERIES}
            or name.endswith((".KS", ".KQ")) or name[:6].isdigit())


def drop_unclosed(raw: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """아직 마감되지 않은(또는 미래 날짜의) 봉을 NaN으로. 한국 열은 서울 15:50, 나머지는 뉴욕 16:15 기준."""
    raw = raw.copy()
    for (tz, hh, mm), kr in ((KR_CLOSE, True), (US_CLOSE, False)):
        local = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo(tz))
        today = pd.Timestamp(local.date())
        closed = (local.hour, local.minute) >= (hh, mm)
        cols = [c for c in raw.columns if is_kr_column(str(c)) == kr]
        late = raw.index > today if closed else raw.index >= today
        if cols and late.any() and raw.loc[late, cols].notna().any().any():
            log.info("미마감 봉 제외(%s): %s", tz, ", ".join(f"{d:%Y-%m-%d}" for d in raw.index[late]))
            raw.loc[late, cols] = np.nan
    return raw.dropna(how="all")


def fetch_raw(use_cache: bool = True, max_age_hours: float = 6) -> tuple[pd.DataFrame, dict]:
    """원시 종가 테이블(열=심볼, 휴장일 NaN 유지)과 메타 정보를 반환."""
    if use_cache and CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            with CACHE_FILE.open("rb") as f:
                return pickle.load(f)

    start = datetime.fromisoformat(config.HISTORY_START)
    raw = fetch_yfinance(config.YF_TICKERS | config.BREADTH_TICKERS, start)
    for name, sym in config.FDR_SYMBOLS.items():
        try:
            raw = raw.join(fetch_fdr(sym, start).rename(name), how="outer")
        except Exception as exc:  # noqa: BLE001
            log.error("FDR %s 수집 실패: %s", sym, exc)
    for name, sid in config.FRED_SERIES.items():
        try:
            raw = raw.join(fetch_fred(sid, start).rename(name), how="outer")
        except Exception as exc:  # noqa: BLE001
            log.error("FRED %s 수집 실패: %s", sid, exc)
    ecos = fetch_ecos_all(start.date())
    if not ecos.empty:
        raw = raw.join(ecos, how="outer")
    raw = drop_unclosed(raw)
    base = raw["KOSPI200"] if raw.get("KOSPI200", pd.Series()).notna().sum() > 60 else raw["KOSPI"]
    vk, vk_src = load_vkospi(base)
    raw = raw.join(vk, how="outer")
    raw = raw.join(vkospi_proxy_flag(vk, vk_src), how="outer")
    meta = {
        "fetched_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes"),
        "vkospi_source": vk_src,
        "kr_credit_source": "ecos" if "KR_CREDIT" in raw else None,
        "tickers": config.YF_TICKERS | config.FDR_SYMBOLS | {k: f"FRED:{v}" for k, v in config.FRED_SERIES.items()},
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as f:
        pickle.dump((raw, meta), f)
    return raw, meta


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """연율화 실현변동성 (소수, 0.20 = 20%)."""
    ret = np.log(close).diff()
    return ret.rolling(window).std() * np.sqrt(config.TRADING_DAYS)


def compute_features(close: pd.Series) -> pd.DataFrame:
    """단일 지수의 SMA, 52주 MDD, 20일 변동성, 3개월 ROC.

    입력은 해당 시장의 실제 거래일만 담긴 시리즈여야 한다.
    """
    close = close.dropna()
    df = pd.DataFrame({"close": close})
    for w in config.SMA_WINDOWS:
        df[f"sma{w}"] = close.rolling(w).mean()
    df["high_52w"] = close.rolling(config.TRADING_DAYS, min_periods=60).max()
    df["drawdown"] = close / df["high_52w"] - 1.0
    df["vol_20d"] = realized_vol(close, 20)
    df[f"roc_{config.TREND.roc_window}"] = close.pct_change(config.TREND.roc_window)
    # 장기 추세(실험용): 200일선, 12-1개월 모멘텀 (t−21일 종가 / t−252일 종가 − 1)
    df[f"sma{LONG_SMA}"] = close.rolling(LONG_SMA).mean()
    df["mom_12_1"] = close.shift(MOM_SKIP) / close.shift(MOM_LOOKBACK) - 1.0
    return df


def build_market_data(raw: pd.DataFrame, meta: dict | None = None, markets: dict = config.MARKETS) -> MarketData:
    """원시 데이터 → 공통 달력 정렬 + ffill + 지표 계산."""
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    # 공통 달력: 미국/한국 주식 거래일의 합집합 (주말 FX 등 잡음 제거)
    equity_cols = [c for m in markets.values() for c in m["indices"] if c in raw]
    calendar = raw[equity_cols].dropna(how="all").index

    features: dict[str, pd.DataFrame] = {}
    trade_dates: dict[str, pd.DatetimeIndex] = {}
    for col in equity_cols:
        features[col] = compute_features(raw[col]).reindex(calendar).ffill()
        trade_dates[col] = raw[col].dropna().index

    # 2003년 이전 원/달러는 FRED로 보완 (yfinance 시작 이전 구간에만)
    if "USDKRW_FRED" in raw and "USDKRW" in raw:
        first = raw["USDKRW"].first_valid_index()
        early = raw["USDKRW_FRED"].where(raw.index < first) if first is not None else raw["USDKRW_FRED"]
        raw["USDKRW"] = raw["USDKRW"].combine_first(early)

    prices = raw.reindex(raw.index.union(calendar)).ffill().reindex(calendar)
    return MarketData(prices=prices, features=features, meta=meta or {}, trade_dates=trade_dates)


def load_market_data(use_cache: bool = True) -> MarketData:
    raw, meta = fetch_raw(use_cache=use_cache)
    return build_market_data(raw, meta)


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    md = load_market_data(use_cache=False)
    print(md.prices.tail())
    print(json.dumps(md.to_payload(tail=1), ensure_ascii=False, indent=2)[:2000])
