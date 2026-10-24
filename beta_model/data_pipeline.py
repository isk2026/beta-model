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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "cache" / "raw.pkl"
VKOSPI_CSV = DATA_DIR / "vkospi.csv"
VKOSPI_PROXY_RATIO = 1.15  # 내재변동성은 통상 실현변동성보다 10~20% 높다


@dataclass
class MarketData:
    """스코어링 엔진에 넘기는 정제된 데이터 묶음."""

    prices: pd.DataFrame                       # 공통 달력, ffill 된 종가 (wide)
    features: dict[str, pd.DataFrame]          # 지수별 SMA/MDD/변동성/ROC
    meta: dict = field(default_factory=dict)   # 출처, proxy 여부 등

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


def load_vkospi(kospi200: pd.Series) -> tuple[pd.Series, str]:
    """VKOSPI 시리즈와 출처('csv' | 'proxy')를 반환."""
    if VKOSPI_CSV.exists():
        df = pd.read_csv(VKOSPI_CSV, parse_dates=[0], index_col=0)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if len(s) > config.TRADING_DAYS:
            return s.rename("VKOSPI"), "csv"
        log.warning("vkospi.csv 데이터가 252일 미만이라 proxy 사용")
    rv = realized_vol(kospi200.dropna(), 20) * 100 * VKOSPI_PROXY_RATIO
    return rv.rename("VKOSPI"), "proxy"


def fetch_raw(use_cache: bool = True, max_age_hours: float = 6) -> tuple[pd.DataFrame, dict]:
    """원시 종가 테이블(열=심볼, 휴장일 NaN 유지)과 메타 정보를 반환."""
    if use_cache and CACHE_FILE.exists():
        age_h = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            with CACHE_FILE.open("rb") as f:
                return pickle.load(f)

    start = datetime.now() - timedelta(days=int(365 * config.HISTORY_YEARS) + 30)
    raw = fetch_yfinance(config.YF_TICKERS, start)
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
    base = raw["KOSPI200"] if raw.get("KOSPI200", pd.Series()).notna().sum() > 60 else raw["KOSPI"]
    vk, vk_src = load_vkospi(base)
    raw = raw.join(vk, how="outer")
    meta = {
        "fetched_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="minutes"),
        "vkospi_source": vk_src,
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
    return df


def build_market_data(raw: pd.DataFrame, meta: dict | None = None) -> MarketData:
    """원시 데이터 → 공통 달력 정렬 + ffill + 지표 계산."""
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    # 공통 달력: 미국/한국 주식 거래일의 합집합 (주말 FX 등 잡음 제거)
    equity_cols = [c for m in config.MARKETS.values() for c in m["indices"] if c in raw]
    calendar = raw[equity_cols].dropna(how="all").index

    features: dict[str, pd.DataFrame] = {}
    for col in equity_cols:
        features[col] = compute_features(raw[col]).reindex(calendar).ffill()

    prices = raw.reindex(raw.index.union(calendar)).ffill().reindex(calendar)
    return MarketData(prices=prices, features=features, meta=meta or {})


def load_market_data(use_cache: bool = True) -> MarketData:
    raw, meta = fetch_raw(use_cache=use_cache)
    return build_market_data(raw, meta)


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    md = load_market_data(use_cache=False)
    print(md.prices.tail())
    print(json.dumps(md.to_payload(tail=1), ensure_ascii=False, indent=2)[:2000])
