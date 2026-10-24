"""한국 데이터 소스: ECOS(한국은행) 시장금리, KRX Open API VKOSPI.

둘 다 무료 개인 인증키가 필요하다. 키가 없으면 None/빈 결과를 돌려주고,
파이프라인은 기존 대체값(VKOSPI proxy, 미국 Baa 스프레드, 월간 3개월 금리)을 쓴다.

    ECOS_API_KEY  https://ecos.bok.or.kr/api  → 인증키 신청 (즉시 발급)
    KRX_API_KEY   https://openapi.krx.co.kr   → 인증키 신청 + '파생상품지수 시세정보' 이용신청

키는 환경변수 또는 프로젝트 루트의 `.env` (KEY=VALUE, git 제외)에서 읽는다.

    python -m beta_model.kr_sources --update-vkospi   # data/vkospi.csv 증분 갱신 (최초 실행 시 2010년~ 백필)
    python -m beta_model.kr_sources --check           # 키·연결 상태 점검
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
VKOSPI_CSV = ROOT / "data" / "vkospi.csv"
VKOSPI_START = date(2010, 1, 4)  # KRX Open API 제공 시작

ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{start}/{end}/{stat}/D/{frm}/{to}/{item}"
KRX_URL = "https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd"

# name: (통계표, 항목) — ECOS 817Y002 시장금리(일별)
ECOS_SERIES: dict[str, tuple[str, str]] = {
    "KR_AA3Y": ("817Y002", "010300000"),   # 회사채(3년, AA-)  1995~
    "KTB3Y": ("817Y002", "010200000"),     # 국고채(3년)       1998~
    "CD91": ("817Y002", "010502000"),      # CD(91일)          1995~  → 원화 현금 수익률
}


def get_key(name: str) -> str | None:
    """환경변수 → 프로젝트 루트 .env 순서로 키를 찾는다."""
    if os.environ.get(name):
        return os.environ[name].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


def _get_json(url: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "beta-model/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# ECOS
# ---------------------------------------------------------------------------
def parse_ecos(payload: dict) -> pd.Series:
    """ECOS StatisticSearch 응답 → 날짜 인덱스 Series. 데이터 없음/오류면 빈 Series."""
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows:
        msg = payload.get("RESULT", {}).get("MESSAGE")
        if msg:
            log.warning("ECOS: %s", msg[:80])
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(r["TIME"]): pd.to_numeric(r["DATA_VALUE"], errors="coerce") for r in rows}, dtype=float,
    )
    return s.sort_index()


def fetch_ecos(stat: str, item: str, start: date, key: str, end: date | None = None) -> pd.Series:
    end = end or date.today()
    url = ECOS_URL.format(key=key, start=1, end=100000, stat=stat, frm=f"{start:%Y%m%d}", to=f"{end:%Y%m%d}", item=item)
    return parse_ecos(_get_json(url, timeout=60))


def fetch_ecos_all(start: date) -> pd.DataFrame:
    """ECOS_SERIES 전체 + 파생 KR_CREDIT(= AA− − 국고 3년, %p). 키가 없으면 빈 DataFrame."""
    key = get_key("ECOS_API_KEY")
    if not key:
        log.info("ECOS_API_KEY 없음 → 국내 신용 스프레드/CD 금리 생략")
        return pd.DataFrame()
    cols = {}
    for name, (stat, item) in ECOS_SERIES.items():
        try:
            cols[name] = fetch_ecos(stat, item, start, key)
        except Exception as exc:  # noqa: BLE001
            log.error("ECOS %s 실패: %s", name, exc)
    df = pd.DataFrame(cols)
    if {"KR_AA3Y", "KTB3Y"} <= set(df.columns):
        df["KR_CREDIT"] = df["KR_AA3Y"] - df["KTB3Y"]
    return df


# ---------------------------------------------------------------------------
# KRX VKOSPI
# ---------------------------------------------------------------------------
def parse_vkospi(payload: dict) -> float | None:
    """KRX 파생상품지수 일별시세 응답에서 '코스피 200 변동성지수' 종가를 찾는다."""
    if str(payload.get("respCode", "200")) not in ("200", ""):
        raise PermissionError(f"KRX: {payload.get('respMsg')} ({payload.get('respCode')})")
    for row in payload.get("OutBlock_1", []) or []:
        name = str(row.get("IDX_NM", "")).replace(" ", "")
        if "VKOSPI" in name.upper() or ("변동성" in name and "200" in name and "코스피" in name):
            val = str(row.get("CLSPRC_IDX", "")).replace(",", "")
            try:
                return float(val)
            except ValueError:
                return None
    return None


def fetch_vkospi_day(d: date, key: str, retries: int = 3) -> float | None:
    url = f"{KRX_URL}?{urllib.parse.urlencode({'AUTH_KEY': key, 'basDd': f'{d:%Y%m%d}'})}"
    for attempt in range(retries):
        try:
            return parse_vkospi(_get_json(url))
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                log.warning("KRX %s 실패: %s", d, exc)
                return None
            time.sleep(1 + attempt)
    return None


def update_vkospi_csv(key: str | None = None, workers: int = 4) -> int:
    """data/vkospi.csv 에 빠진 영업일을 채운다. 추가된 행 수를 반환.

    KRX 데이터는 다음 날 08시(KST)에 갱신되므로 오늘 값은 보통 아직 없다.
    """
    key = key or get_key("KRX_API_KEY")
    if not key:
        log.info("KRX_API_KEY 없음 → VKOSPI 갱신 생략 (proxy 사용)")
        return 0
    existing = pd.read_csv(VKOSPI_CSV, parse_dates=["Date"]) if VKOSPI_CSV.exists() else pd.DataFrame(columns=["Date", "Close"])
    last = existing["Date"].max().date() if len(existing) else VKOSPI_START - timedelta(days=1)
    days = [d.date() for d in pd.bdate_range(last + timedelta(days=1), date.today())]
    if not days:
        return 0
    # 키·서비스 승인 여부를 먼저 한 번 확인 (실패 시 PermissionError로 즉시 중단)
    fetch_vkospi_day(days[-1], key)
    log.info("VKOSPI %d영업일 조회 (%s ~ %s)", len(days), days[0], days[-1])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        values = list(ex.map(lambda d: fetch_vkospi_day(d, key), days))
    new = pd.DataFrame({"Date": pd.to_datetime(days), "Close": values}).dropna()
    if new.empty:
        return 0
    out = pd.concat([existing, new]).drop_duplicates("Date", keep="last").sort_values("Date")
    VKOSPI_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(VKOSPI_CSV, index=False, date_format="%Y-%m-%d")
    return len(new)


# ---------------------------------------------------------------------------
def check() -> None:
    ecos, krx = get_key("ECOS_API_KEY"), get_key("KRX_API_KEY")
    print(f"ECOS_API_KEY: {'있음' if ecos else '없음'}")
    if ecos:
        s = fetch_ecos(*ECOS_SERIES["KR_AA3Y"], date.today() - timedelta(days=14), ecos)
        print(f"  회사채 AA− 3년 최근: {s.tail(1).to_dict() if len(s) else '조회 실패'}")
    print(f"KRX_API_KEY : {'있음' if krx else '없음'}")
    if krx:
        d = (pd.Timestamp.today() - pd.offsets.BDay(1)).date()
        try:
            print(f"  VKOSPI {d}: {fetch_vkospi_day(d, krx)}")
        except PermissionError as exc:
            print(f"  {exc} → openapi.krx.co.kr 에서 '파생상품지수 시세정보' 이용신청 승인 여부를 확인하세요")
    if VKOSPI_CSV.exists():
        df = pd.read_csv(VKOSPI_CSV)
        print(f"data/vkospi.csv: {len(df)}행 ({df['Date'].min()} ~ {df['Date'].max()})")


def main() -> None:
    ap = argparse.ArgumentParser(description="한국 데이터 소스 (ECOS, KRX)")
    ap.add_argument("--update-vkospi", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.update_vkospi:
        n = update_vkospi_csv()
        print(f"VKOSPI {n}행 추가 → {VKOSPI_CSV}")
    if args.check or not args.update_vkospi:
        check()


if __name__ == "__main__":
    main()
