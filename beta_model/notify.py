"""알림 메시지 서식과 텔레그램 전송.

- 신호 알림: signals.build_alert 가 GitHub Issue용 마크다운(data/alert.md)과
  텔레그램용 HTML(data/alert_telegram.html)을 같은 내용으로 만든다.
- 주간 요약(텔레그램 전용): 일요일 실행에서 `python -m beta_model.notify weekly`.
- 주도 섹터 변화 추적: data/sector_leaders.json 에 기준일별 주도 섹터를 남긴다(워크플로가 커밋).

전송은 실패해도 예외를 밖으로 내보내지 않고(워크플로를 깨지 않음), 오류 메시지에서 봇 토큰을 가린다.
텔레그램 설정(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)이 없으면 아무것도 하지 않는다.

    python -m beta_model.notify send data/alert_telegram.html
    python -m beta_model.notify weekly [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .allocation import REGIME_INFO, Regime
from .data_pipeline import ROOT

DASHBOARD_URL = "https://isk2026.github.io/beta-model/"
DISCLAIMER = "투자 참고용 신호이며 투자 권유가 아닙니다."
LEADERS_FILE = ROOT / "data" / "sector_leaders.json"
TELEGRAM_FILE = ROOT / "data" / "alert_telegram.html"
PAYLOAD_FILE = ROOT / "dashboard" / "public" / "data" / "beta_model.json"
SECTOR_MARKETS = ("US", "US_TECH")      # 주도 섹터(미국 SPDR)를 함께 보여줄 시장
LEADERS_KEEP = 60
TG_LIMIT = 4096


# ---------------------------------------------------------------------------
# 서식: 같은 내용을 마크다운(Issue) / HTML(텔레그램 parse_mode=HTML) 으로
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fmt:
    html: bool

    def t(self, s) -> str:
        """일반 텍스트 (HTML이면 &, <, > 이스케이프)."""
        return html.escape(str(s), quote=False) if self.html else str(s)

    def b(self, s) -> str:
        return f"<b>{self.t(s)}</b>" if self.html else f"**{s}**"

    def i(self, s) -> str:
        return f"<i>{self.t(s)}</i>" if self.html else f"_{s}_"

    def head(self, s) -> str:
        return self.b(s) if self.html else f"### {s}"

    def link(self, text, url: str) -> str:
        return f'<a href="{html.escape(url, quote=True)}">{self.t(text)}</a>' if self.html else f"[{text}]({url})"

    def bullets(self, items: list[str]) -> str:
        mark = "• " if self.html else "- "
        return "\n".join(mark + s for s in items)


MD, HTML = Fmt(False), Fmt(True)


def regime_label(r) -> str:
    r = int(r)
    return f"{r} {REGIME_INFO[Regime(r)]['name']}"


def pct(x) -> str:
    return f"{float(x):.0%}"


def trade_text(f: Fmt, action: str, cur: float, nxt: float) -> str:
    if action == "SELL":
        return f"다음 거래일 주식 {pct(cur)} → {f.b(pct(nxt))} {f.b('일괄 줄이기')}"
    if action == "BUY":
        return f"다음 거래일 주식 {pct(cur)} → {f.b(pct(nxt))} {f.b('분할 늘리기')}"
    return f"주식 {pct(cur)} 유지 (매매 없음)"


# ---------------------------------------------------------------------------
# 주도 섹터
# ---------------------------------------------------------------------------
def sector_names(sectors: dict | None) -> dict[str, str]:
    if not sectors:
        return {}
    return {x["ticker"]: x["name"] for x in [*sectors.get("sectors", []), *sectors.get("tech", [])]}


def update_leaders(sectors: dict | None, path: Path | None = None) -> dict | None:
    """기준일별 주도 섹터를 기록하고 직전 기준일과의 차이를 돌려준다.

    같은 기준일로 다시 실행되면 마지막 항목을 덮어쓰므로 '직전 기준일 대비' 변화가 유지된다.
    반환: {as_of, leaders, prev_as_of, prev_leaders, added, removed, history}
    """
    if not sectors or "leaders" not in sectors:
        return None
    path = path or LEADERS_FILE
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        state = {}
    hist = [h for h in state.get("history", []) if isinstance(h, dict) and "as_of" in h]
    entry = {"as_of": sectors["as_of"], "leaders": list(sectors["leaders"])}
    hist = [h for h in hist if h["as_of"] != entry["as_of"]]
    hist = sorted([*hist, entry], key=lambda h: h["as_of"])[-LEADERS_KEEP:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"history": hist}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return leaders_diff(hist, entry["as_of"])


def leaders_diff(hist: list[dict], as_of: str) -> dict | None:
    hist = sorted(hist, key=lambda h: h["as_of"])
    idx = next((i for i, h in enumerate(hist) if h["as_of"] == as_of), None)
    if idx is None:
        return None
    cur = hist[idx]
    prev = hist[idx - 1] if idx > 0 else None
    pl = prev["leaders"] if prev else None
    return {
        "as_of": cur["as_of"], "leaders": cur["leaders"],
        "prev_as_of": prev["as_of"] if prev else None, "prev_leaders": pl,
        "added": [t for t in cur["leaders"] if pl is not None and t not in pl],
        "removed": [t for t in (pl or []) if t not in cur["leaders"]],
        "history": hist,
    }


def _tickers(f: Fmt, tickers: list[str], names: dict[str, str]) -> str:
    return ", ".join(f.t(f"{t} {names.get(t, '')}".strip()) for t in tickers) or f.t("없음")


def sector_items(f: Fmt, market: str, sectors: dict | None, leaders: dict | None = None) -> list[str]:
    """US/US_TECH 시장에 붙일 주도 섹터 줄 (참고 정보)."""
    if market not in SECTOR_MARKETS or not sectors:
        return []
    names = sector_names(sectors)
    items = [f"주도 섹터: {_tickers(f, sectors.get('leaders', []), names)}"]
    if leaders and leaders.get("prev_leaders") is not None and (leaders["added"] or leaders["removed"]):
        ch = []
        if leaders["added"]:
            ch.append(f"신규 {_tickers(f, leaders['added'], names)}")
        if leaders["removed"]:
            ch.append(f"이탈 {_tickers(f, leaders['removed'], names)}")
        items.append(f"주도 섹터 변화 ({f.t(leaders['prev_as_of'])} 대비): " + " / ".join(ch))
    if market == "US_TECH":
        tech = [f.t(f"{x['ticker']} 추세 {x['trend']:+.2f} (6M 대비 SPY {x['rel_6m']:+.0%})") for x in sectors.get("tech", [])]
        xlk = next((x for x in sectors.get("sectors", []) if x["ticker"] == "XLK"), None)
        if xlk:
            tech.append(f.t(f"XLK 순위 {xlk['rank']}/{len(sectors['sectors'])}"))
        if tech:
            items.append(" · ".join(tech))
    return items


# ---------------------------------------------------------------------------
# 신호 알림
# ---------------------------------------------------------------------------
def alert_sections(log: pd.DataFrame, labels: dict[str, str], markets: set[str]) -> list[dict]:
    """알릴 시장별 (현재 행, 직전 행, 국면 변화, 매매) 정리. 알릴 것이 없으면 빈 리스트."""
    out = []
    for m, g in log.groupby("market"):
        if m not in markets:
            continue
        g = g.sort_values("as_of")
        cur = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        changed = prev is not None and int(prev.regime) != int(cur.regime)
        if changed or cur.action != "HOLD":
            out.append({"market": m, "label": labels.get(m, m), "cur": cur,
                        "prev_regime": int(prev.regime) if changed else None})
    order = list(labels)        # 대시보드와 같은 시장 순서 (미국 → 미국(기술) → 한국)
    return sorted(out, key=lambda s: order.index(s["market"]) if s["market"] in order else len(order))


def render_alert(f: Fmt, sections: list[dict], sectors: dict | None = None, leaders: dict | None = None) -> str:
    blocks = [f.b("Beta Model 신호 알림")] if f.html else []
    for s in sections:
        c = s["cur"]
        items = []
        if s["prev_regime"] is not None:
            items.append(f"국면 전환 {f.t(regime_label(s['prev_regime']))} → {f.b(regime_label(c.regime))}")
        else:
            items.append(f"국면 {f.t(regime_label(c.regime))} (유지)")
        items.append(trade_text(f, c.action, c.executed, c.next_weight))
        items.append(f.t(f"추세 {c.trend:+.2f} · 위험 {c.risk:.2f} · 목표 {pct(c.target)}"))
        items += sector_items(f, s["market"], sectors, leaders)
        blocks.append(f.head(f"{s['label']} ({c.as_of} 종가 기준)") + "\n" + f.bullets(items))
    blocks.append(f"대시보드: {f.link(DASHBOARD_URL, DASHBOARD_URL)}" if f.html else f"대시보드: {DASHBOARD_URL}")
    blocks.append(f.i(DISCLAIMER))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# 주간 요약 (텔레그램 전용)
# ---------------------------------------------------------------------------
def week_events(g: pd.DataFrame, start: str) -> tuple[list[str], list[str]]:
    """한 시장 로그에서 start(포함) 이후 국면 전환·매매 목록 (평문)."""
    g = g.sort_values("as_of").reset_index(drop=True)
    changes, trades = [], []
    for i, r in g.iterrows():
        if r.as_of < start:
            continue
        d = r.as_of[5:].replace("-", "/")
        if i > 0 and int(g.regime[i - 1]) != int(r.regime):
            changes.append(f"{d} {regime_label(g.regime[i - 1])} → {regime_label(r.regime)}")
        if r.action != "HOLD":
            verb = "일괄 줄이기" if r.action == "SELL" else "분할 늘리기"
            trades.append(f"{d} {pct(r.executed)} → {pct(r.next_weight)} {verb}")
    return changes, trades


def build_weekly(payload: dict, log: pd.DataFrame, today: date, leaders_hist: list[dict] | None = None) -> str:
    """주간 요약 (텔레그램 HTML). today 포함 최근 7일."""
    f = HTML
    start = (today - timedelta(days=6)).isoformat()
    blocks = [f.b(f"Beta Model 주간 요약 ({start[5:].replace('-', '/')} ~ {today.strftime('%m/%d')})")]
    sectors = payload.get("sectors")
    for m, d in payload.get("markets", {}).items():
        a = d["allocation"]
        items = [
            f"국면 {f.b(regime_label(a['regime']))}",
            f"주식 {f.b(pct(d['executed_weight']))}"
            + (f.t(f" (다음 거래일 {pct(d['next_weight'])})") if abs(d["next_weight"] - d["executed_weight"]) > 0.005 else ""),
            f.t(f"추세 {a['trend_score']:+.2f} · 위험 {a['risk_score']:.2f} · 목표 {pct(a['stock_weight'])}"),
        ]
        g = log[log["market"] == m] if len(log) else log
        changes, trades = week_events(g, start) if len(g) else ([], [])
        if changes:
            items.append("이번 주 국면 전환: " + f.t("; ".join(changes)))
        if trades:
            items.append("이번 주 매매: " + f.t("; ".join(trades)))
        if not changes and not trades:
            items.append(f.t("이번 주 국면 전환·매매 없음"))
        items += sector_items(f, m, sectors)
        blocks.append(f.head(f"{d.get('label', m)} ({d['as_of']} 기준)") + "\n" + f.bullets(items))
    if sectors and leaders_hist:
        wk = [h for h in sorted(leaders_hist, key=lambda h: h["as_of"]) if h["as_of"] >= start]
        before = [h for h in sorted(leaders_hist, key=lambda h: h["as_of"]) if h["as_of"] < start]
        base = before[-1] if before else (wk[0] if wk else None)
        if base and wk:
            names = sector_names(sectors)
            added = [t for t in wk[-1]["leaders"] if t not in base["leaders"]]
            removed = [t for t in base["leaders"] if t not in wk[-1]["leaders"]]
            if added or removed:
                ch = ([f"신규 {_tickers(f, added, names)}"] if added else []) + \
                     ([f"이탈 {_tickers(f, removed, names)}"] if removed else [])
                blocks.append(f"주도 섹터 주간 변화 ({f.t(base['as_of'])} 대비): " + " / ".join(ch))
    blocks.append(f"대시보드: {f.link(DASHBOARD_URL, DASHBOARD_URL)}")
    blocks.append(f.i(DISCLAIMER))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# 텔레그램 전송
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


def redact(msg: str, token: str | None) -> str:
    msg = str(msg)
    if token:
        msg = msg.replace(token, "***")
        msg = msg.replace(urllib.parse.quote(token, safe=""), "***")
    return _TOKEN_RE.sub("bot***", msg)


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """빈 줄(문단) 경계로 limit 이하 조각으로 나눈다 (태그가 문단을 넘지 않으므로 HTML이 깨지지 않음)."""
    parts, cur = [], ""
    for para in text.strip().split("\n\n"):
        while len(para) > limit:                    # 한 문단이 너무 길면 줄 단위로
            cut = para.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(para[:cut])
            para = para[cut:].lstrip("\n")
        cand = f"{cur}\n\n{para}" if cur else para
        if len(cand) > limit:
            parts.append(cur)
            cur = para
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None,
                  opener=None, timeout: float = 20) -> bool:
    """parse_mode=HTML 로 전송. 설정이 없으면 건너뛰고, 어떤 오류도 밖으로 내보내지 않는다."""
    token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("텔레그램 설정 없음 — 전송 건너뜀")
        return False
    if not text or not text.strip():
        print("보낼 내용 없음")
        return False
    opener = opener or urllib.request.urlopen
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok = True
    for part in split_message(text):
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": part, "parse_mode": "HTML",
                                       "disable_web_page_preview": "true"}).encode()
        try:
            with opener(urllib.request.Request(url, data=data, method="POST"), timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
            if not body.get("ok", False):
                ok = False
                print("텔레그램 전송 실패: " + redact(body.get("description", "unknown"), token), file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — 알림 실패가 워크플로를 깨면 안 된다
            ok = False
            detail = ""
            try:  # HTTPError 본문에 텔레그램 오류 설명이 있다
                detail = exc.read().decode("utf-8", "replace")[:300]  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            print("텔레그램 전송 실패: " + redact(f"{type(exc).__name__}: {exc} {detail}".strip(), token), file=sys.stderr)
    if ok:
        print("텔레그램 전송 완료")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_weekly_inputs(payload_file: Path = PAYLOAD_FILE) -> tuple[dict, pd.DataFrame, list[dict]]:
    from .signals import LOG_FILE
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    log = pd.read_csv(LOG_FILE, dtype={"as_of": str}) if LOG_FILE.exists() else pd.DataFrame(
        columns=["as_of", "market", "regime", "action", "executed", "next_weight"])
    hist = json.loads(LEADERS_FILE.read_text(encoding="utf-8")).get("history", []) if LEADERS_FILE.exists() else []
    return payload, log, hist


def kst_today() -> date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
    except Exception:  # noqa: BLE001
        return (datetime.utcnow() + timedelta(hours=9)).date()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Beta Model 텔레그램 알림")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send", help="HTML 파일 내용을 텔레그램으로 전송")
    s.add_argument("file", type=Path)
    w = sub.add_parser("weekly", help="주간 요약 전송")
    w.add_argument("--dry-run", action="store_true", help="전송하지 않고 출력만")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "send":
            if not args.file.exists():
                print(f"{args.file} 없음 — 전송 건너뜀")
                return 0
            send_telegram(args.file.read_text(encoding="utf-8"))
        else:
            payload, log, hist = load_weekly_inputs()
            text = build_weekly(payload, log, kst_today(), hist)
            print(text)
            if not args.dry_run:
                send_telegram(text)
    except Exception as exc:  # noqa: BLE001 — 알림은 어떤 경우에도 워크플로를 실패시키지 않는다
        print("알림 처리 실패: " + redact(f"{type(exc).__name__}: {exc}", os.environ.get("TELEGRAM_BOT_TOKEN")),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
