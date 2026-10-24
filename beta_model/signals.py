"""실전 기록과 알림.

매 실행마다 data/signal_log.csv 에 (시장, 기준일)별 한 줄을 남긴다(같은 기준일은 덮어씀).
나중에 '그때 모델이 실제로 뭐라고 했나'를 백테스트가 아닌 기록으로 검증하기 위함.

알림 조건 — 국면이 바뀌었거나 다음 거래일에 매매가 필요할 때 data/alert.md 를 쓴다.
워크플로가 이 파일을 GitHub Issue(·텔레그램)로 보낸다.
"""

from __future__ import annotations

import pandas as pd

from .allocation import REGIME_INFO, Regime, smooth_weights
from .data_pipeline import ROOT

LOG_FILE = ROOT / "data" / "signal_log.csv"
ALERT_FILE = ROOT / "data" / "alert.md"
COLUMNS = ["as_of", "market", "close", "trend", "risk", "regime", "target", "executed", "next_weight", "action", "run_at"]


def next_weight(target: pd.Series) -> float:
    """오늘 종가 신호를 반영해 '다음 거래일에 들고 있어야 할' 비중 (속도 조절·밴드 포함)."""
    nxt = target.index[-1] + pd.offsets.BDay(1)
    ext = pd.concat([target, pd.Series([target.iloc[-1]], index=[nxt])])
    return float(smooth_weights(ext).iloc[-1])


def action_of(cur: float, nxt: float, tol: float = 0.005) -> str:
    return "SELL" if nxt < cur - tol else "BUY" if nxt > cur + tol else "HOLD"


def update_log(payload: dict, run_at: str) -> tuple[pd.DataFrame, set[str]]:
    """payload(export.build_payload 결과)를 로그에 반영. (전체 로그, 이번에 새 기준일이 생긴 시장)을 반환.

    하루 두 번 실행되므로 한 시장은 같은 기준일로 두 번 기록될 수 있다 → 덮어쓰고 알림은 처음 한 번만.
    """
    rows = []
    for m, d in payload["markets"].items():
        h = d["history"][-1]
        rows.append({
            "as_of": d["as_of"], "market": m, "close": h["index"],
            "trend": h["trend"], "risk": h["risk"], "regime": d["allocation"]["regime"],
            "target": h["target"], "executed": d["executed_weight"], "next_weight": round(d["next_weight"], 4),
            "action": action_of(d["executed_weight"], d["next_weight"]), "run_at": run_at,
        })
    new = pd.DataFrame(rows, columns=COLUMNS)
    log = pd.read_csv(LOG_FILE, dtype={"as_of": str}) if LOG_FILE.exists() else pd.DataFrame(columns=COLUMNS)
    seen = set(zip(log["market"], log["as_of"]))
    fresh = {r["market"] for r in rows if (r["market"], r["as_of"]) not in seen}
    out = pd.concat([log, new]).drop_duplicates(["market", "as_of"], keep="last").sort_values(["as_of", "market"])
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG_FILE, index=False)
    return out, fresh


def build_alert(log: pd.DataFrame, labels: dict[str, str], markets: set[str]) -> str | None:
    """`markets`의 최신 행을 직전 행과 비교. 알릴 것이 없으면 None (alert.md 삭제)."""
    lines = []
    for m, g in log.groupby("market"):
        if m not in markets:
            continue
        g = g.sort_values("as_of")
        cur = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        msgs = []
        if prev is not None and int(prev.regime) != int(cur.regime):
            msgs.append(f"국면 전환 {int(prev.regime)} {REGIME_INFO[Regime(int(prev.regime))]['name']} → "
                        f"**{int(cur.regime)} {REGIME_INFO[Regime(int(cur.regime))]['name']}**")
        if cur.action != "HOLD":
            verb = "줄이기(일괄)" if cur.action == "SELL" else "늘리기(분할)"
            msgs.append(f"다음 거래일 주식 {cur.executed:.0%} → **{cur.next_weight:.0%}** {verb}")
        if msgs:
            lines.append(f"### {labels.get(m, m)} ({cur.as_of} 종가 기준)\n" + "\n".join(f"- {s}" for s in msgs)
                         + f"\n- 추세 {cur.trend:+.2f} · 위험 {cur.risk:.2f} · 목표 {cur.target:.0%}")
    if not lines:
        if ALERT_FILE.exists():
            ALERT_FILE.unlink()
        return None
    text = "\n\n".join(lines) + "\n\n대시보드: https://isk2026.github.io/beta-model/\n\n_투자 참고용 신호이며 투자 권유가 아닙니다._\n"
    ALERT_FILE.write_text(text, encoding="utf-8")
    return text


def live_payload(log: pd.DataFrame, n: int = 30) -> dict:
    """대시보드용: 시장별 최근 n개 기록 + 기록 시작 이후 실전 성과(주식 부분만, 현금 수익 0 가정)."""
    out = {}
    for m, g in log.groupby("market"):
        g = g.sort_values("as_of").reset_index(drop=True)
        ret = g["close"].pct_change().fillna(0)
        held = g["next_weight"].shift(1).fillna(g["executed"].iloc[0])   # 전날 결정한 비중으로 오늘 수익
        model = float((1 + held * ret).prod() - 1)
        bh = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1) if len(g) > 1 else 0.0
        out[m] = {
            "since": g["as_of"].iloc[0], "days": len(g), "model_return": round(model, 4), "index_return": round(bh, 4),
            "regime_changes": int((g["regime"].diff().fillna(0) != 0).sum()),
            "trades": int((g["action"] != "HOLD").sum()),
            "rows": g.tail(n).iloc[::-1][["as_of", "close", "trend", "risk", "regime", "target", "executed", "next_weight", "action"]]
            .to_dict("records"),
        }
    return out
