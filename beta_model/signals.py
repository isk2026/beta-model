"""실전 기록과 알림.

매 실행마다 data/signal_log.csv 에 (시장, 기준일)별 한 줄을 남긴다.
나중에 '그때 모델이 실제로 뭐라고 했나'를 백테스트가 아닌 기록으로 검증하기 위함.

하루 두 번 실행되므로 같은 (시장, 기준일)이 다시 계산될 수 있다 — 예: 16:30 KST 한국 신호는 그날 밤
미국 종가(VIX·금리·신용)가 나오기 전 값이고, 다음날 07:30 실행에서 미국 종가를 반영해 다시 계산된다.
- 기본 열(trend·risk·…·action)은 최신 재계산 값으로 갱신한다 (백테스트와 같은 정보 시점).
- first_* 열은 그 기준일을 처음 기록했을 때(=최초 알림 시점)의 값으로, 덮어쓰지 않는다.
- alert_* 열은 마지막으로 알린 신호. 재계산으로 매매·국면·다음날 비중이 실질적으로 바뀌면 '정정' 알림을 낸다.

알림 조건 — 새 기준일에서 국면이 바뀌었거나 다음 거래일에 매매가 필요할 때, 또는 위의 정정이 있을 때
data/alert.md 를 쓴다. 워크플로가 이 파일을 GitHub Issue로, data/alert_telegram.html 을 텔레그램으로 보낸다.
"""

from __future__ import annotations

import pandas as pd

from . import notify
from .allocation import smooth_weights
from .data_pipeline import ROOT

LOG_FILE = ROOT / "data" / "signal_log.csv"
ALERT_FILE = ROOT / "data" / "alert.md"
BASE_COLUMNS = ["as_of", "market", "close", "trend", "risk", "regime", "target", "executed", "next_weight", "action", "run_at"]
FIRST_COLUMNS = ["first_run_at", "first_trend", "first_risk", "first_regime", "first_next_weight", "first_action"]
ALERT_COLUMNS = ["alert_regime", "alert_action", "alert_next_weight", "revisions"]
COLUMNS = BASE_COLUMNS + FIRST_COLUMNS + ALERT_COLUMNS
# 재계산 정정 알림 기준: 매매 방향·국면이 바뀌거나, 매매가 있는 날 다음날 비중이 이만큼 이상 달라질 때
REVISION_TOL = 0.02


def next_weight(target: pd.Series) -> float:
    """오늘 종가 신호를 반영해 '다음 거래일에 들고 있어야 할' 비중 (속도 조절·밴드 포함)."""
    nxt = target.index[-1] + pd.offsets.BDay(1)
    ext = pd.concat([target, pd.Series([target.iloc[-1]], index=[nxt])])
    return float(smooth_weights(ext).iloc[-1])


def action_of(cur: float, nxt: float, tol: float = 0.005) -> str:
    return "SELL" if nxt < cur - tol else "BUY" if nxt > cur + tol else "HOLD"


def _with_tracking_columns(log: pd.DataFrame) -> pd.DataFrame:
    """이전 형식 로그(first_*/alert_* 없음)를 현재 형식으로. 없는 값은 기본 열로 채운다(최선의 근사)."""
    log = log.copy()
    for first, base in (("first_run_at", "run_at"), ("first_trend", "trend"), ("first_risk", "risk"),
                        ("first_regime", "regime"), ("first_next_weight", "next_weight"), ("first_action", "action"),
                        ("alert_regime", "regime"), ("alert_action", "action"), ("alert_next_weight", "next_weight")):
        if first not in log:
            log[first] = log[base] if base in log else None
        else:
            log[first] = log[first].fillna(log[base])
    log["revisions"] = pd.to_numeric(log.get("revisions"), errors="coerce").fillna(0).astype(int) \
        if "revisions" in log else 0
    return log[COLUMNS]


def is_material_change(alerted: dict, new: dict, tol: float = REVISION_TOL) -> bool:
    """마지막으로 알린 신호(alerted) 대비 재계산 신호(new)가 행동을 바꿀 만큼 다른가."""
    if new["action"] != alerted["action"] or int(new["regime"]) != int(alerted["regime"]):
        return True
    return new["action"] != "HOLD" and abs(float(new["next_weight"]) - float(alerted["next_weight"])) >= tol


def update_log(payload: dict, run_at: str) -> tuple[pd.DataFrame, set[str], dict[str, dict]]:
    """payload(export.build_payload 결과)를 로그에 반영.

    Returns:
        (전체 로그, 새 기준일이 생긴 시장, 재계산으로 신호가 바뀐 시장 → 그 전에 알렸던 신호)
    """
    log = pd.read_csv(LOG_FILE, dtype={"as_of": str}) if LOG_FILE.exists() else pd.DataFrame(columns=BASE_COLUMNS)
    log = _with_tracking_columns(log)
    prev = {(r.market, r.as_of): r for r in log.itertuples(index=False)}

    rows, fresh, revised = [], set(), {}
    for m, d in payload["markets"].items():
        h = d["history"][-1]
        row = {
            "as_of": d["as_of"], "market": m, "close": h["index"],
            "trend": h["trend"], "risk": h["risk"], "regime": d["allocation"]["regime"],
            "target": h["target"], "executed": d["executed_weight"], "next_weight": round(d["next_weight"], 4),
            "action": action_of(d["executed_weight"], d["next_weight"]), "run_at": run_at,
        }
        old = prev.get((m, d["as_of"]))
        if old is None:                       # 처음 기록하는 기준일 = 알림 시점 스냅샷
            fresh.add(m)
            row |= {"first_run_at": run_at, "first_trend": row["trend"], "first_risk": row["risk"],
                    "first_regime": row["regime"], "first_next_weight": row["next_weight"],
                    "first_action": row["action"], "alert_regime": row["regime"], "alert_action": row["action"],
                    "alert_next_weight": row["next_weight"], "revisions": 0}
        else:                                 # 재계산: 최초값은 보존, 실질 변화면 정정
            row |= {c: getattr(old, c) for c in FIRST_COLUMNS + ALERT_COLUMNS}
            alerted = {"regime": old.alert_regime, "action": old.alert_action, "next_weight": old.alert_next_weight}
            if is_material_change(alerted, row):
                revised[m] = alerted
                row |= {"alert_regime": row["regime"], "alert_action": row["action"],
                        "alert_next_weight": row["next_weight"], "revisions": int(old.revisions) + 1}
        rows.append(row)

    new = pd.DataFrame(rows, columns=COLUMNS)
    out = pd.concat([log, new]) if len(log) else new
    out = out.drop_duplicates(["market", "as_of"], keep="last").sort_values(["as_of", "market"])
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG_FILE, index=False)
    return out, fresh, revised


def build_alert(log: pd.DataFrame, labels: dict[str, str], markets: set[str],
                sectors: dict | None = None, leaders: dict | None = None,
                revised: dict[str, dict] | None = None) -> str | None:
    """`markets`(새 기준일)의 최신 행을 직전 행과 비교하고, `revised`(재계산 정정)를 덧붙인다.
    알릴 것이 없으면 None (알림 파일 삭제).

    GitHub Issue용 마크다운(alert.md)과 텔레그램용 HTML(alert_telegram.html)을 함께 쓰고 마크다운을 돌려준다.
    sectors/leaders(선택)가 있으면 미국·미국(기술)에 주도 섹터와 그 변화를 덧붙인다.
    """
    sections = notify.alert_sections(log, labels, markets, revised)
    tg_file = ALERT_FILE.with_name(notify.TELEGRAM_FILE.name)
    if not sections:
        for f in (ALERT_FILE, tg_file):
            if f.exists():
                f.unlink()
        return None
    text = notify.render_alert(notify.MD, sections, sectors, leaders)
    ALERT_FILE.write_text(text, encoding="utf-8")
    tg_file.write_text(notify.render_alert(notify.HTML, sections, sectors, leaders), encoding="utf-8")
    return text


def live_payload(log: pd.DataFrame, n: int = 30) -> dict:
    """대시보드용: 시장별 최근 n개 기록 + 기록 시작 이후 실전 성과(주식 부분만, 현금 수익 0 가정).

    시차는 백테스트와 같다: t일 종가 신호 → t+1일 종가 집행 → t+2일 수익부터 반영.
    즉 t일 수익에는 t−2일에 마지막으로 알린 '다음날 비중'(alert_next_weight)이 적용된다.
    처음 이틀은 로그에 그 신호가 없으므로 직전 행의 집행 비중(executed)으로 대신한다.
    """
    log = _with_tracking_columns(log) if len(log) else log
    out = {}
    for m, g in log.groupby("market"):
        g = g.sort_values("as_of").reset_index(drop=True)
        ret = g["close"].pct_change().fillna(0)
        held = (g["alert_next_weight"].shift(2)
                .fillna(g["executed"].shift(1))
                .fillna(g["executed"].iloc[0])).astype(float)
        model = float((1 + held * ret).prod() - 1)
        bh = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1) if len(g) > 1 else 0.0
        out[m] = {
            "since": g["as_of"].iloc[0], "days": len(g), "model_return": round(model, 4), "index_return": round(bh, 4),
            "regime_changes": int((g["regime"].diff().fillna(0) != 0).sum()),
            "trades": int((g["action"] != "HOLD").sum()),
            "revised_days": int((g["revisions"] > 0).sum()),
            "rows": g.tail(n).iloc[::-1][["as_of", "close", "trend", "risk", "regime", "target", "executed",
                                          "next_weight", "action", "first_action", "first_next_weight", "revisions"]]
            .to_dict("records"),
        }
    return out
