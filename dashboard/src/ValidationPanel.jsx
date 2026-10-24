import { useEffect, useState } from 'react'
import { CheckCircle2, Globe2, ShieldQuestion, XCircle } from 'lucide-react'

const pct = (x, d = 1) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
const pp = (x) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}%p`)

function useValidation() {
  const [data, setData] = useState(null)
  useEffect(() => {
    fetch(`./data/validation.json?t=${Date.now()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null))
  }, [])
  return data
}

function Verdict({ ok, children }) {
  const Icon = ok ? CheckCircle2 : XCircle
  return (
    <p className={`flex items-start gap-1.5 text-xs ${ok ? 'text-emerald-300' : 'text-amber-300'}`}>
      <Icon size={14} className="mt-0.5 shrink-0" aria-hidden /> <span>{children}</span>
    </p>
  )
}

function ProbBar({ label, p }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs"><span className="text-slate-300">{label}</span><span className="num font-mono text-slate-100">{pct(p, 0)}</span></div>
      <div className="h-2 rounded-full bg-slate-800">
        <div className="h-2 rounded-full" style={{ width: `${p * 100}%`, background: p >= 0.9 ? '#10b981' : p >= 0.5 ? '#f59e0b' : '#f43f5e' }} />
      </div>
    </div>
  )
}

export default function ValidationPanel() {
  const v = useValidation()
  if (!v) return null
  const all = { ...v.home, ...v.external }
  const rows = Object.entries(all)
  const wins = rows.filter(([, e]) => e.model.mdd > e.static.mdd).length
  const sharpeOk = rows.filter(([, e]) => e.model.sharpe >= e.static.sharpe - 0.005).length
  const folds = v.walk_forward.folds
  const counts = folds.reduce((acc, f) => { const k = JSON.stringify(f.chosen); acc[k] = (acc[k] ?? 0) + 1; return acc }, {})
  const modal = Math.max(...Object.values(counts))
  const sens = [...v.sensitivity.one_at_a_time]
    .map((r) => ({ ...r, spread: Math.max(...['US', 'KR'].map((m) => Math.abs(r.hi[m].sharpe - r.lo[m].sharpe))) }))
    .sort((a, b) => b.spread - a.spread)
    .slice(0, 5)

  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
          <ShieldQuestion size={16} className="text-slate-400" aria-hidden /> 견고성 검증 — 과적합이 아닌가?
        </h2>
        <p className="mt-1 text-[11px] text-slate-500">
          비교 기준 '고정'은 모델과 평균 주식 비중이 같은 고정 포트폴리오 — 타이밍 효과만 분리해서 봅니다 · {v.generated_at} 실행
        </p>
      </header>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {/* 1. 다른 나라 */}
        <div className="min-w-0">
          <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-slate-300"><Globe2 size={14} aria-hidden /> ① 다른 나라 지수에 그대로 적용</h3>
          <div className="overflow-x-auto">
            <table className="num w-full text-right font-mono text-xs">
              <thead className="font-sans text-slate-500">
                <tr><th className="pb-1 text-left font-medium">시장</th><th className="font-medium">보유</th><th className="font-medium">고정</th><th className="font-medium">모델</th><th className="font-medium">샤프 모델/고정</th></tr>
              </thead>
              <tbody className="text-slate-300">
                {rows.map(([k, e]) => (
                  <tr key={k} className="border-t border-line">
                    <td className="py-1 text-left font-sans">{e.label}{v.home[k] ? '' : ' *'}</td>
                    <td className="text-rose-300">{pct(e.bh.mdd, 0)}</td>
                    <td>{pct(e.static.mdd, 0)}</td>
                    <td className={e.model.mdd > e.static.mdd ? 'text-emerald-300' : 'text-amber-300'}>{pct(e.model.mdd, 0)}</td>
                    <td>{e.model.sharpe.toFixed(2)} / {e.static.sharpe.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-[10px] text-slate-500">최대낙폭 비교 · * 파라미터를 한 번도 맞춰보지 않은 시장 (현금 수익 0%, 원/달러 요소 제외)</p>
          <div className="mt-2 space-y-1">
            <Verdict ok={wins === rows.length}>{rows.length}개 시장 중 {wins}곳에서 고정 비중보다 최대낙폭이 작음</Verdict>
            <Verdict ok={sharpeOk >= rows.length - 1}>샤프는 {sharpeOk}곳에서 고정 비중 이상 — 수익 개선 도구는 아님</Verdict>
          </div>
        </div>

        {/* 2~3. 워크포워드 + 부트스트랩 */}
        <div className="min-w-0 space-y-5">
          <div>
            <h3 className="mb-2 text-xs font-semibold text-slate-300">② 롤링 워크포워드 (5년마다 과거 데이터만으로 재선택)</h3>
            <div className="flex flex-wrap gap-1.5">
              {folds.map((f) => (
                <span key={f.test_from} className="rounded-md border border-line px-2 py-0.5 text-[11px] text-slate-400">
                  {f.test_from}~ <span className="font-mono text-slate-200">{f.chosen.alignment_scale}/{f.chosen.smooth_span}d/{f.chosen.band_in_execution ? '밴드' : '–'}</span>
                </span>
              ))}
            </div>
            <div className="mt-2 space-y-1">
              <Verdict ok={modal >= folds.length - 1}>{folds.length}번 중 {modal}번 같은 설정 선택 → {modal >= folds.length - 1 ? '파라미터 선택이 안정적' : '시기별로 달라짐, 파라미터 불안정'}</Verdict>
              {Object.entries(v.walk_forward.markets).map(([k, e]) => (
                <p key={k} className="num text-[11px] text-slate-400">
                  {e.label}: 모델 MDD <span className="text-emerald-300">{pct(e.model.mdd, 0)}</span> · 고정 {pct(e.static.mdd, 0)} · 보유 {pct(e.bh.mdd, 0)}
                </p>
              ))}
            </div>
          </div>
          <div>
            <h3 className="mb-2 text-xs font-semibold text-slate-300">③ 블록 부트스트랩 {v.bootstrap.US.n}회 — 우위가 우연일 확률</h3>
            <div className="space-y-2.5">
              {Object.entries(v.bootstrap).map(([k, b]) => (
                <div key={k} className="space-y-1.5">
                  <ProbBar label={`${v.home[k].label}: 최대낙폭이 고정보다 작을 확률 (중앙 ${pp(b.mdd_diff_ci[1])})`} p={b.p_mdd_better} />
                  <ProbBar label={`${v.home[k].label}: 샤프가 고정보다 높을 확률`} p={b.p_sharpe_better} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 4. 민감도 */}
        <div className="min-w-0">
          <h3 className="mb-2 text-xs font-semibold text-slate-300">④ 민감도 — 파라미터를 흔들어도 결과가 유지되나</h3>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(v.sensitivity.joint).map(([k, j]) => (
              <div key={k} className="rounded-xl bg-ink/70 p-3 text-[11px] text-slate-400">
                <p className="mb-1 font-semibold text-slate-200">{v.home[k].label} · 전부 동시 ±30% {v.sensitivity.n_random}회</p>
                <p className="num">샤프 <span className="font-mono text-slate-100">{j.sharpe[0].toFixed(2)}~{j.sharpe[2].toFixed(2)}</span></p>
                <p className="num">최대낙폭 <span className="font-mono text-slate-100">{pct(j.mdd[0], 0)}~{pct(j.mdd[2], 0)}</span></p>
                <p className="num">회전율 <span className="font-mono text-slate-100">{pct(j.turnover[0], 0)}~{pct(j.turnover[2], 0)}</span></p>
                <p className="num">고정보다 낙폭 작음 <span className="font-mono text-emerald-300">{pct(j.p_mdd_better_than_static, 0)}</span></p>
              </div>
            ))}
          </div>
          <p className="mt-3 mb-1 text-[11px] text-slate-500">하나씩 ×0.5 / ×1.5 — 샤프 변화가 큰 순 (US · KR)</p>
          <ul className="num space-y-0.5 font-mono text-[11px] text-slate-300">
            {sens.map((r) => (
              <li key={r.param} className="flex justify-between gap-2">
                <span className="truncate font-sans text-slate-400">{r.param}</span>
                <span>{r.lo.US.sharpe.toFixed(2)}↔{r.hi.US.sharpe.toFixed(2)} · {r.lo.KR.sharpe.toFixed(2)}↔{r.hi.KR.sharpe.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* 5. 비용·지연 */}
        <div className="min-w-0">
          <h3 className="mb-2 text-xs font-semibold text-slate-300">⑤ 매매비용·집행 지연 스트레스</h3>
          <div className="overflow-x-auto">
            <table className="num w-full text-right font-mono text-xs">
              <thead className="font-sans text-slate-500">
                <tr><th className="pb-1 text-left font-medium">조건</th><th className="font-medium">미국 샤프</th><th className="font-medium">MDD</th><th className="font-medium">한국 샤프</th><th className="font-medium">MDD</th></tr>
              </thead>
              <tbody className="text-slate-300">
                {v.cost_delay.US.map((r, i) => {
                  const kr = v.cost_delay.KR[i]
                  return (
                    <tr key={i} className="border-t border-line">
                      <td className="py-1 text-left font-sans">비용 {r.cost_bps}bp · 지연 +{r.delay}일</td>
                      <td>{r.sharpe.toFixed(2)}</td><td>{pct(r.mdd, 0)}</td>
                      <td>{kr.sharpe.toFixed(2)}</td><td>{pct(kr.mdd, 0)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-[10px] text-slate-500">
            고정 비중 기준: 미국 샤프 {v.cost_delay.US[0].static_sharpe.toFixed(2)} · MDD {pct(v.cost_delay.US[0].static_mdd, 0)} / 한국 샤프 {v.cost_delay.KR[0].static_sharpe.toFixed(2)} · MDD {pct(v.cost_delay.KR[0].static_mdd, 0)}
          </p>
        </div>
      </div>
    </section>
  )
}
