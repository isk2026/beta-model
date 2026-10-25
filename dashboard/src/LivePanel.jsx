import { ClipboardList } from 'lucide-react'

const pct = (x, d = 0) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
const REGIME_COLOR = { 1: '#10b981', 2: '#f59e0b', 3: '#22d3ee', 4: '#f43f5e' }
const ACTION = {
  BUY: ['늘리기', 'text-emerald-300'],
  SELL: ['줄이기', 'text-rose-300'],
  HOLD: ['유지', 'text-slate-500'],
}

// 매 자동 갱신 때 남긴 신호 기록 — 백테스트가 아니라 '그때 실제로 낸 신호'
export default function LivePanel({ log, market, label }) {
  const g = log?.[market]
  if (!g) return null
  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
            <ClipboardList size={16} className="text-slate-400" aria-hidden /> 실전 신호 기록 · {label}
          </h2>
          <p className="mt-1 text-[11px] text-slate-500">
            {g.since}부터 매 갱신마다 저장 — 최초 알림 값은 보존, 같은 기준일 재계산으로 신호가 바뀐 날은 <span className="text-amber-300">정정</span> 표시
          </p>
        </div>
        <dl className="num flex gap-3 font-mono text-xs">
          <div><dt className="font-sans text-[10px] text-slate-500">기록일</dt><dd className="text-slate-200">{g.days}</dd></div>
          <div><dt className="font-sans text-[10px] text-slate-500">모델(주식분)</dt><dd className="text-emerald-300">{pct(g.model_return, 1)}</dd></div>
          <div><dt className="font-sans text-[10px] text-slate-500">지수</dt><dd className="text-slate-200">{pct(g.index_return, 1)}</dd></div>
          <div><dt className="font-sans text-[10px] text-slate-500">매매/전환</dt><dd className="text-slate-200">{g.trades}/{g.regime_changes}</dd></div>
        </dl>
      </header>
      <div className="max-h-72 overflow-auto">
        <table className="num w-full text-right font-mono text-xs">
          <thead className="sticky top-0 bg-panel font-sans text-slate-500">
            <tr>
              <th className="pb-1.5 text-left font-medium">기준일</th><th className="hidden font-medium sm:table-cell">추세</th><th className="hidden font-medium sm:table-cell">위험</th>
              <th className="font-medium">국면</th><th className="font-medium">목표</th><th className="font-medium">보유</th>
              <th className="font-medium">다음날</th><th className="font-medium">신호</th>
            </tr>
          </thead>
          <tbody className="text-slate-300">
            {g.rows.map((r) => {
              const [txt, cls] = ACTION[r.action] ?? ACTION.HOLD
              const revised = r.revisions > 0
              const firstTxt = (ACTION[r.first_action] ?? ACTION.HOLD)[0]
              return (
                <tr key={r.as_of} className="border-t border-line">
                  <td className="py-1 text-left">{r.as_of}</td>
                  <td className="hidden sm:table-cell">{r.trend?.toFixed(2)}</td><td className="hidden sm:table-cell">{r.risk?.toFixed(2)}</td>
                  <td style={{ color: REGIME_COLOR[r.regime] }}>{r.regime}</td>
                  <td>{pct(r.target)}</td><td>{pct(r.executed)}</td><td>{pct(r.next_weight)}</td>
                  <td className={`font-sans ${cls}`}>
                    {txt}
                    {revised && (
                      <span className="ml-1 text-amber-300" title={`최초 알림: ${firstTxt} · 다음날 ${pct(r.first_next_weight)}`}>정정</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {g.days < 20 && (
        <p className="mt-2 text-[11px] text-slate-500">기록이 쌓이는 중입니다. 수개월 이상 지나야 백테스트와 비교할 의미가 생깁니다.</p>
      )}
    </section>
  )
}
