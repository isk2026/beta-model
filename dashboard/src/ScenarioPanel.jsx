import { Zap } from 'lucide-react'

const pct = (x, d = 0) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
const REGIME_COLOR = { 1: '#10b981', 2: '#f59e0b', 3: '#22d3ee', 4: '#f43f5e' }

export default function ScenarioPanel({ scenarios, market, label }) {
  if (!scenarios) return null
  const today = scenarios.today[market]
  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
          <Zap size={16} className="text-slate-400" aria-hidden /> 스트레스 시나리오 · {label}
        </h2>
        <p className="mt-1 text-[11px] text-slate-500">
          오늘 이후 {scenarios.days}거래일 동안 아래 충격이 오면 모델이 제시할 비중 (실제 엔진·국면 히스테리시스·속도 조절 그대로 적용)
        </p>
      </header>
      <div className="overflow-x-auto">
        <table className="num w-full text-right font-mono text-xs">
          <thead className="font-sans text-slate-500">
            <tr>
              <th className="pb-1.5 text-left font-medium">시나리오</th>
              <th className="hidden font-medium sm:table-cell">추세</th><th className="hidden font-medium sm:table-cell">위험</th>
              <th className="whitespace-nowrap pl-3 text-left font-medium">국면</th>
              <th className="whitespace-nowrap font-medium">목표 주식</th><th className="pl-2 font-medium">변화</th>
            </tr>
          </thead>
          <tbody className="text-slate-300">
            <tr className="border-t border-line bg-white/[0.03]">
              <td className="py-1.5 text-left font-sans text-slate-100">오늘</td>
              <td className="hidden sm:table-cell">{today.trend.toFixed(2)}</td><td className="hidden sm:table-cell">{today.risk.toFixed(2)}</td>
              <td className="pl-3 text-left font-sans"><span style={{ color: REGIME_COLOR[today.regime] }}>{today.regime}</span><span className="hidden sm:inline"> {today.regime_name}</span></td>
              <td className="text-emerald-300">{pct(today.target)}</td><td>–</td>
            </tr>
            {scenarios.scenarios.map((s) => {
              const r = s.markets[market]
              const d = r.target - today.target
              return (
                <tr key={s.id} className="border-t border-line">
                  <td className="py-1.5 text-left font-sans">
                    <span className="text-slate-100">{s.label}</span>
                    <span className="block text-[10px] text-slate-500">{s.desc}</span>
                  </td>
                  <td className="hidden sm:table-cell">{r.trend.toFixed(2)}</td><td className="hidden sm:table-cell">{r.risk.toFixed(2)}</td>
                  <td className="pl-3 text-left font-sans"><span style={{ color: REGIME_COLOR[r.regime] }}>{r.regime}</span><span className="hidden sm:inline"> {r.regime_name}</span></td>
                  <td>{pct(r.target)}</td>
                  <td className={`pl-2 ${d < -0.005 ? 'text-rose-300' : d > 0.005 ? 'text-emerald-300' : ''}`}>{d >= 0 ? '+' : ''}{(d * 100).toFixed(0)}%p</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        최악의 경우에도 주식 {pct(0.6)} 아래로는 내려가지 않습니다(원칙 1). 공포 국면(4)은 과거 이후 수익이 가장 높았던 구간이기도 해서, 위험 점수가 꺾이면 분할 매수를 검토합니다.
      </p>
    </section>
  )
}
