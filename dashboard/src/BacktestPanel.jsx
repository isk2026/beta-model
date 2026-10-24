import { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { FlaskConical, Info } from 'lucide-react'

const C = { model: '#10b981', bh: '#94a3b8', grid: 'rgba(148,163,184,0.10)', tick: '#94a3b8' }
const PARTS = [['full', '전체'], ['in_sample', '표본 내'], ['out_sample', '표본 외']]
const pct = (x, d = 1) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)

function useBacktest() {
  const [data, setData] = useState(null)
  useEffect(() => {
    fetch(`./data/backtest.json?t=${Date.now()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null))
  }, [])
  return data
}

function Curve({ curve, label }) {
  const data = {
    labels: curve.map((p) => p.date),
    datasets: [
      { label: '베타 모델', data: curve.map((p) => p.model), borderColor: C.model, borderWidth: 2 },
      { label: `${label} 보유`, data: curve.map((p) => p.bh), borderColor: C.bh, borderWidth: 1.5 },
    ],
  }
  const options = {
    maintainAspectRatio: false, animation: false,
    interaction: { mode: 'index', intersect: false },
    elements: { point: { radius: 0, hoverRadius: 4, hitRadius: 8 }, line: { tension: 0 } },
    scales: {
      x: {
        ticks: { color: C.tick, maxRotation: 0, autoSkipPadding: 24, callback(v) { return this.getLabelForValue(v).slice(0, 4) } },
        grid: { display: false }, border: { color: C.grid },
      },
      y: { type: 'logarithmic', ticks: { color: C.tick, callback: (v) => `${v}×` }, grid: { color: C.grid }, border: { display: false } },
    },
    plugins: {
      legend: { position: 'top', align: 'end', labels: { color: '#cbd5e1', boxWidth: 12, boxHeight: 2, font: { size: 11 } } },
      tooltip: {
        backgroundColor: '#0b0f19ee', borderColor: '#334155', borderWidth: 1, padding: 10,
        callbacks: { label: (c) => ` ${c.dataset.label} ${c.parsed.y.toFixed(2)}배` },
      },
    },
  }
  return <div className="h-56 sm:h-72"><Line data={data} options={options} aria-label="누적 성과 비교 (로그 스케일)" /></div>
}

export default function BacktestPanel({ market }) {
  const bt = useBacktest()
  const [part, setPart] = useState('full')
  const mk = bt?.markets?.[market]
  if (!mk) return null

  const specs = mk.specs
  const current = specs[bt.current]
  const rows = [
    ['보유 그대로', current[part].bh, null],
    [`고정 비중 ${pct(mk.static_weight, 0)}`, specs['고정 비중']?.[part].model, 'static'],
    ['기존 모델', specs['기존 모델']?.[part].model, 'old'],
    ['현재 모델', current[part].model, 'cur'],
  ].filter((r) => r[1])
  const worst = mk.episodes.reduce((a, e) => (e.bh < a.bh ? e : a), mk.episodes[0])

  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
            <FlaskConical size={16} className="text-slate-400" aria-hidden /> 모델 검증 · {mk.label} {mk.index} ({mk.start.slice(0, 4)}~)
          </h2>
          <p className="mt-1 text-[11px] text-slate-500">
            표본 내 = ~{Number(bt.split_date.slice(0, 4)) - 1} (파라미터 선택), 표본 외 = {bt.split_date.slice(0, 4)}~ (검증만)
          </p>
        </div>
        <div className="flex self-start rounded-lg border border-line p-0.5 text-xs" role="group" aria-label="검증 구간">
          {PARTS.map(([k, lab]) => (
            <button key={k} onClick={() => setPart(k)}
              className={`rounded-md px-2.5 py-1 font-medium transition ${part === k ? 'bg-cyan-500/15 text-cyan-300' : 'text-slate-400 hover:text-slate-200'}`}>
              {lab}
            </button>
          ))}
        </div>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <div className="min-w-0 lg:col-span-3">
          <p className="mb-1 text-xs text-slate-400">1원이 몇 배가 됐나 (로그 스케일, 비용·현금이자 반영)</p>
          <Curve curve={mk.curve} label={mk.index} />
        </div>
        <div className="min-w-0 lg:col-span-2">
          <div className="overflow-x-auto">
            <table className="num w-full text-right font-mono text-xs">
              <thead className="font-sans text-slate-500">
                <tr>
                  <th className="pb-1.5 text-left font-medium">전략</th>
                  <th className="font-medium">연수익</th><th className="font-medium">최대낙폭</th>
                  <th className="font-medium">샤프</th><th className="font-medium">회전율</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                {rows.map(([name, r, kind]) => (
                  <tr key={name} className={`border-t border-line ${kind === 'cur' ? 'text-emerald-300' : ''}`}>
                    <td className="py-1.5 text-left font-sans">{name}</td>
                    <td>{pct(r.cagr)}</td><td>{pct(r.mdd)}</td><td>{r.sharpe.toFixed(2)}</td>
                    <td>{r.turnover == null ? '–' : pct(r.turnover, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 mb-1.5 text-xs text-slate-400">최악의 하락 구간 (자동 추출, 고점→저점)</p>
          <div className="overflow-x-auto">
            <table className="num w-full text-right font-mono text-xs">
              <thead className="font-sans text-slate-500">
                <tr><th className="pb-1 text-left font-medium">구간</th><th className="font-medium">보유</th><th className="font-medium">고정</th><th className="font-medium">모델</th></tr>
              </thead>
              <tbody className="text-slate-300">
                {mk.episodes.map((e) => (
                  <tr key={e.peak} className="border-t border-line">
                    <td className="py-1 text-left">{e.peak.slice(2, 7)}~{e.trough.slice(2, 7)}</td>
                    <td className="text-rose-300">{pct(e.bh, 0)}</td>
                    <td>{pct(e.static, 0)}</td>
                    <td className="text-emerald-300">{pct(e.model, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="mt-5 flex gap-3 rounded-xl border border-line bg-ink/60 p-3 text-xs leading-relaxed text-slate-300">
        <Info size={16} className="mt-0.5 shrink-0 text-cyan-400" aria-hidden />
        <p>
          이 모델은 <b className="text-slate-100">수익을 높이는 도구가 아니라 큰 하락을 줄이는 도구</b>입니다.
          연수익은 보유 그대로보다 약간 낮지만, 최악 구간({worst.peak.slice(0, 7)}) 손실을 {pct(worst.bh, 0)} → {pct(worst.model, 0)}로 줄였고,
          평균 비중이 같은 고정 포트폴리오({pct(worst.static, 0)})보다도 방어가 좋았습니다. 단, 공포 국면(4)은 이후 수익이 가장 높은 구간이기도 해서
          위험 점수가 꺾이면 분할 매수로 되돌리는 판단이 중요합니다.
          <span className="mt-1 block text-slate-500">{bt.assumptions}. 과거 성과가 미래를 보장하지 않습니다.</span>
        </p>
      </div>
    </section>
  )
}
