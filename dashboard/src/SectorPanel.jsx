import { Layers, Star } from 'lucide-react'

const pct = (x, d = 0) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`)
const trendCls = (t) => (t >= 0.3 ? 'text-emerald-300' : t > 0 ? 'text-emerald-200/70' : t > -0.3 ? 'text-rose-200/70' : 'text-rose-300')

// 상대강도 막대: 0 기준 좌우 (최대 ±60%)
function ScoreBar({ v }) {
  const w = Math.min(Math.abs(v) / 0.6, 1) * 50
  return (
    <div className="relative h-2 w-full rounded-full bg-slate-800" aria-hidden>
      <div className="absolute top-0 h-2 w-px bg-slate-500" style={{ left: '50%' }} />
      <div className={`absolute top-0 h-2 rounded-full ${v >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
        style={v >= 0 ? { left: '50%', width: `${w}%` } : { right: '50%', width: `${w}%` }} />
    </div>
  )
}

// 섹터 ETF 상대강도·추세 (미국·한국 공용) — 참고 정보 (배분 규칙에는 포함하지 않음)
const NOTES = {
  US: <>주식의 일부(고정 40% 또는 추세 비례 0~40%)를 상위 2~4개 섹터에 월 1회 배분하는 규칙 6가지를 1999년부터 검증한 결과,
    2012년까지는 SPY보다 나았지만 2013년 이후에는 모두 SPY에 못 미쳤습니다(연 14.8% vs 13.7~14.5%).</>,
  KR: <>같은 규칙 6가지를 TIGER 200 섹터 ETF로 2012년부터 검증한 결과, 2018년까지는 KODEX 200과 비슷하거나 낮았고
    2019년 이후에는 모두 KODEX 200에 못 미쳤습니다(연 19.6% vs 15.9~17.7%). 이력이 짧아(헬스케어·산업재·생활소비재 2015년,
    커뮤니케이션 2019년 상장) 검증력도 제한적입니다.</>,
}

export default function SectorPanel({ sectors }) {
  if (!sectors) return null
  const months = Object.keys(sectors.rank_history)
  const n = sectors.n_sectors ?? sectors.sectors.length
  const bench = sectors.benchmark.name ?? sectors.benchmark.ticker
  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
          <Layers size={16} className="text-slate-400" aria-hidden /> 주도 섹터 모니터 · {sectors.label ?? '미국 섹터 ETF'}
        </h2>
        <p className="mt-1 text-[11px] text-slate-500">
          상대강도 = (6개월 + 12개월(최근 1개월 제외) 수익률) ÷ 2 · ★ 주도 = 상대강도 상위 중 추세 점수 &gt; 0 이고 200일선 위인 3개 · {sectors.as_of} 기준
        </p>
      </header>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-5">
        <div className="min-w-0 xl:col-span-3">
          <table className="num w-full text-right font-mono text-xs">
            <thead className="font-sans text-slate-500">
              <tr>
                <th className="pb-1.5 text-left font-medium">섹터</th>
                <th className="px-2 font-medium sm:w-1/4">상대강도</th>
                <th className="font-medium">추세</th>
                <th className="hidden font-medium sm:table-cell">6M vs 기준</th>
                <th className="hidden font-medium sm:table-cell">1M</th>
                <th className="font-medium">고점比</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              {sectors.sectors.map((x) => (
                <tr key={x.ticker} className={`border-t border-line ${x.leader ? 'bg-emerald-500/[0.06]' : ''}`}>
                  <td className="py-1.5 text-left font-sans">
                    <span className="mr-1 inline-block w-4 text-right font-mono text-slate-500">{x.rank}</span>
                    {x.leader && <Star size={11} className="mr-1 inline fill-amber-300 text-amber-300" aria-label="주도" />}
                    <span className="text-slate-100">{x.name}</span> <span className={`text-[10px] text-slate-500 ${sectors.universe === 'KR' ? 'hidden sm:inline' : ''}`}>{x.ticker}</span>
                  </td>
                  <td className="px-2">
                    <div className="flex items-center justify-end gap-2"><div className="hidden w-full sm:block"><ScoreBar v={x.score} /></div><span className="w-12 shrink-0">{pct(x.score)}</span></div>
                  </td>
                  <td className={trendCls(x.trend)}>{x.trend >= 0 ? '+' : ''}{x.trend.toFixed(2)}</td>
                  <td className={`hidden sm:table-cell ${x.rel_6m >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{pct(x.rel_6m, 1)}</td>
                  <td className="hidden sm:table-cell">{pct(x.ret_1m, 1)}</td>
                  <td className={x.above_sma200 ? '' : 'text-rose-300'} title={x.above_sma200 ? '200일선 위' : '200일선 아래'}>{pct(x.drawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[10px] text-slate-500">고점比 빨간색 = 200일 이동평균 아래 · 기준 {bench} 6개월 {pct(sectors.benchmark.ret_6m, 1)}, 추세 {sectors.benchmark.trend.toFixed(2)}{sectors.leaders.length === 0 && ' · 지금은 주도 조건(추세 > 0, 200일선 위)을 충족하는 섹터 없음'}</p>
        </div>

        <div className="min-w-0 space-y-4 xl:col-span-2">
          <div>
            <h3 className="mb-2 text-xs font-semibold text-slate-300">{sectors.extras_title ?? '기술주 포트 구성'}</h3>
            <div className="grid grid-cols-2 gap-2">
              {sectors.tech.map((x) => (
                <div key={x.ticker} className="rounded-xl bg-ink/70 p-3 text-[11px] text-slate-400">
                  <p className="truncate font-semibold text-slate-200">{x.name} <span className="font-mono text-slate-500">{x.ticker}</span></p>
                  <p className="num mt-1">추세 <span className={`font-mono ${trendCls(x.trend)}`}>{x.trend >= 0 ? '+' : ''}{x.trend.toFixed(2)}</span></p>
                  <p className="num">상대강도 <span className="font-mono text-slate-100">{pct(x.score)}</span></p>
                  <p className="num">1개월 <span className="font-mono text-slate-100">{pct(x.ret_1m, 1)}</span> · 고점比 <span className="font-mono text-slate-100">{pct(x.drawdown)}</span></p>
                </div>
              ))}
            </div>
            <p className="mt-1.5 text-[10px] text-slate-500">상대강도(6~12개월)는 높아도 추세(최근 이평·낙폭)가 꺾이면 단기 조정 중이라는 뜻입니다.</p>
          </div>

          <div className="min-w-0">
            <h3 className="mb-2 text-xs font-semibold text-slate-300">월말 순위 추이 (1 = 최강)</h3>
            <div className="overflow-x-auto">
              <table className="num w-full text-center font-mono text-[10px]">
                <thead className="text-slate-500">
                  <tr><th className="text-left font-sans font-medium" />{months.map((mo) => <th key={mo} className="px-0.5 font-medium">{mo.slice(5)}</th>)}</tr>
                </thead>
                <tbody>
                  {sectors.sectors.map((x) => (
                    <tr key={x.ticker}>
                      <td className="whitespace-nowrap pr-1 text-left font-sans text-slate-400">{sectors.universe === 'KR' ? x.name : x.ticker}</td>
                      {months.map((mo) => {
                        const r = sectors.rank_history[mo][x.ticker]
                        return (
                          <td key={mo} className={`px-0.5 py-px ${r <= 3 ? 'font-semibold text-emerald-300' : r > n - 3 ? 'text-slate-600' : 'text-slate-400'}`}>{r ?? '–'}</td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <p className="mt-4 rounded-lg bg-white/5 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
        <span className="font-semibold text-amber-300">참고용 · 배분 규칙에 미포함.</span> {NOTES[sectors.universe ?? 'US']}
        {' '}섹터 순위는 시장 분위기를 읽는 용도로만 보세요.
      </p>
    </section>
  )
}
