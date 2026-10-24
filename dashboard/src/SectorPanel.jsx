import { useState } from 'react'
import { AlertTriangle, Layers, Star } from 'lucide-react'

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

// ---------------------------------------------------------------- 옵트인 위성 비중
const SAT_KEY = 'beta-model:sector-satellite'

// 기본 꺼짐. 브라우저 저장소가 막혀 있어도 동작 (토글만 기억 못 함)
export function useSatelliteToggle() {
  const [on, setOn] = useState(() => {
    try { return localStorage.getItem(SAT_KEY) === '1' } catch { return false }
  })
  const set = (v) => {
    setOn(v)
    try { localStorage.setItem(SAT_KEY, v ? '1' : '0') } catch { /* 무시 */ }
  }
  return [on, set]
}

export const coreLabel = (market) => (market === 'US_TECH' ? 'QQQ/SOXX 기술 포트' : 'SPY/지수 ETF')

// 주식 비중(stock)을 코어 + 주도 섹터로 나눈 포트 전체 대비 비중. 주식/현금 합계는 그대로.
export function satelliteLines(sat, stock, market) {
  if (!sat) return []
  return [
    { key: 'core', label: coreLabel(market), weight: stock * sat.core, core: true },
    ...sat.legs.map((l) => ({ key: l.ticker, label: `${l.name} ${l.ticker}`, weight: stock * l.weight })),
  ]
}

const p1 = (x) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`)

function EvidenceTable({ bt }) {
  const rows = [['표본 내', bt.in_sample], ['표본 외', bt.out_of_sample]]
  return (
    <table className="num mt-2 w-full text-right font-mono text-[10px] sm:text-[11px]">
      <thead className="font-sans text-slate-500">
        <tr><th className="text-left font-medium">구간</th><th className="font-medium">전략</th><th className="font-medium">연복리</th><th className="font-medium">샤프</th><th className="font-medium">MDD</th></tr>
      </thead>
      <tbody className="text-slate-300">
        {rows.flatMap(([lab, seg]) => [['SPY', seg.spy], ['위성 규칙', seg.rule]].map(([name, v], i) => {
          const worse = name !== 'SPY' && v.cagr < seg.spy.cagr
          return (
            <tr key={lab + name} className={i === 0 ? 'border-t border-line' : ''}>
              <td className="py-0.5 text-left font-sans text-slate-400">{i === 0 ? <>{lab} <span className="block text-slate-500 sm:inline">{seg.period}</span></> : ''}</td>
              <td className="font-sans">{name}</td>
              <td className={worse ? 'text-rose-300' : ''}>{p1(v.cagr)}</td>
              <td>{v.sharpe.toFixed(2)}</td>
              <td>{p1(v.mdd)}</td>
            </tr>
          )
        }))}
      </tbody>
    </table>
  )
}

function SatelliteBox({ sat, on, setOn, stock, market }) {
  const lines = satelliteLines(sat, stock, market)
  return (
    <div className="mt-4 rounded-xl border border-line bg-ink/60 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-slate-200">주도 섹터 위성 비중 <span className="font-normal text-slate-500">(옵트인 · 기본 꺼짐)</span></p>
          <p className="text-[10px] text-slate-500">주식 중 {Math.round(sat.max_fraction * 100)}% × 미국 추세(0~1)를 주도 섹터 {sat.n}개에 균등 · 주식/현금 비중은 그대로</p>
        </div>
        <button type="button" role="switch" aria-checked={on} onClick={() => setOn(!on)}
          className={`relative h-6 w-11 shrink-0 rounded-full transition ${on ? 'bg-amber-500/70' : 'bg-slate-700'}`}>
          <span className="sr-only">주도 섹터 위성 비중 표시</span>
          <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all ${on ? 'left-[22px]' : 'left-0.5'}`} />
        </button>
      </div>

      {on && (
        <div className="mt-3 space-y-3">
          <p className="flex items-start gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1.5 text-[11px] font-semibold text-rose-200">
            <AlertTriangle size={13} className="mt-px shrink-0" aria-hidden /> 검증상 근거 없음 — 2013년 이후 SPY 대비 열위
          </p>
          <div>
            <p className="mb-1 text-[11px] text-slate-400">
              현재 주식 {p1(stock)}를 나누면 (위성 = 주식의 {p1(sat.fraction)}, 미국 추세 {sat.us_trend >= 0 ? '+' : ''}{sat.us_trend.toFixed(2)}) · 포트 전체 대비
            </p>
            <ul className="space-y-1 text-[11px]">
              {lines.map((l) => (
                <li key={l.key} className="flex min-w-0 items-center gap-2">
                  <span className={`min-w-0 flex-1 truncate ${l.core ? 'text-slate-200' : 'text-amber-200'}`}>{l.core ? '' : '★ '}{l.label}</span>
                  <span className="hidden h-1.5 w-24 shrink-0 rounded-full bg-slate-800 sm:block" aria-hidden>
                    <span className={`block h-1.5 rounded-full ${l.core ? 'bg-emerald-500' : 'bg-amber-400'}`} style={{ width: `${Math.min(l.weight, 1) * 100}%` }} />
                  </span>
                  <span className="num w-14 shrink-0 text-right font-mono text-slate-100">{p1(l.weight)}</span>
                </li>
              ))}
              <li className="flex items-center gap-2 border-t border-line pt-1 text-slate-500">
                <span className="flex-1">현금</span><span className="num w-14 shrink-0 text-right font-mono">{p1(1 - stock)}</span>
              </li>
            </ul>
            {sat.legs.length < sat.n && (
              <p className="mt-1 text-[10px] text-slate-500">주도 섹터가 {sat.legs.length}개뿐이라 남는 몫은 코어에 둡니다.</p>
            )}
            {market === 'US_TECH' && (
              <p className="mt-1 text-[10px] text-slate-500">백테스트는 SPY 코어 기준입니다. 기술주 포트 코어로는 검증하지 않았습니다.</p>
            )}
          </div>
          {sat.backtest && (
            <div className="min-w-0 overflow-x-auto">
              <p className="text-[10px] text-slate-500">{sat.backtest.rule}</p>
              <EvidenceTable bt={sat.backtest} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// 미국 섹터 ETF 상대강도·추세 — 참고 정보 (배분 규칙에는 포함하지 않음)
export default function SectorPanel({ sectors, market, stock, satOn = false, setSatOn }) {
  if (!sectors || market === 'KR') return null
  const months = Object.keys(sectors.rank_history)
  return (
    <section className="mt-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5">
      <header className="mb-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
          <Layers size={16} className="text-slate-400" aria-hidden /> 주도 섹터 모니터 · 미국 섹터 ETF
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
                <th className="hidden font-medium sm:table-cell">6M vs SPY</th>
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
                    <span className="text-slate-100">{x.name}</span> <span className="text-[10px] text-slate-500">{x.ticker}</span>
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
          <p className="mt-1 text-[10px] text-slate-500">고점比 빨간색 = 200일 이동평균 아래 · SPY 6개월 {pct(sectors.benchmark.ret_6m, 1)}, 추세 {sectors.benchmark.trend.toFixed(2)}</p>
        </div>

        <div className="min-w-0 space-y-4 xl:col-span-2">
          <div>
            <h3 className="mb-2 text-xs font-semibold text-slate-300">기술주 포트 구성</h3>
            <div className="grid grid-cols-2 gap-2">
              {sectors.tech.map((x) => (
                <div key={x.ticker} className="rounded-xl bg-ink/70 p-3 text-[11px] text-slate-400">
                  <p className="font-semibold text-slate-200">{x.name} <span className="font-mono text-slate-500">{x.ticker}</span></p>
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
                      <td className="pr-1 text-left font-sans text-slate-400">{x.ticker}</td>
                      {months.map((mo) => {
                        const r = sectors.rank_history[mo][x.ticker]
                        return (
                          <td key={mo} className={`px-0.5 py-px ${r <= 3 ? 'font-semibold text-emerald-300' : r >= 9 ? 'text-slate-600' : 'text-slate-400'}`}>{r ?? '–'}</td>
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
        <span className="font-semibold text-amber-300">참고용 · 배분 규칙에 미포함.</span> 주식의 일부(고정 40% 또는 추세 비례 0~40%)를 상위 2~4개 섹터에 월 1회
        배분하는 규칙 6가지를 1999년부터 검증한 결과, 2012년까지는 SPY보다 나았지만 2013년 이후에는 모두 SPY에 못 미쳤습니다(연 14.8% vs 13.7~14.5%).
        섹터 순위는 시장 분위기를 읽는 용도로만 보세요.
      </p>

      {sectors.satellite && setSatOn && stock != null && (
        <SatelliteBox sat={sectors.satellite} on={satOn} setOn={setSatOn} stock={stock} market={market} />
      )}
    </section>
  )
}
