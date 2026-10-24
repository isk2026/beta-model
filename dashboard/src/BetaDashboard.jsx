import { useEffect, useMemo, useState } from 'react'
import {
  ArcElement, CategoryScale, Chart as ChartJS, Filler, Legend, LinearScale, LineElement, LogarithmicScale, PointElement, Tooltip,
} from 'chart.js'
import { Doughnut, Line } from 'react-chartjs-2'
import {
  Activity, AlertTriangle, Calculator, Compass, Gauge, Info, RotateCcw, ShieldAlert, ShieldCheck,
  TrendingDown, TrendingUp, Zap,
} from 'lucide-react'
import { allocate, DEFAULT_PARAMS, FALLBACK_REGIMES, rebalance } from './model.js'
import BacktestPanel from './BacktestPanel.jsx'
import ScenarioPanel from './ScenarioPanel.jsx'
import ValidationPanel from './ValidationPanel.jsx'
import LivePanel from './LivePanel.jsx'

ChartJS.register(ArcElement, CategoryScale, LinearScale, LogarithmicScale, PointElement, LineElement, Filler, Tooltip, Legend)

// ---------------------------------------------------------------- tokens
const C = {
  emerald: '#10b981', rose: '#f43f5e', amber: '#f59e0b', cyan: '#22d3ee',
  slate: '#64748b', violet: '#a78bfa', grid: 'rgba(148,163,184,0.10)', tick: '#94a3b8', panel: '#111827',
}
const REGIME_STYLE = {
  1: { color: C.emerald, Icon: TrendingUp, ring: 'ring-emerald-500/40', glow: 'shadow-[0_0_40px_-12px_#10b981]' },
  2: { color: C.amber, Icon: AlertTriangle, ring: 'ring-amber-500/40', glow: 'shadow-[0_0_40px_-12px_#f59e0b]' },
  3: { color: C.cyan, Icon: ShieldCheck, ring: 'ring-cyan-400/40', glow: 'shadow-[0_0_40px_-12px_#22d3ee]' },
  4: { color: C.rose, Icon: ShieldAlert, ring: 'ring-rose-500/40', glow: 'shadow-[0_0_40px_-12px_#f43f5e]' },
}
const PRESETS = [
  { id: 'us-bull', label: 'US Bull Market', sub: '미국 강세', trend: 0.8, risk: 0.15, Icon: TrendingUp, color: C.emerald },
  { id: 'pre-cut', label: 'Late-Cycle Caution', sub: '상승 중 불안', trend: 0.5, risk: 0.7, Icon: AlertTriangle, color: C.amber },
  { id: 'kr-def', label: 'KR Defensive Hold', sub: '한국 하락 방어', trend: -0.6, risk: 0.3, Icon: ShieldCheck, color: C.cyan },
  { id: 'panic', label: 'Panic / Buy the Dip', sub: '공포 국면', trend: -0.9, risk: 0.9, Icon: Zap, color: C.rose },
]
const TREND_LABELS = { alignment: '이동평균 정배열', drawdown: '52주 고점 대비', momentum: '3개월 모멘텀' }
const RISK_LABELS = { vol_index: '변동성 지수', realized_vol: '실현 변동성', rates: '미 10년물 급등', fx: '원/달러 급등', credit: '신용 스프레드 (Baa)' }
const MACRO_META = {
  VIX: { label: 'VIX', fmt: (v) => v.toFixed(2), chg: (v) => v.toFixed(2), bad: 1 },
  VKOSPI: { label: 'VKOSPI', fmt: (v) => v.toFixed(2), chg: (v) => v.toFixed(2), bad: 1 },
  US10Y: { label: '미 10년물', fmt: (v) => `${v.toFixed(2)}%`, chg: (v) => `${(v * 100).toFixed(0)}bp`, bad: 1 },
  USDKRW: { label: '원/달러', fmt: (v) => v.toLocaleString('ko-KR', { maximumFractionDigits: 1 }), chg: (v) => v.toFixed(1), bad: 1 },
  DXY: { label: '달러 인덱스', fmt: (v) => v.toFixed(2), chg: (v) => v.toFixed(2), bad: 1 },
  BAA_SPREAD: { label: 'Baa 스프레드', fmt: (v) => `${v.toFixed(2)}%p`, chg: (v) => `${(v * 100).toFixed(0)}bp`, bad: 1 },
  KR_CREDIT: { label: '국내 신용 AA−', fmt: (v) => `${v.toFixed(2)}%p`, chg: (v) => `${(v * 100).toFixed(0)}bp`, bad: 1 },
  HY_SPREAD: { label: 'HY 스프레드(참고)', fmt: (v) => `${v.toFixed(2)}%`, chg: (v) => `${(v * 100).toFixed(0)}bp`, bad: 1 },
}

const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`
const signed = (x, d = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(d)}`
const riskColor = (r) => (r < 0.33 ? C.emerald : r < 0.66 ? C.amber : C.rose)

// ---------------------------------------------------------------- data hook
function useModelData() {
  const [state, setState] = useState({ loading: true, data: null, error: null })
  useEffect(() => {
    fetch(`./data/beta_model.json?t=${Date.now()}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data) => setState({ loading: false, data, error: null }))
      .catch((error) => setState({ loading: false, data: null, error }))
  }, [])
  return state
}

// ---------------------------------------------------------------- primitives
function Panel({ title, icon: Icon, right, children, className = '' }) {
  return (
    <section className={`min-w-0 rounded-2xl border border-line bg-panel/80 p-4 sm:p-5 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-slate-200">
            {Icon && <Icon size={16} className="text-slate-400" aria-hidden />}
            {title}
          </h2>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

// 슬라이더 눈금: [값, 라벨]. 구간 경계는 인접 눈금의 중간값.
const TREND_SCALE = [[-1, '강한 하락'], [-0.5, '하락'], [0, '중립'], [0.5, '상승'], [1, '강한 상승']]
const RISK_SCALE = [[0, '안정'], [0.25, '낮은 불안'], [0.5, '경계'], [0.75, '높은 불안'], [1, '공포']]
const levelOf = (scale, v) => scale.reduce((best, s) => (Math.abs(s[0] - v) < Math.abs(best[0] - v) ? s : best))

function ScoreSlider({ label, desc, value, min, max, step, onChange, color, scale, live }) {
  const fill = ((value - min) / (max - min)) * 100
  const livePos = live == null ? null : ((live - min) / (max - min)) * 100
  const level = levelOf(scale, value)
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label className="text-xs font-medium text-slate-300">{label}</label>
        <span className="flex items-baseline gap-2">
          <span className="rounded-md px-1.5 py-0.5 text-[11px] font-semibold" style={{ color, background: `${color}1f` }}>{level[1]}</span>
          <span className="num font-mono text-2xl font-semibold" style={{ color, textShadow: `0 0 18px ${color}66` }}>
            {min < 0 ? signed(value) : value.toFixed(2)}
          </span>
        </span>
      </div>
      <p className="mb-3 text-[11px] leading-relaxed text-slate-500">{desc}</p>
      <div className="relative">
        <input
          type="range" min={min} max={max} step={step} value={value} aria-label={label}
          aria-valuetext={`${value.toFixed(2)} ${level[1]}`}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{ '--thumb': color, '--track': `linear-gradient(90deg, ${color} 0%, ${color} ${fill}%, #1f2937 ${fill}%)` }}
        />
        {livePos != null && (
          <span
            className="pointer-events-none absolute top-[-3px] h-3 w-0.5 rounded bg-slate-300"
            style={{ left: `calc(${livePos}% - 1px)` }} title={`실측값 ${live.toFixed(2)}`}
          />
        )}
      </div>
      <div className="relative mt-1.5 h-8 text-[10px] leading-tight sm:text-[11px]">
        {scale.map(([v, text], i) => {
          const pos = ((v - min) / (max - min)) * 100
          const align = i === 0 ? 'translate-x-0 text-left' : i === scale.length - 1 ? '-translate-x-full text-right' : '-translate-x-1/2 text-center'
          const active = level[0] === v
          return (
            <span key={v} className={`absolute top-0 ${align} ${active ? 'font-semibold text-slate-100' : 'text-slate-500'}`} style={{ left: `${pos}%` }}>
              <span className="num block font-mono">{min < 0 && v > 0 ? `+${v}` : v}</span>
              <span className="block whitespace-nowrap">{text}</span>
            </span>
          )
        })}
      </div>
    </div>
  )
}

function ComponentBar({ label, value, diverging, weight }) {
  if (value == null) {
    return (
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>{label}</span><span>데이터 없음</span>
      </div>
    )
  }
  const color = diverging ? (value >= 0 ? C.emerald : C.rose) : riskColor(value)
  const width = diverging ? Math.abs(value) * 50 : value * 100
  const left = diverging ? (value >= 0 ? 50 : 50 - width) : 0
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="text-slate-300">
          {label}
          {weight != null && <span className="ml-1.5 text-slate-500">×{weight.toFixed(2)}</span>}
        </span>
        <span className="num font-mono text-slate-200">{diverging ? signed(value) : value.toFixed(2)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-slate-800">
        {diverging && <span className="absolute left-1/2 top-[-2px] h-3 w-px bg-slate-500" />}
        <span className="absolute top-0 h-2 rounded-full" style={{ left: `${left}%`, width: `${width}%`, background: color }} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- charts
// 현금 구성: usdShare > 0 이면 원화/달러로 나눈다 (한국 시장)
function cashParts(cash, usdShare) {
  return usdShare > 0
    ? [['원화 현금', cash * (1 - usdShare), C.slate], ['달러 현금', cash * usdShare, C.violet]]
    : [['현금(MMF)', cash, C.slate]]
}

function AllocationDonut({ stock, cash, usdShare }) {
  const parts = [['주식', stock, C.emerald], ...cashParts(cash, usdShare)]
  const data = {
    labels: parts.map((p) => p[0]),
    datasets: [{
      data: parts.map((p) => p[1] * 100),
      backgroundColor: parts.map((p) => p[2]),
      borderColor: C.panel, borderWidth: 2, borderRadius: 4, spacing: 2,
      hoverOffset: 4,
    }],
  }
  const options = {
    cutout: '74%', rotation: -90, circumference: 360, maintainAspectRatio: false,
    animation: { duration: 350 },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (c) => ` ${c.label} ${c.parsed.toFixed(1)}%` } },
    },
  }
  return (
    <div className="relative mx-auto aspect-square w-full max-w-[240px]">
      <Doughnut data={data} options={options} aria-label={`주식 ${pct(stock)}, 현금 ${pct(cash)}`} />
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-xs text-slate-400">주식 비중</span>
        <span className="num font-mono text-4xl font-bold text-emerald-400" style={{ textShadow: `0 0 24px ${C.emerald}88` }}>
          {pct(stock)}
        </span>
        <span className="num mt-0.5 text-xs text-slate-400">현금 {pct(cash)}</span>
      </div>
    </div>
  )
}

function RangeBar({ floor, cap, base, stock }) {
  // 60~100% 구간에서 base(추세만) → stock(위험 차감 후) 위치를 보여준다
  const pos = (v) => `${((v - floor) / (cap - floor)) * 100}%`
  return (
    <div className="mt-4">
      <div className="relative h-2 rounded-full bg-slate-800">
        <span className="absolute top-0 h-2 rounded-full bg-rose-500/50"
          style={{ left: pos(Math.min(stock, base)), width: `calc(${pos(Math.max(Math.min(base, cap), stock))} - ${pos(stock)})` }} />
        <span className="absolute top-[-4px] h-4 w-1 rounded bg-slate-300" style={{ left: `calc(${pos(Math.min(base, cap))} - 2px)` }} title="추세 기준 비중" />
        <span className="absolute top-[-5px] h-[18px] w-[18px] -translate-x-1/2 rounded-full border-2 border-emerald-300 bg-emerald-500"
          style={{ left: pos(stock), boxShadow: `0 0 12px ${C.emerald}` }} />
      </div>
      <div className="mt-2 flex justify-between text-[11px] text-slate-500">
        <span>60% 최대 방어</span><span>100% 풀 투자</span>
      </div>
    </div>
  )
}

function QuadrantMap({ trend, risk, live, threshold }) {
  const W = 220, H = 160, pad = 6
  const x = (t) => pad + ((t + 1) / 2) * (W - 2 * pad)
  const y = (r) => pad + (1 - r) * (H - 2 * pad)
  const ty = y(threshold)
  const cells = [
    { n: 2, x: x(0), y: pad, w: x(1) - x(0), h: ty - pad, c: C.amber },
    { n: 1, x: x(0), y: ty, w: x(1) - x(0), h: H - pad - ty, c: C.emerald },
    { n: 4, x: pad, y: pad, w: x(0) - pad, h: ty - pad, c: C.rose },
    { n: 3, x: pad, y: ty, w: x(0) - pad, h: H - pad - ty, c: C.cyan },
  ]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="추세-위험 국면 지도">
      {cells.map((c) => (
        <g key={c.n}>
          <rect x={c.x + 1} y={c.y + 1} width={c.w - 2} height={c.h - 2} rx="6" fill={c.c} fillOpacity="0.08" stroke={c.c} strokeOpacity="0.25" />
          <text x={c.x + 8} y={c.y + 16} fill={c.c} fontSize="11" fontWeight="600">{c.n}</text>
        </g>
      ))}
      {live && (
        <circle cx={x(live.trend)} cy={y(live.risk)} r="5" fill="none" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="2 2">
          <title>실측 {signed(live.trend)} / {live.risk.toFixed(2)}</title>
        </circle>
      )}
      <circle cx={x(trend)} cy={y(risk)} r="6" fill="#f8fafc" stroke={C.panel} strokeWidth="2" style={{ filter: 'drop-shadow(0 0 6px #f8fafc)' }} />
      <text x={W - pad - 6} y={H - pad - 6} textAnchor="end" fill={C.tick} fontSize="9">추세 →</text>
      <text x={W - pad - 6} y={pad + 14} textAnchor="end" fill={C.tick} fontSize="9">↑ 위험</text>
    </svg>
  )
}

const lineBase = {
  maintainAspectRatio: false, animation: false,
  interaction: { mode: 'index', intersect: false },
  elements: { point: { radius: 0, hoverRadius: 4, hitRadius: 8 }, line: { borderWidth: 2, tension: 0.15 } },
  scales: {
    x: { ticks: { color: C.tick, maxTicksLimit: typeof window !== 'undefined' && window.innerWidth < 640 ? 3 : 6, maxRotation: 0, autoSkipPadding: 12 }, grid: { display: false }, border: { color: C.grid } },
    y: { ticks: { color: C.tick }, grid: { color: C.grid }, border: { display: false } },
  },
  plugins: {
    legend: { display: true, position: 'top', align: 'end', labels: { color: '#cbd5e1', boxWidth: 12, boxHeight: 2, font: { size: 11 } } },
    tooltip: { backgroundColor: '#0b0f19ee', borderColor: '#334155', borderWidth: 1, padding: 10 },
  },
}

function HistoryCharts({ history, label }) {
  const labels = history.map((h) => h.date.slice(2))
  const weight = {
    labels,
    datasets: [
      { label: '집행 비중', data: history.map((h) => h.executed * 100), borderColor: C.emerald,
        backgroundColor: 'rgba(16,185,129,0.10)', fill: { target: { value: 60 } } },
      { label: '목표 비중', data: history.map((h) => h.target * 100), borderColor: C.slate, borderDash: [4, 3], borderWidth: 1.5 },
    ],
  }
  const scores = {
    labels,
    datasets: [
      { label: '추세 점수', data: history.map((h) => h.trend), borderColor: C.cyan },
      { label: '위험 점수', data: history.map((h) => h.risk), borderColor: C.rose },
    ],
  }
  const index = {
    labels,
    datasets: [{ label, data: history.map((h) => h.index), borderColor: '#cbd5e1', borderWidth: 1.5 }],
  }
  const pctY = { ...lineBase, scales: { ...lineBase.scales, y: { ...lineBase.scales.y, min: 55, max: 100, ticks: { color: C.tick, callback: (v) => `${v}%` } } },
    plugins: { ...lineBase.plugins, tooltip: { ...lineBase.plugins.tooltip, callbacks: { label: (c) => ` ${c.dataset.label} ${c.parsed.y.toFixed(1)}%` } } } }
  const scoreY = { ...lineBase, scales: { ...lineBase.scales, y: { ...lineBase.scales.y, min: -1, max: 1 } },
    plugins: { ...lineBase.plugins, tooltip: { ...lineBase.plugins.tooltip, callbacks: { label: (c) => ` ${c.dataset.label} ${c.parsed.y.toFixed(2)}` } } } }
  const idxY = { ...lineBase, plugins: { ...lineBase.plugins, legend: { display: false } } }
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="min-w-0 lg:col-span-3">
        <p className="mb-1 text-xs text-slate-400">주식 비중 (종가 신호 → 익일 집행 · 목표와 3%p 이상 차이날 때만 매매 · 줄일 땐 즉시, 채울 땐 하루 +2%p)</p>
        <div className="h-56 sm:h-64"><Line data={weight} options={pctY} /></div>
      </div>
      <div className="min-w-0 lg:col-span-2">
        <p className="mb-1 text-xs text-slate-400">추세 · 위험 점수</p>
        <div className="h-44"><Line data={scores} options={scoreY} /></div>
      </div>
      <div className="min-w-0">
        <p className="mb-1 text-xs text-slate-400">{label}</p>
        <div className="h-44"><Line data={index} options={idxY} /></div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- calculator
function RebalanceCalculator({ targetWeight, params, usdShare = 0 }) {
  const [currency, setCurrency] = useState('KRW')
  const [total, setTotal] = useState('100000000')
  const [current, setCurrent] = useState('90000000')
  const fmt = useMemo(() => new Intl.NumberFormat(currency === 'KRW' ? 'ko-KR' : 'en-US', {
    style: 'currency', currency, maximumFractionDigits: currency === 'KRW' ? 0 : 2,
  }), [currency])
  const num = (s) => parseFloat(String(s).replace(/[^0-9.]/g, '')) || 0
  const res = rebalance(num(total), num(current), targetWeight, params)

  const switchCurrency = (c) => {
    if (c === currency) return
    setCurrency(c)
    if (c === 'USD') { setTotal('100000'); setCurrent('90000') } else { setTotal('100000000'); setCurrent('90000000') }
  }
  const actionStyle = { BUY: 'text-emerald-400', SELL: 'text-rose-400', HOLD: 'text-slate-300' }
  const actionText = { BUY: '매수', SELL: '매도', HOLD: '유지' }

  return (
    <Panel title="리밸런싱 계산기" icon={Calculator}
      right={
        <div className="flex rounded-lg border border-line p-0.5 text-xs" role="group" aria-label="통화">
          {['KRW', 'USD'].map((c) => (
            <button key={c} onClick={() => switchCurrency(c)}
              className={`rounded-md px-2.5 py-1 font-medium transition ${currency === c ? 'bg-cyan-500/15 text-cyan-300' : 'text-slate-400 hover:text-slate-200'}`}>
              {c}
            </button>
          ))}
        </div>
      }>
      <div className="grid gap-3 sm:grid-cols-2">
        {[['총 자산', total, setTotal], ['현재 주식 평가액', current, setCurrent]].map(([lab, v, set]) => (
          <label key={lab} className="block">
            <span className="text-xs text-slate-400">{lab}</span>
            <input inputMode="decimal" value={Number(num(v)).toLocaleString(currency === 'KRW' ? 'ko-KR' : 'en-US')}
              onChange={(e) => set(e.target.value)}
              className="num mt-1 w-full rounded-lg border border-line bg-ink px-3 py-2 font-mono text-slate-100 outline-none focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20" />
          </label>
        ))}
      </div>
      {res ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl bg-ink/70 p-3">
            <p className="text-xs text-slate-400">목표 주식</p>
            <p className="num font-mono text-lg text-emerald-300">{fmt.format(res.targetValue)}</p>
            <p className="text-[11px] text-slate-500">{pct(targetWeight, 1)} (현재 {pct(res.curWeight, 1)})</p>
          </div>
          <div className="rounded-xl bg-ink/70 p-3">
            <p className="text-xs text-slate-400">목표 현금 {pct(1 - targetWeight, 1)}</p>
            {usdShare > 0 ? (
              <>
                <p className="num font-mono text-sm text-slate-200">원화 {fmt.format(res.cashValue * (1 - usdShare))}</p>
                <p className="num font-mono text-sm text-violet-300">달러 {fmt.format(res.cashValue * usdShare)}</p>
                <p className="text-[11px] text-slate-500">원화: MMF·CD·단기채 / 달러: 달러 MMF·미 초단기국채</p>
              </>
            ) : (
              <>
                <p className="num font-mono text-lg text-slate-200">{fmt.format(res.cashValue)}</p>
                <p className="text-[11px] text-slate-500">MMF·단기국채 등 즉시 투입 가능한 자산</p>
              </>
            )}
          </div>
          <div className="rounded-xl bg-ink/70 p-3">
            <p className="text-xs text-slate-400">다음 거래일 주문</p>
            <p className={`num font-mono text-lg font-semibold ${actionStyle[res.action]}`}>
              {actionText[res.action]} {res.action !== 'HOLD' && fmt.format(res.amount)}
            </p>
            <p className="text-[11px] text-slate-500">
              {res.action === 'BUY' && `천천히: 약 ${res.days}거래일 분할 (하루 ${fmt.format(res.amount / res.days)})`}
              {res.action === 'SELL' && '빠르게: 다음 거래일 일괄 집행'}
              {res.action === 'HOLD' && `차이 ${pct(params.rebalance_band)} 미만 — 매매 불필요`}
            </p>
          </div>
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-500">총 자산을 입력하세요.</p>
      )}
    </Panel>
  )
}

const MARKET_TABS = { US: '🇺🇸 미국', US_TECH: '🇺🇸 미국(기술)', KR: '🇰🇷 한국' }

// 가중 시장(미국 기술주)의 구성 비중 — 내 포트폴리오 비율로 바꾸면 추세 점수를 다시 가중평균한다.
// 브라우저에만 저장 (기록·백테스트·알림은 beta_model/config.py 의 index_weights 기준)
function useIndexWeights(market, defaults) {
  const key = `beta.weights.${market}`
  const [w, setW] = useState(null)
  useEffect(() => {
    if (!defaults) { setW(null); return }
    let saved = null
    try { saved = JSON.parse(localStorage.getItem(key)) } catch { /* 저장소 없음 */ }
    setW(saved && Object.keys(defaults).every((k) => k in saved) ? saved : defaults)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [market, JSON.stringify(defaults)])
  const save = (next) => {
    setW(next)
    try { localStorage.setItem(key, JSON.stringify(next)) } catch { /* 무시 */ }
  }
  return [w, save]
}

function WeightEditor({ weights, defaults, indices, onChange, liveTrend }) {
  const names = Object.keys(weights)
  const [a, b] = names
  const custom = names.some((n) => Math.abs(weights[n] - defaults[n]) > 1e-9)
  return (
    <section className="mb-4 min-w-0 rounded-2xl border border-line bg-panel/80 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-200">내 기술주 포트 비중 → 추세 점수 가중평균</h2>
        {custom && (
          <button onClick={() => onChange(defaults)} className="flex items-center gap-1 text-xs text-cyan-300 hover:text-cyan-200">
            <RotateCcw size={12} aria-hidden /> 기본값 {names.map((n) => `${n} ${pct(defaults[n])}`).join(' · ')}
          </button>
        )}
      </div>
      <div className="mt-3 grid grid-cols-1 items-center gap-3 sm:grid-cols-[1fr_auto]">
        <div>
          <div className="mb-1 flex justify-between font-mono text-xs">
            <span className="text-slate-200">{a} {pct(weights[a])}</span>
            <span className="text-slate-200">{b} {pct(weights[b])}</span>
          </div>
          <input type="range" min={0} max={100} step={5} value={Math.round(weights[a] * 100)}
            aria-label={`${a} 비중`}
            onChange={(e) => { const x = +e.target.value / 100; onChange({ [a]: x, [b]: +(1 - x).toFixed(2) }) }}
            className="w-full accent-emerald-400" />
        </div>
        <dl className="num flex gap-4 font-mono text-xs">
          {names.map((n) => (
            <div key={n}><dt className="font-sans text-[10px] text-slate-500">{n} 추세</dt><dd className="text-slate-200">{indices[n]?.trend_score?.toFixed(2) ?? '–'}</dd></div>
          ))}
          <div><dt className="font-sans text-[10px] text-slate-500">가중 추세</dt><dd className="text-emerald-300">{liveTrend.toFixed(2)}</dd></div>
        </dl>
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        위험 점수는 VIX·금리·신용·바스켓 변동성 기준으로 공통입니다. 비중은 이 브라우저에만 저장되며,
        {custom ? ' 기본값과 달라 국면은 오늘 점수로만 판정합니다(히스테리시스·실전 기록·알림은 기본 비중 기준).' : ' 실전 기록·백테스트·알림과 같은 기준입니다.'}
      </p>
    </section>
  )
}

// ---------------------------------------------------------------- page
export default function BetaDashboard() {
  const { loading, data, error } = useModelData()
  const [market, setMarket] = useState('US')
  const [trend, setTrend] = useState(0.5)
  const [risk, setRisk] = useState(0.3)
  const [activePreset, setActivePreset] = useState(null)

  const params = data?.params ?? DEFAULT_PARAMS
  const usdShare = market === 'KR' ? (params.cash_usd_share ?? 0) : 0
  const regimes = data?.regimes ?? FALLBACK_REGIMES
  const m = data?.markets?.[market]
  const [idxW, setIdxW] = useIndexWeights(market, m?.index_weights)
  const customW = !!(m?.index_weights && idxW && Object.keys(idxW).some((n) => Math.abs(idxW[n] - m.index_weights[n]) > 1e-9))
  const weightedTrend = () => {
    const ns = Object.keys(idxW).filter((n) => m.indices[n]?.trend_score != null)
    const tot = ns.reduce((s, n) => s + idxW[n], 0)
    return tot > 0 ? ns.reduce((s, n) => s + idxW[n] * m.indices[n].trend_score, 0) / tot : m.allocation.trend_score
  }
  const live = m ? { trend: customW ? weightedTrend() : m.allocation.trend_score, risk: m.allocation.risk_score } : null
  const isLive = live && Math.abs(trend - live.trend) < 1e-9 && Math.abs(risk - live.risk) < 1e-9

  // 데이터 로드 또는 시장 전환 시 실측 점수로 슬라이더를 맞춘다
  useEffect(() => {
    if (live) { setTrend(live.trend); setRisk(live.risk); setActivePreset(null) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, market, live?.trend])

  const a = allocate(trend, risk, params)
  // 실측값일 때는 히스테리시스가 적용된 국면(이력 기반), 시뮬레이션은 단순 판정
  const regime = isLive && m && !customW ? m.allocation.regime : a.regime
  const reg = regimes[regime]
  const rs = REGIME_STYLE[regime]

  const applyPreset = (p) => { setTrend(p.trend); setRisk(p.risk); setActivePreset(p.id) }
  const resetLive = () => { if (live) { setTrend(live.trend); setRisk(live.risk); setActivePreset(null) } }

  return (
    <div className="min-h-screen bg-ink bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.08),transparent_55%)]">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:py-10">
        {/* Header */}
        <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400">
              <Compass size={14} aria-hidden /> Beta Model
            </p>
            <h1 className="mt-1 text-2xl font-bold text-slate-50 sm:text-3xl">추세 × 위험 자산배분 대시보드</h1>
            <p className="mt-1 text-sm text-slate-400">
              핸들(추세 −1~+1)로 방향을, 브레이크(위험 0~1)로 속도를 정한다 · 주식 60~100%, 나머지는 MMF
            </p>
          </div>
          <div className="flex flex-col items-start gap-2 sm:items-end">
            <div className="flex rounded-xl border border-line bg-panel p-1" role="tablist" aria-label="시장">
              {Object.keys(data?.markets ?? { US: 1, KR: 1 }).map((k) => (
                <button key={k} role="tab" aria-selected={market === k} onClick={() => setMarket(k)}
                  className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-semibold transition sm:px-4 ${market === k ? 'bg-emerald-500/15 text-emerald-300 shadow-[0_0_16px_-6px_#10b981]' : 'text-slate-400 hover:text-slate-200'}`}>
                  {MARKET_TABS[k] ?? data?.markets?.[k]?.label ?? k}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              {loading ? '데이터 로딩 중…' : m ? `기준일 ${m.as_of} 종가 · 갱신 ${data.meta.fetched_at?.slice(0, 16).replace('T', ' ')} KST` : '데모 모드'}
            </p>
          </div>
        </header>

        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
            <Info size={18} className="mt-0.5 shrink-0" aria-hidden />
            <div>
              실측 데이터를 찾지 못해 데모 모드로 표시합니다. 프로젝트 루트에서 <code className="rounded bg-black/30 px-1.5 py-0.5 font-mono">python -m beta_model.export</code> 를 실행한 뒤 새로고침하세요.
            </div>
          </div>
        )}
        {data?.meta?.vkospi_source === 'proxy' && market === 'KR' && (
          <div className="mb-6 flex items-start gap-3 rounded-xl border border-line bg-panel p-3 text-xs text-slate-400">
            <Info size={16} className="mt-0.5 shrink-0 text-cyan-400" aria-hidden />
            VKOSPI는 무료 API가 없어 KOSPI200 20일 실현변동성×1.15로 대체 계산 중입니다. 실제 값을 쓰려면 <code className="font-mono">data/vkospi.csv</code>(Date,Close)를 넣으세요.
          </div>
        )}

        {m?.index_weights && idxW && (
          <WeightEditor weights={idxW} defaults={m.index_weights} indices={m.indices} onChange={setIdxW} liveTrend={live.trend} />
        )}

        {/* Row 1: Regime / Allocation / Controls */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          <section className={`min-w-0 rounded-2xl border border-line bg-panel/80 p-5 ring-1 ${rs.ring} ${rs.glow} md:col-span-2 xl:col-span-1`}>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-medium text-slate-400">현재 국면</p>
                <h2 className="mt-1 flex items-center gap-2 text-xl font-bold" style={{ color: rs.color }}>
                  <rs.Icon size={22} aria-hidden /> 국면 {regime} · {reg.name}
                </h2>
              </div>
              {!isLive && live && (
                <span className="shrink-0 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-0.5 text-[11px] text-cyan-300">시뮬레이션</span>
              )}
            </div>
            <p className="mt-2 text-sm text-slate-300">{reg.summary}</p>
            {isLive && !customW && m.allocation.regime_raw != null && m.allocation.regime_raw !== regime && (
              <p className="mt-2 rounded-lg bg-white/5 px-2.5 py-1.5 text-[11px] text-slate-400">
                경계 근처: 오늘 점수만 보면 국면 {m.allocation.regime_raw}이지만, 이중 임계값(추세 ±{params.trend_hysteresis}, 위험 ±{params.risk_hysteresis})을 넘지 않아 국면 {regime}을 유지합니다.
              </p>
            )}
            <ol className="mt-4 space-y-2">
              {reg.actions.map((act, i) => (
                <li key={act} className="flex gap-3 text-sm text-slate-200">
                  <span className="num flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-ink" style={{ background: rs.color }}>{i + 1}</span>
                  {act}
                </li>
              ))}
            </ol>
            <div className="mt-5 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
              <QuadrantMap trend={a.trend} risk={a.risk} live={isLive ? null : live} threshold={params.risk_threshold} />
              <ul className="space-y-1 text-[11px] text-slate-400">
                {[1, 2, 3, 4].map((n) => (
                  <li key={n} className={`flex items-center gap-1.5 ${n === regime ? 'font-semibold text-slate-100' : ''}`}>
                    <span className="h-2 w-2 rounded-sm" style={{ background: REGIME_STYLE[n].color }} />{n}. {regimes[n].name}
                  </li>
                ))}
                {!isLive && live && <li className="pt-1 text-slate-500">◌ 실측 위치</li>}
              </ul>
            </div>
          </section>

          <Panel title="주식 vs 현금 배분" icon={Gauge}>
            <AllocationDonut stock={a.stock} cash={a.cash} usdShare={usdShare} />
            <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-slate-300">
              {[['주식', a.stock, C.emerald], ...cashParts(a.cash, usdShare)].map(([lab, v, col]) => (
                <span key={lab} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: col }} />{lab} {pct(v, 1)}</span>
              ))}
            </div>
            <RangeBar floor={params.floor} cap={params.cap} base={a.base} stock={a.stock} />
            <dl className="num mt-4 grid grid-cols-3 gap-2 text-center font-mono text-xs">
              <div className="rounded-lg bg-ink/70 p-2"><dt className="font-sans text-slate-500">추세 기준</dt><dd className="text-slate-200">{pct(a.base, 1)}</dd></div>
              <div className="rounded-lg bg-ink/70 p-2"><dt className="font-sans text-slate-500">위험 차감</dt><dd className="text-rose-300">−{pct(a.deduction, 1)}</dd></div>
              <div className="rounded-lg bg-ink/70 p-2"><dt className="font-sans text-slate-500">최종</dt><dd className="text-emerald-300">{pct(a.stock, 1)}</dd></div>
            </dl>
            {m && isLive && m.executed_weight != null && Math.abs(m.executed_weight - a.stock) > 0.001 && (
              <p className="mt-3 text-[11px] text-slate-400">
                속도 조절 반영 현재 집행 비중: <span className="num font-mono text-emerald-300">{pct(m.executed_weight, 1)}</span>
                {m.executed_weight < a.stock ? ' — 목표까지 천천히 채우는 중' : ' — 다음 거래일 목표로 하향'}
              </p>
            )}
          </Panel>

          <Panel title="점수 시뮬레이터" icon={Activity}
            right={live && (
              <button onClick={resetLive} disabled={isLive}
                className="flex items-center gap-1 rounded-lg border border-line px-2 py-1 text-xs text-slate-300 transition hover:border-emerald-500/50 hover:text-emerald-300 disabled:opacity-40">
                <RotateCcw size={12} aria-hidden /> 실측값
              </button>
            )}>
            <div className="space-y-6">
              <ScoreSlider label="추세 점수 (핸들 · 방향)" value={trend}
                desc="이동평균 정배열 50% + 52주 고점 대비 낙폭 25% + 3개월 모멘텀 25%. 0 이상이면 상승 국면, 높을수록 기본 주식 비중↑" min={-1} max={1} step={0.01}
                onChange={(v) => { setTrend(v); setActivePreset(null) }}
                color={trend >= 0 ? C.emerald : C.rose} scale={TREND_SCALE} live={live?.trend} />
              <ScoreSlider label="위험 점수 (브레이크 · 불안감)" value={risk}
                desc={`변동성 지수·실현변동성·금리/환율 급등·하이일드 스프레드 종합. ${params.risk_threshold} 이상이면 '높은 불안' 국면, 1.0이면 주식 −${pct(params.risk_span)}p`} min={0} max={1} step={0.01}
                onChange={(v) => { setRisk(v); setActivePreset(null) }}
                color={riskColor(risk)} scale={RISK_SCALE} live={live?.risk} />
            </div>
            {live && (
              <p className="mt-4 flex items-center gap-1.5 text-[11px] text-slate-500">
                <span className="inline-block h-3 w-0.5 rounded bg-slate-300" aria-hidden /> 슬라이더 위 흰 눈금 = 오늘 실측값 · 움직여서 "만약 ~라면"을 시뮬레이션
              </p>
            )}
            <div className="mt-4 grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button key={p.id} onClick={() => applyPreset(p)}
                  className={`flex items-center gap-2 rounded-xl border p-2.5 text-left transition hover:bg-white/5 ${activePreset === p.id ? 'bg-white/5' : 'border-line'}`}
                  style={activePreset === p.id ? { borderColor: p.color, boxShadow: `0 0 16px -6px ${p.color}` } : undefined}>
                  <p.Icon size={16} style={{ color: p.color }} className="shrink-0" aria-hidden />
                  <span className="min-w-0">
                    <span className="block truncate text-xs font-semibold text-slate-100">{p.label}</span>
                    <span className="block text-[11px] text-slate-500">{p.sub}</span>
                  </span>
                </button>
              ))}
            </div>
          </Panel>
        </div>

        {/* Row 2: Score breakdown + macro */}
        {m && (
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <Panel title={`추세 구성 · ${m.label}`} icon={m.allocation.trend_score >= 0 ? TrendingUp : TrendingDown}
              right={<span className="num font-mono text-sm" style={{ color: m.allocation.trend_score >= 0 ? C.emerald : C.rose }}>{signed(m.allocation.trend_score)}</span>}>
              <div className="space-y-3">
                {Object.entries(m.trend_components).map(([k, v]) => (
                  <ComponentBar key={k} label={TREND_LABELS[k] ?? k} value={v} diverging />
                ))}
              </div>
              <div className="mt-4 overflow-x-auto">
                <table className="num w-full text-right font-mono text-xs">
                  <thead className="font-sans text-slate-500">
                    <tr><th className="pb-1 text-left font-medium">지수</th><th className="font-medium">종가</th><th className="font-medium">고점比</th><th className="font-medium">3M</th><th className="font-medium">점수</th></tr>
                  </thead>
                  <tbody className="text-slate-300">
                    {Object.entries(m.indices).map(([k, f]) => (
                      <tr key={k} className="border-t border-line">
                        <td className="py-1.5 text-left font-sans">{k}</td>
                        <td>{f.close.toLocaleString()}</td>
                        <td className={f.drawdown < -0.1 ? 'text-rose-300' : ''}>{pct(f.drawdown, 1)}</td>
                        <td className={f.roc_3m >= 0 ? 'text-emerald-300' : 'text-rose-300'}>{pct(f.roc_3m, 1)}</td>
                        <td>{signed(f.trend_score)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel title={`위험 구성 · ${m.label}`} icon={ShieldAlert}
              right={<span className="num font-mono text-sm" style={{ color: riskColor(m.allocation.risk_score) }}>{m.allocation.risk_score.toFixed(2)}</span>}>
              <div className="space-y-3">
                {Object.entries(m.risk_components).map(([k, v]) => (
                  <ComponentBar key={k} value={v} weight={m.risk_weights[k]}
                    label={k === 'credit' && market === 'KR' && data.meta.kr_credit_blend > 0 ? '신용 스프레드 (Baa·국내 AA−)' : (RISK_LABELS[k] ?? k)} />
                ))}
              </div>
              <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
                변동성 지수는 252일 백분위와 절대 레벨을, 신용(Baa 회사채−미 10년물{market === 'KR' && data.meta.kr_credit_blend > 0 ? ' + 국내 AA−−국고 3년' : ''})은 Z점수와 절대 레벨을 반반 반영. 금리·환율은 10일 상승분만 위험으로 봅니다.
              </p>
            </Panel>

            <Panel title="매크로 체크" icon={Activity} className="md:col-span-2 xl:col-span-1">
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-2">
                {Object.entries(data.macro).map(([k, v]) => {
                  const meta = MACRO_META[k]
                  if (!meta || v.value == null) return null
                  const up = (v.chg_10d ?? 0) > 0
                  return (
                    <div key={k} className="rounded-xl bg-ink/70 p-3">
                      <p className="text-[11px] text-slate-400">{meta.label}{k === 'VKOSPI' && data.meta.vkospi_source === 'proxy' ? '*' : ''}</p>
                      <p className="num font-mono text-base text-slate-100">{meta.fmt(v.value)}</p>
                      {v.chg_10d != null && (
                        <p className={`num flex items-center gap-1 font-mono text-[11px] ${up ? 'text-rose-300' : 'text-emerald-300'}`}>
                          {up ? <TrendingUp size={11} aria-hidden /> : <TrendingDown size={11} aria-hidden />}
                          {up ? '+' : ''}{meta.chg(v.chg_10d)} <span className="font-sans text-slate-500">10일</span>
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            </Panel>
          </div>
        )}

        {/* Row 3: history */}
        {m && (
          <Panel title={`최근 1년 모델 추이 · ${m.label}`} icon={TrendingUp} className="mt-4">
            <HistoryCharts history={m.history} label={m.indices[Object.keys(m.indices)[0]] ? `${Object.keys(m.indices)[0]} 지수` : '지수'} />
          </Panel>
        )}

        {m && <LivePanel log={data?.signal_log} market={market} label={m.label} />}
        {m && <ScenarioPanel scenarios={data?.scenarios} market={market} label={m.label} />}
        <BacktestPanel market={market} />
        <ValidationPanel />

        {/* Row 4: calculator + principles */}
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="min-w-0 lg:col-span-2"><RebalanceCalculator targetWeight={a.stock} params={params} usdShare={usdShare} /></div>
          <Panel title="계좌 운용 4원칙" icon={ShieldCheck}>
            <ul className="space-y-3 text-sm">
              {[
                ['비중 범위', '주식 60~100%. 레버리지·마이너스 현금 금지'],
                ['속도 조절', '줄일 때는 빠르게, 채울 때는 천천히'],
                ['매매 시점', '신호는 종가 확인, 매매는 다음 거래일'],
                ['현금 관리', 'MMF·단기채로 보관해 즉시 투입 대기'],
              ].map(([t, d]) => (
                <li key={t}>
                  <p className="font-semibold text-slate-100">{t}</p>
                  <p className="text-slate-400">{d}</p>
                </li>
              ))}
            </ul>
          </Panel>
        </div>

        <footer className="mt-8 text-center text-[11px] leading-relaxed text-slate-600">
          성상현 「매크로비욘드」 베타 모델 아이디어를 구현한 개인 참고용 도구입니다. 투자 권유가 아니며, 점수 산식과 임계값은 임의로 설정한 값입니다.
        </footer>
      </div>
    </div>
  )
}
