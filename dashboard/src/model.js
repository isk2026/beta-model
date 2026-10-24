// Python beta_model/allocation.py 와 같은 공식. 파라미터는 JSON(params)에서 받아 일치시킨다.

export const DEFAULT_PARAMS = {
  base_mid: 0.85, trend_span: 0.15, risk_span: 0.2, floor: 0.6, cap: 1.0,
  trend_threshold: 0.0, risk_threshold: 0.5, max_daily_increase: 0.02, rebalance_band: 0.03,
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))

export function allocate(trend, risk, p = DEFAULT_PARAMS) {
  const t = clamp(trend, -1, 1)
  const r = clamp(risk, 0, 1)
  const base = p.base_mid + p.trend_span * t
  const deduction = p.risk_span * r
  const stock = clamp(base - deduction, p.floor, p.cap)
  const up = t >= p.trend_threshold
  const calm = r < p.risk_threshold
  const regime = up ? (calm ? 1 : 2) : (calm ? 3 : 4)
  return { trend: t, risk: r, base, deduction, stock, cash: 1 - stock, regime }
}

export function rebalance(total, currentStock, targetWeight, p = DEFAULT_PARAMS) {
  if (!(total > 0)) return null
  const cur = clamp(currentStock || 0, 0, total)
  const targetValue = total * targetWeight
  const diff = targetValue - cur
  const curWeight = cur / total
  const action = Math.abs(targetWeight - curWeight) < p.rebalance_band ? 'HOLD' : diff > 0 ? 'BUY' : 'SELL'
  // 채울 때는 천천히: 하루 max_daily_increase 만큼씩 분할 매수
  const days = action === 'BUY' ? Math.ceil((targetWeight - curWeight) / p.max_daily_increase) : action === 'SELL' ? 1 : 0
  return { action, amount: action === 'HOLD' ? 0 : Math.abs(diff), curWeight, targetValue, cashValue: total - targetValue, days }
}

export const FALLBACK_REGIMES = {
  1: { name: '상승 따라가기', summary: '추세 상승 + 불안감 낮음 → 주식 비중을 채우는 구간',
    actions: ['목표 비중까지 주식을 천천히(분할) 채운다', '추세 이탈(SMA20 하향) 여부만 점검', '현금은 MMF에서 대기'] },
  2: { name: '미리 줄이기', summary: '추세는 살아있지만 불안감 상승 → 하락 전환 전 선제적 일부 축소',
    actions: ['위험 점수만큼 주식 비중을 즉시 덜어낸다', '금리·환율·신용 스프레드 중 무엇이 위험을 올렸는지 확인', '추세가 유지되면 위험 하락 시 천천히 복원'] },
  3: { name: '하락 방어', summary: '추세 하락 + 불안감 낮음 → 방향에 맞춰 줄이되 과도한 현금화 지양',
    actions: ['추세 점수에 맞춰 비중 축소 (70%대 유지)', '공포가 아닌 추세 조정이므로 투매 금지', '추세 반전(SMA20 회복) 시 바로 채울 준비'] },
  4: { name: '최대 방어 / 공포', summary: '추세 하락 + 불안감 높음 → 현금 최대 40%, 동시에 Buy the Dip 후보 탐색',
    actions: ['현금(MMF·단기채) 비중을 최대 40%까지 확대', '깊게 빠진 우량 자산 매수 리스트 작성', '위험 점수 꺾이는 시점부터 분할 매수 개시'] },
}
