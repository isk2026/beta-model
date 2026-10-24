# Beta Model 대시보드

성상현 「매크로비욘드」 6-2부의 **베타 모델**(추세 점수 × 위험 점수 → 주식/현금 비중)을 구현한 개인 투자 참고용 도구.

```
beta_model/            Python 엔진
  config.py            모든 파라미터 (가중치·임계값·배분 공식)
  data_pipeline.py     1단계: 데이터 수집 + SMA/MDD/변동성/ROC
  scoring.py           2단계: BetaModelScoringEngine (추세 -1~+1, 위험 0~1)
  allocation.py        3단계: 비중·국면·속도조절·리밸런싱
  export.py            실행 → dashboard/public/data/beta_model.json
tests/test_model.py    오프라인 단위 테스트 (합성 데이터)
dashboard/             4단계: React + Tailwind + Chart.js + lucide
```

## 실행

```bash
pip install -r requirements.txt
python -m beta_model.export --refresh   # 데이터 수집 + JSON 생성 (매일 장 마감 후)
python -m pytest -q                     # 테스트

cd dashboard
npm install
npm run dev                             # http://localhost:5173
```

## 모바일 조회 · 자동 갱신

- 대시보드: **https://isk2026.github.io/beta-model/** (휴대폰 브라우저 → 공유 → "홈 화면에 추가"하면 앱처럼 열림)
- GitHub Actions(`.github/workflows/update.yml`)가 평일 두 번 데이터 수집 → 테스트 → 점수 계산 → 배포:
  - **16:30 KST** — 한국장 마감(15:30) 후
  - **07:30 KST** — 미국장 마감(05:00 서머타임 / 06:00) 후
- 실행마다 `dashboard/public/data/beta_model.json`을 커밋해 일별 모델 기록을 남긴다.
- 수동 갱신: GitHub 저장소 → Actions → *Update & Deploy* → *Run workflow*
  (또는 `gh workflow run update.yml`)
- GitHub 예약 실행은 수 분~수십 분 늦게 시작될 수 있다.

## 원래 프롬프트 대비 변경점

| 항목 | 원 프롬프트 | 변경 | 이유 |
|---|---|---|---|
| 배분 공식 | Base = 0.80 + 0.20·trend | **Base = 0.85 + 0.15·trend** (70~100%) | 원 공식은 추세 −1이면 위험과 무관하게 60%가 되어 원문의 '하락 방어(과도한 현금화 지양)'와 '최대 방어(현금 40%)'가 구분되지 않음 |
| 국면 | 대시보드에만 표시 | Python/JS 공통 로직 (`classify_regime`) | 국면 기준을 한 곳에서 관리 |
| 속도 조절 | 없음 | 줄일 땐 즉시, 채울 땐 하루 +2%p, 종가 신호 → 익일 집행 | 원문 계좌 운용 원칙 2·3번 |
| 변동성 지수 | 252일 백분위 | 백분위 50% + 절대 레벨 50% | 조용한 1년 뒤에는 VIX 16도 상위 백분위가 됨 |
| HY 스프레드 | Z점수 | Z점수 50% + 절대 레벨(3%→0, 6%→1) 50% | 스프레드가 역사적 저점(3.4%)인데 Z점수만으로 위험 1.0이 나오는 문제 |
| 금리·환율 | 10일 변화율 | 10일 **상승분만** 위험으로 반영 | 금리 하락·원화 강세는 위험이 아님 |
| 시장 | 단일 | 미국(S&P·나스닥+VIX) / 한국(코스피·코스닥+VKOSPI) 별도 | 원문이 미국·한국을 다른 국면으로 평가 |
| 지표 계산 | 공통 달력 ffill 후 계산 | 시장별 거래일 기준 계산 후 정렬·ffill | 휴장일 ffill이 변동성을 과소평가함 |
| 대시보드 | 슬라이더 시뮬레이터 | 실측 점수 + 시뮬레이터 + 구성요소 분해 + 1년 추이 + 국면 지도 | 투자 참고용으로 "왜 이 비중인지"가 보여야 함 |
| 데이터 추가 | — | 달러 인덱스(DXY) | 원문에서 위험 모니터링 지표로 언급 |

## 데이터 한계

- **VKOSPI**: 무료 API(yfinance/FDR)가 없어 KOSPI200 20일 실현변동성 × 1.15로 대체한다.
  KRX 등에서 받은 실제 값을 `data/vkospi.csv`(첫 열 Date, 둘째 열 Close, 252행 이상)로 넣으면 자동으로 그 값을 쓴다.
- 데이터는 6시간 캐시된다(`data/cache/`). `--refresh` 플래그를 주면 강제로 다시 수집한다.
- 점수 산식과 임계값은 영상에 공개되지 않아 임의로 설정한 값이다. `beta_model/config.py`에서 조정할 수 있다. 투자 권유가 아니다.
