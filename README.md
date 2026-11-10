<div align="center">

<sub>SLIDE 01 / 08</sub>

# 🎸 오늘의 록 분석 대시보드

🔗 [배포된 오늘의 록 대시보드 바로가기](https://todaysrockdashboard-n9jt6w83iqnkusa5a3sqkj.streamlit.app/)

### 무엇을 추천하고 있나 — 추천 목록 933곡의 성향 보기

`Python` · `Pandas` · `Altair` · `Streamlit` · `RAG` · `UMAP` · `BERTopic`

</div>

<br>

---

<sub>SLIDE 02 / 08</sub>

## 🔎 출발 질문

> **「오늘의 록」은 어떤 날씨와 기분에 어떤 록을 추천하고 있을까?**

추천 목록 **933곡 · 329명(팀) · 74개 장르**.
기분·날씨·에너지는 편집자가 붙인 라벨이라, 이 분석은 음악 자체가 아니라 **사이트의 큐레이션 성향**을 보는 작업이다.

```
원본 읽기 → CSV 만들기 → 기준별 집계 → 교차표·Lift → 순위 비교 → 근거 기반 Q&A
data_backup.json  build_data.py   Pandas groupby  기분×날씨  표 2개   임베딩 검색 + 답변 생성
```

<br>

---

<sub>SLIDE 03 / 08</sub>

## 🌦️ 기분 × 날씨 — 우연보다 자주 붙는 조합

<div align="center">

| 조합 | 곡 수 | Lift |
| :--- | ---: | ---: |
| 눈 × 피곤 | 162곡 | **1.80** |
| 맑음 × 신남 | 345곡 | **1.71** |
| 눈 × 우울 | 239곡 | **1.61** |

</div>

- Lift = 실제 곡 수 ÷ 우연히 겹칠 때 예상 곡 수
- 1보다 크면 우연보다 자주 함께 붙어 있다는 뜻
- 화면에서 **곡 수 · 날씨 기준 비율 · 기분 기준 비율 · Lift**로 바꿔 볼 수 있음

<br>

---

<sub>SLIDE 04 / 08</sub>

## 🕳️ 추천 후보가 얇은 칸

<div align="center">

| 조합 | 곡 수 | Lift |
| :--- | ---: | ---: |
| 맑음 × 우울 | **39곡** | 0.23 |
| 맑음 × 피곤 | 41곡 | 0.40 |
| 눈 × 답답 | 55곡 | 0.47 |

</div>

- 맑은 날 우울한 사용자는 39곡 안에서만 추천을 받는다
- 곡을 보강할 우선순위 칸

<br>

---

<sub>SLIDE 05 / 08</sub>

## ⚡ 많은 장르 ≠ 강렬한 장르

- 곡 수 1위: **Indie Rock** (157곡)
- 평균 에너지 1위: **Punk Rock** (4.71, 10곡 이상 장르 기준)
- 에너지 4~5인 곡이 **511곡(54.8%)** — 추천 목록이 고에너지 쪽으로 기울어 있음
- 화면에서 **평균 막대 ↔ 100% 누적 분포**를 바꿔 볼 수 있음

<br>

---

<sub>SLIDE 06 / 08</sub>

## 💬 분석 Q&A (RAG)

`data/todays_rock_facts.csv`의 사실 12개를 임베딩해 질문과 가까운 근거를 찾고, 그 안에서만 답한다.

| 채널 | 사용하는 자료 |
| :--- | :--- |
| A | 사실표 |
| B | 사실표 + 라벨 설명 |
| C | 사실표 + 라벨 설명 + 해석 기준 |

<br>

---

<sub>SLIDE 07 / 08</sub>

## 🗺️ 텍스트 지도 (임베딩 + UMAP + BERTopic)

> **편집자가 붙인 기분 라벨과, 소개 문장이 실제로 말하는 감정은 같을까?**

```
소개 문장 → 임베딩 → UMAP 2D 지도 → BERTopic 주제 묶기 → AI 주제 이름 → 이웃 곡 5개
theme·tone·reason  text-embedding-3-small  좌표 x,y   한국어 형태소(Kiwi) 키워드   gpt-4.1-mini   코사인 유사도
```

- 곡명·아티스트는 빼고 **소개 문장만** 계산에 넣음
- 지도 색을 **주제 · 편집자 기분 라벨 · 에너지 · 국내/해외 · 연대**로 바꿔 볼 수 있음
- **주제 × 기분 히트맵**으로 텍스트 주제와 편집자 라벨이 얼마나 겹치는지 비교
- 곡을 고르면 소개 문장이 가장 비슷한 곡 5개를 보여줌

무거운 계산은 로컬에서 한 번만 하고 결과 CSV만 배포한다.

```bash
pip install -r requirements-analysis.txt
python build_text_map.py
```

<br>

---

<sub>SLIDE 08 / 08</sub>

## 🗂️ 파일 구성 · 실행

```
app.py                  대시보드 첫 화면
pages/2_텍스트_지도.py    임베딩 지도 · 주제 · 비슷한 곡
pages/3_분석_QA.py       RAG Q&A
build_text_map.py       텍스트 지도 CSV 만들기 (로컬에서 한 번)
utils/todays_rock_rag.py 사실표 읽기 · 유사도 · 답변 생성
build_data.py           source/ → data/ CSV 다시 만들기
source/                 원본 (data_backup.json, 후보 목록 CSV)
data/                   분석용 CSV 5개
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

Q&A 페이지는 `.streamlit/secrets.toml`에 `OPENAI_API_KEY`가 필요하다. 
`.streamlit/secrets.toml.example`을 복사해서 쓰고, 키는 GitHub에 올리지 않는다.

곡 데이터가 바뀌면 `source/data_backup.json`을 교체하고 `python build_data.py`를 실행한다.
