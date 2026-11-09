from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="오늘의 록 텍스트 지도", page_icon="🗺️", layout="wide")

app_folder = Path(__file__).resolve().parent.parent
data_folder = app_folder / "data"

MOODS = ["calm", "excited", "focused", "blue", "tired", "frustrated"]
MOOD_KO = {"calm": "평온", "excited": "신남", "focused": "집중",
           "blue": "우울", "tired": "피곤", "frustrated": "답답"}
MOOD_ORDER = [MOOD_KO[m] for m in MOODS]

st.title("🗺️ 오늘의 록 텍스트 지도")
st.caption(
    "곡마다 붙은 소개 문장(주제 · 분위기 · 추천 이유)을 임베딩으로 바꿔 2차원 지도에 찍었어. "
    "가까이 있는 점일수록 소개 문장의 뜻이 비슷해. 곡명과 아티스트는 계산에 넣지 않았어."
)

map_path = data_folder / "todays_rock_text_map.csv"
topic_path = data_folder / "todays_rock_topics.csv"

if not map_path.exists() or not topic_path.exists():
    st.info("텍스트 지도 데이터가 아직 없어. 로컬에서 아래 명령을 한 번 실행해 줘.")
    st.code("pip install -r requirements-analysis.txt\npython build_text_map.py", language="bash")
    st.caption("실행하면 data/todays_rock_text_map.csv와 data/todays_rock_topics.csv가 만들어져.")
    st.stop()


@st.cache_data
def load_data():
    songs = pd.read_csv(data_folder / "todays_rock_songs.csv")
    text_map = pd.read_csv(map_path)
    topics = pd.read_csv(topic_path)
    merged = songs.merge(text_map, on="song_id", how="inner")
    return merged, topics


songs, topics = load_data()
total = len(songs)

c1, c2, c3 = st.columns(3)
c1.metric("지도에 찍은 곡", f"{total:,}곡")
c2.metric("주제 수", f"{len(topics)}개")
c3.metric("가장 큰 주제", topics.iloc[0]["topic_name"], f"{topics.iloc[0]['songs']}곡", delta_color="off")

st.divider()

# 첫째 줄: 지도
st.header("소개 문장으로 그린 지도")

map_col, map_note = st.columns([3, 1])

color_options = ["주제 (BERTopic)", "편집자 기분 라벨", "에너지", "국내·해외", "연대", "추가 시기 (곡 번호)"]

with map_note:
    color_by = st.radio("점 색 기준", color_options, key="map_color")

    picked_mood = None
    if color_by == "편집자 기분 라벨":
        picked_mood = st.selectbox("강조할 기분", MOODS, format_func=MOOD_KO.get)
        st.caption("한 곡에 기분이 여러 개라서, 고른 기분이 붙은 곡만 색칠해.")
    if color_by == "추가 시기 (곡 번호)":
        st.caption(
            "1~500번은 처음 만든 목록, 501번 이후는 후보 목록으로 추가한 곡이야. "
            "두 시기의 소개 문장 문체가 달라서 지도에서도 갈라져 보여."
        )

    picked_topics = st.multiselect(
        "주제만 골라 보기",
        topics["topic_name"].tolist(),
        placeholder="전체",
    )
    st.caption("마우스 휠로 확대, 드래그로 이동할 수 있어. 축 숫자 자체에는 의미가 없어.")

with map_col:
    plot = songs.copy()
    if picked_topics:
        plot = plot.loc[plot["topic_name"].isin(picked_topics)]

    if color_by == "주제 (BERTopic)":
        color = alt.Color("topic_name:N", title="주제", scale=alt.Scale(scheme="category20"),
                          legend=alt.Legend(columns=1, symbolLimit=40))
    elif color_by == "편집자 기분 라벨":
        plot["label"] = plot[f"mood_{picked_mood}"].map(
            {1: f"{MOOD_KO[picked_mood]} 라벨 있음", 0: "없음"})
        color = alt.Color("label:N", title="기분 라벨",
                          scale=alt.Scale(domain=[f"{MOOD_KO[picked_mood]} 라벨 있음", "없음"],
                                          range=["#e4572e", "#d0d0d0"]))
    elif color_by == "에너지":
        color = alt.Color("energy:O", title="에너지", scale=alt.Scale(scheme="oranges"))
    elif color_by == "국내·해외":
        color = alt.Color("region:N", title="구분")
    elif color_by == "연대":
        color = alt.Color("decade:N", title="연대", scale=alt.Scale(scheme="viridis"))
    else:
        plot["batch"] = (plot["song_id"] > 500).map({False: "1~500번", True: "501번 이후"})
        color = alt.Color("batch:N", title="추가 시기",
                          scale=alt.Scale(domain=["1~500번", "501번 이후"], range=["#4c78a8", "#f58518"]))

    chart = (
        alt.Chart(plot)
        .mark_circle(size=55, opacity=0.8)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=color,
            tooltip=[
                alt.Tooltip("title:N", title="곡명"),
                alt.Tooltip("artist:N", title="아티스트"),
                alt.Tooltip("topic_name:N", title="주제"),
                alt.Tooltip("theme:N", title="소개"),
                alt.Tooltip("moods:N", title="기분 라벨"),
                alt.Tooltip("energy:Q", title="에너지"),
            ],
        )
        .properties(height=560)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"{len(plot)}곡 표시 중")

st.divider()

# 둘째 줄: 주제표
st.header("BERTopic이 찾은 주제")
st.caption("문장이 비슷한 곡끼리 자동으로 묶고, 묶음마다 자주 나온 단어(키워드)와 AI가 지은 이름을 붙였어.")

stats = (
    songs.groupby("topic_id")
    .agg(avg_energy=("energy", "mean"),
         korean_pct=("region", lambda s: (s == "국내").mean() * 100))
    .reset_index()
)
topic_view = topics.merge(stats, on="topic_id", how="left")
st.dataframe(
    topic_view[["topic_name", "songs", "avg_energy", "korean_pct", "keywords", "representative_songs"]],
    hide_index=True,
    column_config={
        "topic_name": "주제",
        "songs": st.column_config.NumberColumn("곡 수"),
        "avg_energy": st.column_config.NumberColumn("평균 에너지", format="%.2f"),
        "korean_pct": st.column_config.NumberColumn("국내 비율", format="%.1f%%"),
        "keywords": "키워드",
        "representative_songs": "대표 곡",
    },
)

st.divider()

# 셋째 줄: 주제 × 편집자 기분
st.header("텍스트 주제와 편집자의 기분 라벨은 일치할까?")
st.caption(
    "주제마다 각 기분 라벨이 붙은 곡의 비율이야. 한 주제에서 특정 기분이 압도적이면 "
    "문장과 라벨이 같은 이야기를 하고 있다는 뜻이고, 고르게 퍼져 있으면 라벨이 문장과 따로 논다는 뜻이야."
)

mood_cols = [f"mood_{m}" for m in MOODS]
topic_mood = (
    songs.groupby("topic_name")[mood_cols].mean().mul(100)
    .reset_index()
    .melt(id_vars="topic_name", var_name="mood", value_name="pct")
)
topic_mood["mood_ko"] = topic_mood["mood"].str.replace("mood_", "").map(MOOD_KO)
topic_order = topics["topic_name"].tolist()

base = alt.Chart(topic_mood).encode(
    x=alt.X("mood_ko:N", axis=alt.Axis(labelAngle=0), title="편집자 기분 라벨", sort=MOOD_ORDER),
    y=alt.Y("topic_name:N", title="텍스트 주제", sort=topic_order),
)
heat = base.mark_rect().encode(
    color=alt.Color("pct:Q", title="라벨 비율 (%)", scale=alt.Scale(scheme="oranges", domain=[0, 100])),
    tooltip=[
        alt.Tooltip("topic_name:N", title="주제"),
        alt.Tooltip("mood_ko:N", title="기분"),
        alt.Tooltip("pct:Q", title="비율 (%)", format=".1f"),
    ],
)
text = base.mark_text(fontSize=12).encode(text=alt.Text("pct:Q", format=".0f"))
st.altair_chart((heat + text).properties(height=max(260, 32 * len(topic_order))),
                width="stretch")

with st.expander("주제별로 가장 많이 붙은 기분"):
    dominant = topic_mood.loc[topic_mood.groupby("topic_name")["pct"].idxmax()]
    dominant = dominant.set_index("topic_name").loc[topic_order].reset_index()
    for row in dominant.itertuples():
        st.write(f"**{row.topic_name}** · {row.mood_ko} {row.pct:.0f}%")
    st.info("곡마다 기분이 여러 개 붙을 수 있어서 한 주제 안의 비율을 더하면 100%를 넘어.")

st.divider()

# 넷째 줄: 비슷한 곡 찾기
st.header("소개 문장이 비슷한 곡 찾기")

songs["label"] = songs["title"] + " — " + songs["artist"]
picked = st.selectbox("곡 고르기", songs.sort_values("title")["label"].tolist())
target = songs.loc[songs["label"] == picked].iloc[0]

st.write(f"**{target['title']}** ({target['artist']}) · 주제 **{target['topic_name']}**")
st.caption(f"소개: {target['theme']} / {target['tone']}")

neighbor_ids = [int(i) for i in str(target["neighbors"]).split("|")]
neighbor_scores = [float(s) for s in str(target["neighbor_scores"]).split("|")]
by_id = songs.set_index("song_id")
neighbors = by_id.loc[neighbor_ids, ["title", "artist", "topic_name", "theme", "moods"]].reset_index()
neighbors["similarity"] = neighbor_scores
neighbors["same_topic"] = neighbors["topic_name"].eq(target["topic_name"]).map({True: "같음", False: "다름"})

st.dataframe(
    neighbors[["title", "artist", "similarity", "same_topic", "theme", "moods"]],
    hide_index=True,
    column_config={
        "title": "곡명",
        "artist": "아티스트",
        "similarity": st.column_config.ProgressColumn("문장 유사도", min_value=0.0, max_value=1.0, format="%.3f"),
        "same_topic": "같은 주제",
        "theme": "소개",
        "moods": "기분 라벨",
    },
)
st.caption("유사도는 임베딩 벡터의 코사인 유사도야. 1에 가까울수록 소개 문장의 뜻이 비슷해.")
