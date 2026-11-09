from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="오늘의 록 분석 대시보드", page_icon="🎸", layout="wide")

app_folder = Path(__file__).resolve().parent
data_folder = app_folder / "data"

MOODS = ["calm", "excited", "focused", "blue", "tired", "frustrated"]
WEATHERS = ["clear", "cloudy", "rain", "snow"]
MOOD_KO = {"calm": "평온", "excited": "신남", "focused": "집중",
           "blue": "우울", "tired": "피곤", "frustrated": "답답"}
WEATHER_KO = {"clear": "맑음", "cloudy": "흐림", "rain": "비", "snow": "눈"}
MOOD_ORDER = [MOOD_KO[m] for m in MOODS]
WEATHER_ORDER = [WEATHER_KO[w] for w in WEATHERS]


@st.cache_data
def load_data():
    songs = pd.read_csv(data_folder / "todays_rock_songs.csv")
    mood_weather = pd.read_csv(data_folder / "todays_rock_mood_weather.csv")
    return songs, mood_weather


songs, mood_weather = load_data()
total = len(songs)

st.title("🎸 오늘의 록 분석 대시보드")
st.caption(
    f"「오늘의 록」 추천 목록 {total}곡 기준 · 기분·날씨·에너지는 편집자가 붙인 라벨이라 "
    "음악 자체의 특성이 아니라 이 사이트의 큐레이션 성향을 보여줘."
)

# 첫째 줄: 전체 규모
korean_count = int((songs["region"] == "국내").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("곡", f"{total:,}곡")
c2.metric("아티스트", f"{songs['artist'].nunique():,}명(팀)")
c3.metric("장르", f"{songs['genre'].nunique()}개")
c4.metric("국내 곡 비율", f"{korean_count / total * 100:.1f}%", help=f"국내 {korean_count}곡")

st.divider()

# 둘째 줄: 연대별 분포 (절대값 vs 비율)
st.header("어느 시대의 록이 많을까?")

decade_col, decade_note = st.columns([3, 1])

with decade_note:
    region = st.radio("범위", ["전체", "국내", "해외"], key="decade_region")
    measure = st.radio("기준", ["곡 수", "비율 (%)"], key="decade_measure")

with decade_col:
    target = songs if region == "전체" else songs.loc[songs["region"] == region]
    decade = target.groupby("decade").size().rename("songs").reset_index()
    decade["pct"] = decade["songs"] / decade["songs"].sum() * 100
    y_field = "songs" if measure == "곡 수" else "pct"
    y_title = "곡 수" if measure == "곡 수" else "비율 (%)"

    chart = (
        alt.Chart(decade)
        .mark_bar()
        .encode(
            x=alt.X("decade:N", title="연대", sort=None),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("decade:N", title="연대"),
                alt.Tooltip("songs:Q", title="곡 수"),
                alt.Tooltip("pct:Q", title="비율 (%)", format=".1f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"{region} {len(target)}곡 기준 · 국내와 해외는 곡 수 차이가 커서 "
        "둘을 비교할 땐 비율로 보는 게 공정해."
    )

st.divider()

# 셋째 줄: 기분 × 날씨 히트맵
st.header("날씨와 기분은 어떻게 짝지어져 있을까?")

basis_labels = {
    "songs": "곡 수",
    "pct_of_weather": "날씨 기준 비율 (%)",
    "pct_of_mood": "기분 기준 비율 (%)",
    "lift": "Lift (우연 대비 배수)",
}
basis_help = {
    "songs": "두 태그가 모두 붙은 곡 수야. 앱에서 그 조합으로 추천할 수 있는 후보 수와 같아.",
    "pct_of_weather": "그 날씨 곡 중 이 기분이 붙은 비율이야. 가로 한 줄을 더하면 100%를 넘을 수 있어 (곡마다 기분이 여러 개라서).",
    "pct_of_mood": "그 기분 곡 중 이 날씨가 붙은 비율이야.",
    "lift": "1보다 크면 우연히 겹칠 때보다 자주 함께 붙어 있다는 뜻이야. 곡 수가 많은 태그의 영향을 걷어낸 값이야.",
}

heat_col, heat_note = st.columns([3, 1])

with heat_note:
    basis = st.radio(
        "색으로 보여줄 값",
        list(basis_labels),
        format_func=basis_labels.get,
        key="heat_basis",
    )
    st.caption(basis_help[basis])

with heat_col:
    is_lift = basis == "lift"
    color_scale = (
        alt.Scale(scheme="redblue", domainMid=1, reverse=True)
        if is_lift
        else alt.Scale(scheme="oranges")
    )
    text_format = ".2f" if is_lift else (".0f" if basis == "songs" else ".1f")

    base = alt.Chart(mood_weather).encode(
        x=alt.X("mood_ko:N", title="기분", sort=MOOD_ORDER),
        y=alt.Y("weather_ko:N", title="날씨", sort=WEATHER_ORDER),
    )
    rects = base.mark_rect().encode(
        color=alt.Color(f"{basis}:Q", title=basis_labels[basis], scale=color_scale),
        tooltip=[
            alt.Tooltip("weather_ko:N", title="날씨"),
            alt.Tooltip("mood_ko:N", title="기분"),
            alt.Tooltip("songs:Q", title="곡 수"),
            alt.Tooltip("expected:Q", title="우연일 때 예상 곡 수"),
            alt.Tooltip("lift:Q", title="Lift"),
        ],
    )
    labels = base.mark_text(fontSize=14).encode(
        text=alt.Text(f"{basis}:Q", format=text_format),
    )
    st.altair_chart((rects + labels).properties(height=300), use_container_width=True)

with st.expander("추천 후보가 가장 적은 조합은?"):
    thin = mood_weather.sort_values("songs").head(3)
    for row in thin.itertuples():
        st.write(
            f"**{row.weather_ko} × {row.mood_ko}** · {row.songs}곡 "
            f"(우연일 때 예상 {row.expected}곡, Lift {row.lift})"
        )
    st.info("이 조합으로 접속한 사용자는 같은 곡을 자주 추천받게 돼. 곡을 보강할 후보 칸이야.")

st.divider()

# 넷째 줄: 기분별 에너지 (평균 vs 분포)
st.header("기분 태그별 에너지는 얼마나 다를까?")

energy_col, energy_note = st.columns([3, 1])

mood_energy = (
    songs.melt(
        id_vars=["song_id", "energy", "region"],
        value_vars=[f"mood_{m}" for m in MOODS],
        var_name="mood",
        value_name="has_mood",
    )
    .query("has_mood == 1")
    .assign(mood_ko=lambda df: df["mood"].str.replace("mood_", "").map(MOOD_KO))
)

with energy_note:
    view = st.radio("보는 방법", ["평균", "분포 (100% 누적)"], key="energy_view")
    st.caption(
        "평균은 한 숫자로 요약하지만 퍼진 정도를 숨겨. "
        "분포로 보면 에너지 1~5가 각각 몇 %인지 보여."
    )

with energy_col:
    if view == "평균":
        avg = mood_energy.groupby("mood_ko")["energy"].agg(["mean", "count"]).reset_index()
        chart = (
            alt.Chart(avg)
            .mark_bar()
            .encode(
                x=alt.X("mood_ko:N", title="기분", sort=MOOD_ORDER),
                y=alt.Y("mean:Q", title="평균 에너지", scale=alt.Scale(domain=[0, 5])),
                tooltip=[
                    alt.Tooltip("mood_ko:N", title="기분"),
                    alt.Tooltip("mean:Q", title="평균 에너지", format=".2f"),
                    alt.Tooltip("count:Q", title="곡 수"),
                ],
            )
        )
    else:
        dist = mood_energy.groupby(["mood_ko", "energy"]).size().rename("songs").reset_index()
        chart = (
            alt.Chart(dist)
            .mark_bar()
            .encode(
                x=alt.X("mood_ko:N", title="기분", sort=MOOD_ORDER),
                y=alt.Y("songs:Q", stack="normalize", title="비율", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "energy:O",
                    title="에너지",
                    scale=alt.Scale(scheme="oranges"),
                    sort="ascending",
                ),
                order=alt.Order("energy:O", sort="ascending"),
                tooltip=[
                    alt.Tooltip("mood_ko:N", title="기분"),
                    alt.Tooltip("energy:O", title="에너지"),
                    alt.Tooltip("songs:Q", title="곡 수"),
                ],
            )
        )
    st.altair_chart(chart.properties(height=320), use_container_width=True)
    st.caption("곡마다 기분이 여러 개 붙을 수 있어서, 한 곡이 여러 막대에 함께 들어가.")

st.divider()

# 다섯째 줄: 국내 vs 해외
st.header("국내 록과 해외 록은 어떻게 다를까?")

region_energy_col, region_mood_col = st.columns(2)

with region_energy_col:
    st.subheader("에너지 분포")
    re_dist = songs.groupby(["region", "energy"]).size().rename("songs").reset_index()
    re_dist["pct"] = re_dist["songs"] / re_dist.groupby("region")["songs"].transform("sum") * 100
    chart = (
        alt.Chart(re_dist)
        .mark_bar()
        .encode(
            x=alt.X("energy:O", title="에너지"),
            xOffset="region:N",
            y=alt.Y("pct:Q", title="비율 (%)"),
            color=alt.Color("region:N", title="구분"),
            tooltip=[
                alt.Tooltip("region:N", title="구분"),
                alt.Tooltip("energy:O", title="에너지"),
                alt.Tooltip("songs:Q", title="곡 수"),
                alt.Tooltip("pct:Q", title="비율 (%)", format=".1f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    region_avg = songs.groupby("region")["energy"].mean()
    st.caption(
        f"평균 에너지 · 국내 {region_avg.get('국내', 0):.2f} · 해외 {region_avg.get('해외', 0):.2f}"
    )

with region_mood_col:
    st.subheader("기분 태그 비율")
    rm = (
        songs.groupby("region")[[f"mood_{m}" for m in MOODS]].mean().mul(100)
        .reset_index()
        .melt(id_vars="region", var_name="mood", value_name="pct")
    )
    rm["mood_ko"] = rm["mood"].str.replace("mood_", "").map(MOOD_KO)
    chart = (
        alt.Chart(rm)
        .mark_bar()
        .encode(
            x=alt.X("mood_ko:N", title="기분", sort=MOOD_ORDER),
            xOffset="region:N",
            y=alt.Y("pct:Q", title="그 기분이 붙은 곡 비율 (%)"),
            color=alt.Color("region:N", title="구분"),
            tooltip=[
                alt.Tooltip("region:N", title="구분"),
                alt.Tooltip("mood_ko:N", title="기분"),
                alt.Tooltip("pct:Q", title="비율 (%)", format=".1f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"국내 {korean_count}곡 · 해외 {total - korean_count}곡 각각을 100으로 놓은 비율")

st.divider()

# 여섯째 줄: 순위 비교
st.header("곡 수 1위 장르와 에너지 1위 장르는 같을까?")

min_songs = st.slider("최소 곡 수 (이보다 적은 장르는 제외)", 5, 30, 10)

genres = (
    songs.groupby("genre")
    .agg(songs=("song_id", "count"), avg_energy=("energy", "mean"))
    .reset_index()
    .query("songs >= @min_songs")
)
genres["평균 에너지"] = genres["avg_energy"].round(2)
genres = genres.rename(columns={"genre": "장르", "songs": "곡 수"})
display_columns = ["장르", "곡 수", "평균 에너지"]

count_col, energy_rank_col = st.columns(2)

with count_col:
    st.subheader("곡 수 순위")
    st.dataframe(genres.sort_values("곡 수", ascending=False)[display_columns], hide_index=True)

with energy_rank_col:
    st.subheader("평균 에너지 순위")
    st.dataframe(genres.sort_values("avg_energy", ascending=False)[display_columns], hide_index=True)

st.caption(f"곡 수 {min_songs}곡 이상인 장르 {len(genres)}개 기준")

with st.expander("답 확인하기"):
    if genres.empty:
        st.warning("조건에 맞는 장르가 없어. 최소 곡 수를 낮춰 봐.")
    else:
        count_leader = genres.loc[genres["곡 수"].idxmax(), "장르"]
        energy_leader = genres.loc[genres["avg_energy"].idxmax(), "장르"]
        st.write(f"**곡 수 1위** · {count_leader}")
        st.write(f"**평균 에너지 1위** · {energy_leader}")
        if count_leader == energy_leader:
            st.success("두 기준에서 1위인 장르가 같아.")
        else:
            st.info("두 기준에서 1위인 장르가 달라. 많이 담긴 장르가 가장 강렬한 장르는 아니야.")

st.divider()

# 전체 너비: 곡 찾기
st.header("조건으로 곡 찾기")

f1, f2, f3, f4 = st.columns(4)
pick_mood = f1.selectbox("기분", ["전체"] + MOODS, format_func=lambda v: MOOD_KO.get(v, v))
pick_weather = f2.selectbox("날씨", ["전체"] + WEATHERS, format_func=lambda v: WEATHER_KO.get(v, v))
pick_region = f3.selectbox("구분", ["전체", "국내", "해외"])
pick_energy = f4.slider("에너지", 1, 5, (1, 5))

found = songs.loc[songs["energy"].between(*pick_energy)]
if pick_mood != "전체":
    found = found.loc[found[f"mood_{pick_mood}"] == 1]
if pick_weather != "전체":
    found = found.loc[found[f"weather_{pick_weather}"] == 1]
if pick_region != "전체":
    found = found.loc[found["region"] == pick_region]

st.caption(f"조건에 맞는 곡 {len(found)}곡")
st.dataframe(
    found[["title", "artist", "year", "genre", "energy", "theme", "lyrics_url"]],
    hide_index=True,
    column_config={
        "title": "곡명",
        "artist": "아티스트",
        "year": st.column_config.NumberColumn("연도", format="%d"),
        "genre": "장르",
        "energy": st.column_config.ProgressColumn("에너지", min_value=0, max_value=5, format="%d"),
        "theme": "주제",
        "lyrics_url": st.column_config.LinkColumn("가사", display_text="열기"),
    },
)
