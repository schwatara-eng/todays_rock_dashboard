"""
오늘의 록 분석용 CSV 만들기

입력: source/data_backup.json (933곡), source/todays_rock_song_candidates.csv (국내/해외 구분 보조)
출력: data/ 폴더의 CSV 5개
실행: python build_data.py
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

MOODS = ["calm", "excited", "focused", "blue", "tired", "frustrated"]
WEATHERS = ["clear", "cloudy", "rain", "snow"]
MOOD_KO = {"calm": "평온", "excited": "신남", "focused": "집중",
           "blue": "우울", "tired": "피곤", "frustrated": "답답"}
WEATHER_KO = {"clear": "맑음", "cloudy": "흐림", "rain": "비", "snow": "눈"}


# ── 1. 곡 데이터 읽기 ─────────────────────────────────────
# data_backup.json의 순서가 DB(todays_rock.sql)의 id 순서와 같다.
songs = json.loads((BASE / "source" / "data_backup.json").read_text(encoding="utf-8"))
df = pd.DataFrame(songs)
df.insert(0, "song_id", range(1, len(df) + 1))
df = df.rename(columns={"lyricsUrl": "lyrics_url"})

# 국내/해외: 장르가 Korean으로 시작하거나, 후보 목록에서 '한국'으로 분류된 곡
cand = pd.read_csv(BASE / "source" / "todays_rock_song_candidates.csv", encoding="utf-8-sig")
cand = cand[pd.to_numeric(cand["DB번호"], errors="coerce").notna()]
korean_ids = set(cand.loc[cand["구분"] == "한국", "DB번호"].astype(int))
is_korean = df["genre"].str.startswith("Korean") | df["song_id"].isin(korean_ids)
df["region"] = is_korean.map({True: "국내", False: "해외"})

df["decade"] = (df["year"] // 10 * 10).astype(str) + "년대"

for m in MOODS:
    df[f"mood_{m}"] = df["moods"].apply(lambda xs, m=m: int(m in xs))
for w in WEATHERS:
    df[f"weather_{w}"] = df["weathers"].apply(lambda xs, w=w: int(w in xs))
df["mood_count"] = df["moods"].str.len()
df["weather_count"] = df["weathers"].str.len()

# 리스트는 CSV에서 '|'로 이어 붙인다 (다시 나눌 때: df["moods"].str.split("|"))
df["moods"] = df["moods"].str.join("|")
df["weathers"] = df["weathers"].str.join("|")

cols = (["song_id", "title", "artist", "year", "decade", "region", "genre", "energy",
         "moods", "weathers", "mood_count", "weather_count"]
        + [f"mood_{m}" for m in MOODS] + [f"weather_{w}" for w in WEATHERS]
        + ["theme", "tone", "reason", "lyrics_url"])
df = df[cols]
df.to_csv(OUT / "todays_rock_songs.csv", index=False, encoding="utf-8-sig")


# ── 2. 태그 long 형식 (곡 1개 × 태그 1개 = 1행) ───────────
tags = pd.concat([
    df[["song_id", "moods"]].assign(tag=df["moods"].str.split("|"))
      .explode("tag").assign(tag_type="mood")[["song_id", "tag_type", "tag"]],
    df[["song_id", "weathers"]].assign(tag=df["weathers"].str.split("|"))
      .explode("tag").assign(tag_type="weather")[["song_id", "tag_type", "tag"]],
])
tags["tag_ko"] = tags["tag"].map({**MOOD_KO, **WEATHER_KO})
tags = tags.sort_values(["song_id", "tag_type"]).reset_index(drop=True)
tags.to_csv(OUT / "todays_rock_song_tags.csv", index=False, encoding="utf-8-sig")


# ── 3. 기분 × 날씨 교차표 (개수 · 비율 · Lift) ────────────
total = len(df)
rows = []
for m in MOODS:
    for w in WEATHERS:
        n_m = df[f"mood_{m}"].sum()
        n_w = df[f"weather_{w}"].sum()
        n_both = ((df[f"mood_{m}"] == 1) & (df[f"weather_{w}"] == 1)).sum()
        expected = n_m * n_w / total
        rows.append({
            "mood": m, "mood_ko": MOOD_KO[m],
            "weather": w, "weather_ko": WEATHER_KO[w],
            "songs": int(n_both),
            "pct_of_mood": round(n_both / n_m * 100, 1),      # 이 기분 곡 중 이 날씨 비율
            "pct_of_weather": round(n_both / n_w * 100, 1),   # 이 날씨 곡 중 이 기분 비율
            "expected": round(expected, 1),                    # 우연히 겹칠 때 예상 곡 수
            "lift": round(n_both / expected, 2),               # 1보다 크면 우연보다 자주 붙음
        })
mw = pd.DataFrame(rows)
mw.to_csv(OUT / "todays_rock_mood_weather.csv", index=False, encoding="utf-8-sig")


# ── 4. 연대별 요약 ────────────────────────────────────────
dec = (df.groupby("decade")
         .agg(songs=("song_id", "count"),
              avg_energy=("energy", "mean"),
              median_energy=("energy", "median"),
              korean_songs=("region", lambda s: (s == "국내").sum()),
              artists=("artist", "nunique"),
              genres=("genre", "nunique"))
         .reset_index())
dec["pct_of_all"] = (dec["songs"] / total * 100).round(1)
dec["avg_energy"] = dec["avg_energy"].round(2)
for m in MOODS:
    dec[f"pct_{m}"] = (df.groupby("decade")[f"mood_{m}"].mean().values * 100).round(1)
dec.to_csv(OUT / "todays_rock_decade_summary.csv", index=False, encoding="utf-8-sig")


# ── 5. 팩트 시트 (RAG 근거용) ─────────────────────────────
def pct(n, d):
    return f"{n / d * 100:.1f}%"

facts = []

def add(field, text, number, basis, source):
    facts.append({"사실ID": f"F{len(facts) + 1:02d}", "분야": field, "내용": text,
                  "숫자": number, "기준": basis, "출처": source})

S = "todays_rock_songs.csv"
BASIS_ALL = f"「오늘의 록」 추천 목록 전체 {total}곡. 기분·날씨·에너지는 편집자가 붙인 라벨."

# 전체 규모
add("전체", f"추천 목록에는 {total}곡, {df['artist'].nunique()}명(팀)의 아티스트, "
    f"{df['genre'].nunique()}개 장르가 들어 있다.",
    f"곡 = {total}; 아티스트 = {df['artist'].nunique()}; 장르 = {df['genre'].nunique()}",
    BASIS_ALL, S)

# 국내/해외
n_kr = (df["region"] == "국내").sum()
add("국내·해외", f"국내 곡은 {n_kr}곡({pct(n_kr, total)}), 해외 곡은 {total - n_kr}곡"
    f"({pct(total - n_kr, total)})이다.",
    f"국내 = {n_kr}곡; 해외 = {total - n_kr}곡",
    BASIS_ALL + " 국내는 장르명이 Korean으로 시작하거나 후보 목록에서 한국으로 분류된 곡.", S)

# 연대
top_dec = dec.sort_values("songs", ascending=False)
d1, dl = top_dec.iloc[0], top_dec.iloc[-1]
add("연대", f"{d1['decade']} 곡이 {d1['songs']}곡으로 가장 많고, "
    f"{dl['decade']} 곡이 {dl['songs']}곡으로 가장 적다.",
    "; ".join(f"{r.decade} = {r.songs}곡({r.pct_of_all}%)" for r in dec.itertuples()),
    BASIS_ALL + " 연대는 발매 연도 기준.", "todays_rock_decade_summary.csv")

# 에너지
ec = df["energy"].value_counts().sort_index()
hi = ec.get(4, 0) + ec.get(5, 0)
add("에너지", f"에너지 4~5인 곡이 {hi}곡({pct(hi, total)})으로 절반이 넘는다. "
    f"에너지 1인 곡은 {ec.get(1, 0)}곡이다.",
    "; ".join(f"에너지 {k} = {v}곡" for k, v in ec.items())
    + f"; 평균 = {df['energy'].mean():.2f}",
    BASIS_ALL + " 에너지는 1(잔잔함)~5(강렬함).", S)

# 기분 태그
mc = {m: df[f"mood_{m}"].sum() for m in MOODS}
mtop = max(mc, key=mc.get); mlow = min(mc, key=mc.get)
add("기분", f"기분 태그는 {MOOD_KO[mtop]}({mtop})이 {mc[mtop]}곡으로 가장 많고, "
    f"{MOOD_KO[mlow]}({mlow})이 {mc[mlow]}곡으로 가장 적다.",
    "; ".join(f"{MOOD_KO[m]} = {mc[m]}곡({pct(mc[m], total)})" for m in MOODS),
    BASIS_ALL + " 한 곡에 기분이 여러 개 붙을 수 있어 합계가 곡 수보다 크다.", S)

# 날씨 태그
wc = {w: df[f"weather_{w}"].sum() for w in WEATHERS}
wtop = max(wc, key=wc.get); wlow = min(wc, key=wc.get)
add("날씨", f"날씨 태그는 {WEATHER_KO[wtop]}이 {wc[wtop]}곡으로 가장 많고, "
    f"{WEATHER_KO[wlow]}이 {wc[wlow]}곡으로 가장 적다.",
    "; ".join(f"{WEATHER_KO[w]} = {wc[w]}곡({pct(wc[w], total)})" for w in WEATHERS),
    BASIS_ALL + " 한 곡에 날씨가 여러 개 붙을 수 있다.", S)

# 기분별 평균 에너지
me = {m: df.loc[df[f"mood_{m}"] == 1, "energy"].mean() for m in MOODS}
mhi = max(me, key=me.get); mlo = min(me, key=me.get)
add("기분·에너지", f"{MOOD_KO[mhi]} 곡의 평균 에너지가 {me[mhi]:.2f}로 가장 높고, "
    f"{MOOD_KO[mlo]} 곡이 {me[mlo]:.2f}로 가장 낮다.",
    "; ".join(f"{MOOD_KO[m]} 평균 에너지 = {me[m]:.2f}" for m in MOODS),
    BASIS_ALL + " 각 기분 태그가 붙은 곡의 에너지 평균.", S)

# 기분×날씨 Lift
mw_sorted = mw.sort_values("lift", ascending=False)
top = mw_sorted.iloc[0]; low = mw_sorted.iloc[-1]
add("기분×날씨", f"{top['weather_ko']}×{top['mood_ko']} 조합은 우연히 겹칠 때보다 "
    f"{top['lift']}배 자주 함께 붙어 있다. 반대로 {low['weather_ko']}×{low['mood_ko']} "
    f"조합은 {low['lift']}배에 그친다.",
    f"{top['weather_ko']}×{top['mood_ko']} = {top['songs']}곡(예상 {top['expected']}곡); "
    f"{low['weather_ko']}×{low['mood_ko']} = {low['songs']}곡(예상 {low['expected']}곡)",
    BASIS_ALL + " Lift = 실제 곡 수 ÷ (기분 곡 수 × 날씨 곡 수 ÷ 전체 곡 수).",
    "todays_rock_mood_weather.csv")

# 추천 풀이 가장 얇은 조합
thin = mw.sort_values("songs").iloc[0]
add("기분×날씨", f"추천 후보가 가장 적은 조합은 {thin['weather_ko']}×{thin['mood_ko']}로 "
    f"{thin['songs']}곡이다.",
    f"{thin['weather_ko']}×{thin['mood_ko']} = {thin['songs']}곡",
    BASIS_ALL + " 해당 기분과 날씨 태그가 모두 붙은 곡 수.",
    "todays_rock_mood_weather.csv")

# 장르·아티스트
gc = df["genre"].value_counts()
add("장르", f"가장 많은 장르는 {gc.index[0]}({gc.iloc[0]}곡), 다음은 {gc.index[1]}"
    f"({gc.iloc[1]}곡)이다.",
    "; ".join(f"{g} = {n}곡" for g, n in gc.head(5).items()),
    BASIS_ALL + " 장르는 곡마다 하나.", S)

ac = df["artist"].value_counts()
add("아티스트", f"곡 수가 가장 많은 아티스트는 {ac.index[0]}({ac.iloc[0]}곡)이다.",
    "; ".join(f"{a} = {n}곡" for a, n in ac.head(5).items()),
    BASIS_ALL, S)

# 국내·해외 에너지 비교
re_ = df.groupby("region")["energy"].mean()
add("국내·해외", f"국내 곡 평균 에너지는 {re_['국내']:.2f}, 해외 곡은 {re_['해외']:.2f}이다.",
    f"국내 = {re_['국내']:.2f}; 해외 = {re_['해외']:.2f}",
    BASIS_ALL, S)

# ── 텍스트 지도 팩트 (build_text_map.py 결과가 있을 때만) ──
map_path, topic_path = OUT / "todays_rock_text_map.csv", OUT / "todays_rock_topics.csv"
if map_path.exists() and topic_path.exists():
    tm = df.merge(pd.read_csv(map_path, encoding="utf-8-sig")[["song_id", "topic_id", "topic_name"]],
                  on="song_id")
    tp = pd.read_csv(topic_path, encoding="utf-8-sig")
    T = "todays_rock_topics.csv"
    BASIS_TOPIC = (BASIS_ALL + " 텍스트 주제는 곡 소개 문장(주제·분위기·추천 이유)을 OpenAI 임베딩으로 바꾼 뒤 "
                   "BERTopic(K-means)으로 묶은 결과. 주제 이름은 AI가 짓고 사람이 일부 수정.")

    big, small = tp.sort_values("songs").iloc[-1], tp.sort_values("songs").iloc[0]
    add("텍스트 주제", f"소개 문장을 비슷한 것끼리 묶으면 {len(tp)}개 주제로 나뉜다. "
        f"가장 큰 주제는 '{big['topic_name']}'({big['songs']}곡), 가장 작은 주제는 "
        f"'{small['topic_name']}'({small['songs']}곡)이다.",
        "; ".join(f"{r.topic_name} = {r.songs}곡" for r in tp.sort_values("songs", ascending=False).itertuples()),
        BASIS_TOPIC, T)

    # 주제별 기분 라벨 비율
    share = tm.groupby("topic_name")[[f"mood_{m}" for m in MOODS]].mean() * 100
    top_mood = share.idxmax(axis=1).str.replace("mood_", "")
    top_pct = share.max(axis=1)
    hi = top_pct.sort_values(ascending=False).head(2)
    add("텍스트 주제×기분", "소개 문장으로 묶은 주제 중 일부는 편집자 기분 라벨과 거의 일치한다. "
        + ", ".join(f"'{t}' 주제는 {top_pct[t]:.1f}%가 {MOOD_KO[top_mood[t]]} 라벨" for t in hi.index) + "이다.",
        "; ".join(f"{t}: {MOOD_KO[top_mood[t]]} {top_pct[t]:.1f}%" for t in hi.index),
        BASIS_TOPIC + " 각 주제 곡 중 해당 기분 라벨이 붙은 곡의 비율.", T)

    lo = top_pct.idxmin()
    row = share.loc[lo].sort_values(ascending=False)
    add("텍스트 주제×기분", f"'{lo}' 주제는 기분 라벨이 가장 흩어져 있다. 가장 많은 라벨도 "
        f"{row.iloc[0]:.1f}%({MOOD_KO[row.index[0].replace('mood_', '')]})에 그친다.",
        "; ".join(f"{MOOD_KO[c.replace('mood_', '')]} = {v:.1f}%" for c, v in row.items()),
        BASIS_TOPIC + " 문장은 비슷하지만 라벨이 제각각이라 라벨 재검토 후보.", T)

    te = tm.groupby("topic_name")["energy"].mean().round(2).sort_values()
    tops = te[te == te.max()].index.tolist()          # 공동 1위가 있을 수 있다
    lows = te[te == te.min()].index.tolist()
    names = lambda ts: ", ".join(f"'{t}'" for t in ts)
    top_note = " (모든 곡이 에너지 5)" if te.max() == 5 else ""
    add("텍스트 주제×에너지", f"평균 에너지가 가장 높은 주제는 {names(tops)}({te.max():.2f}){top_note}, "
        f"가장 낮은 주제는 {names(lows)}({te.min():.2f})이다.",
        "; ".join(f"{t} = {v:.2f}" for t, v in te.sort_values(ascending=False).items()),
        BASIS_TOPIC, T)

    # 곡이 추가된 시기(곡 번호 1~500 / 501 이후)에 따른 쏠림
    tm["batch"] = (tm["song_id"] > 500).map({False: "1~500번", True: "501번 이후"})
    late_all = (tm["batch"] == "501번 이후").mean() * 100
    late = tm.groupby("topic_name")["batch"].apply(lambda b: (b == "501번 이후").mean() * 100).sort_values()
    add("텍스트 주제×추가 시기", f"전체 곡 중 501번 이후 곡은 {late_all:.1f}%지만, 주제별로는 "
        f"'{late.index[0]}' {late.iloc[0]:.0f}%에서 '{late.index[-1]}' {late.iloc[-1]:.0f}%까지 갈린다. "
        "두 시기에 소개 문장을 쓴 방식이 달라 주제 묶기에 영향을 준 것으로 보인다.",
        "; ".join(f"{t}: 501번 이후 {v:.0f}%" for t, v in late.items()),
        BASIS_TOPIC + " 곡 번호 501번 이후는 후보 목록(todays_rock_song_candidates.csv)으로 추가한 곡. "
        "문체 차이는 추정이며 인과관계는 확인되지 않음.", T)

pd.DataFrame(facts).to_csv(OUT / "todays_rock_facts.csv", index=False, encoding="utf-8-sig")

print("완료:", ", ".join(p.name for p in sorted(OUT.glob("*.csv"))))
