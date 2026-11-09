"""
오늘의 록 텍스트 지도 만들기 (임베딩 + UMAP + BERTopic)

곡마다 붙은 소개 문장(theme · tone · reason)을 임베딩으로 바꾸고,
- UMAP으로 2차원 지도 좌표를 만들고
- BERTopic으로 비슷한 문장끼리 주제를 묶고
- OpenAI로 주제마다 한국어 이름을 붙이고
- 곡마다 문장이 가장 비슷한 이웃 곡 5개를 찾는다.

결과는 data/ 폴더의 CSV 2개로 저장한다. Streamlit 페이지는 이 CSV만 읽는다.
  - data/todays_rock_text_map.csv : 곡별 좌표 · 주제 · 이웃 곡
  - data/todays_rock_topics.csv   : 주제별 이름 · 키워드 · 곡 수 · 대표 문장

실행 (로컬에서 한 번만):
  pip install -r requirements-analysis.txt
  python build_text_map.py

API 키는 환경변수 OPENAI_API_KEY 또는 .streamlit/secrets.toml에서 읽는다.
임베딩은 data/todays_rock_embeddings.npy에 저장해 두고, 다음 실행부터는 다시 만들지 않는다.
(문장을 고쳤다면 이 .npy 파일을 지우고 다시 실행)
"""

import os
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from utils.todays_rock_rag import CHAT_MODEL, EMBEDDING_MODEL

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
EMBEDDING_CACHE = DATA / "todays_rock_embeddings.npy"

# ── 조절할 수 있는 값 ─────────────────────────────────────
CLUSTER_METHOD = "kmeans"  # "kmeans": 주제 수를 직접 정함 / "hdbscan": 덩어리를 스스로 찾음
N_TOPICS = 12           # kmeans일 때 주제 수
MIN_TOPIC_SIZE = 15     # hdbscan일 때 주제 하나에 들어갈 최소 곡 수
NEIGHBORS = 5           # 곡마다 저장할 비슷한 곡 수
RANDOM_STATE = 42       # 같은 결과를 다시 얻기 위한 고정값
NAME_TOPICS = True      # False면 주제 이름 대신 키워드만 쓴다 (API 비용 없음)


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    secrets = BASE / ".streamlit" / "secrets.toml"
    if not key and secrets.exists():
        key = tomllib.loads(secrets.read_text(encoding="utf-8")).get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY가 없어. 환경변수나 .streamlit/secrets.toml에 넣어 줘.")
    return key


def song_text(row) -> str:
    # 곡명·아티스트는 넣지 않는다. 이름이 아니라 '소개 문장'끼리 비슷한지 보려는 것.
    return f"주제: {row.theme}\n분위기: {row.tone}\n추천 이유: {row.reason}"


# ── 한국어 키워드 추출 ────────────────────────────────────
# BERTopic은 주제마다 대표 단어를 뽑을 때 문장을 단어로 쪼갠다.
# 한국어는 '관계의', '관계를'처럼 조사가 붙어 있어서 형태소 분석기로 명사·어근만 남긴다.
STOPWORDS = {"분위기", "느낌", "곡", "노래", "순간", "마음", "하루", "기분", "주제", "추천", "이유", "것", "수", "때"}
# 악기·연주 단어는 주제 이름보다 연주 설명에 가까워서 키워드에서 뺀다 (주제를 나누는 계산에는 영향 없음)
STOPWORDS |= {"기타", "보컬", "리듬", "드럼", "베이스", "사운드", "멜로디", "후렴", "목소리",
              "비트", "리프", "소리", "연주", "편곡", "신스", "화음"}

try:
    from kiwipiepy import Kiwi

    _kiwi = Kiwi()

    def korean_tokenizer(text: str) -> list[str]:
        words = [t.form for t in _kiwi.tokenize(text) if t.tag in ("NNG", "NNP", "XR")]
        return [w for w in words if len(w) > 1 and w not in STOPWORDS]

except ImportError:
    print("kiwipiepy가 없어 띄어쓰기 기준으로 단어를 나눠. (키워드 품질이 떨어질 수 있어)")
    _PARTICLES = re.compile(r"(으로|에서|에게|처럼|보다|까지|부터|의|을|를|이|가|은|는|에|와|과|도|로)$")

    def korean_tokenizer(text: str) -> list[str]:
        words = [_PARTICLES.sub("", w) for w in re.findall(r"[가-힣A-Za-z]+", text)]
        return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    if EMBEDDING_CACHE.exists():
        cached = np.load(EMBEDDING_CACHE)
        if len(cached) == len(texts):
            print(f"저장된 임베딩을 다시 써: {EMBEDDING_CACHE.name}")
            return cached
    print(f"임베딩 만드는 중... ({len(texts)}곡, {EMBEDDING_MODEL})")
    vectors = []
    for start in range(0, len(texts), 500):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[start:start + 500])
        vectors.extend(item.embedding for item in response.data)
    array = np.array(vectors, dtype=np.float32)
    np.save(EMBEDDING_CACHE, array)
    return array


def name_topic(client: OpenAI, keywords: list[str], examples: list[str]) -> str:
    prompt = (
        "아래는 록 음악 소개 문장을 비슷한 것끼리 묶은 한 그룹이야.\n"
        f"대표 키워드: {', '.join(keywords)}\n"
        "대표 문장:\n" + "\n".join(f"- {e}" for e in examples) + "\n\n"
        "이 그룹의 감정이나 주제를 2~6단어의 한국어 이름으로 지어 줘. "
        "따옴표나 설명 없이 이름만 답해."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip().strip('"「」')


def main():
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    songs = pd.read_csv(DATA / "todays_rock_songs.csv")
    docs = [song_text(row) for row in songs.itertuples()]

    client = OpenAI(api_key=get_api_key())
    embeddings = embed(client, docs)

    # 1) 지도용 2차원 좌표
    print("UMAP으로 2차원 지도 만드는 중...")
    coords = UMAP(
        n_neighbors=15, n_components=2, min_dist=0.1,
        metric="cosine", random_state=RANDOM_STATE,
    ).fit_transform(embeddings)

    # 2) 주제 묶기: BERTopic 안에서는 5차원으로 줄인 뒤 군집
    # 소개 문장끼리 고르게 비슷해서 HDBSCAN은 덩어리를 2~3개밖에 못 찾는다.
    # 그래서 기본은 주제 수를 직접 정하는 K-means.
    if CLUSTER_METHOD == "hdbscan":
        from hdbscan import HDBSCAN
        cluster_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True)
        print(f"BERTopic으로 주제 묶는 중... (HDBSCAN, 최소 {MIN_TOPIC_SIZE}곡)")
    else:
        cluster_model = KMeans(n_clusters=N_TOPICS, random_state=RANDOM_STATE, n_init=10)
        print(f"BERTopic으로 주제 묶는 중... (K-means, 주제 {N_TOPICS}개)")

    # min_df는 쓰지 않는다. BERTopic은 주제별로 문장을 하나로 합친 뒤 단어를 세서,
    # min_df=2면 '주제 2개 이상에 나온 단어'만 남아 주제가 적을 때 단어가 전부 사라진다.
    vectorizer = CountVectorizer(tokenizer=korean_tokenizer, token_pattern=None)
    topic_model = BERTopic(
        # 기본값 "english"는 단어를 세기 전에 영어·숫자 외 글자를 전부 지운다.
        # 한국어 문장이 통째로 사라지지 않도록 multilingual로 둔다.
        # (임베딩은 직접 넘기므로 다국어 모델을 따로 내려받지 않는다)
        language="multilingual",
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=RANDOM_STATE),
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        # 모든 주제에 고르게 나오는 단어(기타, 리듬 등)의 가중치를 낮춘다
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=8,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    # 어느 주제에도 못 들어간 곡(-1)은 임베딩이 가장 가까운 주제로 보낸다.
    outliers = sum(1 for t in topics if t == -1)
    if outliers:
        print(f"주제 밖 곡 {outliers}개를 가까운 주제로 재배치...")
        topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer)

    topics = np.array(topics)
    topic_ids = sorted(t for t in set(topics) if t != -1)

    # 3) 주제 정보
    rows = []
    for t in topic_ids:
        keywords = [w for w, _ in (topic_model.get_topic(t) or [])][:8]
        member_idx = np.where(topics == t)[0]
        # 대표 문장: 주제 중심(평균 벡터)에 가장 가까운 곡 3개
        center = embeddings[member_idx].mean(axis=0)
        sims = embeddings[member_idx] @ center / (
            np.linalg.norm(embeddings[member_idx], axis=1) * np.linalg.norm(center))
        rep_idx = member_idx[np.argsort(-sims)[:3]]
        examples = [songs.loc[i, "theme"] for i in rep_idx]
        rows.append({
            "topic_id": int(t),
            "keywords": ", ".join(keywords),
            "songs": int(len(member_idx)),
            "representative_songs": " | ".join(
                f"{songs.loc[i, 'title']} - {songs.loc[i, 'artist']}" for i in rep_idx),
            "representative_themes": " | ".join(examples),
            "_kw": keywords, "_ex": examples,
        })

    if NAME_TOPICS:
        print(f"주제 {len(rows)}개에 이름 붙이는 중... ({CHAT_MODEL})")
    used_names = {}
    for row in rows:
        name = (name_topic(client, row["_kw"], row["_ex"])
                if NAME_TOPICS else " · ".join(row["_kw"][:3])) or f"주제 {row['topic_id']}"
        # 이름이 겹치면 페이지에서 두 주제가 하나로 합쳐지니 번호를 붙인다
        used_names[name] = used_names.get(name, 0) + 1
        row["topic_name"] = name if used_names[name] == 1 else f"{name} ({used_names[name]})"

    topic_table = (pd.DataFrame(rows)
                   .drop(columns=["_kw", "_ex"])
                   .sort_values("songs", ascending=False))
    topic_table = topic_table[["topic_id", "topic_name", "songs", "keywords",
                               "representative_songs", "representative_themes"]]
    topic_table.to_csv(DATA / "todays_rock_topics.csv", index=False, encoding="utf-8-sig")

    # 4) 곡마다 문장이 가장 비슷한 이웃 곡
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -1)
    nearest = np.argsort(-similarity, axis=1)[:, :NEIGHBORS]

    name_by_id = dict(zip(topic_table["topic_id"], topic_table["topic_name"]))
    text_map = pd.DataFrame({
        "song_id": songs["song_id"],
        "x": coords[:, 0].round(4),
        "y": coords[:, 1].round(4),
        "topic_id": topics,
        "topic_name": [name_by_id.get(t, "기타") for t in topics],
        "neighbors": ["|".join(str(songs.loc[j, "song_id"]) for j in row) for row in nearest],
        "neighbor_scores": ["|".join(f"{similarity[i, j]:.3f}" for j in row)
                            for i, row in enumerate(nearest)],
    })
    text_map.to_csv(DATA / "todays_rock_text_map.csv", index=False, encoding="utf-8-sig")

    print(f"완료: 주제 {len(topic_ids)}개")
    print(topic_table[["topic_id", "topic_name", "songs", "keywords"]].to_string(index=False))


if __name__ == "__main__":
    main()
