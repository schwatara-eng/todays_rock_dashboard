from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

LABEL_GUIDE = {
    "id": "G01",
    "title": "라벨 데이터 안내",
    "text": (
        "「오늘의 록」의 기분(calm, excited, focused, blue, tired, frustrated), "
        "날씨(clear, cloudy, rain, snow), 에너지(1~5)는 편집자가 곡마다 붙인 라벨이다. "
        "한 곡에 기분과 날씨가 여러 개 붙을 수 있어 태그별 곡 수를 더하면 전체 곡 수보다 크다. "
        "이 값들은 음악의 객관적 특성이 아니라 사이트의 큐레이션 성향을 보여준다."
    ),
    "kind": "guide",
}

INTERPRETATION_GUIDE = {
    "id": "R01",
    "title": "분석 결과 해석 기준",
    "text": (
        "이 앱은 추천 목록 데이터의 관찰 결과를 설명한다. 집단 간 차이만으로 원인을 "
        "단정하지 않는다. 두 비율의 차이는 퍼센트(%)가 아니라 퍼센트포인트(%p)로 표시한다. "
        "Lift는 실제 곡 수를 우연히 겹칠 때의 예상 곡 수로 나눈 값으로, 1보다 크면 "
        "우연보다 자주 함께 붙어 있다는 뜻이다. 자료에 근거가 없는 질문에는 확인할 수 "
        "없다고 답한다."
    ),
    "kind": "guide",
}


def load_fact_documents(path: str | Path) -> list[dict[str, str]]:
    """todays_rock_facts.csv를 RAG 문서 목록으로 읽는다."""
    table = pd.read_csv(path, keep_default_na=False)
    required_columns = {"사실ID", "분야", "내용", "숫자", "기준", "출처"}
    missing = required_columns.difference(table.columns)
    if missing:
        raise ValueError(f"사실표에 필요한 열이 없어: {', '.join(sorted(missing))}")

    documents = []
    for row in table.to_dict(orient="records"):
        content = str(row["내용"]).strip()
        documents.append({
            "id": str(row["사실ID"]).strip(),
            "title": content,
            "text": (
                f"분야: {row['분야']} 내용: {content} 숫자: {row['숫자']} "
                f"기준: {row['기준']} 출처: {row['출처']}"
            ),
            "kind": "fact",
            "category": str(row["분야"]).strip(),
        })
    return documents


def documents_for_channel(
    facts: list[dict[str, str]], channel: str
) -> list[dict[str, str]]:
    documents = list(facts)
    if channel in {"B", "C"}:
        documents.append(LABEL_GUIDE)
    if channel == "C":
        documents.append(INTERPRETATION_GUIDE)
    return documents


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def answer_question(
    client: OpenAI,
    question: str,
    retrieved: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    evidence = "\n\n".join(
        f"[{item['id']}] {item['title']}\n{item['text']}" for item in retrieved
    )
    system_prompt = (
        "너는 「오늘의 록」 추천 목록 데이터 분석 Q&A 도우미야. 답은 아래 검색된 근거 안에서만 해. "
        "근거에 필요한 정보가 없으면 '제공된 자료로는 확인할 수 없습니다.'라고 말해. "
        "수치 질문에는 분자와 분모 또는 계산 방법을 밝혀. 비율 차이는 퍼센트포인트(%p)로 표현해. "
        "기분·날씨·에너지는 편집자가 붙인 라벨이라는 점을 혼동하지 말고, "
        "근거 ID를 문장 끝에 [F01]처럼 붙여. 검색 근거에 없는 ID를 만들지 마. "
        "인과관계는 단정하지 말고 한국어로 간결하게 답해.\n\n"
        f"검색된 근거:\n{evidence}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=messages,
    )
    return response.choices[0].message.content or "답변을 만들지 못했어."


def cache_key_for_documents(documents: list[dict[str, str]]) -> str:
    return json.dumps(documents, ensure_ascii=False, sort_keys=True)
