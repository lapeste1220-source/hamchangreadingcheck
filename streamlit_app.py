# -*- coding: utf-8 -*-
import os
import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ---------------- 기본 설정 ----------------
st.set_page_config(
    page_title="함창고 글 타당성 검사",
    layout="wide"
)

st.title("함창고 박호종 선생님과 함께하는 글의 타당성 검사")
st.caption("고2 비판적 독해 수업용 · 생성형 AI를 활용한 주장–근거 타당성 점검 도구")

TODAY_STR = datetime.date.today().isoformat()

# ---- 모델 설정: 1단계(분석)는 상위 모델, 3단계(완성 글)는 가성비 모델 ----
ANALYSIS_MODEL = "gpt-4o"        # 타당성 분석용 (깊은 논리)
FINAL_MODEL = "gpt-4o-mini"      # 완성 글 작성용 (가성비)

# 교사용 비밀번호 (선생님이 코드에서 언제든 변경 가능)
ADMIN_PASSWORD = "hamchang123"

# 세션당 최대 API 호출 횟수
MAX_CALLS = 3

# 학번 사용 기록 파일 (이미 제출된 학번 관리용)
USED_IDS_FILE = Path("used_ids.txt")

# 반별 최대 번호 (2학년 1~4반)
CLASS_MAX = {1: 23, 2: 24, 3: 22, 4: 22}  # 2-1,2-2,2-3,2-4


# ---------------- 학번 관련 유틸 ----------------
def build_student_code(class_no: int, number: int) -> str:
    """
    2학년 / 반 / 번호를 받아서 '2111' 형식의 학번 코드 생성
    예: 2학년 1반 11번 -> 2111 (2 + 1 + 11)
    """
    return f"2{class_no}{number:02d}"


def load_used_ids() -> set[str]:
    """이미 제출된 학번 코드 집합을 파일에서 읽어온다."""
    if USED_IDS_FILE.exists():
        with USED_IDS_FILE.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_used_id(student_code: str):
    """새로 제출된 학번 코드를 파일에 추가한다."""
    with USED_IDS_FILE.open("a", encoding="utf-8") as f:
        f.write(student_code + "\n")


# ---------------- OpenAI 관련 함수 ----------------
def get_api_key(user_input_key: str | None) -> str:
    """
    API 키 선택 규칙
    1순위: 학생이 입력한 API 키 (user_input_key)
    2순위: 교사용 비밀번호를 맞춘 세션일 때, 서버에 저장된 OPENAI_API_KEY
    둘 다 없으면 에러 발생.
    """
    # 1) 학생 개인 키가 있는 경우 → 그 키 사용
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()

    # 2) 학생 키는 없지만, 교사용 비밀번호를 맞춘 세션인 경우 → 선생님 키 사용
    if st.session_state.get("is_admin", False):
        env_key = os.getenv("OPENAI_API_KEY")
        if not env_key:
            try:
                env_key = st.secrets.get("OPENAI_API_KEY", None)
            except Exception:
                env_key = None

        if env_key:
            return env_key
        else:
            raise ValueError(
                "교사용 비밀번호는 맞았지만, 서버에 OPENAI_API_KEY 시크릿이 설정되어 있지 않습니다."
            )

    # 3) 둘 다 아니면 → 사용 불가
    raise ValueError(
        "OpenAI API 키가 필요합니다.\n"
        "- 학생: 개인 OpenAI API 키를 입력하세요.\n"
        "- 교사: 교사용 비밀번호를 입력해 학교 공용 키를 사용할 수 있습니다."
    )


def call_openai_text(
    model: str,
    instructions: str,
    user_input: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 1400,
) -> str:
    """
    OpenAI Chat Completions API를 사용해 텍스트를 반환.
    - temperature를 낮게(0.2 전후) 설정해 논리 일관성·채점 엄격성을 강화.
    """
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_input},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI 호출 중 오류가 발생했습니다: {e}")

    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return str(resp)


def can_call_api() -> bool:
    """남은 호출 가능 횟수가 있는지 확인하고, 없으면 경고를 띄운다."""
    used = st.session_state.get("usage_count", 0)
    if used >= MAX_CALLS:
        st.warning(f"이 세션에서 사용할 수 있는 최대 호출 횟수({MAX_CALLS}회)를 초과했습니다.")
        return False
    return True


def increase_api_count():
    """API 호출 횟수 1회 증가."""
    st.session_state["usage_count"] = st.session_state.get("usage_count", 0) + 1


# ---------------- 타당성 평가용 시스템 프롬프트 (논리 강도 강화 버전) ----------------
ANALYSIS_INSTRUCTIONS = f"""
당신은 고등학교 2학년 학생을 돕는 '비판적 독해·타당성 평가 전문 조교'입니다.
목표는 **주장–근거–검증–점수**를 구조적으로, 빠짐없이, 일관되게 분석하는 것입니다.

# 역할
- 학생이 올린 지문에 대해,
  1) 타당성 검사가 꼭 필요한 주장/부분을 찾고,
  2) 각 부분에 대해 검사·검증 결과를 논리적으로 서술하며,
  3) 5점 척도로 일관되게 평가합니다.
- 답변은 "친절하지만 단단한 논리 선생님" 느낌으로, 감상보다 분석에 집중합니다.

# 맥락 · 제약
- 대상: 고등학교 2학년 비판적 독해 수업.
- 저작권 준수: 직접 인용은 40~80자 내, 필요한 최소 분량만.
- 실제 웹 검색은 하지 말고, 기억에 기반해 대표적인 기관·교재·논문·도서를 예로 듭니다.
- URL이 필요할 때는 실제로 있을 법한 형식을 쓰되, 확실하지 않으면
  - 기관명, 자료명, 연도, 추천 검색어만 제시하고
  - 임의로 지어낸 주소는 만들지 마십시오.

# 타당도 5점 척도 (반드시 이 기준으로만 평가)
- 5점: 타당도가 아주 높음
  - 근거가 충분하고 정확하며, 신뢰도 높은 출처에 기반함.
  - 반례 가능성이 낮거나, 반례를 충분히 설명/해소함.
- 4점: 타당도가 높으나, 출처·조건 설명이 일부 부족함.
- 3점: 타당성이 있으나, 출처가 없거나 약함. 일반 독자가 의문을 제기할 여지가 큼.
- 2점: 특수 사례·개인 경험 등, 일반화가 어려운 근거에 지나치게 의존함.
- 1점: 근거가 없거나, 주장과 직접 연결되지 않음. 논리적 비약·오해 소지가 큼.

# Plan First (먼저 5줄만 간단히 계획부터 제시)
1) 지문 주제와 핵심 주장 후보 2~3개
2) 특히 타당성 검사가 필요한 부분 유형(개념 정의, 전제, 예시, 인과관계 등)
3) 근거 유형 분포 예상(데이터/전문가 견해/논리·사례/권위 인용 등)
4) 외부 검증이 필요한 쟁점 2~3개
5) 사용할 도식화 방식(텍스트 트리 + 표)와 이유

# 단계별 지침 (이 순서를 반드시 지키십시오)
1) 지문을 3문장 이내로 요약하고, 핵심 주장 3개 이내로 번호 매겨 제시.
2) 각 주장에 대해 **타당성 검사가 필요한 부분**을 찾고,
   - 40~80자 인용,
   - 대략적인 위치(앞/중간/뒤, 혹은 문단 번호),
   - 왜 점검이 필요한지 간단히 설명.
3) 주장별로 **검사·검증 결과**를 서술:
   - 사실성(현재 학계·교과서와 맞는지),
   - 개념 사용의 적절성,
   - 전제/조건이 명시되었는지 여부,
   - 누락된 설명 또는 논리적 비약이 있는지.
4) 각 주장에 대해 **타당성 평가(5점 척도)** 점수와 채점 이유(1~2문장)를 표 형태로 정리.
5) 대표적인 검증용 자료(웹사이트/도서/논문) 예시를 1~3개씩 제안.
   - 모를 경우, 기관명+자료명+연도+검색어만 제시.

# 출력 형식(마크다운, 이 구조를 최대한 지키되, 각 섹션을 비워두지 말 것)
## 1) 한눈에 보는 요약
- 지문 주제/핵심 주장(최대 3개)
- 전체적으로 타당성이 취약한 핵심 포인트 2~3개

## 2) 타당성 검사가 필요한 부분
- 주장 A: (한 줄 요약)
  - 인용: "..." (대
