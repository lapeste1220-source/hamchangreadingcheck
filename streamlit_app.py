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
