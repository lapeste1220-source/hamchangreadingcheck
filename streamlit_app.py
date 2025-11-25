# -*- coding: utf-8 -*-
import os
import datetime
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
DEFAULT_MODEL = "gpt-4.1-mini"  # 필요 시 다른 모델명으로 교체 가능


# ---------------- OpenAI 클라이언트 관련 함수 ----------------
def get_api_key(user_input_key: str | None) -> str:
    """
    1순위: 학생이 입력한 API 키
    2순위: 환경변수 OPENAI_API_KEY
    3순위: st.secrets["OPENAI_API_KEY"]
    중 하나를 찾아서 반환. 없으면 에러 발생.
    """
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        secrets_key = None

    if secrets_key:
        return secrets_key

    raise ValueError("OpenAI API 키가 설정되어 있지 않습니다.")


def call_openai_text(model: str, instructions: str, user_input: str, api_key: str) -> str:
    """
    OpenAI Responses API를 사용해 텍스트를 한 덩어리로 반환.
    """
    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model=model,
            input=user_input,
            instructions=instructions,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI 호출 중 오류: {e}")

    # output에서 text만 모아서 반환
    texts = []
    try:
        for item in resp.output:
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    texts.append(content.text)
    except Exception:
        # 혹시 구조가 달라져도 최소한 repr이라도 보여주기
        texts.append(str(resp))

    return "\n".join(texts).strip()


# ---------------- 타당성 평가용 시스템 프롬프트 ----------------
ANALYSIS_INSTRUCTIONS = f"""
당신은 고등학교 2학년 학생을 돕는 '비판적 독해·타당성 평가 도우미'입니다.
아래 규칙과 출력 형식을 반드시 따르십시오.

# 목표 · 성과기준
- 목표: 학생이 제시한 '입력 지문'을 대상으로 주장–근거의 타당성을 점검하고,
  의심스럽거나 미흡한 근거를 식별해 비판적 읽기의 필요성을 체감하도록 돕는다.
- 산출물 구성:
  (1) 주장–근거 도식화(텍스트 트리/표),
  (2) 주장별 타당도 5점 척도 평가표,
  (3) 점수대별 분류 및 검증 작업 설계·일부 실행 요약,
  (4) 검증용 링크·출처 목록,
  (5) 최종 피드백·자기점검 체크리스트.
- 성공기준(체크리스트)
  1) 모든 핵심 주장에 대응하는 근거가 표로 정리될 것
  2) 각 근거의 출처 유형/신뢰도/검증 난도 명시
  3) 5점 척도 기준에 따라 일관된 채점 근거 제시
  4) 점수대별 검증 작업(사실 확인/추가 근거/반증) 제시 및 일부 실행
  5) 인용은 필요한 최소 분량·정확한 출전 표기, 표절/환각 경고 포함

# 맥락 · 제약
- 배경: 고2 비판적 독해 수업. 학생이 지문 전문 또는 충분한 발췌를 업로드함.
- 제약:
  - 저작권 준수(직접 인용은 40~80자·필요 최소),
  - 출처/링크 필수(웹 검색 가능 시 공신력 있는 자료 우선: 정부·공식 통계·학술지·대학 강의 노트·표준 교과서 등),
  - 웹 검색 불가 시 ‘추천 검색어/데이터베이스/저자·연도’로 대체.
- Do:
  - 주장–근거 매핑,
  - 근거 유형 구분(데이터/전문가 견해/논리·사례/권위 인용 등),
  - 반대 가능성 탐색,
  - 쉬운 말 설명.
- Don’t:
  - 지문을 임의로 바꾸거나 확대 해석하지 말 것,
  - 모호한 단정, 근거 없는 평가 금지.

# 대상 · 톤
- 독자: 고등학교 2학년
- 톤: 또렷하고 친절하게, 쉬운 어휘 우선(전문 용어는 짧게 풀이)

# Plan First (반드시 먼저 5줄 계획 제시)
1) 지문 주제·핵심 주장 후보
2) 근거 유형 분포 예상
3) 외부 검증이 필요한 쟁점 목록
4) 사용할 도식화 방식(트리/표/다이어그램)과 이유
5) 브라우징(웹 검색) 계획이 가능하다고 가정했을 때의 키워드/데이터베이스, 또는 대체 검색어

※ 실제 웹 검색은 하지 말고, “이런 걸 찾으면 좋겠다” 수준으로 작성.

# 단계별 지침(5~10단계)
1) 규칙 확인: 저작권/인용, 평가 척도 의미를 스스로 재확인.
2) 지문 요약(3문장 이내) → 핵심 주장 목록화(번호 매김).
3) 주장별 근거 추출·분류:
   - 40~80자 이내로 짧게 인용,
   - 문단/줄 위치(대략) 표기,
   - 근거 유형 표기(데이터/전문가 견해/논리·사례/권위 인용 등).
4) 논리 점검:
   - 주장–이유–근거–함의 연결,
   - 누락/비약 표시(예: 장·단기 혼동, 정의 불명확, 조건 누락 등).
5) 5점 척도로 채점, “왜 그 점수인지” 1~2문장 근거 서술.
6) 점수대별 분류(5/4/3/2/1점) 후 검증 작업 설계:
   - 사실 여부 확인, 추가 근거 조사, 반증 사례 찾기(과제 형태).
7) 검증 일부 실행:
   - 주장당 1~3개 신뢰 가능한 링크가 있다고 가정하고,
   - 링크 제목·설명·신뢰 이유를 형식에 맞게 제시
     (실제 주소 대신 [기관명, 연도, 자료명] 등으로 대체해도 됨).
8) 구조화 출력:
   - 도식화 → 평가표 → 점수대별 분류·검증 → 링크·출처 → 피드백.
9) 리스크·대안:
   - 불확실/쟁점, 추가 필요 데이터, 상반된 해석 가능성 제시.
10) Self-Check 통과 후 최종 요약(3문장).

# 타당도 5점 척도(평가 기준 고정)
- 5점: 타당도가 아주 높음 (근거가 충분·정확, 신뢰도 높은 출처, 반례 가능성 낮음)
- 4점: 타당도가 높으나 명확한 출처 확인 필요
- 3점: 타당도가 있으나 출처가 없음 또는 약함
- 2점: 특수 사례이거나 일반화가 어려운 근거에 의존
- 1점: 근거가 없거나 주장과 직접 연결되지 않음

# 출력 형식(마크다운, 반드시 이 순서와 제목 사용)
## 1) 한눈에 보는 요약
- 지문 주제/핵심 주장(최대 3개)
- 취약 근거 Top 3와 이유(각 1문장)

## 2) 주장–근거 도식화
- 텍스트 트리 형식
- 이후 표 형식

## 3) 타당도 평가표(5점 척도)

## 4) 점수대별 분류 및 검증작업

## 5) 검증용 링크·출처(형식 유지, 실제 URL 대신 출처 설명만 써도 됨)

## 6) 최종 피드백
- 글의 논증 강·약점 요약(3~5문장)
- 추가 읽기 제안
- 수업 토론 질문 3개

## 7) Self-Check

# 시간 감각
- 현재일: {TODAY_STR}.
- 최신성 필요한 항목은 최근 3~5년 자료를 먼저 고려하라고 권장하되, 실제 검색은 하지 말고 추천 형식으로만 제시할 것.

# 중요
- 결과는 고2 학생이 그대로 읽고 사용할 수 있도록, 너무 어려운 수식·전문 용어는 피하고, 나올 경우 짧게 풀이를 함께 제시.
"""


# ---------------- 예시 지문(기본 값) ----------------
DEFAULT_PASSAGE = """가계, 기업, 정부는 경제 주체로서 가계는 소비, 기업은 생산, 정부는 정책 결정 시 합리적인 선택을 하기 위해 노력한다. 이때 합리적인 선택을 하려면 편익과 비용을 충분히 고려하여 편익에서 비용을 뺀 순편익이 가장 큰 대안을 선택해야 한다. 편익이란 어떤 선택을 할 때 얻는 이득으로, 기업의 판매 수입과 같은 금전적인 것이나 소비자가 상품을 소비함으로써 얻는 정신적 만족감과 같은 비금전적인 것을 말한다. 비용이란 암묵적 비용 중 가장 큰 것과 명시적 비용을 합친 것이다. 암묵적 비용은 어떤 선택으로 인해 포기한 다른 대안의 가치를, 명시적 비용은 그 선택을 할 때 화폐로 직접 지불하는 비용을 말한다.
순편익은 한계편익과 한계비용이 같을 때 가장 커지는데, 한계편익은 어떤 선택에 의해 추가로 발생하는 편익이며 한계비용은 그 선택에 의해 추가로 발생하는 비용이다. 예를 들어, 볼펜을 1개 더 살지 고민하고 있는 소비자의 한계편익은 볼펜을 1개 더 사는 데에서 추가로 얻는 만족감이며, 한계비용은 볼펜을 1개 더 사기 위해 추가로 드는 비용이다.
[A]

기업은 상품을 얼마나 생산하면 이윤을 극대화할 수 있을지 한계비용과 한계수입을 고려해 합리적인 판단을 내릴 수 있다. 기업 입장에서 한계비용은 상품 생산량을 한 단위 증가시키는 데 추가로 드는 비용이며, 한계수입은 상품을 한 단위 더 생산하여 판매할 때 추가로 얻는 수입이다. 완전경쟁시장에 있는 기업이라면 상품의 시장 가격 그 자체가 한계수입이 된다. 완전경쟁시장은 많은 수의 공급자와 수요자로 구성되어 있고 거래되는 상품이 동질적이므로 개별 공급자나 수요자가 시장 가격에 영향을 미칠 수 없다. 즉 기업이나 소비자는 시장에서 결정된 상품 가격을 주어진 것으로 받아들이며 이 가격이 기업의 한계수입이 된다. 상품을 사려는 사람들이 많아져 시장 수요가 증가하여 상품 가격이 오른다면, 한계수입도 그만큼 동일하게 오른다.
생산을 계속할 때 손실이 발생하는 상황이 아니라면, 기업은 한계비용과 한계수입이 일치하도록 생산량을 조절해 이윤을 극대화할 수 있다. 한계비용이 한계수입보다 큰 경우에는 상품 생산량을 한 단위 더 줄일 때 그로 인해 추가로 절약되는 비용이 줄어들 수입보다 크므로 생산량을 줄여 이윤을 증가시킬 수 있다. 이와 반대로 한계수입이 한계비용보다 큰 경우에는 생산량을 늘려 이윤을 증가시킬 수 있다.
그런데 생산을 계속할 때 이윤이 남는 것이 아니라 오히려 손실을 볼 수도 있기 때문에 어떤 상황에서 손실이 발생하는지 판단하는 것도 기업 입장에서 중요하다. 이때 고려할 수 있는 것 중 하나가 평균비용이다. 평균비용은 어떤 양의 상품을 생산하는 데 투입된 총비용을 생산량으로 나눈 것으로, 상품을 한 단위 생산하는 데 드는 평균적인 비용을 말한다. 여기에서 총비용은 고정비용과 가변비용으로 구분된다. 한계비용이 총비용 중 가변비용에만 영향을 받는 것과 달리, 평균비용은 고정비용과 가변비용에 모두 영향을 받는다. 고정비용은 생산량에 따라 변하지 않고 일정한 크기를 유지하는 비용으로, 생산량이 많든 적든 매달 똑같이 내야 하는 임대료가 그 예이다. 가변비용은 생산량에 따라 달라지는 비용으로, 각종 재료비, 상품 생산을 늘리기 위해 추가로 고용하는 직원에게 지급되는 보수 등이 그 예이다.
그렇다면 기업은 손실이 발생하는지 평균비용을 통해 어떻게 알 수 있을까? 총비용을 전부 회수하는 것이 언제라도 가능한 기업이 완전경쟁시장에 있다고 가정해 보자. 이 기업은 평균비용을 상품의 시장 가격과 비교해 보고 만약 가격이 평균비용곡선의 최저점에도 미치지 못한다면, 생산량이 얼마이든 그 가격에 상품을 판매해 보았자 손실을 피할 수 없다고 판단할 것이다. 그렇다면 투입된 총비용을 전부 회수하여 손실 발생을 막는 것이 이 기업에 합리적인 결정일 수 있다. 기업이 의도한 생산량에서의 평균비용이 시장 가격보다는 낮아야 이윤이 남는데, 어떻게 해도 손실을 피할 수 없다면 생산을 계속할 것인지 신중하게 고민해야 하는 것이다. ㉠이처럼 평균비용은 한계비용과 더불어 기업이 생산에 관한 의사 결정을 내릴 때 유용하게 활용된다.
합리적 선택을 중심으로 생산에 관한 기업의 의사 결정을 살펴보는 것은 경제 활동을 더 잘 이해하게 한다는 점에서 의미가 있다. 특히, 기업의 생산 활동은 소비자의 수요를 충족해 주고 고용 증가, 경제 성장 등 사회 전체에 미치는 영향이 크다는 점에서 주의 깊게 살펴볼 필요가 있을 것이다.
"""


# ---------------- 세션 상태 초기화 ----------------
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = ""

if "final_report" not in st.session_state:
    st.session_state["final_report"] = ""


# ---------------- 사이드바: 사용 안내 ----------------
with st.sidebar:
    st.header("사용 안내")
    st.markdown(
        """
학생은 아래 순서대로 사용합니다:

1. **선정 동기**를 간단히 적습니다.
2. **분석할 글(지문)**을 붙여넣습니다.
3. **타당성 조사 포인트**(보고 싶은 부분)를 선택합니다.
4. (선택) 자신의 **OpenAI API 키**를 입력하거나,  
   선생님이 미리 설정한 키를 사용합니다.
5. **[1단계: 타당성 분석 실행]** 버튼을 누릅니다.
6. 결과를 읽고, 자신의 **활동 결과/느낀 점**을 입력합니다.
7. **[2단계: 완성된 글 생성]** 버튼으로 한 편의 보고서를 만듭니다.
8. 아래 버튼을 눌러 **텍스트 파일로 다운로드** 후 출력합니다.
        """
    )
    st.markdown("---")
    st.markdown("**API 키 설정**")
    st.caption("없어도 선생님 서버에 키가 설정되어 있다면 그대로 이용 가능합니다.")


# ---------------- 메인 입력 영역 ----------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("1. 학생 입력 영역")

    selected_motivation = st.text_area(
        "① 이 글(또는 책/자료)을 선택한 이유(선정 동기)를 적어 보세요.",
        height=80,
        placeholder="예) 경제에서 '합리적 선택'이 실제 기업 행동과 연결되는 방식이 궁금해서 선택했다."
    )

    passage_text = st.text_area(
        "② 타당성을 평가하고 싶은 글(지문)을 붙여 넣으세요.",
        value=DEFAULT_PASSAGE,
        height=260,
    )

    st.markdown("**③ 특히 어떤 점의 타당성을 점검해 보고 싶나요? (복수 선택 가능)**")

    validity_options = [
        "주장–근거 연결이 논리적으로 타당한지",
        "한계비용·한계수입·평균비용 개념 사용이 정확한지",
        "완전경쟁시장 설명의 전제가 잘 드러나는지",
        "손실 판단 기준(평균비용, 평균가변비용 등)이 정확한지",
        "용어 정의(합리적 선택, 순편익 등)가 명확한지",
        "기타(학생이 따로 적을 부분)"
    ]
    selected_points = st.multiselect(
        "타당성 조사 포인트 선택",
        options=validity_options,
        default=[
            "주장–근거 연결이 논리적으로 타당한지",
            "손실 판단 기준(평균비용, 평균가변비용 등)이 정확한지",
        ]
    )

    extra_point = st.text_input(
        "④ 위에 없는 다른 조사 포인트가 있다면 적어 주세요 (선택).",
        placeholder="예) 교과서에서 배운 내용과 다른 부분이 있는지 등"
    )

with col_right:
    st.subheader("2. OpenAI API 설정")
    user_api_key_input = st.text_input(
        "OpenAI API 키 (선택, 없으면 서버 기본 키 사용)",
        type="password",
        help="원하면 각자 자신의 OpenAI API 키를 넣어 사용할 수 있습니다."
    )

    st.markdown("---")
    st.subheader("3. 활동 결과 메모 (2단계에서 활용)")
    activity_notes = st.text_area(
        "타당성 분석 결과를 읽고, 스스로 정리한 활동 결과·느낀 점을 간단히 적어 보세요.",
        height=160,
        placeholder="예) A 주장에 비해 B 주장은 근거가 약하다는 느낌을 받았고, 앞으로 경제 기사를 읽을 때도 근거의 출처를 더 꼼꼼히 보아야겠다고 생각했다."
    )


# ---------------- 1단계: 타당성 분석 실행 ----------------
st.markdown("---")
st.subheader("4. 1단계: 생성형 AI를 활용한 타당성 분석")

if st.button("🧪 1단계: 타당성 분석 실행", type="primary"):
    if not passage_text.strip():
        st.error("지문(분석할 글)을 먼저 입력해 주세요.")
    else:
        try:
            api_key = get_api_key(user_api_key_input)
        except ValueError as e:
            st.error(str(e))
        else:
            with st.spinner("타당성 분석을 수행하고 있습니다. 잠시만 기다려 주세요..."):
                # 학생이 선택한 포인트를 문자열로 정리
                points_text = ", ".join(selected_points) if selected_points else "학생이 별도 포인트를 선택하지 않음"
                if extra_point.strip():
                    points_text += f"; 추가 포인트: {extra_point.strip()}"

                user_input_for_analysis = f"""
[학생 선정 동기]
{selected_motivation}

[학생이 특히 점검하고 싶은 타당성 포인트]
{points_text}

[분석 대상 지문]
{passage_text}
"""

                try:
                    analysis_result = call_openai_text(
                        model=DEFAULT_MODEL,
                        instructions=ANALYSIS_INSTRUCTIONS,
                        user_input=user_input_for_analysis,
                        api_key=api_key,
                    )
                    st.session_state["analysis_result"] = analysis_result
                except RuntimeError as e:
                    st.error(str(e))


# ---------------- 분석 결과 표시 ----------------
if st.session_state["analysis_result"]:
    st.success("1단계 타당성 분석이 완료되었습니다.")
    st.markdown("### 🔍 AI 기반 타당성 분석 결과")
    st.markdown(st.session_state["analysis_result"])
else:
    st.info("아직 1단계 분석 결과가 없습니다. 위 버튼으로 먼저 타당성 분석을 실행해 주세요.")


# ---------------- 2단계: 완성된 글(보고서) 생성 ----------------
st.markdown("---")
st.subheader("5. 2단계: 학생 활동까지 반영한 완성 글 생성")

FINAL_REPORT_INSTRUCTIONS = f"""
당신은 '비판적 독해 활동 보고서'를 작성하는 조교입니다.
아래 정보를 바탕으로, 고등학교 2학년 학생의 활동 결과를 정리한 글을 써 주세요.

# 역할
- 고2 학생이 제출할 수 있는 '비판적 독해·타당성 검사 활동 보고서' 초안을 작성합니다.
- 교사가 요구한 형식에 맞추되, 학생의 목소리가 느껴지도록 1인칭(저는/나는) 시점도 자연스럽게 섞어도 좋습니다.

# 구성
1) 활동 배경·선정 동기 (1~2문단)
2) 지문 핵심 내용과 주요 주장 정리 (1문단)
3) 주장–근거 타당성 분석 과정 요약
   - 어떤 주장에 근거가 튼튼했는지
   - 어떤 주장에 근거가 약했는지
   - 논리적 비약·모호한 표현 예시
4) 외부 검증(출처·자료 조사) 계획 또는 예시 1~2개
5) 활동을 통해 배운 점·앞으로의 다짐 (1~2문단)

# 톤
- 또렷하고 진지하지만, 고2 학생의 자연스러운 글 느낌 유지
- 너무 과장된 문장보다는 실제 수업 활동처럼 담백하게

# 길이
- 대략 A4 기준 1~2쪽 분량(문단 6~10개 정도)

# 현재일
- {TODAY_STR}
"""

if st.button("📝 2단계: 완성된 글 생성", type="secondary"):
    if not st.session_state["analysis_result"]:
        st.error("먼저 1단계 타당성 분석을 실행해 주세요.")
    else:
        try:
            api_key = get_api_key(user_api_key_input)
        except ValueError as e:
            st.error(str(e))
        else:
            with st.spinner("완성된 보고서를 생성하는 중입니다..."):
                user_input_for_final = f"""
[학생 선정 동기]
{selected_motivation}

[분석 대상 지문]
{passage_text}

[AI 타당성 분석 결과 요약]
{st.session_state['analysis_result']}

[학생 활동 결과/느낀 점]
{activity_notes}
"""
                try:
                    final_report = call_openai_text(
                        model=DEFAULT_MODEL,
                        instructions=FINAL_REPORT_INSTRUCTIONS,
                        user_input=user_input_for_final,
                        api_key=api_key,
                    )
                    st.session_state["final_report"] = final_report
                except RuntimeError as e:
                    st.error(str(e))


# ---------------- 완성 글 표시 및 다운로드 ----------------
if st.session_state["final_report"]:
    st.success("2단계 완성 글 생성이 완료되었습니다.")
    st.markdown("### 📄 완성된 글 (보고서 초안)")
    st.markdown(st.session_state["final_report"])

    st.download_button(
        label="💾 완성된 글 다운로드 (.txt)",
        data=st.session_state["final_report"],
        file_name="함창고_글_타당성검사_활동보고서.txt",
        mime="text/plain",
    )
else:
    st.info("아직 완성된 글이 없습니다. 위의 [2단계: 완성된 글 생성] 버튼을 눌러 주세요.")


# ---------------- 화면 우측 하단 '만든이' 표시 ----------------
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px; 
                font-size: 0.9rem; color: gray; background-color: rgba(255,255,255,0.7);
                padding: 4px 8px; border-radius: 4px;">
        만든이: 함창고 교사 박호종
    </div>
    """,
    unsafe_allow_html=True,
)
