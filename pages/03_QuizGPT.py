import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    HAS_LANGCHAIN_OPENAI = True
except ImportError:  # pragma: no cover - 호환성을 위한 폴백
    from langchain.chat_models import ChatOpenAI  # type: ignore
    HAS_LANGCHAIN_OPENAI = False
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage
from langchain.retrievers import WikipediaRetriever
import streamlit as st


load_dotenv()

DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

GITHUB_REPO_URL = "https://github.com/yy/full-gpt"
QUESTION_COUNT = 10

DIFFICULTY_OPTIONS: Dict[str, str] = {
    "쉬움": "Focus on fundamental facts and straightforward recall.",
    "보통": "Mix recall with basic application and understanding questions.",
    "어려움": "Prioritise analytical and multi-step reasoning with subtle distractors.",
}

QUIZ_FUNCTION_SPEC = {
    "name": "generate_quiz",
    "description": "Return a list of multiple-choice questions with four answers and one correct answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Optional explanation that clarifies the correct answer.",
                        },
                    },
                    "required": ["question", "answers"],
                },
                "minItems": QUESTION_COUNT,
                "maxItems": QUESTION_COUNT,
            }
        },
        "required": ["questions"],
    },
}

QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert teacher crafting multiple-choice quizzes.\n"
                "Generate {question_count} questions targeting {difficulty_label} difficulty.\n"
                "Difficulty guidance: {difficulty_instruction}\n"
                "Each question must include exactly four answer options labelled with text only, "
                "and exactly one option must be correct. Provide the strongest pedagogical quality."
            ),
        ),
        (
            "human",
            "Use the following study material to create the quiz.\n---\n{context}",
        ),
    ]
)


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file) -> List[Any]:
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term: str) -> List[Any]:
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def build_llm(api_key: str) -> ChatOpenAI:
    kwargs: Dict[str, Any] = {"temperature": 0.1}
    if HAS_LANGCHAIN_OPENAI:
        kwargs["model"] = "gpt-4o-mini"
    else:
        kwargs["model_name"] = "gpt-4o-mini"
    if api_key:
        try:
            return ChatOpenAI(**kwargs, openai_api_key=api_key)
        except TypeError:
            return ChatOpenAI(**kwargs, api_key=api_key)
    return ChatOpenAI(**kwargs)


def parse_quiz_payload(message: AIMessage) -> List[Dict[str, Any]]:
    fn_call = message.additional_kwargs.get("function_call")
    if not fn_call or fn_call.get("name") != "generate_quiz":
        raise ValueError("Function call result not available from LLM response")
    arguments = fn_call.get("arguments", "{}")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM 함수 호출 응답을 파싱하지 못했습니다.") from exc
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError("LLM did not return any questions")
    return questions


def generate_quiz(docs: List[Any], difficulty_label: str, api_key: str) -> List[Dict[str, Any]]:
    context = format_docs(docs)
    if not context.strip():
        raise ValueError("선택한 자료에서 텍스트를 찾지 못했습니다.")
    llm = build_llm(api_key)
    bound_llm = llm.bind(functions=[QUIZ_FUNCTION_SPEC], function_call={"name": "generate_quiz"})
    chain = QUIZ_PROMPT | bound_llm
    ai_message = chain.invoke(
        {
            "context": context,
            "difficulty_label": difficulty_label,
            "difficulty_instruction": DIFFICULTY_OPTIONS.get(difficulty_label, ""),
            "question_count": QUESTION_COUNT,
        }
    )
    return parse_quiz_payload(ai_message)


def reset_quiz_state(clear_questions: bool = False) -> None:
    for key in st.session_state.get("question_keys", []):
        st.session_state.pop(key, None)
    st.session_state["answers_submitted"] = False
    st.session_state["score"] = None
    st.session_state["total_questions"] = None
    st.session_state["feedback"] = []
    st.session_state["balloons_shown"] = False
    if clear_questions:
        st.session_state["quiz_data"] = None
        st.session_state["question_keys"] = []


def ensure_session_defaults() -> None:
    st.session_state.setdefault("quiz_data", None)
    st.session_state.setdefault("question_keys", [])
    st.session_state.setdefault("answers_submitted", False)
    st.session_state.setdefault("score", None)
    st.session_state.setdefault("total_questions", None)
    st.session_state.setdefault("feedback", [])
    st.session_state.setdefault("balloons_shown", False)
    st.session_state.setdefault("quiz_context_id", None)


def render_results() -> None:
    if not st.session_state.get("answers_submitted"):
        return
    score = st.session_state.get("score") or 0
    total = st.session_state.get("total_questions") or 0
    feedback = st.session_state.get("feedback", [])
    st.metric("점수", f"{score} / {total}")
    for item in feedback:
        status = item["status"]
        question_text = item["question"]
        correct_answer = item["correct"]
        explanation = item.get("explanation")
        if status == "correct":
            st.success(f"정답! {question_text}")
        else:
            st.error(f"오답! {question_text}\n정답: {correct_answer}")
        if explanation:
            st.caption(f"해설: {explanation}")
    if total and score == total:
        st.success("만점입니다! 축하합니다 🎉")
        if not st.session_state.get("balloons_shown"):
            st.balloons()
            st.session_state["balloons_shown"] = True
    else:
        st.info("만점이 아닙니다. 다시 도전해 보세요!")
        if st.button("다시 풀기"):
            reset_quiz_state(clear_questions=False)


def render_quiz() -> None:
    questions = st.session_state.get("quiz_data")
    if not questions:
        return
    st.subheader("📋 Quiz")
    with st.form("quiz_form"):
        for idx, question in enumerate(questions):
            key = f"question_{idx}"
            if key not in st.session_state["question_keys"]:
                st.session_state["question_keys"].append(key)
            st.markdown(f"**문제 {idx + 1}. {question['question']}**")
            options = [answer["answer"] for answer in question["answers"]]
            current_selection = st.session_state.get(key)
            default_index = options.index(current_selection) if current_selection in options else None
            if default_index is None:
                st.radio(
                    "보기 중 하나를 선택하세요.",
                    options,
                    index=None,
                    key=key,
                )
            else:
                st.radio(
                    "보기 중 하나를 선택하세요.",
                    options,
                    index=default_index,
                    key=key,
                )
        submitted = st.form_submit_button("채점하기")
    if submitted:
        unanswered = [
            key
            for key in st.session_state["question_keys"]
            if st.session_state.get(key) is None
        ]
        if unanswered:
            st.warning("모든 문항에 답변해 주세요.")
            return
        score = 0
        feedback: List[Dict[str, Any]] = []
        for idx, question in enumerate(questions):
            key = st.session_state["question_keys"][idx]
            user_answer = st.session_state.get(key)
            correct_option = next((ans["answer"] for ans in question["answers"] if ans["correct"]), None)
            explanation = question.get("explanation")
            status = "correct" if user_answer == correct_option else "incorrect"
            if status == "correct":
                score += 1
            feedback.append(
                {
                    "status": status,
                    "question": question["question"],
                    "correct": correct_option,
                    "explanation": explanation,
                }
            )
        st.session_state["score"] = score
        st.session_state["total_questions"] = len(questions)
        st.session_state["feedback"] = feedback
        st.session_state["answers_submitted"] = True


def main() -> None:
    ensure_session_defaults()
    st.title("QuizGPT")
    st.write("학습 자료로부터 난이도별 객관식 퀴즈를 자동으로 생성합니다.")

    with st.sidebar:
        st.header("설정")
        st.markdown(f"[📁 GitHub 리포지토리]({GITHUB_REPO_URL})")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="개인 OpenAI API Key를 입력하면 해당 키로 퀴즈를 생성합니다.",
        )
        difficulty = st.selectbox("난이도 선택", list(DIFFICULTY_OPTIONS.keys()))
        st.caption("난이도에 따라 문제의 깊이와 난이도가 달라집니다.")

        docs: List[Any] | None = None
        file = None
        topic = ""
        choice = st.selectbox("자료 선택", ("파일", "위키피디아"))
        if choice == "파일":
            file = st.file_uploader(".pdf, .txt, .docx 파일 업로드", type=["pdf", "txt", "docx"])
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("위키피디아 검색어")
            if topic:
                docs = wiki_search(topic)

    context_id: Tuple[Any, ...] = (
        choice,
        getattr(file, "name", None),
        topic,
        difficulty,
    )
    if st.session_state.get("quiz_context_id") != context_id:
        reset_quiz_state(clear_questions=True)
        st.session_state["quiz_context_id"] = context_id

    api_key_to_use = api_key_input or DEFAULT_OPENAI_KEY
    if not api_key_to_use:
        st.warning("OpenAI API Key를 입력하거나 .env에 설정해 주세요.")
        return

    if not docs:
        st.info("사이드바에서 파일을 업로드하거나 위키피디아 검색어를 입력해 주세요.")
        return

    if st.button("퀴즈 생성하기", type="primary"):
        try:
            questions = generate_quiz(docs, difficulty, api_key_to_use)
        except Exception as exc:  # pragma: no cover - UI 에러 표시
            st.error(f"퀴즈 생성 중 오류가 발생했습니다: {exc}")
        else:
            reset_quiz_state(clear_questions=True)
            st.session_state["quiz_data"] = questions
            st.session_state["question_keys"] = [f"question_{i}" for i in range(len(questions))]
            st.success("퀴즈가 생성되었습니다! 아래에서 풀어보세요.")

    render_quiz()
    render_results()


if __name__ == "__main__":
    main()
