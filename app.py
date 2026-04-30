"""Streamlit UI for VibeFinder."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ai_client import GeminiClient
from src.evaluate import EVAL_CASES
from src.rag_system import generate_recommendation


PROJECT_ROOT = Path(__file__).parent
SONGS_PATH = PROJECT_ROOT / "data" / "songs.csv"
CONTEXTS_PATH = PROJECT_ROOT / "data" / "listening_contexts.csv"

EXAMPLE_QUERIES = [
    "Give me high energy pop songs for a workout.",
    "I need calm lofi music for coding and deep study.",
    "Recommend gentle acoustic music for a calm evening.",
]


def main() -> None:
    st.set_page_config(page_title="VibeFinder", page_icon="🎧", layout="wide")
    st.title("VibeFinder")
    st.caption("RAG music recommendation system with explainable scoring and guardrails")

    if "query" not in st.session_state:
        st.session_state.query = EXAMPLE_QUERIES[0]

    gemini_ready = GeminiClient().is_configured

    with st.sidebar:
        st.header("Run Settings")
        use_gemini = st.toggle("Use Gemini when available", value=gemini_ready)
        show_debug = st.toggle("Show RAG evidence", value=True)
        top_k = st.slider("Recommendations", min_value=1, max_value=5, value=3)

        st.divider()
        st.subheader("Examples")
        for query in EXAMPLE_QUERIES:
            if st.button(query, use_container_width=True):
                st.session_state.query = query

        st.divider()
        st.write("Gemini status")
        st.success("Configured") if gemini_ready else st.info("Using local fallback")

    query = st.text_area(
        "Music request",
        key="query",
        height=90,
        placeholder="Ask for a playlist situation, mood, genre, or energy level.",
    )

    run_clicked = st.button("Generate recommendations", type="primary")

    if run_clicked or query:
        response = generate_recommendation(
            query=query,
            songs_path=str(SONGS_PATH),
            contexts_path=str(CONTEXTS_PATH),
            use_gemini=use_gemini,
            k=top_k,
        )
        render_response(response, show_debug=show_debug)

    st.divider()
    render_evaluation_panel()


def render_response(response, show_debug: bool) -> None:
    status_text = "PASS" if response.guardrails.passed else "CHECK"
    metric_cols = st.columns(3)
    metric_cols[0].metric("Provider", response.provider)
    metric_cols[1].metric("Guardrails", status_text)
    metric_cols[2].metric("Confidence", f"{response.guardrails.confidence:.2f}")

    if response.guardrails.issues:
        st.warning("Guardrail issues: " + "; ".join(response.guardrails.issues))

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Ranked Songs")
    for rank, (song, score, reason) in enumerate(response.recommendations, start=1):
        with st.expander(f"{rank}. {song['title']} by {song['artist']} - score {score:.2f}", expanded=rank == 1):
            detail_cols = st.columns(5)
            detail_cols[0].metric("Genre", song["genre"])
            detail_cols[1].metric("Mood", song["mood"])
            detail_cols[2].metric("Energy", f"{song['energy']:.2f}")
            detail_cols[3].metric("Dance", f"{song['danceability']:.2f}")
            detail_cols[4].metric("Acoustic", f"{song['acousticness']:.2f}")
            st.write(reason)

    if show_debug:
        st.subheader("RAG Evidence")
        evidence_rows = [
            {
                "source": item.source,
                "title": item.title,
                "retrieval_score": item.score,
                "evidence": item.text,
            }
            for item in response.retrieved_items
        ]
        st.dataframe(evidence_rows, hide_index=True, use_container_width=True)

        st.subheader("Agentic Workflow")
        for index, step in enumerate(response.plan_steps, start=1):
            st.write(f"{index}. {step}")


def render_evaluation_panel() -> None:
    st.subheader("Reliability Check")
    st.caption("Runs the predefined evaluation cases with deterministic local generation.")

    if not st.button("Run evaluation harness"):
        return

    passed = 0
    rows = []
    for case in EVAL_CASES:
        response = generate_recommendation(
            case.query,
            songs_path=str(SONGS_PATH),
            contexts_path=str(CONTEXTS_PATH),
            use_gemini=False,
        )
        titles = [song["title"] for song, _score, _reason in response.recommendations]
        title_hit = any(expected in titles for expected in case.expected_any_title)
        confidence_ok = response.guardrails.confidence >= case.min_confidence
        case_passed = title_hit and confidence_ok and response.guardrails.passed
        passed += int(case_passed)
        rows.append(
            {
                "case": case.name,
                "result": "PASS" if case_passed else "FAIL",
                "confidence": response.guardrails.confidence,
                "top_songs": ", ".join(titles),
            }
        )

    st.metric("Evaluation Result", f"{passed}/{len(EVAL_CASES)}")
    st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
