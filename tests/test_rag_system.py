from src.rag_system import (
    generate_recommendation,
    infer_preferences,
    load_contexts,
    retrieve,
    validate_response,
)
from src.recommender import load_songs


def test_retrieve_finds_context_and_catalog_evidence():
    songs = load_songs("data/songs.csv")
    contexts = load_contexts("data/listening_contexts.csv")

    items = retrieve("high energy workout pop", songs, contexts, k=4)

    assert items
    assert any(item.source == "listening_contexts" for item in items)
    assert any(item.source == "song_catalog" for item in items)


def test_infer_preferences_extracts_query_signals():
    prefs = infer_preferences("I want calm lofi music for coding", [])

    assert prefs["genre"] == "lofi"
    assert prefs["energy"] == 0.42


def test_generate_recommendation_uses_fallback_and_passes_guardrails():
    response = generate_recommendation(
        "Give me high energy pop songs for a workout.",
        use_gemini=False,
    )

    titles = [song["title"] for song, _score, _reason in response.recommendations]
    assert response.provider == "local_fallback"
    assert response.guardrails.passed
    assert response.guardrails.confidence >= 0.75
    assert any(title in titles for title in ["Sunrise City", "Gym Hero", "Bassline Orbit"])


def test_validate_response_catches_unhelpful_answer():
    recommendations = [
        (
            {
                "title": "Sunrise City",
                "artist": "Neon Echo",
                "genre": "pop",
                "mood": "happy",
            },
            6.0,
            "genre match",
        )
    ]

    report = validate_response("Try something fun.", recommendations, [])

    assert not report.passed
    assert report.issues
