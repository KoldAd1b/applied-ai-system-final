"""Retrieval-augmented music recommendation workflow."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ai_client import GeminiClient
from .recommender import load_songs, recommend_songs


LOGGER = logging.getLogger("vibefinder")

GENRE_TERMS = {
    "ambient",
    "classical",
    "country",
    "edm",
    "folk",
    "hip hop",
    "indie pop",
    "jazz",
    "lofi",
    "metal",
    "pop",
    "r&b",
    "reggae",
    "rock",
    "synthwave",
}

MOOD_TERMS = {
    "chill",
    "confident",
    "euphoric",
    "focused",
    "happy",
    "intense",
    "moody",
    "nostalgic",
    "peaceful",
    "relaxed",
    "romantic",
    "warm",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "but",
    "for",
    "from",
    "give",
    "i",
    "me",
    "music",
    "need",
    "of",
    "or",
    "play",
    "playlist",
    "recommend",
    "song",
    "songs",
    "some",
    "that",
    "the",
    "to",
    "want",
    "with",
}


@dataclass
class RetrievedItem:
    source: str
    title: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailReport:
    passed: bool
    confidence: float
    issues: List[str]


@dataclass
class RecommendationResponse:
    query: str
    preferences: Dict[str, Any]
    plan_steps: List[str]
    retrieved_items: List[RetrievedItem]
    recommendations: List[Tuple[Dict[str, Any], float, str]]
    answer: str
    provider: str
    guardrails: GuardrailReport


def setup_logging(log_path: str = "logs/system.log") -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def load_contexts(csv_path: str) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            context = dict(row)
            context["id"] = int(context["id"])
            context["energy_min"] = float(context["energy_min"])
            context["energy_max"] = float(context["energy_max"])
            contexts.append(context)
    return contexts


def infer_preferences(query: str, retrieved_contexts: Sequence[RetrievedItem]) -> Dict[str, Any]:
    normalized = query.lower()
    prefs: Dict[str, Any] = {}

    for genre in sorted(GENRE_TERMS, key=len, reverse=True):
        if genre in normalized:
            prefs["genre"] = genre
            break

    for mood in sorted(MOOD_TERMS, key=len, reverse=True):
        if mood in normalized:
            prefs["mood"] = mood
            break

    if any(term in normalized for term in ("high energy", "workout", "gym", "run", "party", "boost")):
        prefs["energy"] = 0.88
    elif any(term in normalized for term in ("study", "focus", "coding", "homework", "read")):
        prefs["energy"] = 0.42
    elif any(term in normalized for term in ("low energy", "sleep", "calm", "quiet", "relax", "evening")):
        prefs["energy"] = 0.32

    if any(term in normalized for term in ("acoustic", "folk", "classical", "gentle")):
        prefs["likes_acoustic"] = True
    elif any(term in normalized for term in ("edm", "pop", "workout", "gym", "party", "bass")):
        prefs["likes_acoustic"] = False

    if any(term in normalized for term in ("dance", "party", "club")):
        prefs["danceability"] = 0.86

    if any(term in normalized for term in ("happy", "upbeat", "positive", "boost")):
        prefs["valence"] = 0.82

    if retrieved_contexts:
        top_context = retrieved_contexts[0]
        metadata = top_context.metadata
        prefs.setdefault(
            "energy",
            round((metadata["energy_min"] + metadata["energy_max"]) / 2, 2),
        )
        tags = _tokens(metadata.get("tags", ""))
        for genre in GENRE_TERMS:
            if genre in tags and "genre" not in prefs:
                prefs["genre"] = genre
        for mood in MOOD_TERMS:
            if mood in tags and "mood" not in prefs:
                prefs["mood"] = mood

    prefs.setdefault("energy", 0.60)
    prefs.setdefault("likes_acoustic", None)
    return prefs


def retrieve(query: str, songs: Sequence[Dict[str, Any]], contexts: Sequence[Dict[str, Any]], k: int = 6) -> List[RetrievedItem]:
    query_terms = _tokens(query)
    items: List[RetrievedItem] = []

    for context in contexts:
        text = f"{context['name']} {context['description']} {context['tags']} {context['notes']}"
        score = _overlap_score(query_terms, _tokens(text))
        if score:
            items.append(
                RetrievedItem(
                    source="listening_contexts",
                    title=context["name"],
                    text=f"{context['description']} Notes: {context['notes']}",
                    score=score,
                    metadata=context,
                )
            )

    for song in songs:
        text = (
            f"{song['title']} {song['artist']} {song['genre']} {song['mood']} "
            f"energy {song['energy']} valence {song['valence']} danceability {song['danceability']}"
        )
        score = _overlap_score(query_terms, _tokens(text))
        if score:
            items.append(
                RetrievedItem(
                    source="song_catalog",
                    title=song["title"],
                    text=(
                        f"{song['title']} by {song['artist']} is {song['genre']} / {song['mood']} "
                        f"with energy {song['energy']:.2f}, valence {song['valence']:.2f}, "
                        f"danceability {song['danceability']:.2f}, acousticness {song['acousticness']:.2f}."
                    ),
                    score=score,
                    metadata=song,
                )
            )

    return sorted(items, key=lambda item: item.score, reverse=True)[:k]


def generate_recommendation(
    query: str,
    songs_path: str = "data/songs.csv",
    contexts_path: str = "data/listening_contexts.csv",
    use_gemini: bool = True,
    k: int = 3,
) -> RecommendationResponse:
    setup_logging()
    LOGGER.info("starting recommendation query=%s", query)

    songs = load_songs(songs_path)
    contexts = load_contexts(contexts_path)
    initial_retrieval = retrieve(query, songs, contexts, k=6)
    context_items = [item for item in initial_retrieval if item.source == "listening_contexts"]
    preferences = infer_preferences(query, context_items)
    recommendations = recommend_songs(preferences, songs, k=k, mode="balanced")
    evidence = _recommendation_evidence(recommendations, initial_retrieval)
    plan_steps = [
        "Parse the user request into music preferences.",
        "Retrieve matching listening contexts and catalog songs.",
        "Score songs with the content-based recommender.",
        "Draft a grounded answer from retrieved evidence.",
        "Run guardrails for evidence use and catalog-only recommendations.",
    ]

    prompt = _build_prompt(query, preferences, evidence, recommendations)
    ai_result = GeminiClient().generate(prompt) if use_gemini else None

    if ai_result and ai_result.text:
        answer = ai_result.text
        provider = ai_result.provider
    else:
        answer = _fallback_answer(query, recommendations, evidence)
        provider = ai_result.provider if ai_result else "local_fallback"
        if ai_result and ai_result.error:
            LOGGER.warning("gemini fallback reason=%s", ai_result.error)

    guardrails = validate_response(answer, recommendations, evidence)
    LOGGER.info(
        "completed recommendation provider=%s passed=%s confidence=%.2f issues=%s",
        provider,
        guardrails.passed,
        guardrails.confidence,
        json.dumps(guardrails.issues),
    )

    return RecommendationResponse(
        query=query,
        preferences=preferences,
        plan_steps=plan_steps,
        retrieved_items=evidence,
        recommendations=recommendations,
        answer=answer,
        provider=provider,
        guardrails=guardrails,
    )


def validate_response(
    answer: str,
    recommendations: Sequence[Tuple[Dict[str, Any], float, str]],
    evidence: Sequence[RetrievedItem],
) -> GuardrailReport:
    issues: List[str] = []
    answer_lower = answer.lower()
    evidence_titles = {item.title.lower() for item in evidence}
    recommended_titles = {song["title"].lower() for song, _score, _reason in recommendations}

    mentioned_recs = [title for title in recommended_titles if title in answer_lower]
    if not mentioned_recs:
        issues.append("Answer does not name any scored recommendation.")

    unsupported_song_titles = []
    for match in re.findall(r"\b[A-Z][A-Za-z0-9&' ]{2,}\b", answer):
        lowered = match.strip().lower()
        if lowered.endswith((" by", " is", " has")):
            lowered = lowered.rsplit(" ", 1)[0]
        if lowered in recommended_titles or lowered in evidence_titles:
            continue
        if lowered in {"i", "because", "try", "start", "best"}:
            continue
    if unsupported_song_titles:
        issues.append(f"Unsupported song titles mentioned: {', '.join(unsupported_song_titles)}")

    if len(answer.split()) < 35:
        issues.append("Answer is too short to explain the decision.")

    confidence = max(0.0, 1.0 - (0.25 * len(issues)))
    return GuardrailReport(passed=not issues, confidence=confidence, issues=issues)


def format_response(response: RecommendationResponse, include_debug: bool = False) -> str:
    lines = [
        f"Query: {response.query}",
        f"Provider: {response.provider}",
        f"Guardrails: {'PASS' if response.guardrails.passed else 'CHECK'} "
        f"(confidence {response.guardrails.confidence:.2f})",
        "",
        response.answer,
    ]
    if response.guardrails.issues:
        lines.extend(["", "Guardrail issues:"])
        lines.extend(f"- {issue}" for issue in response.guardrails.issues)
    if include_debug:
        lines.extend(["", "Plan:"])
        lines.extend(f"{index}. {step}" for index, step in enumerate(response.plan_steps, start=1))
        lines.extend(["", "Retrieved evidence:"])
        lines.extend(f"- [{item.source}] {item.title} (score {item.score:.2f})" for item in response.retrieved_items)
    return "\n".join(lines)


def _build_prompt(
    query: str,
    preferences: Dict[str, Any],
    evidence: Sequence[RetrievedItem],
    recommendations: Sequence[Tuple[Dict[str, Any], float, str]],
) -> str:
    evidence_text = "\n".join(f"- {item.title}: {item.text}" for item in evidence)
    rec_text = "\n".join(
        f"- {song['title']} by {song['artist']}: score {score:.2f}; {reason}"
        for song, score, reason in recommendations
    )
    return (
        "You are VibeFinder, a careful music recommendation assistant. "
        "Use only the catalog evidence below. Do not invent songs or artists. "
        "Give three recommendations with concise reasons and one caution about limitations.\n\n"
        f"User request: {query}\n"
        f"Inferred preferences: {json.dumps(preferences, sort_keys=True)}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        f"Scored recommendations:\n{rec_text}\n"
    )


def _fallback_answer(
    query: str,
    recommendations: Sequence[Tuple[Dict[str, Any], float, str]],
    evidence: Sequence[RetrievedItem],
) -> str:
    context = next((item for item in evidence if item.source == "listening_contexts"), None)
    opening = "Based on the catalog evidence"
    if context:
        opening += f" and the '{context.title}' listening context"
    lines = [f"{opening}, I would recommend:"]
    for rank, (song, score, reason) in enumerate(recommendations, start=1):
        lines.append(
            f"{rank}. {song['title']} by {song['artist']} - score {score:.2f}. "
            f"It fits because {reason}."
        )
    lines.append(
        "Limitation: this answer is grounded only in the small fictional song catalog, "
        "so missing genres or artists may reduce recommendation quality."
    )
    return "\n".join(lines)


def _recommendation_evidence(
    recommendations: Sequence[Tuple[Dict[str, Any], float, str]],
    retrieved: Sequence[RetrievedItem],
) -> List[RetrievedItem]:
    evidence = list(retrieved)
    existing_titles = {item.title for item in evidence}
    for song, score, reason in recommendations:
        if song["title"] in existing_titles:
            continue
        evidence.append(
            RetrievedItem(
                source="scored_recommendation",
                title=song["title"],
                text=f"{song['title']} by {song['artist']} scored {score:.2f}. Reasons: {reason}.",
                score=score,
                metadata=song,
            )
        )
    return evidence


def _tokens(text: str) -> Set[str]:
    raw = re.findall(r"[a-z0-9&]+(?:\s+[a-z0-9&]+)?", text.lower())
    tokens = {token.strip() for token in raw if token.strip() and token.strip() not in STOPWORDS}
    words = re.findall(r"[a-z0-9&]+", text.lower())
    tokens.update(word for word in words if word not in STOPWORDS)
    return tokens


def _overlap_score(query_terms: Iterable[str], document_terms: Set[str]) -> float:
    query_set = set(query_terms)
    if not query_set:
        return 0.0
    overlap = query_set & document_terms
    phrase_bonus = sum(0.5 for term in overlap if " " in term)
    return float(len(overlap)) + phrase_bonus
