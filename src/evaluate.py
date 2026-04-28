"""Reliability harness for VibeFinder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag_system import generate_recommendation


@dataclass
class EvalCase:
    name: str
    query: str
    expected_any_title: List[str]
    min_confidence: float = 0.75


EVAL_CASES = [
    EvalCase(
        name="workout_energy",
        query="Give me high energy pop songs for a workout.",
        expected_any_title=["Sunrise City", "Gym Hero", "Bassline Orbit"],
    ),
    EvalCase(
        name="study_focus",
        query="I need calm lofi music for coding and deep study.",
        expected_any_title=["Midnight Coding", "Library Rain", "Focus Flow"],
    ),
    EvalCase(
        name="quiet_evening",
        query="Recommend gentle acoustic music for a calm evening.",
        expected_any_title=["Moonlit Strings", "Golden Porch", "Library Rain"],
    ),
]


def run_evaluation() -> int:
    passed = 0
    print("VibeFinder Reliability Evaluation")
    print("=" * 40)
    for case in EVAL_CASES:
        response = generate_recommendation(case.query, use_gemini=False)
        titles = [song["title"] for song, _score, _reason in response.recommendations]
        title_hit = any(expected in titles for expected in case.expected_any_title)
        confidence_ok = response.guardrails.confidence >= case.min_confidence
        case_passed = title_hit and confidence_ok and response.guardrails.passed
        passed += int(case_passed)
        print(f"{case.name}: {'PASS' if case_passed else 'FAIL'}")
        print(f"  titles: {', '.join(titles)}")
        print(f"  confidence: {response.guardrails.confidence:.2f}")
        if response.guardrails.issues:
            print(f"  issues: {'; '.join(response.guardrails.issues)}")
    print("=" * 40)
    print(f"Summary: {passed} out of {len(EVAL_CASES)} cases passed.")
    return 0 if passed == len(EVAL_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
