"""Command line runner for the VibeFinder applied AI system."""

import argparse

from .rag_system import format_response, generate_recommendation
from .recommender import load_songs, recommend_songs


PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "danceability": 0.85,
        "likes_acoustic": False,
    },
    "Chill Acoustic Focus": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "valence": 0.60,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
    },
}

SCORING_MODES = ("balanced", "genre_first", "mood_first")


def run_original_demo() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for profile_name, user_prefs in PROFILES.items():
        print(f"\n{'=' * 72}")
        print(f"Profile: {profile_name}")

        for mode in SCORING_MODES:
            recommendations = recommend_songs(user_prefs, songs, k=5, mode=mode)

            print(f"\nMode: {mode}")
            print("-" * 72)
            for rank, (song, score, explanation) in enumerate(recommendations, start=1):
                print(
                    f"{rank}. {song['title']} by {song['artist']} "
                    f"({song['genre']}, {song['mood']})"
                )
                print(f"   Score: {score:.2f}")
                print(f"   Because: {explanation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VibeFinder AI music recommendations.")
    parser.add_argument("--query", help="Natural-language music request to answer with RAG.")
    parser.add_argument("--debug", action="store_true", help="Show plan and retrieved evidence.")
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Use the deterministic local generator even if GEMINI_API_KEY is set.",
    )
    parser.add_argument(
        "--original-demo",
        action="store_true",
        help="Run the original Module 3 scoring-mode demo.",
    )
    args = parser.parse_args()

    if args.original_demo:
        run_original_demo()
        return

    query = args.query or "I need upbeat pop or EDM songs for a workout playlist."
    response = generate_recommendation(query=query, use_gemini=not args.no_gemini)
    print(format_response(response, include_debug=args.debug))


if __name__ == "__main__":
    main()
