"""Command line runner for the Music Recommender Simulation."""

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


def main() -> None:
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


if __name__ == "__main__":
    main()
