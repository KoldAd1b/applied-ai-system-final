from dataclasses import asdict

from src.recommender import (
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_load_songs_converts_numeric_fields(tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness",
                "1,Typed Track,Test Artist,pop,happy,0.8,120,0.9,0.7,0.2",
            ]
        ),
        encoding="utf-8",
    )

    songs = load_songs(str(csv_path))

    assert songs[0]["id"] == 1
    assert isinstance(songs[0]["energy"], float)
    assert isinstance(songs[0]["tempo_bpm"], float)
    assert isinstance(songs[0]["valence"], float)
    assert isinstance(songs[0]["danceability"], float)
    assert isinstance(songs[0]["acousticness"], float)


def test_score_song_returns_score_and_reasons():
    song = asdict(make_small_recommender().songs[0])
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    score, reasons = score_song(user_prefs, song)

    assert isinstance(score, float)
    assert score > 0
    assert reasons


def test_exact_match_scores_higher_than_mismatch():
    matching_song = asdict(make_small_recommender().songs[0])
    mismatched_song = asdict(make_small_recommender().songs[1])
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    matching_score, _ = score_song(user_prefs, matching_song)
    mismatched_score, _ = score_song(user_prefs, mismatched_song)

    assert matching_score > mismatched_score


def test_recommend_songs_returns_top_k_sorted_by_score():
    songs = [asdict(song) for song in make_small_recommender().songs]
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    results = recommend_songs(user_prefs, songs, k=2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]
    assert results[0][0]["title"] == "Test Pop Track"


def test_oop_recommend_uses_same_top_result_as_functional_api():
    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    user_prefs = {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }

    oop_top = rec.recommend(user, k=1)[0]
    functional_top = recommend_songs(user_prefs, [asdict(song) for song in rec.songs], k=1)[0][0]

    assert oop_top.title == functional_top["title"]


def test_scoring_modes_can_change_ranking_priorities():
    songs = [
        {
            "id": 1,
            "title": "Genre Match",
            "artist": "Test Artist",
            "genre": "pop",
            "mood": "sad",
            "energy": 0.8,
            "tempo_bpm": 120.0,
            "valence": 0.4,
            "danceability": 0.6,
            "acousticness": 0.2,
        },
        {
            "id": 2,
            "title": "Mood Match",
            "artist": "Test Artist",
            "genre": "rock",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120.0,
            "valence": 0.8,
            "danceability": 0.6,
            "acousticness": 0.2,
        },
    ]
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    genre_first_top = recommend_songs(user_prefs, songs, k=1, mode="genre_first")[0][0]
    mood_first_top = recommend_songs(user_prefs, songs, k=1, mode="mood_first")[0][0]

    assert genre_first_top["title"] == "Genre Match"
    assert mood_first_top["title"] == "Mood Match"
