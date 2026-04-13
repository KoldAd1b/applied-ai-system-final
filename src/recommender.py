import csv
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Tuple


NUMERIC_FIELDS = ("energy", "tempo_bpm", "valence", "danceability", "acousticness")

SCORING_MODES = {
    "balanced": {"genre": 2.0, "mood": 1.5, "energy": 2.0},
    "genre_first": {"genre": 3.0, "mood": 1.0, "energy": 1.0},
    "mood_first": {"genre": 1.0, "mood": 3.0, "energy": 1.5},
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5, mode: str = "balanced") -> List[Song]:
        """Return the top k songs sorted by recommendation score."""
        scored_songs = [
            (song, score_song(_user_to_preferences(user), _song_to_dict(song), mode)[0])
            for song in self.songs
        ]
        ranked = sorted(scored_songs, key=lambda item: item[1], reverse=True)
        return [song for song, _score in ranked[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song, mode: str = "balanced") -> str:
        """Explain why a song matches the user's preferences."""
        score, reasons = score_song(_user_to_preferences(user), _song_to_dict(song), mode)
        return f"Score {score:.2f}: {'; '.join(reasons)}"


def load_songs(csv_path: str) -> List[Dict[str, Any]]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song: Dict[str, Any] = dict(row)
            song["id"] = int(song["id"])
            for field in NUMERIC_FIELDS:
                song[field] = float(song[field])
            songs.append(song)
    return songs


def score_song(user_prefs: Dict[str, Any], song: Dict[str, Any], mode: str = "balanced") -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    if mode not in SCORING_MODES:
        valid_modes = ", ".join(sorted(SCORING_MODES))
        raise ValueError(f"Unknown scoring mode '{mode}'. Choose one of: {valid_modes}.")

    prefs = _normalize_preferences(user_prefs)
    weights = SCORING_MODES[mode]
    score = 0.0
    reasons: List[str] = []

    genre = prefs.get("genre")
    if genre:
        if _same_label(song.get("genre"), genre):
            score += weights["genre"]
            reasons.append(f"genre match (+{weights['genre']:.2f})")
        else:
            reasons.append("genre differs (+0.00)")

    mood = prefs.get("mood")
    if mood:
        if _same_label(song.get("mood"), mood):
            score += weights["mood"]
            reasons.append(f"mood match (+{weights['mood']:.2f})")
        else:
            reasons.append("mood differs (+0.00)")

    if prefs.get("energy") is not None:
        points = _closeness_points(song.get("energy"), prefs["energy"], weights["energy"])
        score += points
        reasons.append(f"energy closeness (+{points:.2f})")

    for field in ("valence", "danceability"):
        if prefs.get(field) is not None:
            points = _closeness_points(song.get(field), prefs[field], 1.0)
            score += points
            reasons.append(f"{field} closeness (+{points:.2f})")

    if prefs.get("likes_acoustic") is not None:
        acousticness = float(song.get("acousticness", 0.0))
        if prefs["likes_acoustic"]:
            points = acousticness
            reasons.append(f"acoustic preference (+{points:.2f})")
        else:
            points = (1.0 - acousticness) * 0.5
            reasons.append(f"low-acoustic preference (+{points:.2f})")
        score += points

    if not reasons:
        reasons.append("no matching preferences supplied (+0.00)")

    return score, reasons


def recommend_songs(
    user_prefs: Dict[str, Any],
    songs: List[Dict[str, Any]],
    k: int = 5,
    mode: str = "balanced",
) -> List[Tuple[Dict[str, Any], float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored_songs = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, mode)
        scored_songs.append((song, score, "; ".join(reasons)))

    return sorted(scored_songs, key=lambda item: item[1], reverse=True)[:k]


def _normalize_preferences(user_prefs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "genre": user_prefs.get("genre", user_prefs.get("favorite_genre")),
        "mood": user_prefs.get("mood", user_prefs.get("favorite_mood")),
        "energy": user_prefs.get("energy", user_prefs.get("target_energy")),
        "valence": user_prefs.get("valence"),
        "danceability": user_prefs.get("danceability"),
        "likes_acoustic": user_prefs.get("likes_acoustic"),
    }


def _user_to_preferences(user: UserProfile) -> Dict[str, Any]:
    return {
        "favorite_genre": user.favorite_genre,
        "favorite_mood": user.favorite_mood,
        "target_energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }


def _song_to_dict(song: Song) -> Dict[str, Any]:
    if is_dataclass(song):
        return asdict(song)
    return dict(song)


def _same_label(left: Any, right: Any) -> bool:
    return str(left).strip().lower() == str(right).strip().lower()


def _closeness_points(song_value: Any, target_value: Any, weight: float) -> float:
    similarity = max(0.0, 1.0 - abs(float(song_value) - float(target_value)))
    return similarity * weight
