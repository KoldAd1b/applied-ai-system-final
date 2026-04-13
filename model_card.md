# Model Card: Music Recommender Simulation

## 1. Model Name

VibeFinder 1.0

## 2. Intended Use

This system suggests songs from a small fictional catalog based on a user's preferred genre, mood, energy level, and optional taste signals. It is intended for classroom exploration of recommendation logic. It should not be used as a real music product or as a serious prediction of a person's taste.

## 3. How the Model Works

VibeFinder is a content-based recommender. It compares user preferences to song features and turns those comparisons into a score. Genre and mood matches add fixed points. Energy, valence, and danceability use closeness scoring, so songs near the user's target values score higher. Acousticness is rewarded differently depending on whether the user says they like acoustic music.

The system includes three scoring modes:

- `balanced`: gives genre, mood, and energy all meaningful influence
- `genre_first`: gives exact genre matches the strongest boost
- `mood_first`: gives exact mood matches the strongest boost

Every recommendation includes reasons so the score is easier to understand.

## 4. Data

The catalog is stored in `data/songs.csv` and contains 18 fictional songs. The dataset includes pop, lofi, rock, ambient, jazz, synthwave, indie pop, edm, folk, r&b, hip hop, metal, reggae, classical, and country. Each song has genre, mood, energy, tempo, valence, danceability, and acousticness fields.

The data is very small and hand-made. It does not include lyrics, language, release date, listener history, popularity, artist background, or cultural context.

## 5. Strengths

The recommender works best when the user has clear preferences that match the catalog labels. For example, a high-energy pop profile correctly pushes upbeat pop songs toward the top. A chill acoustic profile favors lofi and acoustic songs with lower energy. The explanations are also useful because users can see why a song ranked highly.

## 6. Limitations and Bias

The system can over-reward exact genre and mood labels. This can create a filter bubble where users mostly see songs that look like their past preferences. It cannot learn from skips, repeated listening, playlists, social behavior, lyrics, or long-term taste changes. It may also miss good recommendations when a song has the wrong label but the right feel.

Because the dataset is small, some genres and moods have only one example. That means the model may treat a single fictional song as if it represents a whole genre.

## 7. Evaluation

I evaluated the system with three profiles: High-Energy Pop, Chill Acoustic Focus, and Deep Intense Rock. I checked whether the top results matched the profile's genre, mood, and energy target. I also compared `balanced`, `genre_first`, and `mood_first` modes to see how the rankings changed.

The main surprise was how strongly a label match can affect the ranking. A mood-first setting can lift songs from different genres if they share the same mood. A genre-first setting can keep songs in the same genre near the top even when another song has similar energy.

The automated tests check CSV loading, typed numeric fields, score explanations, sorted rankings, OOP recommendations, and scoring mode behavior.

## 8. Future Work

Future improvements could include:

- Learning from real user behavior, such as skips, saves, and replays
- Adding diversity-aware ranking so the top results are not too repetitive
- Using richer tags for lyrics, era, instruments, language, and activity
- Letting users adjust the weights directly
- Adding feedback loops so the recommender changes over time

## 9. Personal Reflection

Building this system made recommendation algorithms feel more concrete. A ranked list is not magic. It is the result of choices about what data matters and how much each feature is worth.

The most interesting part was seeing that simple math can still produce results that feel personal. The risky part is that simple math can also hide bias. If the catalog is narrow or the weights are too strong, the system can keep showing the same kind of song instead of helping someone discover something new.
