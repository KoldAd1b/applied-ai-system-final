# Model Card: VibeFinder Applied AI System

## Model Name

VibeFinder RAG 2.0

## Intended Use

VibeFinder recommends fictional songs from a small CSV catalog based on natural-language user requests. It is intended as a classroom and portfolio project showing retrieval, explainable ranking, optional LLM generation, guardrails, and reliability testing.

It should not be used as a production music recommender or as a serious prediction of a person's real taste.

## Base Project

The base project was **Music Recommender Simulation** from Module 3. It accepted structured taste profiles and returned ranked songs with score explanations. This final project adds natural-language input, retrieval over custom context data, optional Gemini response generation, guardrails, logging, and an evaluation script.

## System Behavior

The system follows five observable steps:

1. Parse the user request into preferences.
2. Retrieve relevant listening contexts and catalog songs.
3. Score all songs with the original content-based recommender.
4. Generate a grounded recommendation answer with Gemini or a deterministic fallback.
5. Validate the response and report confidence.

## Data

The main catalog is `data/songs.csv`, which contains 18 fictional songs with genre, mood, energy, tempo, valence, danceability, and acousticness.

The RAG knowledge base is `data/listening_contexts.csv`, which contains custom guidance for workout, study, commute, calm evening, and confidence-boost listening situations.

## Strengths

VibeFinder works best when the user request includes recognizable signals such as "workout," "study," "calm," "lofi," "pop," "acoustic," or "high energy." It is transparent because the ranked songs include score reasons, retrieved evidence can be displayed with `--debug`, and the fallback generator produces reproducible outputs.

## Limitations And Biases

The catalog is too small to represent real music taste. Some genres and moods have only one song, which can make one fictional track stand in for an entire style. The retriever is keyword-based, so it may miss synonyms or more subtle intent. The preference parser uses hand-written rules, which means its priorities reflect the developer's assumptions.

The system may also create a mild filter-bubble effect by rewarding exact genre and mood matches. A real system should add diversity checks, user feedback, consent-aware personalization, and broader evaluation.

## Misuse Risks And Mitigations

This version recommends fictional songs, so direct harm is limited. The larger pattern could be misused if it collected listening behavior without permission or inferred sensitive traits from music preferences.

Mitigations in this project:

- No personal listening history is stored.
- API keys are read from environment variables, not source code.
- Logs are ignored by Git.
- Answers include a limitation note when using the small catalog.
- Guardrails check that generated answers name scored recommendations.

## Evaluation

The automated test suite currently passes:

```text
12 passed
```

The reliability harness runs three predefined cases:

```text
workout_energy: PASS
study_focus: PASS
quiet_evening: PASS
Summary: 3 out of 3 cases passed.
```

The most useful failure during testing was the "calm lofi music for coding" case. The parser originally treated "calm" as a low-energy evening signal before noticing the coding/study intent. The fix was to prioritize explicit study/focus words over generic calm language.

## AI Collaboration Reflection

AI assistance was helpful when turning the original recommender into a full applied system. The strongest suggestion was to keep the deterministic recommender as the trusted ranking core and use retrieval plus generation around it, rather than replacing the core with an opaque LLM answer.

One flawed suggestion was relying on generated prose as if it proved correctness. That was not enough. The better approach was to add guardrails, unit tests, and a repeatable evaluation harness that checks expected titles and confidence scores.

## Future Improvements

- Add embedding-based retrieval for larger catalogs.
- Add diversity-aware ranking to reduce repetitive results.
- Add a small UI for interactive demos.
- Add human evaluation forms for side-by-side recommendation quality.
- Compare Gemini output quality against the local fallback across the same test cases.
