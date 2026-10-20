#!/usr/bin/env python3

import json
import re

import ollama
import requests

MODEL = "qwen3-local:latest"
REWORK_MODEL = "qwen-coder:latest"
EVAL_MODEL = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest"
CATEGORIES = [
    "originality", "imagery", "rhyme", "rhythm",
    "emotion", "coherence", "fidelity",
]
# Fidelity (topic + user constraints) carries by far the most weight
WEIGHTS = {
    "originality": 1,
    "imagery": 1,
    "rhyme": 2,
    "rhythm": 1,
    "emotion": 1,
    "coherence": 1,
    "fidelity": 5,
}

EMBED_MODEL = "nomic-embed-text-v1:latest"
CHROMA_BASE = (
    "http://localhost:8000/api/v2/tenants/default_tenant"
    "/databases/default_database"
)
CHROMA_COLLECTION = "ehi"
CHROMA_RESULTS = 5

TARGET_SCORE = 8.5
MAX_ITERATIONS = 5
EVAL_RETRIES = 3


def call_llm(system, prompt, temperature=0.8, model=MODEL, json_mode=False):
    response = ollama.chat(
        model=model,
        format="json" if json_mode else None,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": temperature,
        },
    )

    return response["message"]["content"]


# ------------------------------------------------------
# GENERATOR
# ------------------------------------------------------

def retrieve_context(query, n_results=CHROMA_RESULTS):
    try:
        embedding = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]

        collection = requests.get(
            f"{CHROMA_BASE}/collections/{CHROMA_COLLECTION}", timeout=30
        )
        collection.raise_for_status()

        response = requests.post(
            f"{CHROMA_BASE}/collections/{collection.json()['id']}/query",
            json={
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["documents"],
            },
            timeout=60,
        )
        response.raise_for_status()
        documents = response.json()["documents"][0]
    except Exception as e:
        print(f"Warning: could not retrieve context from Chroma: {e}")
        return ""

    # Drop the <document_metadata> header added at ingestion time
    cleaned = [
        re.sub(r"<document_metadata>.*?</document_metadata>", "", d, flags=re.S).strip()
        for d in documents
    ]

    return "\n\n---\n\n".join(c for c in cleaned if c)


def constraint_text(indication):
    if not indication:
        return ""
    return (
        "Respect absolutely and totally the following constraint "
        f"from the use: {indication}. You cannot do something that "
        "goes against the constraint."
    )


def generate_poem(topic, indication="", context=""):
    system = """
You are a poet.
Your language is Spanish.

Write high quality poetry.
Avoid clichés.
Use strong imagery.
"""

    prompt = f"""
Topic:
{topic}

Reference material:
{context}

Write a poem.

{constraint_text(indication)}
"""

    return call_llm(system, prompt, 0.9)


# ------------------------------------------------------
# EVALUATOR
# ------------------------------------------------------

def requires_rhyme(indication):
    text = indication or ""
    # "sin rima", "no rimes", "without rhyme" are negations, not requirements
    text = re.sub(
        r"\b(sin|no|without|nunca|ni)\s+(\w+\s+)?(rim\w*|rhym\w*)",
        "", text, flags=re.I,
    )
    return bool(re.search(r"\b(rim\w*|rhym\w*)", text, re.I))


def evaluate_poem(poem, topic, indication=""):

    rhyme = requires_rhyme(indication)

    constraints = (
        f"These are the constraints that the poem must respect: {indication}"
        if indication else ""
    )

    rhyme_rule = (
        "You are extremely strict about rhyme: penalize heavily any verse "
        "that does not rhyme properly, any inconsistent rhyme scheme, and "
        "any forced or imperfect rhyme. "
        if rhyme else
        "Do NOT evaluate rhyme; the user did not ask for it. "
    )
    rhyme_category = "- rhyme (be very strict)\n" if rhyme else ""
    rhyme_json = (
        '  "rhyme": {"reason": "...", "score": <number 0-10>},\n'
        if rhyme else ""
    )

    system = f"""
You are a harsh literary critic.
{rhyme_rule}You are equally strict about fidelity: the poem must
stay on the requested topic and obey every user constraint.
Always include specific criticisms (cite the offending verses).

Return ONLY a single valid JSON object and nothing else:
no markdown, no code fences, no comments, no text before or after it.
Every "score" must be a plain number between 0 and 10 (no quotes, no
text, no "/10"). "criticisms" must be an array of strings.
"""

    prompt = f"""
Evaluate this poem.

Requested topic: {topic}

{constraints}

Score each category from 0 to 10:

- originality
- imagery
{rhyme_category}- rhythm
- emotional impact
- coherence
- fidelity (how faithful the poem is to the requested topic and to the user constraints; any violation of a constraint must score very low)

Return one object per category. Write the "reason" FIRST (one short
sentence quoting the poem), and only then the "score" that follows from
that reason. Use the full 0-10 range; give 0 only if the quality is
completely absent. Judge this specific poem; do not copy example values.

{{
  "originality": {{"reason": "...", "score": <number 0-10>}},
  "imagery": {{"reason": "...", "score": <number 0-10>}},
{rhyme_json}  "rhythm": {{"reason": "...", "score": <number 0-10>}},
  "emotion": {{"reason": "...", "score": <number 0-10>}},
  "coherence": {{"reason": "...", "score": <number 0-10>}},
  "fidelity": {{"reason": "...", "score": <number 0-10>}},
  "criticisms": ["...", "..."]
}}

Poem:

{poem}
"""

    for attempt in range(EVAL_RETRIES):
        result = call_llm(system, prompt, 0.3, EVAL_MODEL, json_mode=True)
        evaluation = parse_evaluation(result, rhyme)

        if evaluation is not None:
            return evaluation

        print(f"  (evaluation not parseable, attempt {attempt + 1}/{EVAL_RETRIES})")

    return {
        "score": 0,
        "criticisms": ["evaluation failed"]
    }


def parse_evaluation(text, rhyme=True):
    """Extract and normalize the evaluation JSON; None if unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None

    if not isinstance(data, dict):
        return None

    evaluation = {}
    found = 0

    for category in CATEGORIES:
        if category == "rhyme" and not rhyme:
            evaluation[category] = None
            continue
        raw = data.get(category)
        if isinstance(raw, dict):
            raw = raw.get("score")
        value = to_score(raw)
        if value is not None:
            found += 1
        evaluation[category] = value

    if found == 0:
        return None

    # Missing categories are ignored instead of counted as 0
    weighted = [
        (evaluation[c], w) for c, w in WEIGHTS.items()
        if evaluation[c] is not None
    ]
    evaluation["score"] = round(
        sum(v * w for v, w in weighted) / sum(w for _, w in weighted), 2
    )

    criticisms = data.get("criticisms", [])
    if isinstance(criticisms, str):
        criticisms = [criticisms]
    elif not isinstance(criticisms, list):
        criticisms = []
    evaluation["criticisms"] = [str(c) for c in criticisms]

    return evaluation


def to_score(value):
    """Coerce numbers, '7', '7/10', '7,5' into a float in [0, 10]."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"\d+(?:[.,]\d+)?", str(value))
        if not match:
            return None
        number = float(match.group().replace(",", "."))
    return max(0.0, min(10.0, number))


# ------------------------------------------------------
# REWORKER
# ------------------------------------------------------

def rework_poem(poem, evaluation):

    system = """
You are a poetry editor.
Improve the poem while preserving the core intent.
"""

    criticisms = "\n".join(
        f"- {x}" for x in evaluation.get("criticisms", [])
    )

    prompt = f"""
Current poem:

{poem}

Criticisms:

{criticisms}

Improve the poem.
"""

    return call_llm(system, prompt, 0.7, REWORK_MODEL)


# ------------------------------------------------------
# FINAL POLISH
# ------------------------------------------------------

def polish_poem(poem, indication=""):

    system = """
You are a final poetry editor.

Improve rhythm, line breaks and punctuation.

Do not change meaning.
"""

    prompt = f"""
{poem}

{constraint_text(indication)}
"""

    return call_llm(system, prompt, 0.2)


# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------

def create_poem(topic, indication=""):

    context = retrieve_context(topic)

    poem = generate_poem(topic, indication, context)

    best_poem = poem
    best_score = 0

    for iteration in range(MAX_ITERATIONS):

        evaluation = evaluate_poem(poem, topic, indication)

        score = evaluation["score"]

        print(
            f"Iteration {iteration + 1}: "
            f"score={score}"
        )

        for category in CATEGORIES:
            print(f"  {category}: {evaluation.get(category, 'n/a')}")

        if score > best_score:
            best_score = score
            best_poem = poem

        if score >= TARGET_SCORE:
            print("Target reached")
            break

        poem = rework_poem(poem, evaluation)

    final_poem = polish_poem(best_poem, indication)

    return final_poem, best_score


# ------------------------------------------------------
# RUN
# ------------------------------------------------------

if __name__ == "__main__":

    topic = input("¿Sobre qué quieres escribir el poema? ").strip()

    if not topic:
        raise SystemExit("No se indicó ningún tema.")

    indication = input("¿Alguna indicación? ").strip()

    poem, score = create_poem(topic, indication)

    print()
    print("=" * 80)
    print(f"FINAL SCORE: {score}")
    print("=" * 80)
    print()
    print(poem)