import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import JaroWinkler

# Import SBERT with graceful fallback if unavailable
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


# 1. Load dataset
tables = pd.read_html("publictaion_titles_real.xls")
df = tables[0].rename(columns={1: "publication_title"})

df = df.dropna(subset=["publication_title"]).copy()
df["publication_title"] = df["publication_title"].astype(str).str.strip()
df = df[df["publication_title"] != ""]
df = df.drop_duplicates(subset=["publication_title"]).reset_index(drop=True)


# 2. Clean titles
def clean_title(title):
    title = str(title).lower()
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


df["cleaned_title"] = df["publication_title"].apply(clean_title)
df = df.drop_duplicates(subset=["cleaned_title"]).reset_index(drop=True)

if df.empty:
    raise ValueError("No publication titles were found in the dataset.")


# 3. PRGI Generic Root Words & Domain Constants
GENERIC_ROOT_WORDS = {
    'manthan', 'darpan', 'chronology', 'inspire', 'success', 'khulasa',
    'rahasya', 'katha', 'herald', 'batkahi', 'nukkad', 'crossroads',
    'times', 'news', 'daily', 'express', 'mirror', 'tribune'
}

CONTRASTIVE_PAIRS = [
    {'detection', 'prevention'},
    {'classification', 'prevention'},
    {'diagnosis', 'treatment'}
]


# 4. Train the TF-IDF vectorizer (Character n-grams)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
tfidf_matrix = vectorizer.fit_transform(df["cleaned_title"])

# Load SBERT model lazy-loaded
sbert_model = SentenceTransformer('all-MiniLM-L6-v2') if SBERT_AVAILABLE else None


# 5. Title combination search
def find_title_combinations(cleaned_query):
    query_words = cleaned_query.split()
    matches = []

    for _, row in df.iterrows():
        existing_words = row["cleaned_title"].split()
        length = len(existing_words)
        if not existing_words:
            continue

        for start in range(len(query_words) - length + 1):
            end = start + length
            if query_words[start:end] == existing_words:
                matches.append({
                    "title": row["publication_title"],
                    "start": start,
                    "end": end,
                    "length": length
                })

    matches.sort(key=lambda item: item["length"], reverse=True)
    used_positions, used_titles = set(), set()

    for match in matches:
        positions = set(range(match["start"], match["end"]))
        if positions.isdisjoint(used_positions):
            used_positions.update(positions)
            used_titles.add(match["title"])

    return list(used_titles)


# 6. Main Adaptive Screening Function with Rejection Recommendations
def evaluate_user_title(user_title, threshold=0.85):
    cleaned_query = clean_title(user_title)

    print("=" * 70)
    print(f"PROPOSED TITLE FOR EVALUATION: '{user_title}'")
    print("=" * 70)

    # Step 1: Input Validation
    if not cleaned_query or len(cleaned_query) < 3:
        print("RECOMMENDATION: REJECT - INVALID INPUT")
        print("VIOLATION REASON: Input is empty, malformed, or too short to constitute a valid publication title.")
        print("=" * 70)
        return

    # Step 2: PRGI Root/Generic Word Rule
    query_tokens = cleaned_query.split()
    if len(query_tokens) == 1 and query_tokens[0] in GENERIC_ROOT_WORDS:
        print("RECOMMENDATION: REJECT - DIRECT GUIDELINE VIOLATION")
        print(f"VIOLATION REASON: PRGI Guideline Rule 5(3) - Proposed title consists solely of generic root word '{query_tokens[0]}'.")
        print("=" * 70)
        return

    # Step 3: Exact Match Check
    exact_matches = df[df["cleaned_title"] == cleaned_query]
    if not exact_matches.empty:
        matched_title = exact_matches.iloc[0]["publication_title"]
        print("RECOMMENDATION: REJECT - EXACT TITLE DUPLICATE")
        print(f"VIOLATION REASON: Identical title already registered in database: '{matched_title}'")
        print("=" * 70)
        return

    # Step 4: Title Combination Check
    component_titles = find_title_combinations(cleaned_query)
    if len(component_titles) >= 2:
        print("RECOMMENDATION: FLAG FOR REVIEW - PROHIBITED TITLE COMBINATION")
        print("VIOLATION REASON: Combines multiple existing registered titles without substantial change.")
        print("EXISTING COMPONENT TITLES DETECTED:")
        for title in component_titles:
            print(f"  - {title}")
        print("=" * 70)
        return

    # Step 5: TF-IDF Candidate Retrieval
    query_vector = vectorizer.transform([cleaned_query])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    res = pd.DataFrame({
        "Existing Title": df["publication_title"],
        "cleaned_title": df["cleaned_title"],
        "TF-IDF Score": tfidf_scores
    }).sort_values("TF-IDF Score", ascending=False).head(20).copy()

    if res.empty or res["TF-IDF Score"].max() < 0.05:
        print("RECOMMENDATION: ACCEPT - NO SIMILARITY DETECTED")
        print("NOTE: No existing registered titles meet minimum candidate similarity threshold.")
        print("=" * 70)
        return

    # Step 6: Jaro-Winkler Similarity
    res["Jaro-Winkler Score"] = res["cleaned_title"].apply(
        lambda title: JaroWinkler.similarity(cleaned_query, title)
    )
    
    top_candidate = res.sort_values("Jaro-Winkler Score", ascending=False).iloc[0]
    best_title = top_candidate["Existing Title"]
    best_cleaned = top_candidate["cleaned_title"]
    jw_score = top_candidate["Jaro-Winkler Score"]

    # Conflict Check (Contrastive terms)
    query_set, best_set = set(query_tokens), set(best_cleaned.split())
    has_conflict = any(
        pair.issubset(query_set.union(best_set)) and 
        not pair.issubset(query_set) and 
        not pair.issubset(best_set) 
        for pair in CONTRASTIVE_PAIRS
    )

    # Adaptive Early Exit Check
    EARLY_EXIT_THRESHOLD = 0.94
    if jw_score >= EARLY_EXIT_THRESHOLD and not has_conflict:
        print("RECOMMENDATION: FLAG FOR REVIEW - HIGH SPELLING SIMILARITY")
        print(f"MATCHED EXISTING TITLE : '{best_title}'")
        print(f"SPELLING SIMILARITY    : {jw_score:.3f} (Jaro-Winkler)")
        print("VIOLATION REASON       : Strong character match indicates potential typo or minor spelling variant.")
        print("SBERT EXECUTED         : False (Skipped via Adaptive Early Exit)")
        print("=" * 70)
        return

    # Step 7: Semantic SBERT Comparison (if early exit didn't trigger)
    sbert_score = 0.0
    if sbert_model:
        emb = sbert_model.encode([cleaned_query, best_cleaned])
        sbert_score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    final_score = max(jw_score, sbert_score)

    if has_conflict:
        print("RECOMMENDATION: FLAG FOR REVIEW - DOMAIN KEYWORD CONFLICT")
        print(f"MATCHED EXISTING TITLE : '{best_title}'")
        print(f"Jaro-Winkler Score     : {jw_score:.3f}")
        if sbert_model:
            print(f"SBERT Score            : {sbert_score:.3f}")
        print("VIOLATION REASON       : High surface match detected, but title contains conflicting domain/scope terms.")
    elif final_score >= threshold:
        print("RECOMMENDATION: FLAG FOR REVIEW - HIGH MULTI-SIGNAL SIMILARITY")
        print(f"MATCHED EXISTING TITLE : '{best_title}'")
        print(f"Jaro-Winkler Score     : {jw_score:.3f}")
        if sbert_model:
            print(f"SBERT Score            : {sbert_score:.3f}")
        print("VIOLATION REASON       : Combined similarity exceeds allowed threshold. Deceptively similar to an existing title.")
    else:
        print("RECOMMENDATION: ACCEPT - LIKELY UNIQUE TITLE")
        print(f"CLOSEST CANDIDATE      : '{best_title}' (Max Similarity: {final_score:.3f})")
        print("NOTE                   : Similarity score is below rejection threshold.")

    print("=" * 70)

# Test example
evaluate_user_title("The Cybernetics Journal of Robotics")