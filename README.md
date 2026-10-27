# PRGI Publication Title Verification & Duplicate Detection System

An intelligent, multi-stage title screening and duplicate detection engine developed for the Press Registrar General of India (PRGI). The system evaluates proposed publication titles against a database of registered titles to enforce regulatory compliance, identify direct/partial duplicates, catch character-level typos, and detect deep semantic similarities.

---

## Key Features & Detection Architecture

The system uses a 4-layer hybrid verification pipeline to balance ultra-fast evaluation times with high semantic accuracy:
Proposed Title Submission
│
▼
[ Layer 1: Rule & Compliance Gate ] ──► Rejects short inputs & PRGI generic root word violations
│
▼
[ Layer 2: Combination & Structural Gate ] ──► Detects titles formed by stitching existing registered titles
│
▼
[ Layer 3: Lexical & Character Candidate Search ]
├── TF-IDF (Char n-grams): Filters 160,000+ database titles down to Top 20 candidate matches
└── Jaro-Winkler: Evaluates spelling variations & triggers Adaptive Early Exit (>= 0.94)
│
▼
[ Layer 4: Deep Semantic AI Evaluation ] ──► SBERT (Sentence-BERT) evaluates contextual similarity

### Integrated Algorithms

| Algorithm | Role in Pipeline | Primary Benefit |
| :--- | :--- | :--- |
| **Rule-Based Filtering** | Early Exit Gate | Instantly catches malformed titles, generic root words, and single-word violations. |
| **TF-IDF (Char $n$-grams)** | Candidate Retrieval | Uses sub-word character $n$-grams (range 2–4) to quickly filter candidates across large datasets. |
| **Jaro-Winkler Distance** | Lexical Matching | Measures string edit distance to detect typos and spelling modifications (e.g., *Namaskar* vs. *Namascar*). |
| **SBERT (`sentence-transformers`)** | Semantic Matching | Encodes contextual meaning into 384-dimensional dense vectors to identify synonym-based duplicates (e.g., *Journal of Cancer* vs. *Oncology Periodical*). |

---

## Verification Logic & Rules Summary

1. **Input Validation**: Rejects titles shorter than 3 characters or containing invalid formatting.
2. **Generic Root Words Rule**: Rejects single-word titles belonging to restricted root sets (`times`, `news`, `manthan`, `darpan`, etc.).
3. **Exact Match Check**: Rejects 100% duplicate titles immediately.
4. **Prohibited Combination Check**: Flags titles formed by combining two or more existing registered titles without substantial change.
5. **Adaptive Early Exit**: If character-level similarity (Jaro-Winkler) reaches $\ge 94\%$, the system flags high spelling similarity and exits early without running deep SBERT embeddings (saving compute).
6. **Domain Keyword Conflict**: Flags high surface matches containing conflicting domain terms (e.g., *detection* vs. *prevention*).
7. **Multi-Signal Similarity Threshold**: Flags titles with combined similarity scores meeting or exceeding the threshold ($\ge 0.85$).

---

## Setup & Installation

### Prerequisites

* **Python 3.11** or **Python 3.13** *(Recommended for PyTorch and Sentence-Transformers stability)*
* VS Code (optional, but recommended)

### 1. Install Dependencies

Install the required Python packages:

```bash
python -m pip install pandas numpy scikit-learn rapidfuzz sentence-transformers lxml

├── duplicate_check.py           # Main evaluation engine and execution logic
├── publictaion_titles_real.xls  # Source database containing existing publication titles
└── README.md                    # Project documentation

