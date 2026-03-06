from pathlib import Path

# Paths
ROOT_DIR     = Path(__file__).resolve().parents[1]
FAISS_DIR = str(Path(__file__).resolve().parent / "faiss_index")
DATA_DIR  = str(Path(__file__).resolve().parent / "data")
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"

DOCUMENT_PATH = str(ROOT_DIR / "experiments" / "data" / "gpu_deep_learning.txt")


# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_SETTINGS = {
    "google": {
        "model_name": "gemini-2.5-flash",
        "temperature": 0.5,
        "max_output_tokens": 512,
    },
    "groq": {
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.5,
        "max_output_tokens": 512,
    },
}


# Defaults
DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_K             = 5
DEFAULT_SEARCH_TYPE   = "similarity"
DEFAULT_FETCH_K       = 20
DEFAULT_LAMBDA_MULT   = 0.5
DEFAULT_PROVIDER      = "groq"


# Experiment Group 1: Baseline
BASELINE = [
    {
        "name":           "baseline",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              DEFAULT_K,
        "search_type":    DEFAULT_SEARCH_TYPE,
        "fetch_k":        DEFAULT_FETCH_K,
        "lambda_mult":    DEFAULT_LAMBDA_MULT,
        "provider":       DEFAULT_PROVIDER,
    },
]


# Experiment Group 2: Chunk Size
CHUNK_SIZE_EXPERIMENTS = [
    {
        "name":           "chunk_500",
        "chunk_size":     500,
        "chunk_overlap":  100,
        "k":              DEFAULT_K,
        "search_type":    DEFAULT_SEARCH_TYPE,
        "fetch_k":        DEFAULT_FETCH_K,
        "lambda_mult":    DEFAULT_LAMBDA_MULT,
        "provider":       DEFAULT_PROVIDER,
    },
    {
        "name":           "chunk_1500",
        "chunk_size":     1500,
        "chunk_overlap":  300,
        "k":              DEFAULT_K,
        "search_type":    DEFAULT_SEARCH_TYPE,
        "fetch_k":        DEFAULT_FETCH_K,
        "lambda_mult":    DEFAULT_LAMBDA_MULT,
        "provider":       DEFAULT_PROVIDER,
    },
]


# Experiment Group 3: Retrieval Depth (k)
K_EXPERIMENTS = [
    {
        "name":           "k_3",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              3,
        "search_type":    DEFAULT_SEARCH_TYPE,
        "fetch_k":        DEFAULT_FETCH_K,
        "lambda_mult":    DEFAULT_LAMBDA_MULT,
        "provider":       DEFAULT_PROVIDER,
    },
    {
        "name":           "k_8",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              8,
        "search_type":    DEFAULT_SEARCH_TYPE,
        "fetch_k":        DEFAULT_FETCH_K,
        "lambda_mult":    DEFAULT_LAMBDA_MULT,
        "provider":       DEFAULT_PROVIDER,
    },
]


# Experiment Group 4: Retrieval Algorithm
MMR_EXPERIMENTS = [
    {
        "name":           "search_mmr_balanced",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              DEFAULT_K,
        "search_type":    "mmr",
        "fetch_k":        20,
        "lambda_mult":    0.5,
        "provider":       DEFAULT_PROVIDER,
    },
    {
        "name":           "search_mmr_relevant",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              DEFAULT_K,
        "search_type":    "mmr",
        "fetch_k":        20,
        "lambda_mult":    0.8,
        "provider":       DEFAULT_PROVIDER,
    },
    {
        "name":           "search_mmr_diverse",
        "chunk_size":     DEFAULT_CHUNK_SIZE,
        "chunk_overlap":  DEFAULT_CHUNK_OVERLAP,
        "k":              DEFAULT_K,
        "search_type":    "mmr",
        "fetch_k":        20,
        "lambda_mult":    0.2,
        "provider":       DEFAULT_PROVIDER,
    },
]

# Which groups are to be used
# CONFIGS = (
#     BASELINE
#     + CHUNK_SIZE_EXPERIMENTS
#     + K_EXPERIMENTS
#     + MMR_EXPERIMENTS
# )

CONFIGS = [BASELINE]
