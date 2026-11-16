# scripts/04_search.py

import os
import numpy as np
import pandas as pd
import torch 
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


# Load .env file
load_dotenv() 

# Configuration   
INDEX_NAME = "arxiv-papers"

MODEL_NAME = "allenai/specter2_base"

QUERY_ADAPTER = "allenai/specter2_adhoc_query"
DOCUMENT_ADAPTER = "allenai/specter2"

PARQUET_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"

TOP_K = 5
VECTOR_DIM = 768
MAX_LENGTH = 512 

 
# connect to PINECONE 
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env")

print("Connecting to Pinecone...")

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

print(f"Connected to index: {INDEX_NAME}") 

 
# Load the dataset and local embeddings for validation and comparison. 
print("\nLoading dataset...")

df = pd.read_parquet(PARQUET_FILE)
df = df.reset_index(drop=True)

print(f"Dataset records: {len(df)}")  

print("\nLoading local embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")

if embeddings.ndim != 2:
    raise ValueError("Embeddings must be a 2D array.")

if embeddings.shape[0] != len(df):
    raise ValueError(
        f"Dataset and embeddings do not match: "
        f"{len(df)} records vs {embeddings.shape[0]} embeddings"
    )

if embeddings.shape[1] != VECTOR_DIM:
    raise ValueError(
        f"Expected {VECTOR_DIM} dimensions, "
        f"got {embeddings.shape[1]}"
    )

if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain NaN or infinite values.")

print("Dataset and embeddings validated successfully.") 

 
# Select device 
if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}") 

 
# Loads the SPECTER2 model with two adapters:
# - adhoc_query for query embeddings
# - proximity for document embeddings 
print("\nLoading SPECTER2 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoAdapterModel.from_pretrained(MODEL_NAME)

# Query adapter
query_adapter_name = model.load_adapter(
    QUERY_ADAPTER,
    source="hf",
    load_as="specter2_adhoc_query"
)

# Document adapter
document_adapter_name = model.load_adapter(
    DOCUMENT_ADAPTER,
    source="hf",
    load_as="specter2_proximity"
)

model.to(device)
model.eval()

print("Query adapter:", query_adapter_name)
print("Document adapter:", document_adapter_name)

print("SPECTER2 loaded successfully.") 

 
# Encode a search query using the SPECTER2 adhoc_query adapter  
def encode_query(query: str) -> np.ndarray:
    """
    Encode a search query using the SPECTER2
    adhoc_query adapter.

    Query input is short raw text,
    not title + abstract.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model.set_active_adapters("specter2_adhoc_query")

    inputs = tokenizer(
        [query.strip()],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_token_type_ids=False
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        # CLS token representation
        query_embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalization
        query_embedding = torch.nn.functional.normalize(
            query_embedding,
            p=2,
            dim=1
        )

    return (
        query_embedding
        .cpu()
        .numpy()[0]
        .astype(np.float32)
    ) 

 
# Display the top-k results from Pinecone with metadata
def display_results(results, title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    matches = results.matches

    if not matches:
        print("No results found.")
        return

    for rank, match in enumerate(matches, start=1):

        metadata = match.metadata or {}

        print(f"\nRank: {rank}")
        print(f"ID: {match.id}")
        print(f"Score: {match.score:.4f}")

        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Category: {metadata.get('category', 'N/A')}")
        print(f"Year: {metadata.get('year', 'N/A')}")

        abstract = metadata.get("abstract", "N/A")

        print("Abstract:", abstract[:250]) 

 
# Semantic search is performed using the SPECTER2 adhoc_query adapter for query embeddings and the document adapter for stored embeddings in Pinecone 
def semantic_search(query, top_k=TOP_K, filters=None):
    """
    Search scientific papers in Pinecone.

    Uses:
    - adhoc_query adapter for query embeddings
    - document adapter for previously stored embeddings
    """

    query_vector = encode_query(query)

    query_args = {
        "vector": query_vector.tolist(),
        "top_k": top_k,
        "include_metadata": True
    }

    if filters is not None:
        query_args["filter"] = filters

    results = index.query(**query_args)

    return results 

 
# Local similarity metrics  
def compare_metrics(query):
    """
    Compare:
    - Cosine similarity
    - Dot product
    - L2 distance

    Uses local document embeddings.
    """

    print("\n" + "=" * 80)
    print("LOCAL SIMILARITY COMPARISON")
    print("=" * 80)

    query_vector = encode_query(query)

    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    document_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    query_norm = np.linalg.norm(query_vector)

    cosine_scores = (
        embeddings @ query_vector
    ) / (
        document_norms * query_norm + 1e-10
    )

    # --------------------------------------------------------
    # DOT PRODUCT
    # --------------------------------------------------------

    dot_scores = embeddings @ query_vector

    # --------------------------------------------------------
    # L2 DISTANCE
    # --------------------------------------------------------

    l2_scores = np.linalg.norm(
        embeddings - query_vector,
        axis=1
    )

    # --------------------------------------------------------
    # RANKINGS
    # --------------------------------------------------------

    rankings = {
        "Cosine Similarity": np.argsort(-cosine_scores)[:TOP_K],
        "Dot Product": np.argsort(-dot_scores)[:TOP_K],
        "L2 Distance": np.argsort(l2_scores)[:TOP_K]
    }

    score_arrays = {
        "Cosine Similarity": cosine_scores,
        "Dot Product": dot_scores,
        "L2 Distance": l2_scores
    }

    top_results = {}

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    for metric, indices in rankings.items():

        print("\n" + "-" * 80)
        print(metric)
        print("-" * 80)

        top_results[metric] = []

        for rank, idx in enumerate(indices, start=1):

            row = df.iloc[idx]
            score = score_arrays[metric][idx]

            title = row["title"]

            top_results[metric].append(title)

            print(f"\nRank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"Title: {title}")
            print(f"Category: {row['category']}")
            print(f"Year: {row['year']}")

   
    # Compare the top-5 results  
    print("\n" + "=" * 80)
    print("TOP-5 RANKING COMPARISON")
    print("=" * 80)

    cosine_set = set(top_results["Cosine Similarity"])
    dot_set = set(top_results["Dot Product"])
    l2_set = set(top_results["L2 Distance"])

    overlap_cosine_dot = len(cosine_set & dot_set)
    overlap_cosine_l2 = len(cosine_set & l2_set)
    overlap_dot_l2 = len(dot_set & l2_set)

    print(
        f"\nCosine vs Dot Product: "
        f"{overlap_cosine_dot}/5 common articles"
    )

    print(
        f"Cosine vs L2 Distance: "
        f"{overlap_cosine_l2}/5 common articles"
    )

    print(
        f"Dot Product vs L2 Distance: "
        f"{overlap_dot_l2}/5 common articles"
    )
 
 
# Experiments   
def main():

    query = "galaxy evolution star formation properties of galaxies" 

    print("\n" + "=" * 80)
    print("ARXIV SEMANTIC SEARCH")
    print("=" * 80) 

    results = semantic_search(query)

    display_results(
        results,
        "A. SEMANTIC SEARCH - TOP 5"
    )

    # --------------------------------------------------------
    # B. RECENT REINFORCEMENT LEARNING PAPERS
    # --------------------------------------------------------

    recent_filter = {
        "$and": [
            {"year": {"$gte": 2021}},
            {"category": {"$eq": "astro-ph"}}
        ]
    }

    results = semantic_search(
        "galaxy evolution",
        filters=recent_filter
    )

    display_results(
        results,
        "B. FILTERED SEARCH - RECENT ASTROPHICS PAPERS"
    )

    # --------------------------------------------------------
    # C. OLDER PAPERS BEFORE 2015
    # --------------------------------------------------------

    old_filter = {
        "year": {"$lt": 2015}
    }

    results = semantic_search(
        query,
        filters=old_filter
    )

    display_results(
        results,
        "C. FILTERED SEARCH - PAPERS BEFORE 2015"
    )

    # --------------------------------------------------------
    # D. COMPARE LOCAL SIMILARITY METRICS
    # --------------------------------------------------------

    compare_metrics(query)

    print("\n" + "=" * 80)
    print("SEARCH EXPERIMENTS COMPLETED")
    print("=" * 80) 

if __name__ == "__main__":
    main()
