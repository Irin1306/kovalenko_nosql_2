# scripts/05_chunking.py 

import os
import re
import time
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from pinecone import Pinecone, ServerlessSpec


# Load .env file
load_dotenv() 

# Configuration
INPUT_FILE = "data/arxiv_subset.parquet"

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

VECTOR_DIM = 768
MAX_LENGTH = 512

MAX_ARTICLES = 30

# Pinecone configuration
CLOUD = "aws"
REGION = "us-east-1"

# Chunking configuration
FIXED_CHUNK_SIZE = 120
FIXED_OVERLAP = 20

SEMANTIC_CHUNK_SIZE = 120

# Processing configuration
EMBEDDING_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 100

TOP_K = 5

TEST_QUERIES = [ 
    "How is dark matter distributed in the Milky Way?", 
    "What are the methods for detecting extrasolar planets?", 
    "What is the relationship between active galactic nuclei and black hole accretion?" 
]

 
# Check API key
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError(
        "PINECONE_API_KEY not found. Check your .env file."
    ) 

 
# Select device  
def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

 
# Load dataset   
def load_dataset():

    print("\nLoading dataset...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. "
            "Run scripts/01_prepare_data.py first."
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Loaded {len(df)} articles.")

    # Required columns from script 01
    required_columns = [
        "id",
        "title",
        "abstract",
        "year",
        "category",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}. "
                f"Available columns: {list(df.columns)}"
            )

    # Clean data
    df["abstract"] = (
        df["abstract"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .fillna("unknown")
        .astype(str)
    )

    # Remove empty abstracts
    df = df[
        (df["abstract"] != "")
        & (df["title"] != "")
    ].copy()

    if df.empty:
        raise ValueError("No valid articles found.")

    # Select longest abstracts
    print(
        f"\nSelecting {MAX_ARTICLES} longest abstracts..."
    )

    df["word_count"] = (
        df["abstract"].str.split().str.len()
    )

    df = (
        df.sort_values(
            "word_count",
            ascending=False
        )
        .head(MAX_ARTICLES)
        .reset_index(drop=True)
    )

    print(f"Selected {len(df)} articles.")

    print(
        "Average abstract length: "
        f"{df['word_count'].mean():.1f} words"
    )

    return df 

 
# Text splitting
def split_sentences(text):

    text = re.sub(r"\s+", " ", text).strip()

    # Split after sentence-ending punctuation
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9\[])',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ] 

 
# Fixed-size chunking  
def fixed_size_chunking(
    text,
    chunk_size=FIXED_CHUNK_SIZE,
    overlap=FIXED_OVERLAP
):

    words = text.split()

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks 

 
# Semantic chunking  
def semantic_chunking(
    text,
    max_words=SEMANTIC_CHUNK_SIZE
):

    sentences = split_sentences(text)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # A sentence longer than max_words must be split
        if sentence_word_count > max_words:

            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_word_count = 0

            # Split oversized sentence to avoid oversized chunks
            for start in range(
                0,
                sentence_word_count,
                max_words
            ):

                part = sentence_words[
                    start:start + max_words
                ]

                chunks.append(" ".join(part))

            continue

        # Add sentence if it fits
        if (
            current_word_count + sentence_word_count
            <= max_words
        ):

            current_sentences.append(sentence)
            current_word_count += sentence_word_count

        else:

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = [sentence]
            current_word_count = sentence_word_count

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks 

 
# SPECTER2 model loading  
def load_model():

    print("\nLoading SPECTER2 model...")

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoAdapterModel.from_pretrained(
        MODEL_NAME
    )

    # Load proximity adapter
    adapter_name = model.load_adapter(
        ADAPTER_NAME,
        source="hf"
    )

    # Explicitly activate adapter
    model.set_active_adapters(adapter_name)

    print("Loaded adapter:", adapter_name)
    print("Active adapter:", model.active_adapters)

    if model.active_adapters is None:
        raise RuntimeError(
            "SPECTER2 adapter is not active!"
        )

    model.to(device)
    model.eval()

    print("SPECTER2 loaded successfully.")

    return tokenizer, model, device  

 
# Tokenize chunks and check their lengths against the model's maximum input length 
def validate_chunks(chunks, tokenizer):

    print("\nValidating chunk token lengths...")

    token_lengths = []

    for chunk in chunks:

        tokens = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False
        )

        token_count = len(tokens["input_ids"])

        token_lengths.append(token_count)

        if token_count > MAX_LENGTH:

            raise ValueError(
                f"Chunk exceeds {MAX_LENGTH} tokens: "
                f"{token_count} tokens."
            )

    if token_lengths:

        print(
            f"Maximum chunk length: "
            f"{max(token_lengths)} tokens"
        )

        print(
            f"Average chunk length: "
            f"{np.mean(token_lengths):.1f} tokens"
        )

    return token_lengths 

 
# Generate embeddings   
def generate_embeddings(
    chunks,
    tokenizer,
    model,
    device
):

    if not chunks:
        raise ValueError("No chunks provided.")

    print("\nGenerating embeddings...")

    all_embeddings = []

    total_batches = (
        len(chunks) + EMBEDDING_BATCH_SIZE - 1
    ) // EMBEDDING_BATCH_SIZE

    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(chunks),
                EMBEDDING_BATCH_SIZE
            ),
            total=total_batches,
            desc="Embedding batches"
        ):

            batch = chunks[
                start:start + EMBEDDING_BATCH_SIZE
            ]

            inputs = tokenizer(
                batch,
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

            outputs = model(**inputs)

            # Extract CLS token embeddings
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

            # L2 normalization
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)

    if embeddings.shape != (
        len(chunks),
        VECTOR_DIM
    ):

        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    print(f"Embeddings shape: {embeddings.shape}")

    print(
        f"First vector norm: "
        f"{np.linalg.norm(embeddings[0]):.6f}"
    )

    return embeddings 

 
#Build records for Pinecone upload, including metadata 
def build_chunk_records(df, strategy):

    print(f"\nGenerating {strategy} chunks...")

    records = []

    chunk_function = (
        fixed_size_chunking
        if strategy == "fixed"
        else semantic_chunking
    )

    for _, article in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{strategy} chunking"
    ):

        chunks = chunk_function(
            article["abstract"]
        )

        for chunk_number, chunk_text in enumerate(
            chunks
        ):

            record_id = (
                f"{article['id']}-{strategy}-{chunk_number}"
            )

            records.append({
                "id": record_id,
                "arxiv_id": str(article["id"]),
                "title": str(article["title"]),
                "text": chunk_text,
                "chunk_number": int(chunk_number),
                "year": int(article["year"]),
                "category": str(article["category"]),
            })

    print(
        f"Generated {len(records)} "
        f"{strategy} chunks."
    )

    return records  

 
# PINECONE initialization  
def initialize_pinecone():

    print("\nConnecting to Pinecone...")

    pc = Pinecone(api_key=api_key)

    return pc  
 
 
# Create index   
def create_clean_index(pc, index_name):

    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]

    if index_name in existing_indexes:

        print(
            f"\nDeleting old index: {index_name}"
        )

        pc.delete_index(index_name)

        # Wait until deletion completes
        while index_name in [
            index.name
            for index in pc.list_indexes()
        ]:

            time.sleep(2)

    print(f"\nCreating index: {index_name}")

    pc.create_index(
        name=index_name,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )

    # Wait until index is ready
    while not pc.describe_index(
        index_name
    ).status["ready"]:

        time.sleep(2)

    print(
        f"Index {index_name} created successfully."
    )

    return pc.Index(index_name) 

 
# PINECONE vectors  
def prepare_vectors(records, embeddings):

    if len(records) != len(embeddings):
        raise ValueError(
            "Records and embeddings count mismatch."
        )

    vectors = []

    for record, embedding in zip(
        records,
        embeddings
    ):

        metadata = {
            "arxiv_id": record["arxiv_id"],
            "title": record["title"],
            "text": record["text"],
            "chunk_number": record["chunk_number"],
            "year": record["year"],
            "category": record["category"],
        }

        vectors.append({
            "id": record["id"],
            "values": embedding.tolist(),
            "metadata": metadata,
        })

    return vectors 

 
# Upload vectors in batches to Pinecone index 
def upload_vectors(index, vectors):

    print("\nUploading chunks to Pinecone...")

    total_batches = (
        len(vectors) + UPLOAD_BATCH_SIZE - 1
    ) // UPLOAD_BATCH_SIZE

    for start in tqdm(
        range(
            0,
            len(vectors),
            UPLOAD_BATCH_SIZE
        ),
        total=total_batches,
        desc="Uploading"
    ):

        batch = vectors[
            start:start + UPLOAD_BATCH_SIZE
        ]

        index.upsert(
            vectors=batch
        )

    print("\nUpload completed.")

    stats = index.describe_index_stats()

    print(
        "Total vectors in index: "
        f"{stats.total_vector_count}"
    )

    if stats.total_vector_count != len(vectors):
        raise ValueError(
            "Uploaded vector count does not match "
            "the expected number of vectors."
        ) 

 
# Process either fixed-size or semantic chunking strategy: generate chunks, create embeddings, and upload 
def process_strategy(
    pc,
    index_name,
    records,
    tokenizer,
    model,
    device
):

    print("\n" + "=" * 70)
    print(index_name.upper())
    print("=" * 70)

    chunks = [
        record["title"] + " [SEP] " + record["text"]
        for record in records
    ]

    # Validate all chunks against tokenizer limit
    validate_chunks(
        chunks,
        tokenizer
    )

    # Generate embeddings
    embeddings = generate_embeddings(
        chunks,
        tokenizer,
        model,
        device
    )

    # Prepare vectors and metadata
    vectors = prepare_vectors(
        records,
        embeddings
    )

    # Create clean Pinecone index
    index = create_clean_index(
        pc,
        index_name
    )

    # Upload vectors
    upload_vectors(
        index,
        vectors
    )

    return index 

 
# SEARCH CHUNKS  
def search_chunks(
    index,
    query,
    tokenizer,
    model,
    device,
    top_k=TOP_K
):

    # Embed query using the same SPECTER2 model
    with torch.inference_mode():

        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        query_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        query_embedding = (
            torch.nn.functional.normalize(
                query_embedding,
                p=2,
                dim=1
            )
        )

        query_vector = (
            query_embedding[0]
            .cpu()
            .numpy()
            .astype(np.float32)
            .tolist()
        )

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return results 

 
# Print search results 
def print_search_results(
    results,
    strategy
):

    print("\n" + "-" * 70)
    print(f"{strategy.upper()} SEARCH")
    print("-" * 70)

    matches = results.matches

    if not matches:
        print("No results found.")
        return

    for rank, match in enumerate(
        matches,
        start=1
    ):

        metadata = match.metadata or {}

        print(f"\nRank: {rank}")
        print(f"Score: {match.score:.4f}")

        print(
            f"Article: "
            f"{metadata.get('title', 'Unknown')}"
        )

        print(
            f"arXiv ID: "
            f"{metadata.get('arxiv_id', 'Unknown')}"
        )

        print(
            f"Year: "
            f"{metadata.get('year', 'Unknown')}"
        )

        print(
            f"Category: "
            f"{metadata.get('category', 'Unknown')}"
        )

        print(
            f"Chunk number: "
            f"{metadata.get('chunk_number', 'Unknown')}"
        )

        chunk_text = metadata.get(
            "text",
            ""
        )

        print(
            f"Text: {chunk_text[:350]}"
        ) 

 
# Compare search results between fixed-size and semantic chunking strategies 
def compare_search(
    fixed_index,
    semantic_index,
    tokenizer,
    model,
    device
):

    print("\n" + "=" * 70)
    print("CHUNK SEARCH COMPARISON")
    print("=" * 70)

    for query in TEST_QUERIES:

        print("\n" + "#" * 70)
        print(f"QUERY: {query}")
        print("#" * 70)

        # Fixed-size search
        fixed_results = search_chunks(
            fixed_index,
            query,
            tokenizer,
            model,
            device
        )

        print_search_results(
            fixed_results,
            "fixed-size"
        )

        # Semantic search
        semantic_results = search_chunks(
            semantic_index,
            query,
            tokenizer,
            model,
            device
        )

        print_search_results(
            semantic_results,
            "semantic"
        ) 

 
# MAIN  
def main():

    print("=" * 70)
    print("ARXIV CHUNKING EXPERIMENT")
    print("=" * 70)

    # Load dataset
    df = load_dataset()

    # Load SPECTER2
    tokenizer, model, device = load_model()

    # Initialize Pinecone
    pc = initialize_pinecone()

    # Generate fixed chunks
    fixed_records = build_chunk_records(
        df,
        "fixed"
    )

    # Generate semantic chunks
    semantic_records = build_chunk_records(
        df,
        "semantic"
    )

    print(
        f"\nGenerated {len(fixed_records)} fixed chunks."
    )

    print(
        f"Generated {len(semantic_records)} semantic chunks."
    )

    # Process fixed-size strategy
    fixed_index = process_strategy(
        pc,
        INDEX_FIXED,
        fixed_records,
        tokenizer,
        model,
        device
    )

    # Process semantic strategy
    semantic_index = process_strategy(
        pc,
        INDEX_SEMANTIC,
        semantic_records,
        tokenizer,
        model,
        device
    )

    # Compare search results
    compare_search(
        fixed_index,
        semantic_index,
        tokenizer,
        model,
        device
    )

    print("\n" + "=" * 70)
    print("CHUNKING EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

