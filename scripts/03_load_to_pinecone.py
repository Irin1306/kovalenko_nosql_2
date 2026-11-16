# scripts/03_load_to_pinecone.py

import os
import numpy as np
import pandas as pd 
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


# Load .env file 
load_dotenv()

# Configuration
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

# Pinecone configuration
CLOUD = "aws"
REGION = "us-east-1"


def main():

    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found. Check your .env file."
        )

    # Load dataset
    print("Loading dataset...")

    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)

    print(f"Dataset records: {len(df)}")

    # Load embeddings
    print("Loading embeddings...")

    embeddings = np.load(INPUT_EMBEDDINGS)

    print(f"Embeddings shape: {embeddings.shape}")

    if embeddings.ndim != 2 or embeddings.shape[1] != VECTOR_DIM:
        raise ValueError(
            f"Expected embeddings with {VECTOR_DIM} dimensions."
        )

    # The embedding script uses the first 5,000 records.
    if len(embeddings) > len(df):
        raise ValueError(
            "More embeddings than dataset records."
        )

    if len(embeddings) != len(df):
        print(
            f"Using the first {len(embeddings)} dataset records "
            "to match the embeddings."
        )
        df = df.head(len(embeddings)).copy()

    if len(embeddings) == 0:
        raise ValueError("No embeddings found.")

    # Initialize Pinecone
    print("\nConnecting to Pinecone...")

    pc = Pinecone(api_key=api_key)

    # Create index if it does not exist
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:

        print(f"Creating index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )

        print("Index created.")

    else:
        print(f"Index {INDEX_NAME} already exists.")

    # Verify index configuration
    index_info = pc.describe_index(INDEX_NAME)

    if index_info.dimension != VECTOR_DIM:
        raise ValueError(
            f"Index dimension is {index_info.dimension}, "
            f"expected {VECTOR_DIM}."
        )

    if index_info.metric != "cosine":
        raise ValueError(
            f"Index metric is {index_info.metric}, expected cosine."
        )

    # Connect to index
    index = pc.Index(INDEX_NAME)

    # Prepare and upload vectors
    print("\nUploading vectors to Pinecone...")

    for start in tqdm(
        range(0, len(df), BATCH_SIZE),
        desc="Uploading batches"
    ):

        end = min(start + BATCH_SIZE, len(df))

        batch = []

        for i in range(start, end):

            row = df.iloc[i]

            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"])[:500],
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"])
            }

            vector = (
                f"paper_{i}",
                embeddings[i].tolist(),
                metadata
            )

            batch.append(vector)

        index.upsert(vectors=batch)

    # Display index statistics
    print("\nUpload completed!")

    stats = index.describe_index_stats()

    print(f"Index name: {INDEX_NAME}")
    print(f"Total vectors in index: {stats.total_vector_count}")


if __name__ == "__main__":
    main()
