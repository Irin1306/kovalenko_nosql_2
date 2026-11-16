# scripts/02_embed.py

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


# Configuration
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"

BATCH_SIZE = 64
VECTOR_DIM = 768 
MAX_LENGTH = 512


def main():

    # Load dataset
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run script 01 first."
        )

    df = pd.read_parquet(INPUT_FILE).copy() 

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(f"Loaded {len(df)} articles.")

    # Prepare texts
    texts = (
        df["title"].fillna("").astype(str)
        + " [SEP] "
        + df["abstract"].fillna("").astype(str)
    ).tolist()

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load SPECTER2
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.load_adapter(
        ADAPTER_NAME,
        source="hf",
        set_active=True
    )

    print("Active adapter:", model.active_adapters)

    model.to(device)
    model.eval()

    # Generate embeddings
    all_embeddings = []

    total_batches = (
        len(texts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    with torch.inference_mode():

        for batch_index, i in enumerate(
            range(0, len(texts), BATCH_SIZE), start=1
        ):

            batch = texts[i:i + BATCH_SIZE]

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
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            print(
                f"Batch {batch_index}/{total_batches} "
                f"- processed {min(i + BATCH_SIZE, len(texts))}"
                f"/{len(texts)} articles"
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)

    # Validate embeddings
    print("\nEmbedding results:")
    print(f"Number of texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(
        f"First vector norm: "
        f"{np.linalg.norm(embeddings[0]):.6f}"
    )

    if embeddings.shape != (len(df), VECTOR_DIM):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}. "
            f"Expected ({len(df)}, {VECTOR_DIM})."
        )

    # Save embeddings
    os.makedirs("embeddings", exist_ok=True)

    np.save(OUTPUT_FILE, embeddings)  

    # Verify saved file
    saved_embeddings = np.load(OUTPUT_FILE)

    print(f"\nEmbeddings saved to {os.path.abspath(OUTPUT_FILE)}")
    print(f"Saved shape: {saved_embeddings.shape}")
    print(f"Saved dtype: {saved_embeddings.dtype}")

    if saved_embeddings.shape != (len(df), VECTOR_DIM):
        raise ValueError("Saved embeddings have an unexpected shape!")

    print("Embedding generation completed successfully!")


if __name__ == "__main__":
    main()

 