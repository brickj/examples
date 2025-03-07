import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

# Set the model to use - phi-2 is a very small (2.7B params) and fast model
MODEL_REPO = "TheBloke/phi-2-GGUF"
MODEL_FILE = "phi-2.Q2_K.gguf"  # Q2_K quantization is smallest/fastest

def download_fast_model(model_dir="model"):
    """Download a small, fast model for embeddings"""
    model_path = os.path.join(model_dir, MODEL_FILE)
    
    # Check if model directory exists, create if it doesn't
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    # Check if model exists locally
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
    else:
        # Download model
        print(f"Downloading small {MODEL_FILE} model...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=model_dir
        )
        print(f"Model downloaded to: {model_path}")
    
    return model_path

def load_netflix_data(csv_path="data/netflix_reviews.csv"):
    """Load Netflix reviews from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Netflix data not found at {csv_path}. Run generate_netflix_data.py first.")
    
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} Netflix shows")
    return data

def generate_embeddings(descriptions, model_path):
    """Generate embeddings using a small fast model"""
    print(f"Initializing model {os.path.basename(model_path)}...")
    
    # Initialize model
    llm = Llama(
        model_path=model_path,
        n_ctx=256,        # Smaller context window for speed
        n_threads=8,      # Use more threads for faster processing
        embedding=True    # Enable embedding functionality
    )
    
    print("Model initialized")
    
    # Generate embeddings
    all_embeddings = []
    total_texts = len(descriptions)
    
    print(f"Generating embeddings for {total_texts} descriptions...")
    
    for i, text in enumerate(descriptions):
        # Show progress
        if i % 2 == 0:
            print(f"Processing {i+1}/{total_texts}")
        
        # Generate embedding
        embedding = llm.embed(text)
        embedding_array = np.array(embedding, dtype=np.float32)
        all_embeddings.append(embedding_array)
    
    # Stack embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings

def save_embeddings(embeddings, output_file="data/embedded.csv"):
    """Save embeddings to CSV file"""
    # Create column names for the embeddings
    embedding_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    
    # Create DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Save to CSV
    embeddings_df.to_csv(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    # Create data directory if needed
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Download fast model
    model_path = download_fast_model()
    
    # Load Netflix data
    netflix_data = load_netflix_data()
    
    # Extract descriptions
    descriptions = netflix_data['description'].tolist()
    
    # Generate embeddings
    embeddings = generate_embeddings(descriptions, model_path)
    
    # Save embeddings
    save_embeddings(embeddings)
    
    print(f"\nComplete! Embedded {len(descriptions)} Netflix show descriptions.")
    print(f"Embedding file saved to data/embedded.csv") 