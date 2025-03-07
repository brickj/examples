import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity

# Same model settings as embedding script
MODEL_REPO = "TheBloke/phi-2-GGUF"
MODEL_FILE = "phi-2.Q2_K.gguf"

def load_model(model_dir="model"):
    """Load the same model used for embedding"""
    model_path = os.path.join(model_dir, MODEL_FILE)
    
    # Check if model exists locally
    if not os.path.exists(model_path):
        print(f"Model not found. Downloading {MODEL_FILE}...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=model_dir
        )
    
    # Initialize model
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        n_threads=8,
        embedding=True
    )
    
    return llm

def load_data():
    """Load Netflix data and embeddings"""
    # Check if files exist
    if not os.path.exists("data/netflix_reviews.csv"):
        raise FileNotFoundError("File data/netflix_reviews.csv not found. Run generate_netflix_data.py first.")
    
    if not os.path.exists("data/embedded.csv"):
        raise FileNotFoundError("File data/embedded.csv not found. Run embed_netflix_fast.py first.")
    
    # Load Netflix shows data
    netflix_data = pd.read_csv("data/netflix_reviews.csv")
    
    # Load embeddings
    embeddings_df = pd.read_csv("data/embedded.csv")
    embeddings = embeddings_df.values
    
    # Verify alignment
    if len(netflix_data) != len(embeddings):
        print(f"WARNING: Netflix data has {len(netflix_data)} entries but embeddings has {len(embeddings)} entries")
        # Make sure we only use as many embeddings as we have Netflix entries
        min_length = min(len(netflix_data), len(embeddings))
        netflix_data = netflix_data.iloc[:min_length]
        embeddings = embeddings[:min_length]
        print(f"Aligned data to {min_length} entries")
    
    print(f"Loaded {len(netflix_data)} Netflix shows and their embeddings")
    return netflix_data, embeddings

def search_shows(query, netflix_data, embeddings, llm, top_n=3):
    """Search for Netflix shows similar to the query"""
    print(f"Searching for: '{query}'")
    
    # Make sure top_n isn't larger than our dataset
    top_n = min(top_n, len(netflix_data))
    
    # Generate embedding for query
    query_embedding = np.array(llm.embed(query), dtype=np.float32).reshape(1, -1)
    
    # Check embedding dimension alignment
    if query_embedding.shape[1] != embeddings.shape[1]:
        print(f"WARNING: Query embedding dimension ({query_embedding.shape[1]}) doesn't match stored embeddings ({embeddings.shape[1]})")
        print("This may cause poor search results")
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top results
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Format results
    results = []
    for idx in top_indices:
        show = netflix_data.iloc[idx]
        results.append({
            'title': show['title'],
            'year': int(show['year']),
            'genre': show['genre'],
            'seasons': int(show['seasons']),
            'similarity': similarities[idx],
            'description': show['description']
        })
    
    return results

def print_results(results):
    """Print search results in a readable format"""
    if not results:
        print("No results found.")
        return
        
    print("\n==== SEARCH RESULTS ====")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} ({result['year']}) - {result['genre']}")
        print(f"   Seasons: {result['seasons']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        
        # Print truncated description if it's long
        desc = result['description']
        if len(desc) > 150:
            print(f"   Description: {desc[:150]}...")
        else:
            print(f"   Description: {desc}")

def search():
    """Search for shows with a hardcoded query"""
    try:
        # Load model, data, and embeddings
        print("Loading model and data...")
        llm = load_model()
        netflix_data, embeddings = load_data()
        
        # Hard-coded query - modify this string as needed
        query = "action"
        print(f"\nSearching for '{query}' shows...")
        
        # Search and display results
        results = search_shows(query, netflix_data, embeddings, llm)
        print_results(results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Make sure you've generated both netflix_reviews.csv and embedded.csv files.")

if __name__ == "__main__":
    search() 