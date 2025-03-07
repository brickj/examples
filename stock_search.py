import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_name="bert-base-uncased", model_dir="model"):
    """Load BERT model and tokenizer from local folder or download if not available"""
    global model, tokenizer
    
    # Check if model exists locally
    if os.path.exists(os.path.join(model_dir, "config.json")):
        # Load from local path
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        print(f"Model loaded from {model_dir}")
    else:
        # Download and save
        os.makedirs(model_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print("Model downloaded and saved")

def fetch_stock_data(ticker="AAPL", days=7):
    """Fetch stock data and create descriptions"""
    # Get data from yfinance
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = stock.history(start=start_date, end=end_date, interval='1h')
    df = df.reset_index()
    
    # Create text descriptions
    df['description'] = df.apply(lambda row: 
        f"At {row['Datetime'].strftime('%Y-%m-%d %H:%M')}, {ticker} stock price was ${row['Close']:.2f} "
        f"with a volume of {row['Volume']:,} shares. The stock opened at ${row['Open']:.2f} "
        f"and reached a high of ${row['High']:.2f} and low of ${row['Low']:.2f}.", axis=1)
    
    return df

def generate_embeddings(texts):
    """Generate embeddings for text descriptions"""
    global model, tokenizer
    
    # Ensure model is loaded
    if model is None or tokenizer is None:
        load_model()
    
    # Generate embeddings
    embeddings = []
    for text in texts:
        # Tokenize and get model output
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token embedding as the sentence embedding
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten())
    
    return np.array(embeddings)

def search(query, stock_data, embeddings, n_results=5):
    """Search for similar stock descriptions based on a query"""
    global model, tokenizer
    
    # Ensure model is loaded
    if model is None or tokenizer is None:
        load_model()
    
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0].reshape(1, -1)
    
    # Calculate similarity scores
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:n_results]
    
    results = []
    for idx in top_indices:
        results.append({
            'description': stock_data.iloc[idx]['description'],
            'similarity': similarities[idx],
            'datetime': stock_data.iloc[idx]['Datetime'],
            'price': stock_data.iloc[idx]['Close']
        })
    
    return results

def save_data(stock_data, embeddings, stock_file="data/stock_data.csv", emb_file="data/embeddings.npy"):
    """Save stock data and embeddings to files"""
    # Save stock data to CSV
    stock_data.to_csv(stock_file, index=False)
    
    # Save embeddings to NumPy file
    np.save(emb_file, embeddings)
    
    print(f"Saved stock data to {stock_file} and embeddings to {emb_file}")

def load_data(stock_file="stock_data.csv", emb_file="embeddings.npy"):
    """Load stock data and embeddings from files"""
    # Load stock data from CSV
    stock_data = pd.read_csv(stock_file)
    stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
    
    # Load embeddings from NumPy file
    embeddings = np.load(emb_file)
    
    print(f"Loaded stock data from {stock_file} and embeddings from {emb_file}")
    return stock_data, embeddings

# Example usage
if __name__ == "__main__":
    # 1. Load the model
    load_model()
    
    # 2. Fetch stock data
    stock_data = fetch_stock_data("AAPL", days=7)
    
    # 3. Generate embeddings
    embeddings = generate_embeddings(stock_data['description'].tolist())
    
    # 4. Save data for later use
    save_data(stock_data, embeddings)
    
    # 5. Search example
    results = search("high price with large volume", stock_data, embeddings, n_results=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Date: {result['datetime']}")
        print(f"Price: ${result['price']:.2f}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Description: {result['description']}")