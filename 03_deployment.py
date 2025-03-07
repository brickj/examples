# 03_deployment.py
import os
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import mlflow
import mlflow.pyfunc

from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec, ParamSchema, ParamSpec

from sklearn.metrics.pairwise import cosine_similarity

# 🏗️ Defining the Netflix Similarity Model Class
class NetflixSimilarityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load precomputed embeddings and Netflix reviews data.
        """
        # Load precomputed embeddings
        self.embeddings_df = pd.read_csv(context.artifacts['embeddings_path'])
        
        # Load Netflix reviews corpus
        self.netflix_df = pd.read_csv(context.artifacts['netflix_reviews_path'])
        
        # Print diagnostics about the loaded data
        print(f"Loaded embeddings shape: {self.embeddings_df.shape}")
        print(f"Loaded Netflix reviews shape: {self.netflix_df.shape}")
        
        # Convert embeddings to numpy array for faster similarity computation
        self.embeddings = self.embeddings_df.values
        
        # Store the smaller of the two shapes to avoid index errors
        self.max_index = min(len(self.netflix_df), len(self.embeddings_df)) - 1
        print(f"Using max safe index: {self.max_index}")
        
        # Store sample embedding index within safe bounds
        self.sample_embedding_index = 0  # Use first row by default
    
    def predict(self, context, model_input, params):
        """
        Find similar Netflix content based on similarity to a sample embedding.
        In a real implementation, we would generate embeddings for new queries.
        """
        # Extract parameters
        top_n = params.get("top_n", 5) if params else 5
        embedding_index = min(params.get("embedding_index", self.sample_embedding_index), self.max_index)
        
        # Make sure we don't request more results than we have data
        top_n = min(top_n, self.max_index)
        
        # Use a specific row from our embeddings as the "query embedding"
        query_embedding = self.embeddings[embedding_index:embedding_index+1]
        
        # Compute cosine similarity between query and all embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices of top N most similar results, ensuring they're within bounds
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Make absolutely sure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx <= self.max_index]
        
        # If no valid indices (unlikely), just use the first few rows
        if not valid_indices:
            valid_indices = list(range(min(top_n, self.max_index + 1)))
        
        # Retrieve corresponding results from the corpus
        results = self.netflix_df.iloc[valid_indices].copy()
        results.loc[:, 'similarity'] = similarities[valid_indices]
        
        # Return results as a dictionary
        return results.to_dict(orient="records")
    
    @classmethod
    def log_model(cls, model_name, embeddings_path, netflix_reviews_path):
        """
        Logs the model to MLflow with appropriate artifacts and schema.
        """
        # Check if the files exist
        for path in [embeddings_path, netflix_reviews_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        # Print file sizes for information
        emb_size = os.path.getsize(embeddings_path) / (1024 * 1024)
        netflix_size = os.path.getsize(netflix_reviews_path) / (1024 * 1024)
        print(f"Embeddings file size: {emb_size:.2f} MB")
        print(f"Netflix reviews file size: {netflix_size:.2f} MB")
        
        # Count lines in each file
        with open(embeddings_path, 'r') as f:
            emb_lines = sum(1 for _ in f)
        
        with open(netflix_reviews_path, 'r') as f:
            netflix_lines = sum(1 for _ in f)
        
        print(f"Embeddings file lines: {emb_lines}")
        print(f"Netflix reviews file lines: {netflix_lines}")
        
        # Simple input schema - just accepting a query string
        input_schema = Schema([ColSpec("string", "query")])
        
        output_schema = Schema([
            TensorSpec(np.dtype("object"), (-1,), "Netflix content with similarity scores")
        ])
        
        # Parameters allow specifying which embedding to use and how many results
        params_schema = ParamSchema([
            ParamSpec("top_n", "integer", 5),
            ParamSpec("embedding_index", "integer", 0)
        ])
        
        # Define model signature
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)
        
        # Define necessary package requirements
        requirements = ["scikit-learn", "pandas", "numpy", "tabulate"]
        
        # Define artifacts dictionary
        artifacts = {
            "embeddings_path": embeddings_path,
            "netflix_reviews_path": netflix_reviews_path
        }
        
        # Log the model in MLflow
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=requirements
        )

# 📜 Logging Model to MLflow
def log_model_to_mlflow():
    # Set the MLflow experiment name
    mlflow.set_experiment(experiment_name="Netflix Similarity Model")

    # Start an MLflow run
    with mlflow.start_run(run_name="Netflix_Similarity_Run") as run:
        # Print the artifact URI for reference
        print(f"Run's Artifact URI: {run.info.artifact_uri}")
        
        # Log the Netflix similarity model to MLflow
        NetflixSimilarityModel.log_model(
            model_name="Netflix_Similarity",
            embeddings_path="data/embedded.csv",
            netflix_reviews_path="data/netflix_reviews.csv",
        )

        # Register the logged model in MLflow Model Registry
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/Netflix_Similarity", 
            name="Netflix_Similarity"
        )
        
        return run.info.run_id

# 📦 Function to load the model and run inference
def get_similar_content(query, run_id=None, embedding_index=0, top_n=5):
    """
    Get similar Netflix content for a given query.
    
    Args:
        query (str): The search query (currently not used for embedding generation)
        run_id (str, optional): MLflow run ID. If None, uses the latest model version
        embedding_index (int): Index of the embedding to use from the dataset
        top_n (int): Number of results to return
        
    Returns:
        DataFrame: Similar Netflix content
    """
    if run_id:
        # Load model from specific run
        model_uri = f"runs:/{run_id}/Netflix_Similarity"
    else:
        # Get latest model version
        client = MlflowClient()
        model_metadata = client.get_latest_versions("Netflix_Similarity", stages=["None"])
        latest_model_version = model_metadata[0].version
        model_uri = f"models:/Netflix_Similarity/{latest_model_version}"
    
    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Prepare simple input data
    input_data = {"query": [query]}
    
    # Run inference with parameters
    result = model.predict(input_data, params={"top_n": top_n, "embedding_index": embedding_index})
    
    # Convert to DataFrame for better display
    return pd.DataFrame(result)

# 📊 Demo: Find similar Netflix content
def run_demo():
    # Log model to MLflow
    run_id = log_model_to_mlflow()
    
    # Use a simple query (the actual content is determined by the embedding_index)
    query = "Show me something like Stranger Things"
    
    try:
        # Get similar content using embedding at index 0
        similar_content = get_similar_content(
            query=query,
            run_id=run_id,
            embedding_index=0,  # Use first row of embeddings
            top_n=5
        )
        
        # Display results
        print(f"\nQuery: {query}")
        print("\nTop similar Netflix content:")
        if 'title' in similar_content.columns:
            print(tabulate(similar_content[['title', 'genre', 'year', 'similarity']], 
                          headers='keys', tablefmt='fancy_grid', showindex=False))
        else:
            # Just display whatever columns are available
            print(tabulate(similar_content, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Displaying sample of the Netflix reviews data instead:")
        netflix_df = pd.read_csv("data/netflix_reviews.csv", nrows=5)
        print(tabulate(netflix_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Run the demo if executed directly
if __name__ == "__main__":
    run_demo()