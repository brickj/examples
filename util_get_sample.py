import pandas as pd

def show_file_structure(file_path):
    # Read just a tiny sample to get structure
    df = pd.read_csv(file_path, nrows=2)
    
    print(f"\n=== {file_path} Structure ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    
    # For embeddings, check if there are many numerical columns
    if len(df.columns) > 10:
        print(f"Total number of columns: {len(df.columns)}")
        # Show first few and last few column names
        print(f"First 5 columns: {df.columns[:5].tolist()}")
        print(f"Last 5 columns: {df.columns[-5:].tolist()}")

# Show structure of Netflix reviews
show_file_structure("data/netflix_reviews.csv")

# Show structure of embeddings
show_file_structure("data/embedded.csv")