import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# -----------------------------
# Config
# -----------------------------
CLUSTER_OUTPUT_FOLDER = "./Cluster_Outputs"  # Folder with individual cluster CSVs
SUMMARY_FILE = "clinical_cluster_summary.csv"  # Summary file from main clustering
OUTPUT_TABLE_FILE = "cluster_analysis_output.csv"  # Final output file

# Mapping of cluster numbers to Main Cluster labels
MAIN_CLUSTER_LABELS = {
    1: "Mental State",
    0: "Relationships"
}

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text_for_keyword(text):
    """Clean text for keyword extraction"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join([w for w in text.split() if len(w) > 2])

def get_most_similar_keyword(text, keywords, model):
    """Find the keyword most similar to the text"""
    if not keywords:
        return ""
    
    # Clean the text and keywords
    cleaned_text = clean_text_for_keyword(text)
    cleaned_keywords = [clean_text_for_keyword(kw) for kw in keywords]
    
    # Get embeddings
    text_embedding = model.encode([cleaned_text])
    keyword_embeddings = model.encode(cleaned_keywords)
    
    # Calculate similarities
    similarities = cosine_similarity(text_embedding, keyword_embeddings).flatten()
    most_similar_idx = np.argmax(similarities)
    
    return keywords[most_similar_idx]

def extract_keywords_from_comment(comment, model, top_n=5):
    """Extract keywords from a single comment using KeyBERT"""
    from keybert import KeyBERT
    kw_model = KeyBERT(model=model)
    keywords = kw_model.extract_keywords(
        comment,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

# -----------------------------
# Main Processing
# -----------------------------
def process_cluster_outputs():
    # Load the summary file
    summary_df = pd.read_csv(SUMMARY_FILE)
    
    # Convert keywords from string to list
    summary_df['keywords'] = summary_df['keywords'].apply(
        lambda x: [kw.strip(" '") for kw in x.strip("[]").split(",")] if isinstance(x, str) else []
    )
    
    # Initialize sentence transformer model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare to collect all data
    all_data = []
    
    # Process each cluster file
    cluster_files = [f for f in os.listdir(CLUSTER_OUTPUT_FOLDER) if f.startswith("cluster_") and f.endswith(".csv")]
    
    for cluster_file in tqdm(cluster_files, desc="Processing cluster files"):
        # Extract cluster number from filename
        cluster_num = int(cluster_file.split("_")[1].split(".")[0])
        
        # Load the cluster data
        cluster_df = pd.read_csv(os.path.join(CLUSTER_OUTPUT_FOLDER, cluster_file))
        
        # Get the summary info for this cluster
        cluster_summary = summary_df[summary_df['cluster'] == cluster_num].iloc[0]
        cluster_keywords = cluster_summary['keywords']
        representative_comment = cluster_summary['representative_comment']
        
        # Get Main Cluster label
        main_cluster_label = MAIN_CLUSTER_LABELS.get(cluster_num, f"Cluster {cluster_num}")
        
        # Process each patient question in the cluster
        for _, row in cluster_df.iterrows():
            patient_question = row['patient_question']
            
            # Get the most similar keyword from cluster keywords
            if cluster_keywords:
                cluster_label = get_most_similar_keyword(patient_question, cluster_keywords, model)
            else:
                # If no keywords in summary, extract from representative comment
                rep_keywords = extract_keywords_from_comment(representative_comment, model)
                cluster_label = get_most_similar_keyword(patient_question, rep_keywords, model)
            
            # Add to our collection
            all_data.append({
                'patient_question': patient_question,
                'Cluster': cluster_num,
                'Cluster_label': cluster_label,
                'Main Cluster': main_cluster_label
            })
    
    # Create final dataframe
    output_df = pd.DataFrame(all_data)
    
    # Save to file
    output_df.to_csv(OUTPUT_TABLE_FILE, index=False)
    print(f"\nOutput saved to {OUTPUT_TABLE_FILE}")
    print(f"Total records processed: {len(output_df)}")

if __name__ == "__main__":
    process_cluster_outputs()