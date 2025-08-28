import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
from keybert import KeyBERT
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances 
from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer 
from sklearn.cluster import KMeans 
# -----------------------------
# Config
# -----------------------------
INPUT_FOLDER = "./df3"  # Contains your CSV
OUTPUT_FOLDER = "./Clinical_Cluster_Outputs"
SUMMARY_FILE = "clinical_cluster_summary.csv"
PLOT_FILE = "clinical_projection.png"

# -----------------------------
# Load Data
# -----------------------------
def load_clinical_data(folder_path):
    """Load data using your exact column names"""
    dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fname))
            
            # Select and rename columns explicitly
            df = df[['Id_Number', 'Patient Question', 'Distorted part', 
                    'Dominant Distortion', 'Secondary Distortion (Optional)']]
            
            # Rename for consistency
            df = df.rename(columns={
                'Patient Question': 'patient_question',
                'Distorted part': 'distorted_part',
                'Dominant Distortion': 'dominant_distortion',
                'Secondary Distortion (Optional)': 'secondary_distortion'
            })
            
            dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} records with columns: {full_df.columns.tolist()}")
    return full_df

# -----------------------------
# Clean Clinical Text
# -----------------------------
stop_words = set(stopwords.words("english"))
medical_stopwords = {"patient", "doctor", "hospital", "medical"}  # Add domain-specific stopwords
stop_words.update(medical_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_clinical_text(text):
    """Specialized cleaning for clinical narratives"""
    if not isinstance(text, str):
        return ""
    
    # Remove sensitive identifiers
    text = re.sub(r'\b\d{2,}\b', '', text)  # Remove numbers (ages, dates)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)  # Replace names
    
    # Medical-specific cleaning
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Remove brackets/parentheses
    text = re.sub(r'\b\w{20,}\b', '', text)  # Remove very long words
    
    # Standard cleaning
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters
    text = text.lower()
    
    # Lemmatize keeping negation context
    tokens = []
    for token in text.split():
        if token.startswith(('not_', 'no_')):  # Preserve negations
            stemmed = lemmatizer.lemmatize(token[3:])
            tokens.append(f"not_{stemmed}")
        else:
            tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join([t for t in tokens if t not in stop_words and len(t) > 2])

# -----------------------------
# Extract Cluster Insights
# -----------------------------
def extract_cluster_insights(df, model, embeddings, cluster_col="cluster", text_col="cleaned_text", top_n_keywords=10, output_folder="Cluster_Outputs"):
    os.makedirs(output_folder, exist_ok=True)
    kw_model = KeyBERT(model=model)
    cluster_summary = []

    df["embedding"] = list(embeddings)

    for cluster_id in tqdm(sorted(df[cluster_col].unique())):
        if cluster_id == -1:
            continue

        cluster_df = df[df[cluster_col] == cluster_id].copy()
        comments = cluster_df[text_col].tolist()
        cluster_embeddings = np.vstack(cluster_df["embedding"].tolist())

        if len(comments) == 0:
            continue

        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        best_idx = np.argmax(sims)
        representative_comment = comments[best_idx]

        merged_text = " ".join(comments[:100])
        keywords = kw_model.extract_keywords(
            merged_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.5,
            top_n=top_n_keywords
        )
        keywords = [kw[0] for kw in keywords if kw]

        cluster_df.to_csv(os.path.join(output_folder, f"cluster_{cluster_id}.csv"), index=False)

        cluster_summary.append({
            "cluster": cluster_id,
            "num_comments": len(cluster_df),
            "keywords": keywords,
            "representative_comment": representative_comment
        })

    return pd.DataFrame(cluster_summary).sort_values("num_comments", ascending=False).reset_index(drop=True)

# -----------------------------
# Cluster Projection Plot
# -----------------------------
def plot_cluster_projections(embeddings, labels, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    projections = {
        "PCA": PCA(n_components=2).fit_transform(embeddings),
        "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings),
        "TSNE": TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    }

    for name, proj in projections.items():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette="tab10", s=30, legend=None)
        plt.title(f"{name} Cluster Projection")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{name.lower()}_projection.png"))
        plt.close()

# -----------------------------
# Main (Optimized Initial Approach)
# -----------------------------
def main():
    print("Loading data...")
    df = load_clinical_data(INPUT_FOLDER)
    
    print("\nCleaning text...")
    tqdm.pandas(desc="Processing")
    df["cleaned_text"] = df["patient_question"].progress_apply(clean_clinical_text)
    df = df[df["cleaned_text"].str.strip().astype(bool)]
    
    print("\nGenerating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Keep original model
    embeddings = model.encode(
        df["cleaned_text"].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print("\nOptimizing dimensionality...")
    # 1. Increased PCA components
    pca = PCA(n_components=min(100, len(embeddings[0])), random_state=42)  # Changed from 50 to 100
    embeddings = pca.fit_transform(embeddings)
    
    # 2. Normalize with L2 norm
    embeddings = Normalizer(norm='l2').fit_transform(embeddings)
    
    print("\nClustering with tuned parameters...")
    # Dynamic cluster sizing
    min_cluster_size = max(3, int(len(df)*0.0075))  # Slightly more clusters than original
    min_samples = 2  # More sensitive to small clusters
    
    # Cosine distance matrix
    cosine_dist = cosine_distances(embeddings).astype(np.float64)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.4,  # Better cluster merging
        alpha=0.8  # Balanced clustering
    )
    labels = clusterer.fit_predict(cosine_dist)
    
    # Post-processing to merge very small clusters
    unique, counts = np.unique(labels, return_counts=True)
    for label in unique[counts < min_cluster_size//2]:  # Merge clusters < half of min size
        labels[labels == label] = -1
    
    # Relabel to get consecutive cluster IDs
    labels = pd.Series(labels).astype('category').cat.codes.values - 1
    
    # Evaluation
    if len(set(labels)) > 1:
        sil_score = silhouette_score(embeddings, labels, metric='cosine')
        print(f"\nSilhouette Score: {sil_score:.4f}")
        
        # Fallback to KMeans if score drops below 0.15
        if sil_score < 0.15:
            print("Falling back to optimized KMeans...")
            optimal_k = min(8, len(df)//20)  # Conservative cluster count
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(embeddings)
            sil_score = silhouette_score(embeddings, labels, metric='cosine')
            print(f"New Silhouette (KMeans): {sil_score:.4f}")
    
    # Cluster analysis
    cluster_counts = pd.Series(labels).value_counts()
    print("\nFinal Cluster Distribution:")
    print(cluster_counts[cluster_counts.index != -1])  # Exclude noise
    
    # Rest of your pipeline remains identical
    df["cluster"] = labels
    
    print("\nExtracting cluster insights...")
    summary_df = extract_cluster_insights(df, model=model, embeddings=embeddings)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    
    print("\nPlotting projections...")
    plot_cluster_projections(embeddings, labels)
    
    print("\nDone. Outputs saved.")

if __name__ == "__main__":
    main()