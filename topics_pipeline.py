# topic_pipeline.py (Final with Optimizations)

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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Config
# -----------------------------
INPUT_FOLDER = "./Reddit_dataset"
OUTPUT_FOLDER = "./Cluster_Outputs"
SUMMARY_FILE = "cluster_summary_overview.csv"
PLOT_FILE = "cluster_projection.png"

# -----------------------------
# Load and Concatenate CSVs
# -----------------------------
def load_and_concatenate_csvs(folder_path):
    dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            file_path = os.path.join(folder_path, fname)
            df = pd.read_csv(file_path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# -----------------------------
# Clean Text
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_comment(text):
    # Remove markdown images and URLs
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Keep hashtags and mentions while removing other special characters
    text = re.sub(r'[^A-Za-z0-9@#\s]', ' ', text)  # Keep letters, numbers, @, and #
    
    # Remove repeated characters only when 4+ consecutive repeats
    text = re.sub(r'\b([a-zA-Z])\1{3,}\b', '', text)  # Allow "cooool" -> "cool" but remove "cooooool"
    
    # Remove action verbs while preserving negation context
    text = re.sub(r'\b(?:up[- ]?voted|shared|posted|reposted|like|vote|click)\b(?![ -]?(?:not|never))\b', 
                 '', text, flags=re.IGNORECASE)
    
    # Clean whitespace and normalize case
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# -----------------------------
# Extract Cluster Insights (Updated)
# -----------------------------
def extract_cluster_insights(df, model, embeddings, cluster_col="cluster", text_col="cleaned_comments", top_n_keywords=10, output_folder="Cluster_Outputs"):
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
        keywords = kw_model.extract_keywords(merged_text, top_n=top_n_keywords, stop_words='english')
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
# Main
# -----------------------------
def main():
    print("Loading data...")
    df = load_and_concatenate_csvs(INPUT_FOLDER)
    tqdm.pandas(desc="Cleaning")
    df["cleaned_comments"] = df["comments"].astype(str).progress_apply(clean_comment)

    df["word_count"] = df["cleaned_comments"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 3].drop_duplicates(subset="cleaned_comments").copy()

    print("Embedding comments...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["cleaned_comments"].tolist(), show_progress_bar=True)
    np.save("embeddings.npy", embeddings)

    print("Reducing dimensionality...")
    pca_model = PCA(n_components=50)
    pca_50 = pca_model.fit_transform(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(pca_50)

    valid_idx = labels != -1
    if valid_idx.sum() == 0:
        raise ValueError("All points labeled as noise. Try adjusting clustering params.")

    sil_score = silhouette_score(pca_50[valid_idx], labels[valid_idx])
    print(f"Silhouette Score: {sil_score:.4f}")

    if sil_score < 0.55:
        print("[WARN] Silhouette score < 0.55.")

    df = df.iloc[:len(labels)].copy().reset_index(drop=True)
    df["cluster"] = labels

    print("Extracting cluster insights...")
    summary_df = extract_cluster_insights(df, model=model, embeddings=embeddings, cluster_col="cluster")
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("Plotting projections...")
    plot_cluster_projections(embeddings, labels)

    print("Done. Outputs saved.")

if __name__ == "__main__":
    main()

