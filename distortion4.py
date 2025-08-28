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
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from umap import UMAP

# -----------------------------
# Config
# -----------------------------
INPUT_FOLDER = "./df3"
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
            
            df = df[['Id_Number', 'Patient Question', 'Distorted part', 
                    'Dominant Distortion', 'Secondary Distortion (Optional)']]
            
            df = df.rename(columns={
                'Patient Question': 'patient_question',
                'Distorted part': 'distorted_part',
                'Dominant Distortion': 'dominant_distortion',
                'Secondary Distortion (Optional)': 'secondary_distortion'
            })
            
            dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df)} records")
    return full_df

# -----------------------------
# Clean Clinical Text
# -----------------------------
stop_words = set(stopwords.words("english"))
medical_stopwords = {"patient", "doctor", "hospital", "medical"}
stop_words.update(medical_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_clinical_text(text):
    """Specialized cleaning for clinical narratives"""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'\b\d{2,}\b', '', text)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\b\w{20,}\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    
    tokens = []
    for token in text.split():
        if token.startswith(('not_', 'no_')):
            stemmed = lemmatizer.lemmatize(token[3:])
            tokens.append(f"not_{stemmed}")
        else:
            tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join([t for t in tokens if t not in stop_words and len(t) > 2])

# -----------------------------
# Enhanced Clustering Functions
# -----------------------------
def optimized_kmeans(embeddings, max_k=15):
    """Find optimal k with silhouette scoring"""
    best_k = 2
    best_score = -1
    best_labels = None
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=20,
            random_state=42
        )
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric='cosine')
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    
    print(f"Best KMeans: k={best_k}, score={best_score:.4f}")
    return best_labels

def enhanced_hdbscan(embeddings):
    """Improved HDBSCAN with UMAP preprocessing"""
    reducer = UMAP(
        n_components=min(50, embeddings.shape[1]),
        metric='cosine',
        random_state=42
    )
    reduced_emb = reducer.fit_transform(embeddings)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.3,
        metric='euclidean',
        cluster_selection_method='leaf'
    )
    return clusterer.fit_predict(reduced_emb)

def hybrid_clustering(embeddings):
    """Combine both methods for best results"""
    # Try KMeans first as it's more stable
    kmeans_labels = optimized_kmeans(embeddings)
    kmeans_score = silhouette_score(embeddings, kmeans_labels, metric='cosine')
    
    # Only try HDBSCAN if KMeans score is low
    if kmeans_score < 0.15:
        print("KMeans score low, trying HDBSCAN...")
        hdb_labels = enhanced_hdbscan(embeddings)
        hdb_score = silhouette_score(embeddings, hdb_labels, metric='cosine')
        
        if hdb_score > kmeans_score:
            print(f"Selected HDBSCAN (score={hdb_score:.4f})")
            return hdb_labels
    
    return kmeans_labels

# -----------------------------
# Extract Cluster Insights
# -----------------------------
def extract_cluster_insights(df, model, embeddings, cluster_col="cluster", 
                           text_col="cleaned_text", top_n_keywords=10, 
                           output_folder="Cluster_Outputs"):
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
        "UMAP": UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(embeddings),
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
    df = load_clinical_data(INPUT_FOLDER)
    
    print("\nCleaning text...")
    tqdm.pandas(desc="Processing")
    df["cleaned_text"] = df["patient_question"].progress_apply(clean_clinical_text)
    df = df[df["cleaned_text"].str.strip().astype(bool)]
    print(f"After cleaning: {len(df)} valid narratives")
    
    print("\nGenerating embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2')  # More powerful model
    embeddings = model.encode(
        df["cleaned_text"].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print("\nOptimizing embeddings...")
    embeddings = Normalizer(norm='l2').fit_transform(embeddings)
    pca = PCA(n_components=min(75, embeddings.shape[1]), random_state=42)
    embeddings = pca.fit_transform(embeddings)
    
    print("\nRunning hybrid clustering...")
    labels = hybrid_clustering(embeddings)
    
    # Post-processing
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for label, count in zip(unique, counts):
        print(f"Cluster {label}: {count} samples")
    
    final_score = silhouette_score(embeddings, labels, metric='cosine')
    print(f"\nFinal Silhouette Score: {final_score:.4f}")
    
    df["cluster"] = labels
    
    print("\nExtracting insights...")
    summary_df = extract_cluster_insights(df, model=model, embeddings=embeddings)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    
    print("\nPlotting projections...")
    plot_cluster_projections(embeddings, labels)
    
    print("\nDone. Outputs saved to:")
    print(f"- Cluster summaries: {SUMMARY_FILE}")
    print(f"- Projection plots: {PLOT_FILE}")
    print(f"- Individual cluster files: {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()