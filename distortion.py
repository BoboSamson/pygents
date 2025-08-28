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
from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

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
# Main
# -----------------------------
def main():
    print("Loading data...")
    df = load_clinical_data(INPUT_FOLDER)
    
    print("\nData Sample:")
    print(df[['patient_question']].head(1).to_dict())
    
    print("\nCleaning text...")
    tqdm.pandas(desc="Processing")
    df["cleaned_text"] = df["patient_question"].progress_apply(clean_clinical_text)
    
    # Filter empty results
    initial_count = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)]
    print(f"After cleaning: {len(df)} valid narratives (removed {initial_count - len(df)} empty)")
    
    print("\nGenerating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        df["cleaned_text"].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    np.save("embeddings.npy", embeddings)
    
    print("\nReducing dimensionality...")
    n_components = min(50, len(df)-1)
    pca_model = PCA(n_components=n_components)
    pca_50 = pca_model.fit_transform(embeddings)
    
    print("\nClustering...")
    min_cluster_size = max(5, int(len(df)*0.01))
    print(f"Using min_cluster_size: {min_cluster_size}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(pca_50)
    
    # Evaluate clusters
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        print("Warning: Insufficient clusters formed - try reducing min_cluster_size")
    else:
        valid_idx = labels != -1
        if sum(valid_idx) > 1:  # Need at least 2 samples
            sil_score = silhouette_score(pca_50[valid_idx], labels[valid_idx])
            print(f"Silhouette Score: {sil_score:.4f}")
            if sil_score < 0.55:
                print("[WARN] Silhouette score < 0.55 - consider parameter adjustment")

    df["cluster"] = labels
    
    print("\nExtracting cluster insights...")
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