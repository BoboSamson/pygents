import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
from keybert import KeyBERT
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

# Initialize tqdm for pandas
tqdm.pandas()

# -----------------------------
# Config (Optimized Parameters)
# -----------------------------
INPUT_FOLDER = "./df3"
OUTPUT_FOLDER = "./Clinical_Cluster_Outputs"
SUMMARY_FILE = "clinical_cluster_summary.csv"
METRICS_FILE = "cluster_metrics.json"
PLOT_DIR = "plots"

# Proven parameters from experiments
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
PCA_COMPONENTS = 75  # Optimized value
HDBSCAN_PARAMS = {
    'min_cluster_size': 'auto',  # Will be set to 1% of data size
    'min_samples': 1,
    'metric': 'euclidean',
    'cluster_selection_method': 'leaf',
    'cluster_selection_epsilon': 0.5
}

# -----------------------------
# Data Loading and Cleaning
# -----------------------------
def load_clinical_data(folder_path):
    dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, fname))
            df = df[['Patient Question', 'Distorted part', 'Dominant Distortion']]
            df = df.rename(columns={
                'Patient Question': 'patient_question',
                'Distorted part': 'distorted_part',
                'Dominant Distortion': 'dominant_distortion'
            })
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

stop_words = set(stopwords.words("english"))
medical_stopwords = {"patient", "doctor", "hospital", "medical"}
stop_words.update(medical_stopwords)
lemmatizer = WordNetLemmatizer()

def clean_clinical_text(text):
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
# Clustering with Proven Parameters
# -----------------------------
def optimized_clustering(embeddings, n_samples):
    # Dimensionality reduction with proven PCA components
    pca = PCA(n_components=min(PCA_COMPONENTS, len(embeddings)-1))
    reduced_embeddings = pca.fit_transform(embeddings)
    reduced_embeddings = StandardScaler().fit_transform(reduced_embeddings)
    
    # Set min_cluster_size dynamically (1% of data but at least 5)
    params = HDBSCAN_PARAMS.copy()
    params['min_cluster_size'] = max(5, int(n_samples * 0.01))
    
    # Cluster with proven parameters
    clusterer = hdbscan.HDBSCAN(**params).fit(reduced_embeddings)
    labels = clusterer.labels_
    
    return labels, clusterer, reduced_embeddings

# -----------------------------
# Analysis and Visualization
# -----------------------------
# -----------------------------
# Analysis and Visualization
# -----------------------------
# -----------------------------
# Analysis and Visualization
# -----------------------------
def extract_cluster_insights(df, model, embeddings, cluster_col="cluster", output_folder=OUTPUT_FOLDER):
    kw_model = KeyBERT(model=model)
    cluster_summary = []
    df["embedding"] = list(embeddings)
    
    # Create cluster details directory
    cluster_details_dir = os.path.join(output_folder, "cluster_details")
    os.makedirs(cluster_details_dir, exist_ok=True)

    # First process regular clusters
    for cluster_id in tqdm(sorted(df[cluster_col].unique())):
        if cluster_id == -1:
            continue

        cluster_df = df[df[cluster_col] == cluster_id].copy()
        comments = cluster_df["cleaned_text"].tolist()
        questions = cluster_df["patient_question"].tolist()
        cluster_embeddings = np.vstack(cluster_df["embedding"].tolist())

        # Get representative example
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        representative_idx = np.argmax(sims)
        representative_comment = comments[representative_idx]
        representative_question = questions[representative_idx]

        # Extract keywords for the entire cluster
        cluster_keywords = kw_model.extract_keywords(
            " ".join(comments[:100]),  # Using first 100 comments to limit processing time
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.5,
            top_n=10
        )
        cluster_keywords = [kw[0] for kw in cluster_keywords if kw]

        # Extract keywords for each individual question
        question_keywords = []
        for question, comment in zip(questions, comments):
            keywords = kw_model.extract_keywords(
                comment,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=5  # Fewer keywords per individual question
            )
            question_keywords.append([kw[0] for kw in keywords if kw])

        # Create detailed cluster CSV
        cluster_details = pd.DataFrame({
            "patient_question": questions,
            "cleaned_text": comments,
            "keywords": question_keywords,
            "distance_to_centroid": sims
        })
        cluster_details.to_csv(os.path.join(cluster_details_dir, f"cluster_{cluster_id}.csv"), index=False)

        # Add to summary
        cluster_summary.append({
            "cluster": cluster_id,
            "size": len(cluster_df),
            "keywords": cluster_keywords,
            "representative_comment": representative_comment,
            "representative_question": representative_question
        })

    # Process noise cluster
    noise_df = df[df[cluster_col] == -1].copy()
    if len(noise_df) > 0:
        noise_comments = noise_df["cleaned_text"].tolist()
        noise_questions = noise_df["patient_question"].tolist()
        noise_embeddings = np.vstack(noise_df["embedding"].tolist())
        
        if len(noise_comments) >= 5:
            centroid = noise_embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(noise_embeddings, centroid).flatten()
            representative_idx = np.argmax(sims) if len(noise_comments) > 0 else None
            representative_comment = noise_comments[representative_idx] if representative_idx is not None else ""
            representative_question = noise_questions[representative_idx] if representative_idx is not None else ""

            noise_keywords = kw_model.extract_keywords(
                " ".join(noise_comments[:100]),
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_mmr=True,
                diversity=0.5,
                top_n=10
            )
            noise_keywords = [kw[0] for kw in noise_keywords if kw]

            # Create noise cluster details
            noise_details = pd.DataFrame({
                "patient_question": noise_questions,
                "cleaned_text": noise_comments,
                "keywords": [[kw[0] for kw in kw_model.extract_keywords(
                    comment,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5)] for comment in noise_comments],
                "distance_to_centroid": sims
            })
            noise_details.to_csv(os.path.join(cluster_details_dir, "noise_cluster.csv"), index=False)

            cluster_summary.append({
                "cluster": "noise_cluster",
                "size": len(noise_df),
                "keywords": noise_keywords,
                "representative_comment": representative_comment,
                "representative_question": representative_question
            })

    return pd.DataFrame(cluster_summary).sort_values("size", ascending=False)

def plot_projections(embeddings, labels):
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Convert noise cluster label for visualization
    plot_labels = np.where(labels == -1, max(labels) + 1, labels)
    
    methods = {
        "PCA": PCA(n_components=2).fit_transform(embeddings),
        "UMAP": umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(embeddings),
        "TSNE": TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    }
    
    for name, proj in methods.items():
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=proj[:, 0], y=proj[:, 1],
            hue=plot_labels, palette="viridis",
            s=30, alpha=0.7, legend=None
        )
        plt.title(f"{name} Projection")
        plt.savefig(f"{PLOT_DIR}/{name.lower()}_projection.png", dpi=300)
        plt.close()

def calculate_metrics(embeddings, reduced_emb, labels, clusterer=None):
    # For metrics calculation, we'll treat noise as its own cluster
    adjusted_labels = np.where(labels == -1, max(labels) + 1, labels)
    valid_idx = adjusted_labels != (max(labels) + 1)  # Exclude noise if needed
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(x) for x in obj]
        return obj
    
    try:
        if len(set(adjusted_labels[valid_idx])) < 2:
            print("Warning: Not enough clusters for metrics calculation")
            return None
            
        cluster_counts = Counter(adjusted_labels)
        metrics = {
            'cluster_stats': {
                'n_clusters': len([k for k in cluster_counts if k != (max(labels) + 1)]),
                'n_noise_points': cluster_counts.get(max(labels) + 1, 0),
                'avg_cluster_size': np.mean([v for k, v in cluster_counts.items() if k != (max(labels) + 1)]),
                'cluster_size_distribution': {str(k): v for k, v in cluster_counts.items() if k != (max(labels) + 1)}
            },
            'silhouette_score': silhouette_score(reduced_emb[valid_idx], adjusted_labels[valid_idx]),
            'calinski_harabasz_score': calinski_harabasz_score(reduced_emb[valid_idx], adjusted_labels[valid_idx]),
            'davies_bouldin_score': davies_bouldin_score(reduced_emb[valid_idx], adjusted_labels[valid_idx])
        }
        
        return convert_numpy_types(metrics)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# -----------------------------
# Main Workflow
# -----------------------------
def main():
    # Load and clean data
    print("Loading data...")
    df = load_clinical_data(INPUT_FOLDER)
    
    print("Cleaning text...")
    df["cleaned_text"] = df["patient_question"].progress_apply(clean_clinical_text)
    df = df[df["cleaned_text"].str.strip().astype(bool)]
    
    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["cleaned_text"].tolist(), show_progress_bar=True)
    
    # Cluster with optimized parameters
    print("Clustering data...")
    labels, clusterer, reduced_emb = optimized_clustering(embeddings, len(df))
    df["cluster"] = labels 
    
    # Save results
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Calculate and save metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(embeddings, reduced_emb, labels)
    if metrics:
        with open(os.path.join(OUTPUT_FOLDER, METRICS_FILE), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Generate insights and plots
    print("Extracting cluster insights...")
    summary = extract_cluster_insights(df, model, embeddings, output_folder=OUTPUT_FOLDER)
    summary.to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_FILE), index=False)
    
    print("Creating visualizations...")
    plot_projections(embeddings, labels)
    
    print("\n=== Cluster Metrics ===")
    if metrics and 'silhouette_score' in metrics:
        print(f"- Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"- Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f}")
        print(f"- Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
        print(f"- Number of clusters: {metrics['cluster_stats']['n_clusters']}")
        print(f"- Noise points: {metrics['cluster_stats']['n_noise_points']}")
        print(f"- Avg cluster size: {metrics['cluster_stats']['avg_cluster_size']:.1f}")
    else:
        print("No valid metrics could be calculated")
    

if __name__ == "__main__":
    main()