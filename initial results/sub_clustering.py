import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -----------------------------
# Config
# -----------------------------
USE_LLM_LABELS = True
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

CLUSTER_SUMMARY_FILE = "./Clinical_Cluster_Outputs/clinical_cluster_summary.csv"
CLUSTER_DETAILS_FOLDER = "./Clinical_Cluster_Outputs/cluster_details"
OUTPUT_TABLE_FILE = "cluster_analysis_output.csv"
MAIN_CLUSTER_MIN_KEYWORDS = 3
SUB_CLUSTER_MIN_KEYWORDS = 2
MAX_QUESTIONS_FOR_CONTEXT = 5

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text_for_keyword(text):
    """Clean text for keyword extraction"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join([w for w in text.split() if len(w) > 2])

def load_cluster_questions(cluster_id):
    """Load questions for a specific cluster from its detail file"""
    if cluster_id == "noise_cluster":
        file_path = os.path.join(CLUSTER_DETAILS_FOLDER, "noise_cluster.csv")
    else:
        file_path = os.path.join(CLUSTER_DETAILS_FOLDER, f"cluster_{cluster_id}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df['patient_question'].tolist()
    return []

def generate_cluster_label(representative_comment, keywords, sample_questions):
    """Generate human-readable cluster label using LLM with context"""
    if not USE_LLM_LABELS:
        return ' '.join(keywords[:2]).title()
    
    try:
        questions_context = "\n".join([f"- {q[:200]}" for q in sample_questions[:MAX_QUESTIONS_FOR_CONTEXT]])
        
        prompt = f"""
        As a clinical psychologist, analyze these patient concerns and create a concise clinical label:
        
        Representative example: "{representative_comment[:500]}"
        
        Key themes identified: {', '.join(keywords[:5])}
        
        Sample patient questions:
        {questions_context}
        
        Create a 2-4 word clinical label that:
        1. Captures the core cognitive distortion
        2. Uses professional terminology
        3. Is specific yet broad enough to cover variations
        
        Clinical label:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(LLM_DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        label = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = label.split("Clinical label:")[-1].strip()
        return label.split('\n')[0].strip('"\'')
    except Exception as e:
        print(f"Label generation failed: {e}")
        return ' '.join(keywords[:2]).title()

def extract_subcluster_keywords(questions, model, top_n=5):
    """Extract subcluster keywords from questions with optimized processing"""
    kw_model = KeyBERT(model=model)
    all_keywords = []
    
    batch_size = 100
    for i in tqdm(range(0, len(questions), batch_size), desc="Extracting subcluster keywords"):
        batch = questions[i:i+batch_size]
        batch_texts = [str(q) for q in batch if isinstance(q, str)]
        
        if batch_texts:
            keywords = kw_model.extract_keywords(
                batch_texts,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            all_keywords.extend([kw[0] for sublist in keywords for kw in sublist])
    
    keyword_counts = Counter(all_keywords)
    return [kw for kw, count in keyword_counts.most_common(20) if count >= SUB_CLUSTER_MIN_KEYWORDS]

def create_subclusters(questions, cluster_keywords, model):
    """Create subclusters and track question counts"""
    if len(questions) < 5:
        return ["General"] * len(questions), [], {"General": len(questions)}
    
    subcluster_keywords = extract_subcluster_keywords(questions, model)
    
    if not subcluster_keywords:
        return ["General"] * len(questions), [], {"General": len(questions)}
    
    question_embeddings = model.encode(questions)
    keyword_embeddings = model.encode(subcluster_keywords)
    
    similarity_matrix = cosine_similarity(question_embeddings, keyword_embeddings)
    subcluster_indices = np.argmax(similarity_matrix, axis=1)
    
    # Track subcluster counts
    subcluster_counts = defaultdict(int)
    subclusters = []
    for idx in subcluster_indices:
        subcluster = subcluster_keywords[idx]
        subclusters.append(subcluster)
        subcluster_counts[subcluster] += 1
    
    return subclusters, subcluster_keywords, dict(subcluster_counts)

# -----------------------------
# Main Processing
# -----------------------------
def process_clusters():
    global USE_LLM_LABELS, tokenizer, model
    
    print("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if USE_LLM_LABELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16 if LLM_DEVICE == "cuda" else torch.float32
            )
            print("LLM model loaded successfully")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            USE_LLM_LABELS = False
    
    print("Loading cluster data...")
    summary_df = pd.read_csv(CLUSTER_SUMMARY_FILE)
    
    # Convert string representation of keywords to list
    def parse_keywords(keyword_str):
        if pd.isna(keyword_str):
            return []
        return [kw.strip(" '") for kw in keyword_str.strip("[]").split(",")]
    
    summary_df['keywords'] = summary_df['keywords'].apply(parse_keywords)
    
    all_data = []
    subcluster_stats = []
    
    for _, cluster_info in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Processing clusters"):
        cluster_num = cluster_info['cluster']
        cluster_keywords = cluster_info['keywords']
        rep_comment = cluster_info['representative_comment']
        rep_question = cluster_info['representative_question']
        
        # Load all questions for this cluster
        cluster_questions = load_cluster_questions(cluster_num)
        
        if not cluster_questions:
            print(f"No questions found for cluster {cluster_num}")
            continue
        
        # Generate main cluster label
        main_label = generate_cluster_label(rep_comment, cluster_keywords, cluster_questions)
        
        # Create subclusters
        subcluster_labels, subcluster_keywords, subcluster_counts = create_subclusters(
            cluster_questions,
            cluster_keywords,
            embedding_model
        )
        
        # Record subcluster statistics
        for sub_label, count in subcluster_counts.items():
            subcluster_stats.append({
                'Main_Cluster': cluster_num,
                'Main_Cluster_Label': main_label,
                'Subcluster_Label': sub_label,
                'Question_Count': count,
                'Keywords': ', '.join(subcluster_keywords) if subcluster_keywords else ''
            })
        
        # Record all questions with their cluster assignments
        for i, question in enumerate(cluster_questions):
            sub_label = subcluster_labels[i] if i < len(subcluster_labels) else "General"
            
            all_data.append({
                'patient_question': question,
                'Cluster': cluster_num,
                'Main_Cluster_Label': main_label,
                'Subcluster_Label': sub_label,
                'Cluster_Keywords': ", ".join(cluster_keywords),
                'Subcluster_Keywords': sub_label
            })
    
    # Create and save outputs
    output_df = pd.DataFrame(all_data)
    output_df.to_csv(OUTPUT_TABLE_FILE, index=False)
    
    subcluster_stats_df = pd.DataFrame(subcluster_stats)
    subcluster_stats_df.to_csv("subcluster_statistics.csv", index=False)
    
    subcluster_counts = output_df.groupby(['Main_Cluster_Label', 'Subcluster_Label']).size().reset_index(name='Count')
    subcluster_counts.to_csv("subcluster_distribution.csv", index=False)
    
    print(f"\nAnalysis complete. Results saved to:")
    print(f"- Main output: {OUTPUT_TABLE_FILE}")
    print(f"- Subcluster statistics: subcluster_statistics.csv")
    print(f"- Subcluster distribution: subcluster_distribution.csv")

if __name__ == "__main__":
    process_clusters()