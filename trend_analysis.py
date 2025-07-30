import re
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import umap

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.dropna(subset=['comment_text'])
    return df

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def cluster_topics(df: pd.DataFrame, text_col: str) -> Tuple[pd.DataFrame, HDBSCAN, TfidfVectorizer]:
    df['clean_text'] = df[text_col].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine'
                        )
    embeddings = reducer.fit_transform(tfidf_matrix)

    clusterer = HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    df['cluster'] = labels
    return df, clusterer, tfidf


def merge_similar_clusters(tfidf_matrix, labels: np.ndarray, similarity_threshold: float = 0.85) -> np.ndarray:
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    cluster_vectors = []

    for cid in unique_clusters:
        idx = np.where(labels == cid)[0]
        mean_vector = tfidf_matrix[idx].mean(axis=0)
        cluster_vectors.append(np.asarray(mean_vector).flatten())

    similarity_matrix = cosine_similarity(normalize(np.vstack(cluster_vectors)))

    merged_map = {}
    merged_to = {}

    for i, cid1 in enumerate(unique_clusters):
        for j, cid2 in enumerate(unique_clusters):
            if i >= j:
                continue
            if similarity_matrix[i, j] >= similarity_threshold:
                target = min(cid1, cid2)
                source = max(cid1, cid2)
                merged_to[source] = target

    for c in np.unique(labels):
        cur = c
        while cur in merged_to:
            cur = merged_to[cur]
        merged_map[c] = cur

    return np.array([merged_map.get(c, c) for c in labels])

def get_top_terms_per_cluster(
    tfidf_matrix,
    labels: np.ndarray,
    tfidf: TfidfVectorizer,
    n_terms: int = 10,
    max_clusters: int = 10,
    banned_words: List[str] = None
) -> Dict[int, List[str]]:
    terms = np.array(tfidf.get_feature_names_out())
    top_clusters = pd.Series(labels).value_counts().sort_values(ascending=False).index
    top_clusters = [cid for cid in top_clusters if cid != -1][:max_clusters]

    banned_words = set(w.lower() for w in (banned_words or []))

    cluster_terms = {}
    for cid in top_clusters:
        idx = np.where(labels == cid)[0]
        cluster_vector = tfidf_matrix[idx].mean(axis=0).A1
        sorted_idx = np.argsort(cluster_vector)[::-1]

        keywords = []
        for i in sorted_idx:
            word = terms[i]
            if word in banned_words or word in keywords:
                continue
            keywords.append(word)
            if len(keywords) >= n_terms:
                break

        cluster_terms[cid] = keywords

    return cluster_terms

def name_and_describe_clusters(cluster_keywords: Dict[int, List[str]]) -> Tuple[List[str], Dict[int, str]]:
    summaries = []
    cluster_names = {}
    for cluster_id, words in cluster_keywords.items():
        if not words:
            continue
        if len(words) == 0:
            continue
        if cluster_id == -1:
            name = "Other"
            summary = f"### Topic -1 \n **Keywords**: None \n **Name**: {name} \n"
            summaries.append(summary)
            cluster_names[cluster_id] = name
            continue
        name = " / ".join(words[:2]).title() if len(words) >= 2 else words[0].title()
        summary = (
            f"""### Topic {cluster_id} \n **Keywords**: {', '.join(words)} \n  \n **Name**: {name} \n"""
        )
        summaries.append(summary)
        cluster_names[cluster_id] = name
    return summaries, cluster_names

def plot_trends_over_time(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    cluster_col: str = 'cluster',
    cluster_names: Dict[int, str] = None
) -> None:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df['date'] = df[timestamp_col].dt.date
    trend_df = df.groupby(['date', cluster_col]).size().unstack(fill_value=0)

    # Keep only top 10 clusters
    top_clusters = trend_df.sum(axis=0).nlargest(10).index
    trend_df = trend_df[top_clusters]

    # Rename columns using the name mapping
    if cluster_names:
        new_columns = [f"{cid}: {cluster_names.get(cid, f'Topic {cid}')}" for cid in top_clusters]
        trend_df.columns = new_columns

    # Plot
    ax = trend_df.plot(figsize=(15, 5), title="Topic Frequency Over Time")
    ax.set_yscale('log')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title="Cluster")
    plt.xlabel("Date")
    plt.ylabel("Comment Count (log scale)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()