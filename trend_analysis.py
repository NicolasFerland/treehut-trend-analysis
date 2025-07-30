# trend_analysis.py

import os
import re
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and parse a CSV file with timestamp and comment text."""
    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df = df.dropna(subset=['comment_text'])
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

def preprocess_text(text: str) -> str:
    """Clean comment text by removing links, punctuation, and extra spaces."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def cluster_topics(df: pd.DataFrame, text_col: str) -> Tuple[pd.DataFrame, DBSCAN, TfidfVectorizer]:
    """Cluster comments into topics using TF-IDF, TSNE, and DBSCAN."""
    df['clean_text'] = df[text_col].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = reducer.fit_transform(tfidf_matrix.toarray())

    clusterer = DBSCAN(eps=2, min_samples=5)
    labels = clusterer.fit_predict(embeddings)

    df['cluster'] = labels
    return df, clusterer, tfidf

def merge_similar_clusters(
    tfidf_matrix,
    labels: np.ndarray,
    similarity_threshold: float = 0.95
) -> np.ndarray:
    """Merge clusters whose mean TF-IDF vectors are very similar."""
    unique_clusters = np.unique(labels[labels != -1])
    cluster_vectors = []

    for cid in unique_clusters:
        idx = np.where(labels == cid)[0]
        cluster_vec = tfidf_matrix[idx].mean(axis=0)
        cluster_vectors.append(cluster_vec)

    merged = labels.copy()
    merged_map = {}
    merged_to = {}

    similarities = cosine_similarity(normalize(np.vstack(cluster_vectors)))

    for i, cid1 in enumerate(unique_clusters):
        for j, cid2 in enumerate(unique_clusters):
            if i < j and similarities[i, j] > similarity_threshold:
                merged_to[cid2] = cid1

    for i, label in enumerate(merged):
        if label in merged_to:
            merged[i] = merged_to[label]


def get_top_terms_per_cluster(
    tfidf_matrix,
    labels: np.ndarray,
    tfidf: TfidfVectorizer,
    n_terms: int = 10,
    max_clusters: int = 10,
    min_df_ratio: float = 0.005,
    max_df_ratio: float = 0.7
) -> Dict[int, List[str]]:
    """
    Extract top N keywords for each cluster, filtering low-information terms based on frequency.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix of documents.
        labels: Cluster labels assigned to documents.
        tfidf: The fitted TfidfVectorizer.
        n_terms: Number of top terms to extract per cluster.
        max_clusters: Number of clusters to return based on frequency.
        min_df_ratio: Minimum % of documents a word must appear in.
        max_df_ratio: Maximum % of documents a word can appear in.

    Returns:
        Dictionary mapping cluster ID to a list of top terms.
    """
    cluster_terms = defaultdict(list)
    terms = tfidf.get_feature_names_out()
    label_counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    top_clusters = [cid for cid in label_counts.index if cid != -1][:max_clusters]

    # Calculate term document frequencies
    doc_freq = np.asarray((tfidf_matrix > 0).sum(axis=0)).ravel()
    doc_count = tfidf_matrix.shape[0]
    term_df_ratio = doc_freq / doc_count

    # Filter based on DF ratio
    valid_term_mask = (term_df_ratio >= min_df_ratio) & (term_df_ratio <= max_df_ratio)
    valid_terms = set(terms[valid_term_mask])

    seen_terms = set()

    for cluster_id in top_clusters:
        idx = np.where(labels == cluster_id)[0]
        cluster_tfidf = tfidf_matrix[idx].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[::-1]

        keywords = []
        for i in top_indices:
            word = terms[i]
            if word not in valid_terms or word in seen_terms:
                continue
            keywords.append(word)
            seen_terms.add(word)
            if len(keywords) >= n_terms:
                break
        cluster_terms[cluster_id] = keywords

    return cluster_terms


def plot_trends_over_time(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    cluster_col: str = 'cluster',
    cluster_names: Dict[int, str] = None
) -> None:
    """Plot topic frequency over time."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df['date'] = df[timestamp_col].dt.date
    trend_df = df.groupby(['date', cluster_col]).size().unstack(fill_value=0)

    if cluster_names:
        trend_df.columns = [cluster_names.get(i, f"Topic {i}") for i in trend_df.columns]

    trend_df.plot(figsize=(15, 5), title="Topic Frequency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Comment Count")
    plt.grid(True)
    plt.show()

def name_and_describe_clusters(cluster_keywords: Dict[int, List[str]]) -> Tuple[List[str], Dict[int, str]]:
    """Heuristic-based topic naming without LLM."""
    summaries = []
    cluster_names = {}
    for cluster_id, words in cluster_keywords.items():
        name = " / ".join(words[:2]).title()
        summary = (
            f"\n### Topic {cluster_id}\n"
            f"**Keywords**: {', '.join(words)}\n"
            f"**Name**: {name}\n"
            f"**Summary**: Comments related to '{name.lower()}', based on frequent word usage."
        )
        summaries.append(summary)
        cluster_names[cluster_id] = name
    return summaries, cluster_names
