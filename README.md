# 🎺 Treehut Trend Insight Report — March 2025

## 📊 Executive Summary

This report analyzes **17,812 Instagram comments** from Treehut's posts during March 2025. The goal is to uncover **actionable trends** and track how discussions evolve over time, helping Treehut’s social team optimize content strategy with data-driven insights.

---

## 🔍 Key Findings

* 📈 A **significant spike in engagement** occurred around **March 21**, with topics focused on **Shaunstyle**, **Sabrina**, **Treehutor**, and **Zicoxuality**.
* 🗖️ A **clear weekly pattern** emerges, with comment activity typically **peaking from Friday through Monday**.

---

## 🎯 Actionable Insights

* **Capitalize on Trending Topics**: Track rising topics (like "Shaunstyle") and encourage content aligned with those themes—especially if associated with strong performance (e.g., high views or engagement).
* **Leverage Weekends**: Since engagement is higher Friday–Monday, schedule key campaigns during these windows.
* **Refine Topic Strategy**: Use identified clusters to tailor captions or influencer collaborations more effectively.

---

## 🧹 Project Overview

### 🗂 Code Structure

| File                        | Purpose                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| `trend_analysis.py`         | Modular pipeline for data loading, preprocessing, clustering, and trends |
| `trend_analysis_demo.ipynb` | Interactive notebook for running the full analysis and generating plots  |

### ⚙️ Methodology

* **Preprocessing**: Lowercases, strips URLs/symbols, and tokenizes text
* **TF-IDF Vectorization**: Converts comments into interpretable term vectors
* **UMAP + HDBSCAN**: Reduces dimensionality and detects dense topic clusters
* **Trend Analysis**: Tracks cluster prevalence over time using timestamps
* **Topic Summarization**: Uses rule-based heuristics to extract keywords per cluster

---

## 🚀 Getting Started

### 1. Set Up Your Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook trend_analysis_demo.ipynb
```

### 3. Add Your Data File

Expected file name: `engagements.csv`
It should contain the following columns:

* `timestamp`
* `media_id`
* `media_caption`
* `comment_text`

---

## 📘 Sample Output

### 🔑 Top Keywords Across Topics

```
love, scent, please, carry, canada, soft, good, smell, try, store
```

### 🧠 Example Topic Summaries

```
### Topic 0
**Keywords**: rose, vanilla, soft, smooth, favorite  
**Name**: Rose / Vanilla  
Summary: Comments frequently referencing scent preferences, especially rose and vanilla.

### Topic 2
**Keywords**: canada, please, carry, store, find  
**Name**: Canada / Please  
Summary: Requests for Treehut products to be available in Canadian stores.

### Topic -1
**Name**: Other  
Summary: Comments not fitting into any dominant cluster.
```

---

## 📈 Visualization

The trend plot shows:

* Topic frequencies over time
* Log-scaled y-axis for better visibility of both large and small clusters
* Descriptive topic names in the legend

Example:

`(figure not shown here)`

---

## 🌱 Extension Plan

> Ranked by impact vs. implementation effort:

| Feature                     | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| Time Granularity Control    | View trends by day, week, or custom intervals   |
| Sentiment Analysis          | Gauge tone and emotion across topic clusters    |
| Named Entity Recognition    | Highlight brands, places, or products           |
| Topic Merging / Hierarchies | Identify overlapping or related topics          |
| Instagram API Integration   | Enable near real-time ingestion of comments     |
| Streamlit Dashboard         | Make exploration easier for non-technical teams |
| LLM Integration             | Automatically name/summarize topics             |
| Metric-Aware Prioritization | Combine trends with view/click data for context |

---

## 🛃 Tool Disclosure

| Tool / Library | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| **Python**     | Core scripting                                         |
| `pandas`       | Data manipulation                                      |
| `matplotlib`   | Plotting                                               |
| `scikit-learn` | TF-IDF, cosine similarity, normalization               |
| `UMAP`         | Dimensionality reduction                               |
| `HDBSCAN`      | Topic clustering                                       |
| **ChatGPT**    | README editing and polish (not used in pipeline logic) |

---

## ✅ Evaluation Checklist

| Criteria              | Status                                                  |
| --------------------- | ------------------------------------------------------- |
| Actionable Insights   | ✅ Extracted with context and recommendations            |
| Clear Code Structure  | ✅ Modular, well-documented components                   |
| Visualization Quality | ✅ Trend plots with log scale and clear legends          |
| Extensibility         | ✅ Multiple future enhancements proposed and prioritized |
| AI Transparency       | ✅ Declared LLM usage for documentation only             |
