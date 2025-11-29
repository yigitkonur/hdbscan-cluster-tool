<h1 align="center">🔬 Embedding Clustering Toolkit 🔬</h1>
<h3 align="center">Stop guessing cluster counts. Start discovering natural groupings.</h3>

<p align="center">
  <strong>
    <em>The ultimate clustering toolkit for high-dimensional text embeddings. It finds the natural structure in your data using DBSCAN & HDBSCAN—no magic numbers required.</em>
  </strong>
</p>

<p align="center">
  <!-- Package Info -->
  <a href="#"><img alt="python" src="https://img.shields.io/badge/python-3.9+-4D87E6.svg?style=flat-square"></a>
  <a href="#"><img alt="embeddings" src="https://img.shields.io/badge/embeddings-3072d-4D87E6.svg?style=flat-square"></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <!-- Features -->
  <a href="#"><img alt="license" src="https://img.shields.io/badge/License-MIT-F9A825.svg?style=flat-square"></a>
  <a href="#"><img alt="platform" src="https://img.shields.io/badge/platform-macOS_|_Linux_|_Windows-2ED573.svg?style=flat-square"></a>
</p>

<p align="center">
  <img alt="no k" src="https://img.shields.io/badge/🎯_no_k_required-automatic_cluster_discovery-2ED573.svg?style=for-the-badge">
  <img alt="jupyter" src="https://img.shields.io/badge/📓_jupyter_ready-interactive_analysis-2ED573.svg?style=for-the-badge">
</p>

<div align="center">

### 🧭 Quick Navigation

[**⚡ Get Started**](#-get-started-in-60-seconds) •
[**✨ Key Features**](#-feature-breakdown-the-secret-sauce) •
[**🎮 Usage & Examples**](#-usage-fire-and-forget) •
[**⚙️ Configuration**](#️-configuration--customization) •
[**🆚 Why This Works**](#-why-this-slaps-k-means)

</div>

---

**Embedding Clustering Toolkit** is the analysis partner your embeddings deserve. Stop arbitrarily picking "k=5" and praying your clusters make sense. This toolkit uses density-based algorithms that discover the natural groupings in your data—automatically identifying how many clusters exist and which points are just noise.

<div align="center">
<table>
<tr>
<td align="center">
<h3>🎯</h3>
<b>Auto Cluster Detection</b><br/>
<sub>No predefined k needed</sub>
</td>
<td align="center">
<h3>🔍</h3>
<b>Parameter Search</b><br/>
<sub>Find optimal thresholds</sub>
</td>
<td align="center">
<h3>📊</h3>
<b>Quality Metrics</b><br/>
<sub>Silhouette scores built-in</sub>
</td>
<td align="center">
<h3>🗑️</h3>
<b>Noise Handling</b><br/>
<sub>Outliers isolated cleanly</sub>
</td>
</tr>
</table>
</div>

How it slaps:
- **You:** Load your OpenAI/Cohere/any embeddings CSV
- **Toolkit:** Searches parameters, finds natural clusters, isolates noise
- **You:** Export to Excel, visualize, analyze
- **Result:** Meaningful groupings without arbitrary decisions. Go grab a coffee. ☕

---

## 💥 Why This Slaps K-Means

Clustering embeddings with K-Means is like forcing your data into boxes it doesn't fit. Density-based clustering finds the boxes that actually exist.

<table align="center">
<tr>
<td align="center"><b>❌ The K-Means Way (Pain)</b></td>
<td align="center"><b>✅ The DBSCAN Way (Glory)</b></td>
</tr>
<tr>
<td>
<ol>
  <li>Guess k=10. Run K-Means.</li>
  <li>Results look weird. Try k=15.</li>
  <li>Still bad. Maybe k=8?</li>
  <li>Elbow method says k=12. Sure, why not.</li>
  <li>Get clusters that mix unrelated items.</li>
</ol>
</td>
<td>
<ol>
  <li>Run the notebook.</li>
  <li>Algorithm finds 47 natural clusters.</li>
  <li>Outliers are flagged as noise.</li>
  <li>Silhouette score confirms quality.</li>
  <li>Export and ship. Done. 🚀</li>
</ol>
</td>
</tr>
</table>

We're not forcing structure. We're **discovering structure** with cosine similarity, density estimation, and automatic parameter optimization that processes your high-dimensional embeddings the right way.

---

## 🚀 Get Started in 60 Seconds

### Prerequisites

- Python 3.9+
- Your embeddings in CSV format

### Installation

```bash
# Clone the repository
git clone https://github.com/yigitkonur/embedding-clustering-toolkit.git
cd embedding-clustering-toolkit

# Install dependencies
pip install -r requirements.txt
```

### Quick Start with Jupyter Notebook

The **recommended way** to use this toolkit is through the interactive Jupyter notebook:

```bash
# Launch Jupyter
jupyter notebook embedding_clustering_toolkit.ipynb
```

The notebook provides:
- 📋 **Configurable parameters** at the top
- 📊 **Interactive visualizations** 
- 🔍 **Parameter search** with visual results
- 💾 **One-click export** to Excel

### Quick Start with Scripts

If you prefer command-line scripts:

```bash
# 1. Edit the input path in classify.py
# 2. Run DBSCAN clustering
python classify.py

# Or run HDBSCAN with PCA
python classify_hdbscan.py

# Or find optimal parameters first
python sweet_spot_finder.py
```

---

## 🎮 Usage: Fire and Forget

### Using the Jupyter Notebook (Recommended)

**1. Configure Your Analysis**

```python
config = ClusteringConfig(
    input_csv_path="your_embeddings.csv",
    vector_dimension=3072,  # Match your embedding model
    similarity_threshold=0.78,  # Higher = tighter clusters
    min_samples=2,  # Minimum points for a cluster
)
```

**2. Run All Cells**

The notebook walks you through:
1. Loading and validating your embeddings
2. Finding optimal parameters (optional but recommended)
3. Running DBSCAN and/or HDBSCAN clustering
4. Visualizing results
5. Exporting to Excel

**3. Analyze Results**

```
📊 DBSCAN Results:
   ├─ Clusters: 47
   ├─ Noise points: 23 (4.2%)
   ├─ Clustered points: 527 (95.8%)
   ├─ Avg cluster size: 11.2
   └─ Silhouette score: 0.634
```

### CSV Format

Your CSV should have embeddings split across columns or in a single column:

**Split format (default):**
```csv
Name,1,2,3,4,5,6
"Document A","0.001,0.023,...","0.045,0.012,...","...",...
```

**Single column format:**
```csv
Name,embedding
"Document A","[0.001, 0.023, 0.045, ...]"
```

---

## ✨ Feature Breakdown: The Secret Sauce

<div align="center">

| Feature | What It Does | Why You Care |
| :---: | :--- | :--- |
| **🎯 DBSCAN Clustering**<br/>Cosine similarity | Groups embeddings by semantic similarity without predefined k | Natural clusters that actually make sense |
| **⚡ HDBSCAN + PCA**<br/>Dimensionality reduction | Reduces 3072D → 30D, then clusters | 10x faster on large datasets |
| **🔍 Parameter Search**<br/>Grid search optimization | Tests hundreds of threshold/min_samples combos | Find the "sweet spot" automatically |
| **📊 Quality Metrics**<br/>Silhouette scoring | Measures how well-separated your clusters are | Know if your clustering is actually good |
| **🗑️ Noise Detection**<br/>Outlier isolation | Flags points that don't belong anywhere | Clean clusters without forced assignments |
| **📈 Visualizations**<br/>PCA projections | 2D scatter plots of your clusters | See the structure in your data |
| **💾 Excel Export**<br/>One-click output | Sorted results with cluster IDs | Ready for downstream analysis |

</div>

---

## ⚙️ Configuration & Customization

### Key Parameters

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `similarity_threshold` | `0.78` | Cosine similarity cutoff (0-1). Higher = tighter clusters. |
| `min_samples` | `2` | Minimum points to form a cluster. |
| `vector_dimension` | `3072` | Expected embedding dimensions. |
| `n_pca_components` | `30` | PCA dimensions for HDBSCAN. |

### Choosing Parameters

<table align="center">
<tr>
<td><b>If you get...</b></td>
<td><b>Try...</b></td>
</tr>
<tr>
<td>Too many tiny clusters</td>
<td>Lower <code>similarity_threshold</code> (e.g., 0.70)</td>
</tr>
<tr>
<td>Everything in one cluster</td>
<td>Raise <code>similarity_threshold</code> (e.g., 0.85)</td>
</tr>
<tr>
<td>Too much noise</td>
<td>Lower <code>min_samples</code> to 1</td>
</tr>
<tr>
<td>Noisy clusters</td>
<td>Raise <code>min_samples</code> to 3-5</td>
</tr>
</table>

### Embedding Model Dimensions

| Model | Dimensions | Set `vector_dimension` to |
|:------|:----------:|:------------------------:|
| OpenAI text-embedding-3-large | 3072 | `3072` |
| OpenAI text-embedding-3-small | 1536 | `1536` |
| OpenAI text-embedding-ada-002 | 1536 | `1536` |
| Cohere embed-english-v3.0 | 1024 | `1024` |
| Voyage voyage-2 | 1024 | `1024` |
| Custom | varies | your dimension |

---

## 📁 Repository Structure

```
embedding-clustering-toolkit/
├── 📓 embedding_clustering_toolkit.ipynb  # Interactive notebook (START HERE)
├── 📜 classify.py                         # DBSCAN clustering script
├── 📜 classify_hdbscan.py                 # HDBSCAN + PCA script
├── 📜 sweet_spot_finder.py                # Parameter optimization script
├── 📋 sample.csv                          # Example embeddings data
├── 📋 requirements.txt                    # Python dependencies
└── 📖 README.md                           # You are here
```

---

## 🔍 Understanding the Output

### Cluster Labels

- **`cluster >= 0`**: Assigned to a specific cluster
- **`cluster = -1`**: Noise/outlier (doesn't fit any cluster)

### Quality Metrics

| Metric | Good | Acceptable | Poor |
|:-------|:----:|:----------:|:----:|
| **Silhouette Score** | > 0.5 | 0.25 - 0.5 | < 0.25 |
| **Noise Ratio** | < 10% | 10-30% | > 30% |
| **Avg Cluster Size** | > 5 | 3-5 | < 3 |

---

## 🔥 Common Issues & Quick Fixes

<details>
<summary><b>Expand for troubleshooting tips</b></summary>

| Problem | Solution |
| :--- | :--- |
| **All points are noise** | Lower `similarity_threshold` significantly (try 0.5-0.6) |
| **One giant cluster** | Raise `similarity_threshold` (try 0.85-0.95) |
| **Out of memory** | Use HDBSCAN with PCA instead of DBSCAN |
| **Invalid vector dimensions** | Check your CSV format matches the expected column structure |
| **hdbscan import error** | Run `pip install hdbscan` (may need C compiler on some systems) |
| **Slow clustering** | Use HDBSCAN + PCA or reduce `n_pca_components` |

</details>

---

## 🆚 DBSCAN vs HDBSCAN: When to Use Which

<table align="center">
<tr>
<td align="center"><b>DBSCAN</b></td>
<td align="center"><b>HDBSCAN + PCA</b></td>
</tr>
<tr>
<td>
<ul>
  <li>Smaller datasets (< 10K points)</li>
  <li>You know a good similarity threshold</li>
  <li>Uniform cluster densities</li>
  <li>Full dimensional analysis needed</li>
</ul>
</td>
<td>
<ul>
  <li>Large datasets (10K+ points)</li>
  <li>Varying cluster densities</li>
  <li>Speed is important</li>
  <li>Very high dimensions (3000+)</li>
</ul>
</td>
</tr>
</table>

---

## 🛠️ Advanced: Using as a Library

```python
from embedding_clustering_toolkit import (
    ClusteringConfig,
    EmbeddingDataLoader,
    DBSCANClusterer,
    ParameterSearcher
)

# Configure
config = ClusteringConfig(
    input_csv_path="my_embeddings.csv",
    vector_dimension=1536
)

# Load data
loader = EmbeddingDataLoader(config)
df, valid_df = loader.load()

# Find best parameters
searcher = ParameterSearcher(loader.get_vector_matrix())
results = searcher.search()
best = searcher.get_best_params(results)

# Cluster
clusterer = DBSCANClusterer(loader.get_vector_matrix())
labels = clusterer.fit(
    similarity_threshold=best['similarity_threshold'],
    min_samples=int(best['min_samples'])
)
```

---

<div align="center">

## 🌟 Star This Repo

If this toolkit saved you from K-Means hell, give it a ⭐

**Built with 🔥 because guessing cluster counts is a soul-crushing waste of time.**

MIT © [Yiğit Konur](https://github.com/yigitkonur)

</div>
