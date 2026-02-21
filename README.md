density-based clustering for text embeddings. takes pre-computed vectors (e.g. OpenAI `text-embedding-3-large`), runs DBSCAN or HDBSCAN, exports labeled clusters to Excel. includes a parameter sweep tool to find the right threshold without guessing.

```bash
pip install -r requirements.txt
```

[![python](https://img.shields.io/badge/python-3.8+-93450a.svg?style=flat-square)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-grey.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## what it does

you already have embeddings in a CSV. this tool clusters them.

- **DBSCAN** with cosine similarity — works directly on high-dimensional vectors, no reduction needed
- **HDBSCAN** with PCA + euclidean — reduces to 30 dims first, handles varying density
- **parameter sweep** — brute-force grid search over similarity thresholds, prints cluster counts so you can find the sweet spot
- **silhouette scoring** — automatic quality metric after each run
- **Excel export** — sorted by cluster ID, noise labeled as `-1`

there's also a Jupyter notebook (`embedding_clustering_toolkit.ipynb`) that wraps everything into classes with visualizations, auto-detected input formats, and side-by-side DBSCAN vs HDBSCAN comparison plots.

## input format

CSV with a `Name` column and six columns (`1` through `6`) of comma-separated floats. the six chunks concatenate into a single 3072-dim vector per row.

```csv
Name,1,2,3,4,5,6
"some entity","0.012,0.034,...","0.056,0.078,...","...","...","...","..."
```

the split-into-6-columns layout is a workaround for Excel's ~32k character cell limit. rows that don't produce exactly 3072 floats after concatenation are silently dropped.

the notebook also supports single-column formats: a column named `embedding` or `vector` containing either a JSON array or comma-separated string.

## scripts

### `classify.py` — DBSCAN clustering

runs DBSCAN with cosine distance on raw 3072-dim vectors. similarity threshold is converted to epsilon: `eps = 1 - threshold`.

```
SIMILARITY_THRESHOLD = 0.78    # cosine similarity cutoff
MIN_SAMPLES = 2                # core point threshold
VECTOR_DIMENSION = 3072        # expected vector length
```

prints: cluster count, noise ratio, average cluster size, per-cluster sizes, silhouette score.

### `classify_hdbscan.py` — HDBSCAN clustering

reduces vectors to 30 dimensions via PCA, then runs HDBSCAN with euclidean distance. PCA is needed because euclidean distance degrades in high dimensions.

```
N_COMPONENTS = 30              # PCA output dims
METRIC = 'euclidean'           # HDBSCAN distance metric
min_cluster_size = 2
min_samples = 1
cluster_selection_epsilon = 0.0
```

### `sweet_spot_finder.py` — parameter search

sweeps similarity thresholds from 0.995 down to 0.800 in steps of 0.005. for each value, runs DBSCAN and prints the cluster count. helps you find where meaningful clusters emerge before committing to a threshold.

```
threshold range: 0.995 → 0.800 (step -0.005)
min_samples: 2
```

### `embedding_clustering_toolkit.ipynb` — full notebook

class-based workflow that unifies everything above:

- `ClusteringConfig` — all parameters in one dataclass
- `EmbeddingDataLoader` — multi-format CSV loader with validation
- `ParameterSearcher` — grid search with 4-panel matplotlib plots
- `DBSCANClusterer` / `HDBSCANClusterer` — clustering with stats
- `ResultsExporter` — single or dual Excel export
- `ClusterVisualizer` — PCA 2D scatter plots, size bar charts, method comparison

HDBSCAN is optional in the notebook — falls back gracefully if not installed.

## configuration

no CLI flags. all parameters are module-level constants or dataclass fields. edit the source files directly.

| parameter | default | used by |
|:---|:---|:---|
| `input_csv_path` | `'path/to/your/input.csv'` | all |
| `output_xlsx_path` | `'path/to/your/output.xlsx'` | all |
| `VECTOR_DIMENSION` | `3072` | all |
| `SIMILARITY_THRESHOLD` | `0.78` | DBSCAN, sweep |
| `MIN_SAMPLES` (DBSCAN) | `2` | DBSCAN, sweep |
| `N_COMPONENTS` | `30` | HDBSCAN |
| `METRIC` | `'euclidean'` | HDBSCAN |
| `min_cluster_size` | `2` | HDBSCAN |
| `min_samples` (HDBSCAN) | `1` | HDBSCAN |
| `cluster_selection_epsilon` | `0.0` | HDBSCAN |
| `threshold_range` | `(0.995, 0.800, -0.005)` | sweep |

## output format

Excel `.xlsx` with two columns:

| column | type | description |
|:---|:---|:---|
| `Name` | string | original label from input |
| `cluster` | integer | cluster ID (`0, 1, 2, ...`) or `-1` for noise |

sorted ascending by cluster ID. noise points appear first.

## dependencies

| package | min version | role |
|:---|:---|:---|
| `pandas` | 1.5.0 | CSV/Excel I/O |
| `numpy` | 1.23.0 | vector assembly |
| `scikit-learn` | 1.2.0 | DBSCAN, PCA, silhouette |
| `hdbscan` | 0.8.33 | HDBSCAN algorithm |
| `openpyxl` | 3.1.0 | Excel write engine (implicit, used by pandas) |
| `matplotlib` | 3.7.0 | visualization (notebook) |
| `seaborn` | 0.12.0 | plot styling (notebook) |
| `tqdm` | 4.65.0 | progress bars (notebook) |
| `jupyter` | 1.0.0 | notebook runtime |

## notes

- the parameter sweep reloads the CSV on every iteration to avoid mutation across runs. works fine, just not optimized for speed
- silhouette score is skipped when there's only 1 cluster or when every point is its own cluster
- sklearn's DBSCAN doesn't use a spatial index for cosine metric, so the sweep scales as O(N^2 * D) per iteration
- this tool doesn't generate embeddings — bring your own vectors

## license

MIT
