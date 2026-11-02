# Week 3

Clustering the STL-10 embeddings.

* **`01_KMeans.ipynb`**: k-means with K = 10, checked against the true classes.
* **`02_KMedoids.ipynb`**: every centre must be a real image (medoid).
* **`03_Hierarchical.ipynb`**: Ward linkage, dendrogram, cut at K = 10.
* **`04_Compare.ipynb`**: all three methods on layer 12 and layer 3.
* **`05_PCA.ipynb`**: how much variance the components keep, 2-D and 3-D projections.
* **`06_PCA_then_Clustering.ipynb`**: keep the first *d* components, then cluster.

## Data

The embeddings come from `week_2`, so run those notebooks first:

* `week_2/01_Download_Data.ipynb`
* `week_2/03_Extract_Features.ipynb`

This week reads them from `../week_2/data/`. No PyTorch is needed here; the features are
already extracted.

## Usage

1. Run `uv sync` to install dependencies.
