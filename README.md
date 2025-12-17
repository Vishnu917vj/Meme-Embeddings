# Movie Embeddings Jupyter Notebook

This repository contains a Jupyter Notebook (`Movie Embeddings.ipynb`) that demonstrates how to train simple movie embeddings using contrastive learning on a dataset of movie pairs. The goal is to learn vector representations (embeddings) for movies based on pairwise similarity labels, which can be used for tasks like finding similar movies or visualizing movie relationships.

## Overview

The notebook performs the following steps:
- Loads a CSV dataset of movie pairs with similarity labels (e.g., "movie_pairs.csv" with columns: `Movie1`, `Movie2`, `Label` where Label is 1 for similar, 0 for dissimilar).
- Preprocesses the data to map movies to integer IDs.
- Defines a custom PyTorch Dataset and DataLoader for training.
- Implements a simple embedding model using PyTorch (an embedding layer followed by a projection head for contrastive learning).
- Trains the model using a pairwise contrastive loss (encouraging similar pairs to have close embeddings and dissimilar pairs to be far apart).
- Extracts learned embeddings.
- Computes cosine similarity to find similar movies.
- Visualizes the embeddings in 2D using t-SNE for clustering analysis.

Example output includes:
- Training loss over 100 epochs.
- Embeddings for individual movies.
- Top similar movies based on cosine similarity.
- A 2D scatter plot of movie embeddings.

This is a basic example inspired by Word2Vec-style embeddings and contrastive learning techniques like SimCLR, adapted for tabular movie pair data.

## Requirements

To run the notebook, you'll need Python 3.x and the following libraries:
- `pandas` (for data loading and manipulation)
- `torch` (for model training and embeddings)
- `scikit-learn` (for t-SNE visualization)
- `matplotlib` (for plotting)

You can install them using pip:

```
pip install pandas torch scikit-learn matplotlib
```

The notebook assumes a small dataset with 30 unique movies (as in the example), but it can scale to larger datasets.

## Dataset

The notebook loads data from `movie_pairs.csv` (not included in this repo). This file should contain:
- `Movie1`: Name or ID of the first movie.
- `Movie2`: Name or ID of the second movie.
- `Label`: Binary label (1 = similar, 0 = dissimilar).

Example structure:
```
Movie1,Movie2,Label
MovieA,MovieB,1
MovieA,MovieC,0
...
```

If you don't have this file, you can create a synthetic one or replace it with your own pairwise similarity data.

## Usage

1. Clone this repository:
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Ensure `movie_pairs.csv` is in the same directory as the notebook (or update the file path in the code).

3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

4. Open `Movie Embeddings.ipynb` and run the cells sequentially.

Key hyperparameters you can tweak:
- Embedding dimension (`emb_dim=128`)
- Batch size (`batch_size=64`)
- Number of epochs (`epochs=100`)
- Contrastive loss margin (`margin=1.0`)

After training, you can:
- Query embeddings for specific movie IDs.
- Find top-k similar movies using cosine similarity.
- Generate a t-SNE plot to visualize clusters (e.g., similar movies grouping together).

## Model Details

- **Model Architecture**: `MovieEmbeddingModel`
  - Embedding layer: Maps movie IDs to dense vectors.
  - Projection head: A small MLP (Linear -> ReLU -> Linear) to refine representations during training.
  - Normalization: L2-normalizes embeddings for cosine similarity.

- **Loss Function**: `ContrastiveLoss`
  - For similar pairs (label=1): Minimizes Euclidean distance.
  - For dissimilar pairs (label=0): Pushes distance beyond a margin.

- **Optimizer**: Adam with learning rate 0.001.

The trained embeddings are stored in `model.movie_embedding.weight` (a tensor of shape `[num_movies, emb_dim]`).

## Visualization

The notebook includes a t-SNE reduction to 2D for embedding visualization. Points are annotated with movie indices (0 to num_movies-1). Adjust perplexity or other t-SNE params for better clustering if needed.

Example plot (generated in the notebook):
- Scatter plot showing movie embeddings in 2D space.
- Helps identify clusters of similar movies.

## Limitations

- Assumes a small number of unique movies (e.g., 30); for larger datasets, consider GPU acceleration or optimizations.
- The dataset is not included; results depend on the quality of pairwise labels.
- No evaluation metrics (e.g., accuracy on held-out pairs) are implemented—add them if needed.
- This is a toy example; for production, consider pre-trained models like those from Sentence Transformers for text-based movie descriptions.

## Contributing

Feel free to fork, submit issues, or pull requests if you have improvements!

## License

This project is licensed under the MIT License.
