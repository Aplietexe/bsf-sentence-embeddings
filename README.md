# bsf-sentence-embeddings

This repository has the code for the blog post "Do block-sparse featurizers find concept manifolds in sentence embeddings?". A block-sparse featurizer (BSF) reconstructs an activation from a few active blocks of directions, where a sparse autoencoder (SAE) uses single directions. [Fel et al. (2026)](https://arxiv.org/abs/2606.25234) found that BSF blocks recover continuous concepts in vision models, such as the full circle of curve orientations in InceptionV1. The post applies their tests to MPNet, MiniLM, E5 and BGE embeddings of documents from 20 Newsgroups, arXiv and DBpedia, and to synthetic circles of known shape. It compares the blocks with SAE features, k-means clusters and untrained random blocks. On text the blocks behave like clusters with local coordinates, and none of the tests identifies a concept manifold.

## Run the experiments

The code needs [uv](https://docs.astral.sh/uv/) and installs on Linux, on Windows on x86 and on Macs with Apple silicon and macOS 14 or later. Install the dependencies with

```
uv sync
```

`embed.py` downloads the three datasets and the four encoders at fixed versions, samples the documents and embeds them into `data/`. It has to run before anything else and takes about 9 minutes on an RTX 4090.

```
uv run python embed.py
```

The experiment scripts can then run in any order. Each prints the numbers from its section of the post and saves the underlying values to `results/<script>.csv`. The times below are for an RTX 4090 with no trained models saved yet.

| Script | Section | Results | Time |
| --- | --- | --- | ---: |
| `detection.py` | 2 | Figures 1 and 3, and Table 1 | 17 min |
| `geometry.py` | 3 | Correlations between encoders, Figure 5, and the variance around category means | 0.5 min |
| `circles.py` | 4 | Figure 6 and the circle-recovery numbers in the text | 1 min |
| `stable_rank.py` | 4 | Table 2 | 11 min |
| `description_length.py` | 5 | Figure 7, the index bits, and the reconstruction R² in the text | 10 min |
| `steering.py` | 6 | Figure 8 | 1 min |

Trained models are saved in `models/` and reused, so `steering.py` loads the models that `detection.py` trained instead of training them again. The scripts also run on a CPU, where training is much slower.

## Draw the figures

The scripts in `plots/` draw the post's eight figures into `figures/`. Each has the same name as the experiment script whose results it draws, and together they take a few seconds. Figures 2, 4 and 6 also read the embeddings or the saved models.

```
for name in detection geometry circles description_length steering; do uv run python -m plots.$name; done
```

## Model and training

`common.py` holds the code the experiments share, which loads the data, splits it, generates the synthetic circles and defines the BSF. The BSF follows `GrassmannianBSF` from [goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer). Each block is a set of orthonormal directions that the encoder and decoder share, scaled by a learned gain, and each input keeps the k blocks whose codes have the largest norm. The SAE baseline is the same model with blocks of size one.

Each training step uses the whole training set, and the model learns to reconstruct each training embedding from a copy with Gaussian noise added. Training uses Adam with a one-cycle learning-rate schedule for 1500 steps and a peak learning rate of 0.001. The circle models in `circles.py` instead use the number of steps and peak learning rate that gave the lowest validation reconstruction error among 1500 or 3000 steps and 0.0003, 0.001 or 0.003.
