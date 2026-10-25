"""
This script takes the activity (positive) networks constructed in
make_networks.py and reduces them to embeddings that can
capture latent community structure. The script also includes a
validation run tested against the data from the Waller and 
Anderson (2021) paper.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, dok_matrix, issparse
from tqdm import tqdm
import os
import io
import json
import zstandard as zstd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import rankdata, pearsonr, spearmanr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, n_subs, hidden_dim, embed_dim, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(n_subs, hidden_dim), nn.ReLU(),
            # nn.Linear(hidden_dim, embed_dim),  # comment this and uncomment below for 3-layer
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_subs)
        )

    def forward(self, x, training=False):
        if training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        z = self.encoder(x)
        return self.decoder(z), z


def uniformity_loss(z):
    z    = nn.functional.normalize(z, dim=-1)
    diff = z.unsqueeze(0) - z.unsqueeze(1)
    sq   = (diff ** 2).sum(-1)
    return torch.log(torch.exp(-2 * sq).mean())


def covariance_penalty(z):
    z        = z - z.mean(dim=0)
    cov      = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov ** 2
    off_diag.fill_diagonal_(0)
    return off_diag.sum() / z.shape[1]


def embed_autoencoder(L, embed_dim=150, hidden_dim=512,
                      epochs=100, lr=1e-3, batch_size=256, patience=5,
                      neg_weight=None, weighted=False, noise_std=0.1,
                      cov_lambda=1e-3):

    L = (L - L.mean(axis=1, keepdims=True)) / (L.std(axis=1, keepdims=True) + 1e-9)

    if weighted:
        if neg_weight is None:
            neg_prop   = (L < 0).mean()
            pos_prop   = 1 - neg_prop
            neg_weight = pos_prop / neg_prop if neg_prop > 0 else 1.0
    else:
        neg_weight = 1.0

    n_subs    = L.shape[0]
    L_t       = torch.tensor(L, dtype=torch.float32)
    model     = Autoencoder(n_subs, hidden_dim, embed_dim, noise_std=noise_std)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = max(1, (n_subs + batch_size - 1) // batch_size)

    best_loss        = float('inf')
    best_state       = None
    patience_counter = 0

    # Precompute off-diagonal indices for full matrix
    diag_idx = torch.arange(n_subs)

    for epoch in range(epochs):
        model.train()
        idx   = torch.randperm(n_subs)
        total = 0
        for start in range(0, n_subs, batch_size):
            rows = idx[start:start + batch_size]
            B    = len(rows)

            # build off-diagonal mask: True = off-diagonal, False = diagonal
            off_diag = torch.ones(B, n_subs, dtype=torch.bool)
            off_diag[torch.arange(B), rows] = False

            # input: exclude diagonal entirely by using off-diagonal values only
            x_full   = L_t[rows]                          # (B, n_subs)
            x        = x_full.clone()
            x[~off_diag] = 0.0                            # zero diagonal in input before noising

            x_hat, z = model(x, training=True)

            # reconstruction loss on off-diagonal entries only
            weights  = torch.where(x_full[off_diag] < 0,
                                   torch.full_like(x_full[off_diag], neg_weight),
                                   torch.ones_like(x_full[off_diag]))
            loss_rec = (weights * (x_hat[off_diag] - x_full[off_diag]) ** 2).mean()

            loss_cov     = covariance_penalty(z)
            loss_uniform = uniformity_loss(z)
            loss         = loss_rec + cov_lambda * loss_cov 

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        avg_loss = total / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss * 0.99:
            best_loss        = avg_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}  best_loss={best_loss:.4f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # exclude diagonal from input at inference
        L_inf = L_t.clone()
        L_inf[diag_idx, diag_idx] = 0.0
        _, embeddings = model(L_inf, training=False)
    return embeddings.numpy()


def analogy(df, positive=None, negative=None, topn=5, metric="cosine"):
    """
    df: pandas DataFrame with embeddings, index = words
    positive: list of words to add
    negative: list of words to subtract
    topn: number of nearest neighbors to return
    metric: 'cosine' or 'euclidean'
    """
    positive = positive or []
    negative = negative or []

    # build query vector
    vec = np.zeros(df.shape[1])

    for w in positive:
        v = df.loc[w].values
        vec += v / np.linalg.norm(v)

    for w in negative:
        v = df.loc[w].values
        vec -= v / np.linalg.norm(v)

    vec = vec.reshape(1, -1)

    if metric == "cosine":
        sims = cosine_similarity(df.values, vec).flatten()
        results = pd.Series(sims, index=df.index)
        results = results.drop(positive + negative, errors="ignore")
        return results.nlargest(topn)

    elif metric == "euclidean":
        dists = euclidean_distances(df.values, vec).flatten()
        results = pd.Series(dists, index=df.index)
        results = results.drop(positive + negative, errors="ignore")
        return results.nsmallest(topn)

    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")


def evaluate_analogies(emb_df, eval_data, topn=5, metric="cosine"):
    """
    emb_df: embedding DataFrame (index = subreddits)
    eval_data: DataFrame with columns [0, 1, 2]
               0 and 1 = positive "words" (subreddits)
               2 = expected answer
    """
    top1_correct = 0
    in_topn_correct = 0
    total = 0

    for _, row in eval_data.iterrows():
        w1, w2, target = row[0], row[1], row[2]

        # skip if any subreddit missing from embeddings
        if not all(w in emb_df.index for w in [w1, w2, target]):
            continue

        results = analogy(
            emb_df,
            positive=[w1, w2],
            negative=None,
            topn=topn,
            metric=metric
        )

        total += 1

        # metric 1: is target the top result?
        if len(results) > 0 and results.index[0] == target:
            top1_correct += 1

        # metric 2: is target anywhere in topn?
        if target in results.index:
            in_topn_correct += 1

    return {
        "top1_accuracy": top1_correct / total if total > 0 else 0,
        "topn_accuracy": in_topn_correct / total if total > 0 else 0,
        "evaluated_samples": total
    }

eval_data = pd.read_csv('/mydata/eval_data.csv', header=None)  # change to eval data file; this is the sports analogies data from Waller and Anderson (2021)


readpath = "/mydata/subreddit_networks"  # change to your subreddit network directory
writepath = "/mydata/subreddit_embeddings"  # change to wherever you want to write embeddings to

readdirs = os.listdir(readpath)
years = [int(dir.split('_')[1]) for dir in readdirs]
years.sort()

for year in years:
    print(f"Reading {year} matrix...")
    filepath = f"{readpath}/nets_{year}/similarity_activity-{year}.csv"

    yearpath = f"{writepath}/nets_{year}"
    if os.path.exists(yearpath):
        print(f"Already found written files for {year}. Skipping...")
        continue
    os.makedirs(yearpath, exist_ok=True)

    activity = pd.read_csv(filepath)
    activity.rename(columns={'Unnamed: 0': 'Subreddit'}, inplace=True)
    activity.set_index('Subreddit', inplace=True)
    
    print(f"Preprocessing matrix: Winsorizing diagonal...")
    A = activity.values
    masked = A.copy()
    np.fill_diagonal(masked, -np.inf)
    # get the max of each row (which is now the second-highest overall)
    second_max = masked.max(axis=1)
    # replace diagonal with second-highest values
    A[np.arange(A.shape[0]), np.arange(A.shape[1])] = second_max

    print(f"Embedding {year} matrix...")
    embeddings = embed_autoencoder(A, epochs=100, weighted=True)

    embeddings = pd.DataFrame(embeddings, index=activity.index)
    
    print(f"Evaluating performance...")
    performance = evaluate_analogies(embeddings, eval_data)
    if performance['evaluated_samples'] == 0:
        performance['top1_accuracy'] = "N/A"
        performance['topn_accuracy'] = "N/A"
    
    print(performance)
    performance = pd.DataFrame(performance, index=range(len(performance)))

    performance.to_csv(f"{yearpath}/activity_performance-{year}.csv")
    embeddings.to_csv(f"{yearpath}/activity_embeddings-{year}.csv")

