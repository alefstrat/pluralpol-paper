"""
This script takes raw data files from the Arctic Shift dumps
and converts them into the subreddit networks (positive and negative).
"""


import zstandard as zstd
import orjson as json
import io
import os
import numpy as np
from tqdm import tqdm
import orjson as json
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='network.log', filemode='a+')

def get_or_create_index(mapping, key):
    """
    Checks if sub or author has been
    encountered and assigns index if not
    """
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def ensure_shape(M, rows, cols):
    """
    Function to dynamically reshape dok_matrix (i.e., expand 
    rows for new subreddits and columns for new users) as we 
    iterate through comments.
    """
    if rows > M.shape[0] or cols > M.shape[1]:
        M.resize((max(rows, M.shape[0]), max(cols, M.shape[1])))


def stream_zst(file_path, dctx):
    total_size = os.path.getsize(file_path)

    with open(file_path, "rb", buffering=10 * 1024 * 1024) as fh, \
         tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Reading {file_path}") as pbar:
        try: 
            with dctx.stream_reader(fh, read_size=10 * 1024 * 1024) as reader:
                buffered = io.BufferedReader(reader)

                last_pos = 0

                for line in buffered:
                    current_pos = fh.tell()
                    pbar.update(current_pos - last_pos)
                    last_pos = current_pos

                    if line:
                        yield json.loads(line)
                        
        except zstd.ZstdError as e:
            logging.error(f"{e}. Skipping {file_path}.")
            return


def compute_z_matrix(A, M):
    """
    Vectorized operations for computing Z scores per
    user per subreddit. Inputs must be sparse matrices.
    """

    # compute global negativity probability per user (across all subreddits)
    total_comments = np.array(A.sum(axis=0)).flatten()
    total_negative = np.array(M.sum(axis=0)).flatten()

    p_global = total_negative / total_comments

    # expected negativity per subreddit per user based on global negativity probability per user
    E = A.multiply(p_global).tocsr()  
    # sigma, i.e., the square root of per-user and per-subreddit variance
    S = A.multiply(p_global * (1 - p_global)).sqrt().tocsr()  

    return (M - E).multiply(S.power(-1))  # return Z score matrix (actual negativity minus expected divided by sigma)

dctx = zstd.ZstdDecompressor(max_window_size=2**31)

data_dir = 'mydata/reddit_files/'  # your data directory for storing the monthly reddit data dumps
output_dir = '/mydata/subreddit_networks/'  # change to whereever you want to store the output files

year_data = {}

for file in os.listdir(data_dir):
    if file.endswith('.zst'):
        fileyear = int(file.split('_')[1][:4])  # based on naming convention, amend as necessary
        if fileyear not in year_data:
            year_data[fileyear] = [file]
        else:
            year_data[fileyear].append(file)

for year in year_data:
    year_data[year].sort()

year_data = dict(sorted(year_data.items()))

for year in year_data:
    writepath = f"{output_dir}/nets_{year}/"
    if os.path.exists(writepath):
        print(f"Already found written files for {year}. Skipping...")
        continue
    # declare empty index maps
    subreddit_index = {}
    author_index = {}

    # dok matrices to make dynamic reshaping as efficient as possible
    activity = dok_matrix((1,1), dtype=np.float64)
    negativity = dok_matrix((1,1), dtype=np.float64)

    for file in year_data[year]:
        filepath = os.path.join(data_dir, file)
        for comment in stream_zst(filepath, dctx):
            subreddit = comment["subreddit"]
            author = comment["author"]
            score = comment["score"]

            if author in ['[deleted]', 'AutoModerator']:
                continue

            # assign indices
            i = get_or_create_index(subreddit_index, subreddit)
            j = get_or_create_index(author_index, author)

            # ensure matrix shape
            ensure_shape(activity, i + 1, j + 1)
            ensure_shape(negativity, i + 1, j + 1)

            # update activity
            activity[i, j] = activity.get((i, j), 0) + 1

            # update negativity
            try:
                if score < 1:
                    negativity[i, j] = negativity.get((i, j), 0) + 1
            except TypeError:
                negativity[i, j] = negativity.get((i, j), 0) + 0

    print("Constructing base matrices...")

    # convert to csr for efficient operations; dok matrix is fast for construction but very slow for maths
    A = activity.tocsr()
    M = negativity.tocsr()

    if A.shape[0] < 2:  # skip years for which we have fewer than 2 retained subreddits
        continue

    # implement masks to drop subreddits with < 1000 comments and < 100 users
    row_numcomments = np.array(A.sum(axis=1)).ravel()
    row_numauthors = A.getnnz(axis=1)

    mask = (row_numcomments >= 1000) & (row_numauthors > 100)
    A = A[mask, :]
    M = M[mask, :]

    premask_labels = list(subreddit_index.keys())
    postmask_labels = np.array(premask_labels)[mask]

    subreddit_index = {label: i for i, label in enumerate(postmask_labels)}
    labels = list(postmask_labels)

    MAX_SUBREDDITS = 45000

    n_subs = A.shape[0]

    if n_subs > MAX_SUBREDDITS:
        print(f"Capping subreddits from {n_subs} to {MAX_SUBREDDITS}")
        with open('./logs/truncation_log.txt', 'a+') as fw:
            fw.write(f"Truncated {n_subs} down to {MAX_SUBREDDITS} for {year}.\n")

        # recompute number of authors per subreddit after initial mask
        row_numauthors = A.getnnz(axis=1)

        # get indices of top 45k subreddits by author count
        top_idx = np.argsort(row_numauthors)[-MAX_SUBREDDITS:]

        # sort to preserve original order (for label consistency)
        top_idx = np.sort(top_idx)

        # apply secondary mask
        A = A[top_idx, :]
        M = M[top_idx, :]

        labels = np.array(labels)[top_idx].tolist()
        subreddit_index = {label: i for i, label in enumerate(labels)}

    # also drop users with no comments on the remaining subreddits for accurate user metadata descriptives
    col_numcomments = np.array(A.sum(axis=0)).ravel()
    user_mask = col_numcomments > 0

    A = A[:, user_mask]
    M = M[:, user_mask]

    # continue computations as normal
    Z = compute_z_matrix(A, M)
    Z.data[np.isnan(Z.data)] = 0  # handle nans to allow for cosine similarity computations later

    # extract subreddits for writing. we are assuming order-preserving dictionaries (check Python version)
    labels = list(subreddit_index.keys())

    # extract subreddit information: number of total comments, number of negative comments, number of unique authors
    attributes = pd.DataFrame.from_dict({
        'labels': labels,
        'total_comments': np.array(A.sum(axis=1)).ravel(),  # total number of comments from activity matrix
        'negative_comments': np.array(M.sum(axis=1)).ravel(),  # total negative of comments from negativity matrix
        'authors': A.getnnz(axis=1)  # total unique authors counted as number of non-zero values from activity matrix, rowwise
    })

    # extract information for the entire year
    metadata = pd.DataFrame.from_dict({
        'total_comments': [A.sum()],
        'negative_comments': [M.sum()],
        'N_subreddits': [A.shape[0]],
        'N_unique_authors': [A.shape[1]]
    })

    tfidf = TfidfTransformer()
    A_tfidf = tfidf.fit_transform(A)

    print("Computing pairwise similarities...")
    xsub_activity = pd.DataFrame(cosine_similarity(A_tfidf), index=labels, columns=labels)
    xsub_alignment = pd.DataFrame(cosine_similarity(Z), index=labels, columns=labels)

    print("Writing to file...")
    
    os.makedirs(writepath, exist_ok=True)

    attributes.to_csv(f"{writepath}/attributes-{year}.csv")
    metadata.to_csv(f"{writepath}/metadata-{year}.csv")
    xsub_activity.to_csv(f"{writepath}/similarity_activity-{year}.csv")
    xsub_alignment.to_csv(f"{writepath}/similarity_alignment-{year}.csv")
