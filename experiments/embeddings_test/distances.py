import editdistance
import numpy as np
from collections import Counter


def tokenize(transcription):
    """
    Tokenize by tabs/newlines for alignment-aware transcriptions.

    Empty tokens are intentionally preserved because they may encode alignment.
    """
    return transcription.replace('\n', '\t').split('\t')


def levenshtein_list(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def edit_distance_tokens(t1, t2):
    tokens1 = tokenize(t1)
    tokens2 = tokenize(t2)
    return editdistance.eval(tokens1, tokens2)


def build_global_vocabulary(list_transcriptions):
    token_set = set()
    for transcription in list_transcriptions:
        tokens = tokenize(transcription)
        token_set.update(tokens)
    return sorted(token_set)  # Stable order.


def build_histogram(tokens, vocabulary):
    counter = Counter(tokens)
    histogram = [counter[token] if token in counter else 0 for token in vocabulary]
    return np.array(histogram)


def histogram_distance(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))  # L1 distance.


def calculateHistogramDistanceWords(list_transcriptions):
    print("Building global vocabulary...")
    vocabulary = build_global_vocabulary(list_transcriptions)

    print("Building histograms...")
    histograms = []
    for transcription in list_transcriptions:
        tokens = tokenize(transcription)
        hist = build_histogram(tokens, vocabulary)
        histograms.append(hist)

    print("Calculating histogram distances...")
    n = len(list_transcriptions)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        print(f"Calculating distances to {i + 1} of {n}")
        for j in range(n):
            distance_matrix[i][j] = histogram_distance(histograms[i], histograms[j])

    return distance_matrix


def calculateEditDistance(list_transcriptions):
    print("Calculating edit distance...")
    edit_distance_matrix = np.ones((len(list_transcriptions), len(list_transcriptions))) * -1

    idx1 = 0
    for transcription1 in list_transcriptions:
        idx2 = 0
        print("Calculating distances to " + str(idx1 + 1) + " of " + str(len(list_transcriptions)))
        for transcription2 in list_transcriptions:
            dist = edit_distance_tokens(transcription1, transcription2)
            edit_distance_matrix[idx1][idx2] = dist
            idx2 += 1
        idx1 += 1

    return edit_distance_matrix


def calculateEuclideanDistance(list_embeddings):
    """
    NxN matrix of Euclidean distances between embeddings.
    """
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(list_embeddings, dtype=np.float32)  # (N, D)

    # Optional L2 normalization.
    #norms = np.linalg.norm(X, axis=1, keepdims=True)   # (N,1)
    #norms[norms == 0] = 1.0
    #X = X / norms

    d = pdist(X, metric="euclidean")
    D = squareform(d)
    return D


# ----------------------------------------------------------------------
# Token lengths and bounded normalizations in [0, 1]
# ----------------------------------------------------------------------

def token_lengths(list_transcriptions):
    """
    Token length, preserving empty tokens when present.
    """
    lens = []
    for t in list_transcriptions:
        lens.append(len(tokenize(t)))
    return np.array(lens, dtype=np.float32)


def normalize_edit_by_maxlen(distance_matrix, list_transcriptions):
    """
    Bounded [0, 1] normalization for Levenshtein distance:
      ED_norm(i,j) = ED(i,j) / max(len_i, len_j)
    """
    dist = np.asarray(distance_matrix, dtype=np.float32)
    lengths = token_lengths(list_transcriptions)  # (N,)

    denom = np.maximum.outer(lengths, lengths)
    denom[denom == 0] = 1.0
    return dist / denom


def normalize_edit_by_meanlen_clipped(distance_matrix, list_transcriptions):
    """
    Mean-length normalization clipped to [0, 1].
    """
    dist = np.asarray(distance_matrix, dtype=np.float32)
    lengths = token_lengths(list_transcriptions)

    denom = 0.5 * (lengths[:, None] + lengths[None, :])
    denom[denom == 0] = 1.0
    out = dist / denom
    out = np.minimum(out, 1.0)
    return out


def normalize_hist_by_sumlen(distance_matrix, list_transcriptions):
    """
    Bounded [0, 1] normalization for histogram L1 distance:
      H_norm(i,j) = L1(hist_i, hist_j) / (len_i + len_j)
    """
    dist = np.asarray(distance_matrix, dtype=np.float32)
    lengths = token_lengths(list_transcriptions)

    denom = lengths[:, None] + lengths[None, :]
    denom[denom == 0] = 1.0
    return dist / denom
