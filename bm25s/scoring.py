from collections import Counter
import math

import numpy as np
import os

def _faketqdm(*args, **kwargs):
    pass
try:
    if os.environ.get("DISABLE_TQDM", False):
        tqdm = _faketqdm
    else:
        from tqdm.auto import tqdm
except ImportError:
    tqdm = _faketqdm

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator to allow definition if numba is missing (code won't be used)
    def njit(*args, **kwargs):
        def decorator(func):
            pass
        return decorator

def _calculate_doc_freqs(
    corpus_tokens, unique_tokens, show_progress=True, leave_progress=False
) -> dict:
    """
    Document Frequency, aka DF, is the number of documents that contain a specific token.
    This function return a dictionary with the document frequency of each token, which is
    why it is called `doc_frequencies`.
    """
    pass


def _build_idf_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable = None,
    dtype="float32",
) -> np.ndarray:
    pass


def _build_nonoccurrence_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable,
    calculate_tfc_fn: callable,
    l_d,
    l_avg,
    k1,
    b,
    delta,
    dtype="float32",
) -> np.ndarray:
    """
    The non-occurrence array is used to store the idf score for tokens that do not occur in the
    document. This is useful for BM25L and BM25+ variants, where we need to calculate the idf
    score for tokens that do not occur in the document, which will be used to calculate the
    final score.

    The nonoccurence array has length |V|, where V is the set of unique tokens in the corpus.

    The `compute_idf_fn` is the function to calculate the idf score for a token that does not occur
    in the document. The `calculate_tfc_fn` is the function to calculate the term frequency component
    of the BM25 score, which is used to calculate the final score for tokens that do not occur in the
    document.
    """
    pass


def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_tfc_bm25l(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_tfc_bm25plus(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _select_tfc_scorer(method) -> callable:
    pass


def _score_idf_robertson(df, N, allow_negative=False):
    """
    Computes the inverse document frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_idf_lucene(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_idf_atire(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_idf_bm25l(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _score_idf_bm25plus(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    pass


def _select_idf_scorer(method) -> callable:
    pass


def _get_counts_from_token_ids(token_ids, dtype, int_dtype):
    pass


def _build_scores_and_indices_for_matrix(
    corpus_token_ids,
    idf_array,
    avg_doc_len,
    doc_frequencies,
    k1,
    b,
    delta,
    nonoccurrence_array,
    method="robertson",
    dtype="float32",
    int_dtype="int32",
    show_progress=True,
    leave_progress=False,
):
    pass


def _compute_relevance_from_scores_legacy(
    data, indptr, indices, num_docs, query_tokens_ids, dtype
):
    """
    The legacy implementation of the `_compute_relevance_from_scores` function. This may
    be faster than the new implementation for some cases, but it cannot benefit from
    numba acceleration, as it uses python lists. This function is kept for reference
    and comparison purposes.
    """
    pass

def _compute_relevance_from_scores_jit_ready(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    query_tokens_ids: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This internal static function calculates the relevance scores for a given query,
    by using the BM25 scores that have been precomputed in the BM25 eager index.
    This version is ready for JIT compilation with numba, but is slow if not compiled.
    """
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(query_tokens_ids)):
        start, end = indptr_starts[i], indptr_ends[i]
        # The following code is slower with numpy, but faster after JIT compilation
        for j in range(start, end):
            scores[indices[j]] += data[j]

    return scores


def _np_csc_jit_ready(data, rows, cols, shape):
    """
    Numba-compilable core implementation of CSC construction.
    Uses Counting Sort (linear time) instead of Argsort (linear-logarithmic time).
    
    This implementation assumes `rows` (doc_ids) are sorted in the input, which is 
    standard for BM25 construction. It performs a stable sort on `cols` to strictly 
    guarantee the final (col, row) sorted order required by CSC matrices.
    """
    pass

def _np_csc_python(data, rows, cols, shape):
    """
    Pure NumPy implementation of CSC construction.
    """
    n_cols = shape[1]

    # 1. Compute indptr
    col_counts = np.bincount(cols, minlength=n_cols)

    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(col_counts, out=indptr[1:])

    # 2. Sort the data
    packed_indices = (cols.astype(np.int64) << 32) | rows.astype(np.int64)
    sorter = np.argsort(packed_indices)

    # 3. Reorder arrays based on the sort
    sorted_data = data[sorter]
    sorted_indices = rows[sorter]
    
    return sorted_data, sorted_indices, indptr
