import numpy as np

try:
    import jax.lax
except (ImportError, RuntimeError):
    JAX_IS_AVAILABLE = False
else:
    JAX_IS_AVAILABLE = True
    # if JAX is available, we need to initialize it with a dummy scores and capture
    # any output to avoid it from saying that gpu is not available
    _ = jax.lax.top_k(np.array([0] * 5), 1)


def _topk_numpy(query_scores, k, sorted):
    # https://stackoverflow.com/questions/65038206/how-to-get-indices-of-top-k-values-from-a-numpy-array
    # np.argpartition is faster than np.argsort, but do not return the values in order
    pass


def _topk_jax(query_scores, k):
    pass


def topk(query_scores, k, backend="auto", sorted=True):
    """
    This function is used to retrieve the top-k results for a single query. It will only work
    on a 1-dimensional array of scores.
    """
    pass
