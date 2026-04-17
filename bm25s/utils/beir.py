import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


from . import json_functions

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
GH_URL = "https://github.com/xhluca/bm25s/releases/download/data/{}.zip"


def clean_results_keys(beir_results):
    pass


def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with BEIR evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """
    pass


def merge_cqa_dupstack(data_path):
    pass


def download_dataset(
    dataset,
    base_url=GH_URL,
    save_dir="./datasets",
    unzip=True,
    redownload=False,
    show_progress=True,
):
    pass


def load_jsonl(
    dataset,
    fname,
    save_dir="./datasets",
    show_progress=True,
    return_dict=True,
    force_title=False,
    remove=None,
):
    pass


def load_corpus(dataset, save_dir="./datasets", show_progress=True, return_dict=True):
    pass


def load_queries(dataset, save_dir="./datasets", show_progress=True, return_dict=True):
    pass


def load_qrels(
    dataset, split="test", save_dir="./datasets", show_progress=True, return_dict=True
):
    """
    This is tsv files
    """
    pass


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
    ignore_identical_ids: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Acknowledgement: This function is adapted from BEIR's EvaluateRetrieval class.
    License for this function: Apache-2.0
    """
    pass
