#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import hashlib
import logging
import multiprocessing as mp
import os
import random
import re
import struct
import time
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import glob

import numpy as np
import typer
from datasets import load_dataset
from tqdm import tqdm
from scipy.integrate import quad as integrate

from rensa import CMinHash, RMinHash


# ---------------- CONFIG ---------------- #

SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- UTILS ---------------- #


def ngrams(sequence: List[str], n: int, min_ngram_size: int) -> Iterable:
    if len(sequence) < min_ngram_size:
        return []
    iterables = tee(sequence, n)
    for i, sub in enumerate(iterables):
        for _ in range(i):
            next(sub, None)
    return zip(*iterables)


def sha1_hash32(data: bytes) -> int:
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


# ---------------- MINHASH ---------------- #


def embed_func_rmin(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> Dict[str, Any]:
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }

    # Rensa based
    m = RMinHash(num_perm=num_perm, seed=SEED)
    m.update(tokens)
    hash_tuple = tuple(m.digest())
    hash_array = np.array(hash_tuple, dtype=np.uint64)
    Hs = [bytes(hash_array[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def embed_func_stack(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> Dict[str, Any]:
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }

    # Stack based
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def embed_func_cmin(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> Dict[str, Any]:
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }

    # Rensa based
    m = CMinHash(num_perm=num_perm, seed=SEED)
    m.update(tokens)
    hash_tuple = tuple(m.digest())
    hash_array = np.array(hash_tuple, dtype=np.uint64)
    Hs = [bytes(hash_array[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


# ---------------- LSH PARAMS ---------------- #


def optimal_param(threshold: float, num_perm: int):
    def fp(s, b, r):
        return 1 - (1 - s**r) ** b

    def fn(s, b, r):
        return 1 - (1 - (1 - s**r) ** b)

    best = (1, num_perm)
    min_err = float("inf")

    for b in range(1, num_perm + 1):
        r = num_perm // b
        if r == 0:
            continue
        fpv, _ = integrate(lambda s: fp(s, b, r), 0, threshold)
        fnv, _ = integrate(lambda s: fn(s, b, r), threshold, 1)
        err = fpv + fnv
        if err < min_err:
            min_err = err
            best = (b, r)
    return best


# ---------------- UNION FIND ---------------- #


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


# ---------------- MAIN ---------------- #


def run(
    dataset: str = typer.Option(..., help="Parquet datasets"),
    column: str = "code",
    output_dir: str = "output",
    ngram_size: int = 5,
    min_ngram_size: int = 5,
    num_perm: int = 256,
    threshold: float = 0.7,
):
    datasets = glob.glob(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B, R = optimal_param(threshold, num_perm)
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES = [defaultdict(set) for _ in range(B)]

    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T

    uf = UnionFind()

    # ---------- PASS 1: BUILD LSH ---------- #

    global_offset = 0
    dataset_sizes = []

    for path in datasets:
        logger.info(f"Indexing {path}")
        ds = load_dataset("parquet", data_files=path, split="train")

        embedded = ds.map(
            embed_func_rmin,
            fn_kwargs=dict(
                num_perm=num_perm,
                ngram_size=ngram_size,
                hashranges=HASH_RANGES,
                permutations=PERMUTATIONS,
                min_ngram_size=min_ngram_size,
            ),
            input_columns=[column],
            with_indices=True,
            remove_columns=ds.column_names,
            num_proc=os.cpu_count(),
        )

        for rec in embedded:
            gid = rec["__id__"] + global_offset
            for h, table in zip(rec["__signatures__"], HASH_TABLES):
                table[h].add(gid)

        dataset_sizes.append(len(ds))
        global_offset += len(ds)

        del ds, embedded
        gc.collect()

    # ---------- CLUSTER ---------- #

    logger.info("Clustering...")
    for table in tqdm(HASH_TABLES):
        for cluster in table.values():
            if len(cluster) > 1:
                root = min(cluster)
                for x in cluster:
                    uf.union(x, root)

    # ---------- PASS 2: FILTER ---------- #

    offset = 0
    for path, size in zip(datasets, dataset_sizes):
        logger.info(f"Filtering {path}")
        ds = load_dataset("parquet", data_files=path, split="train")

        ds = ds.map(
            lambda _, idx: {"__cluster__": uf.find(idx + offset)},
            with_indices=True,
            num_proc=os.cpu_count(),
        )

        ds = ds.filter(
            lambda r, idx: r["__cluster__"] == idx + offset,
            with_indices=True,
            num_proc=os.cpu_count(),
        )

        out = output_dir / Path(path).name
        ds.remove_columns("__cluster__").to_parquet(out)

        offset += size
        del ds
        gc.collect()

    logger.info("✅ Deduplication complete")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    typer.run(run)
