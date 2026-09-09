from __future__ import annotations

import hashlib
import math
import re
from threading import Lock
from typing import List, Sequence

import numpy as np


class TextEmbedder:
    """Sentence-transformer embeddings with an optional deterministic test fallback."""

    def __init__(
        self,
        model_name: str,
        fallback_dimensions: int = 768,
        allow_fallback: bool = True,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.fallback_dimensions = fallback_dimensions
        self.allow_fallback = allow_fallback
        self._model = None
        self._load_attempted = False
        self._load_error: Exception | None = None
        self._encode_lock = Lock()

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self.model_name.startswith("offline:"):
            return
        try:
            from sentence_transformers import SentenceTransformer

            options = {}
            if self.device.strip().lower() != "auto":
                options["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **options)
        except Exception as error:
            self._model = None
            self._load_error = error
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Unable to load embedding model {self.model_name!r}. "
                    "Install the paper dependencies with: "
                    "python -m pip install -e \".[paper]\""
                ) from error

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        if self._model is not None:
            # Some embedding backends are not safe to call concurrently on one
            # model instance, especially when they share a GPU.
            with self._encode_lock:
                value = self._model.encode(
                    list(texts), convert_to_numpy=True, normalize_embeddings=True
                )
            return np.asarray(value, dtype=float)
        return np.vstack([self._hash_embedding(text) for text in texts]) if texts else np.empty((0, self.fallback_dimensions))

    def similarities(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        vectors = self.encode([query, *documents])
        return [float(vectors[0] @ item) for item in vectors[1:]]

    def _hash_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dimensions, dtype=float)
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            index = number % self.fallback_dimensions
            vector[index] += -1.0 if (number >> 8) & 1 else 1.0
        norm = math.sqrt(float(vector @ vector))
        return vector / norm if norm else vector


def cluster_vectors(
    vectors: np.ndarray,
    clusters: int,
    min_size: int = 1,
    seed: int = 42,
) -> List[int]:
    """K-Means with a lower bound on cluster size, but no equal-size constraint."""
    if len(vectors) == 0:
        return []
    min_size = max(1, int(min_size))
    if len(vectors) < min_size:
        raise ValueError(
            f"Clustering requires at least {min_size} papers, but only "
            f"{len(vectors)} were provided"
        )
    max_clusters = max(1, len(vectors) // min_size)
    clusters = max(1, min(clusters, len(vectors), max_clusters))
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(vectors, clusters, rng)
    labels = np.full(len(vectors), -1, dtype=int)
    for _ in range(100):
        distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_labels = _enforce_minimum_size(new_labels, distances, min_size)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(clusters):
            members = vectors[labels == index]
            centers[index] = members.mean(axis=0)
    return labels.tolist()


def _kmeans_plus_plus(
    vectors: np.ndarray, clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = [vectors[int(rng.integers(len(vectors)))]]
    while len(centers) < clusters:
        distances = np.min(
            ((vectors[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        total = float(distances.sum())
        if total == 0:
            centers.append(vectors[len(centers) % len(vectors)])
        else:
            centers.append(vectors[int(rng.choice(len(vectors), p=distances / total))])
    return np.asarray(centers, dtype=float)


def _enforce_minimum_size(
    labels: np.ndarray,
    distances: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Move the least costly papers into undersized clusters."""
    labels = labels.copy()
    cluster_count = distances.shape[1]
    sizes = np.bincount(labels, minlength=cluster_count)
    for target in np.flatnonzero(sizes < min_size):
        while sizes[target] < min_size:
            donors = sizes[labels] > min_size
            candidates = np.flatnonzero(donors & (labels != target))
            if not len(candidates):
                raise ValueError(
                    "Cannot satisfy the requested minimum cluster size with "
                    f"{len(labels)} papers and {cluster_count} clusters"
                )
            current = labels[candidates]
            cost = distances[candidates, target] - distances[candidates, current]
            paper_index = int(candidates[np.argmin(cost)])
            source = int(labels[paper_index])
            labels[paper_index] = target
            sizes[source] -= 1
            sizes[target] += 1
    return labels
