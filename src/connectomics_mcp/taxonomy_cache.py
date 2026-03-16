"""Disk-based taxonomy vocabulary cache.

Caches the distinct (level, name, count) vocabulary for each dataset
to a JSON file with a 24-hour TTL.  Taxonomy, search, and fuzzy-fallback
operations read from this cache — zero API calls on cache hit.

Cache location:
    ``~/.connectomics_mcp/taxonomy_cache/{dataset}_{mat_version}.json``

Overridable via ``CONNECTOMICS_MCP_ARTIFACT_DIR`` (taxonomy_cache/ subdirectory).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Vocabulary cache TTL — taxonomy doesn't change with proofreading.
_VOCAB_CACHE_TTL_SECONDS = 86400  # 24 hours


def _cache_dir() -> Path:
    """Return the taxonomy cache directory, creating it if needed."""
    base = os.environ.get(
        "CONNECTOMICS_MCP_ARTIFACT_DIR",
        str(Path.home() / ".connectomics_mcp" / "artifacts"),
    )
    cache_dir = Path(base).parent / "taxonomy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(dataset: str, mat_version: int | None) -> Path:
    """Build the cache file path for a dataset."""
    version_str = str(mat_version) if mat_version else "latest"
    return _cache_dir() / f"{dataset}_{version_str}.json"


def load_vocab(dataset: str, mat_version: int | None = None) -> dict[str, Any] | None:
    """Load cached vocabulary from disk if fresh enough.

    Returns
    -------
    dict or None
        The cached vocabulary dict, or None if no valid cache exists.
        Structure: ``{"levels": [...], "example_lineages": [...],
        "n_total_neurons": int, "cached_at": float}``
    """
    path = _cache_path(dataset, mat_version)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > _VOCAB_CACHE_TTL_SECONDS:
            logger.debug("Taxonomy cache expired for %s", dataset)
            return None
        return data
    except Exception as e:
        logger.warning("Failed to read taxonomy cache %s: %s", path, e)
        return None


def save_vocab(
    dataset: str,
    mat_version: int | None,
    levels: list[dict],
    example_lineages: list[dict],
    n_total_neurons: int,
) -> None:
    """Save vocabulary to disk cache."""
    path = _cache_path(dataset, mat_version)
    data = {
        "dataset": dataset,
        "mat_version": mat_version,
        "n_total_neurons": n_total_neurons,
        "levels": levels,
        "example_lineages": example_lineages,
        "cached_at": time.time(),
    }
    try:
        path.write_text(json.dumps(data, indent=2))
        logger.debug("Saved taxonomy cache to %s", path)
    except Exception as e:
        logger.warning("Failed to write taxonomy cache %s: %s", path, e)


def get_vocab_for_search(
    dataset: str, mat_version: int | None = None
) -> list[dict] | None:
    """Get flat vocabulary list for fuzzy search.

    Returns a list of ``{"name": str, "level": str, "n_neurons": int}``
    dicts — all distinct annotation values across all levels.
    Returns None if no cache is available.
    """
    cached = load_vocab(dataset, mat_version)
    if cached is None:
        return None

    flat: list[dict] = []
    for level_data in cached.get("levels", []):
        level_name = level_data["level_name"]
        for v in level_data.get("values", []):
            flat.append({
                "name": v["name"],
                "level": level_name,
                "n_neurons": v["n_neurons"],
            })
    return flat
