"""Artifact writer: saves complete DataFrames to Parquet and manages caching.

All tabular tools call ``save_artifact`` to persist their full result set.
The returned ``ArtifactManifest`` is embedded in the tool's context-window
response so the agent knows where to load data from.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from connectomics_mcp.output_contracts.schemas import ArtifactManifest

logger = logging.getLogger(__name__)

# Cache window in seconds — artifacts younger than this are reused.
_CACHE_MAX_AGE_SECONDS = 3600  # 1 hour


def _artifact_dir() -> Path:
    """Return the artifact output directory, creating it if needed."""
    base = os.environ.get(
        "CONNECTOMICS_MCP_ARTIFACT_DIR",
        str(Path.home() / ".connectomics_mcp" / "artifacts"),
    )
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_filename(
    tool: str,
    dataset: str,
    neuron_id: int | str | None,
    materialization_version: int | None,
    timestamp: str,
    extra_key: str | None = None,
) -> str:
    """Build the artifact filename following the naming convention."""
    parts = [dataset, tool]
    if neuron_id is not None:
        parts.append(str(neuron_id))
    if extra_key is not None:
        parts.append(extra_key)
    mat = f"v{materialization_version}" if materialization_version is not None else "vNone"
    parts.append(mat)
    parts.append(timestamp)
    return "_".join(parts) + ".parquet"


def _find_cached(
    tool: str,
    dataset: str,
    neuron_id: int | str | None,
    materialization_version: int | None,
    extra_key: str | None = None,
) -> Path | None:
    """Find a cached artifact matching the query that is less than 1 hour old."""
    out_dir = _artifact_dir()
    prefix_parts = [dataset, tool]
    if neuron_id is not None:
        prefix_parts.append(str(neuron_id))
    if extra_key is not None:
        prefix_parts.append(extra_key)
    mat = f"v{materialization_version}" if materialization_version is not None else "vNone"
    prefix_parts.append(mat)
    prefix = "_".join(prefix_parts) + "_"

    now = time.time()
    for candidate in sorted(out_dir.glob(f"{prefix}*.parquet"), reverse=True):
        age = now - candidate.stat().st_mtime
        if age < _CACHE_MAX_AGE_SECONDS:
            return candidate
    return None


def _describe_columns(df: pd.DataFrame) -> str:
    """Generate a human-readable schema description for a DataFrame."""
    parts: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        parts.append(f"  {col}: {dtype}")
    return "Columns:\n" + "\n".join(parts)


def save_artifact(
    df: pd.DataFrame,
    tool: str,
    dataset: str,
    neuron_id: int | str | None = None,
    materialization_version: int | None = None,
    extra_key: str | None = None,
) -> ArtifactManifest:
    """Save a DataFrame as a Parquet artifact and return a manifest.

    Parameters
    ----------
    df : pd.DataFrame
        The complete result to persist — never truncated.
    tool : str
        Tool name (e.g. "connectivity").
    dataset : str
        Dataset name.
    neuron_id : int | str | None
        Neuron identifier, if applicable.
    materialization_version : int | None
        CAVE materialization version, if applicable.
    extra_key : str | None
        Additional cache key component (e.g. table name) to
        disambiguate queries that share the same tool/dataset/neuron_id.

    Returns
    -------
    ArtifactManifest
        Manifest pointing to the saved (or cached) Parquet file.
    """
    # Check cache first
    cached = _find_cached(tool, dataset, neuron_id, materialization_version, extra_key)
    if cached is not None:
        logger.debug("Cache hit for %s/%s/%s", tool, dataset, neuron_id)
        cached_df = pd.read_parquet(cached)
        return ArtifactManifest(
            artifact_path=str(cached),
            n_rows=len(cached_df),
            columns=list(cached_df.columns),
            schema_description=_describe_columns(cached_df),
            dataset=dataset,
            query_timestamp=datetime.fromtimestamp(
                cached.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            materialization_version=materialization_version,
            cache_hit=True,
        )

    # Write new artifact
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    filename = _build_filename(
        tool, dataset, neuron_id, materialization_version, ts, extra_key
    )
    out_path = _artifact_dir() / filename

    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.debug("Wrote artifact %s (%d rows)", out_path, len(df))

    return ArtifactManifest(
        artifact_path=str(out_path),
        n_rows=len(df),
        columns=list(df.columns),
        schema_description=_describe_columns(df),
        dataset=dataset,
        query_timestamp=now.isoformat(),
        materialization_version=materialization_version,
        cache_hit=False,
    )
