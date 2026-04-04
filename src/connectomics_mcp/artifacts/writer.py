"""Artifact writer: saves complete DataFrames to Parquet and manages caching.

All tabular tools call ``save_artifact`` to persist their full result set.
The returned ``ArtifactManifest`` is embedded in the tool's context-window
response so the agent knows where to load data from.

Cache policy
------------
- **Versioned queries** (``materialization_version`` is not None): the cache
  never expires — same version ≡ same data, because CAVE materialization
  snapshots are immutable.
- **Unversioned queries** (``materialization_version`` is None): cached
  artifacts are reused for up to 7 days.  This covers neuPrint (static
  hemibrain snapshot) and any CAVE query made without an explicit version.
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

# Cache window for *unversioned* queries only.  Versioned queries never expire.
_CACHE_MAX_AGE_SECONDS = 604800  # 7 days


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
    """Build the artifact filename following the naming convention.

    Uses labeled segments (``nid-``, ``xk-``) so that a bare neuron_id can
    never collide with an extra_key of the same string value.
    """
    parts = [dataset, tool]
    parts.append(f"nid-{neuron_id}")
    parts.append(f"xk-{extra_key}")
    mat = f"v{materialization_version}" if materialization_version is not None else "vNone"
    parts.append(mat)
    parts.append(timestamp)
    return "_".join(parts) + ".parquet"


def _build_prefix(
    tool: str,
    dataset: str,
    neuron_id: int | str | None,
    materialization_version: int | None,
    extra_key: str | None = None,
) -> str:
    """Build the cache-lookup prefix (everything before the timestamp)."""
    parts = [dataset, tool]
    parts.append(f"nid-{neuron_id}")
    parts.append(f"xk-{extra_key}")
    mat = f"v{materialization_version}" if materialization_version is not None else "vNone"
    parts.append(mat)
    return "_".join(parts) + "_"


def _find_cached(
    tool: str,
    dataset: str,
    neuron_id: int | str | None,
    materialization_version: int | None,
    extra_key: str | None = None,
) -> Path | None:
    """Find a cached artifact matching the query.

    Versioned queries never expire.  Unversioned queries expire after
    ``_CACHE_MAX_AGE_SECONDS`` (7 days).
    """
    out_dir = _artifact_dir()
    prefix = _build_prefix(tool, dataset, neuron_id, materialization_version, extra_key)

    now = time.time()
    for candidate in sorted(out_dir.glob(f"{prefix}*.parquet"), reverse=True):
        if materialization_version is not None:
            # Versioned → immutable snapshot, never expires
            return candidate
        # Unversioned → apply TTL
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


def _manifest_from_path(
    cached_path: Path,
    dataset: str,
    materialization_version: int | None,
    n_rows: int,
    columns: list[str],
    schema_description: str,
) -> ArtifactManifest:
    """Build an ArtifactManifest for an existing file on disk."""
    return ArtifactManifest(
        artifact_path=str(cached_path),
        n_rows=n_rows,
        columns=columns,
        schema_description=schema_description,
        dataset=dataset,
        query_timestamp=datetime.fromtimestamp(
            cached_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        materialization_version=materialization_version,
        cache_hit=True,
    )


def load_cached_artifact(
    tool: str,
    dataset: str,
    neuron_id: int | str | None = None,
    materialization_version: int | None = None,
    extra_key: str | None = None,
) -> tuple[pd.DataFrame, ArtifactManifest] | None:
    """Look up a cached artifact.  Single entry point for all cache reads.

    Parameters
    ----------
    tool, dataset, neuron_id, materialization_version, extra_key
        Cache key components — same semantics as ``save_artifact``.

    Returns
    -------
    tuple[pd.DataFrame, ArtifactManifest] or None
        ``(df, manifest)`` on cache hit, ``None`` on miss.
    """
    cached_path = _find_cached(tool, dataset, neuron_id, materialization_version, extra_key)
    if cached_path is None:
        return None

    logger.debug("Cache hit for %s/%s/%s", tool, dataset, neuron_id)
    cached_df = pd.read_parquet(cached_path)
    manifest = _manifest_from_path(
        cached_path, dataset, materialization_version,
        n_rows=len(cached_df),
        columns=list(cached_df.columns),
        schema_description=_describe_columns(cached_df),
    )
    return cached_df, manifest


def save_artifact(
    df: pd.DataFrame,
    tool: str,
    dataset: str,
    neuron_id: int | str | None = None,
    materialization_version: int | None = None,
    extra_key: str | None = None,
) -> ArtifactManifest:
    """Save a DataFrame as a Parquet artifact and return a manifest.

    Checks the cache first — if an identical query already has a fresh
    artifact on disk, returns the cached manifest without re-writing.

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
    cached = load_cached_artifact(tool, dataset, neuron_id, materialization_version, extra_key)
    if cached is not None:
        return cached[1]  # return the manifest

    # Write new artifact
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    filename = _build_filename(
        tool, dataset, neuron_id, materialization_version, ts, extra_key
    )
    out_path = _artifact_dir() / filename

    # Coerce root ID columns to nullable Int64 to prevent float64 inference
    # when NaN values are present (pandas infers float64 for int+NaN columns).
    _ROOT_ID_COLUMNS = {
        "pt_root_id", "pre_root_id", "post_root_id", "partner_id",
        "pre_pt_root_id", "post_pt_root_id", "root_id",
        "partner_nucleus_id", "target_id",
    }
    for col in df.columns:
        if col in _ROOT_ID_COLUMNS and df[col].dtype == "float64":
            df[col] = df[col].astype("Int64")

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
