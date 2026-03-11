"""CAVE backend adapter for MICrONS and FlyWire datasets.

Concrete base class ``CAVEBackend`` holds all shared CAVE logic.
Dataset-specific config and behaviour live in two subclasses:

* ``MICrONSBackend`` — nucleus enrichment in connectivity
* ``FlyWireBackend`` — hierarchy cache, NT prediction, NT enrichment,
  neurons-by-type override
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

from connectomics_mcp.backends.base import ConnectomeBackend
from connectomics_mcp.exceptions import BackendConnectionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAVEBackend — concrete base with shared CAVE logic
# ---------------------------------------------------------------------------

class CAVEBackend(ConnectomeBackend):
    """Backend adapter for CAVE-based connectomic datasets.

    Subclasses set class-level attributes for table/column names
    and override template-method hooks for dataset-specific enrichment.
    """

    # -- config (overridden by subclasses) ----------------------------------
    dataset_name: str = ""
    datastack: str = ""
    cell_type_table: str | None = None
    cell_type_column: str = "cell_type"
    synapse_table: str = "synapses"
    nucleus_table: str | None = None
    proofreading_table: str | None = None

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        """Lazily initialize the CAVEclient."""
        if self._client is None:
            try:
                import caveclient

                token = os.environ.get("CAVE_CLIENT_TOKEN")
                self._client = caveclient.CAVEclient(
                    self.datastack, write_server_cache=False
                )
                if token:
                    self._client.auth.token = token
                logger.debug("Initialized CAVEclient for %s", self.datastack)
            except Exception as e:
                raise BackendConnectionError("cave", str(e)) from e
        return self._client

    # -- template-method hooks (no-ops on base) -----------------------------

    def _enrich_neuron_info(
        self, root_id: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Hook for dataset-specific neuron info enrichment.

        Called after the base neuron-info dict is built.  The default
        implementation returns the dict unchanged.
        """
        return result

    def _enrich_connectivity(
        self,
        root_id: int,
        direction: str,
        partners_df: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Hook for dataset-specific connectivity enrichment.

        Called after partner cell types and placeholder columns are added.
        The default implementation returns the DataFrame unchanged.
        """
        return partners_df

    def _interpret_proofreading_row(
        self, row: pd.Series
    ) -> dict[str, Any]:
        """Hook for interpreting a proofreading-table row.

        Returns a dict with keys ``axon_proofread``, ``dendrite_proofread``,
        ``strategy_axon``, ``strategy_dendrite``.
        """
        axon_proofread = None
        dendrite_proofread = None
        if "status_axon" in row.index:
            val = row.get("status_axon")
            axon_proofread = bool(val) if pd.notna(val) else None
        if "status_dendrite" in row.index:
            val = row.get("status_dendrite")
            dendrite_proofread = bool(val) if pd.notna(val) else None
        return {
            "axon_proofread": axon_proofread,
            "dendrite_proofread": dendrite_proofread,
            "strategy_axon": row.get("strategy_axon"),
            "strategy_dendrite": row.get("strategy_dendrite"),
        }

    # -- shared methods -----------------------------------------------------

    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        """Fetch neuron info from CAVE.

        Parameters
        ----------
        neuron_id : int | str
            Root ID of the neuron.

        Returns
        -------
        dict
            Raw neuron info with keys: neuron_id, dataset, cell_type,
            cell_class, soma_position_nm, n_pre_synapses, n_post_synapses,
            is_current, materialization_version, warnings.
        """
        root_id = int(neuron_id)
        logger.debug("get_neuron_info(%d) on %s", root_id, self.dataset_name)

        warnings: list[str] = []

        # Check root ID currency
        is_current = True
        try:
            is_latest = self.client.chunkedgraph.is_latest_roots([root_id])
            is_current = bool(is_latest[0]) if is_latest else True
            if not is_current:
                warnings.append(
                    f"Root ID {root_id} is outdated. "
                    f"Use `validate_root_ids()` to get current IDs."
                )
        except Exception as e:
            logger.warning("Failed to check root ID currency: %s", e)
            warnings.append(f"Could not verify root ID currency: {e}")

        # Query cell type
        cell_type = None
        cell_class = None
        if self.cell_type_table:
            try:
                ct_df = self.client.materialize.query_table(
                    self.cell_type_table,
                    filter_equal_dict={"pt_root_id": root_id},
                )
                if len(ct_df) > 0:
                    row = ct_df.iloc[0]
                    cell_type = row.get(self.cell_type_column, None)
                    cell_class = row.get("classification_system", None) or row.get(
                        "cell_class", None
                    )
            except Exception as e:
                logger.warning(
                    "Failed to query cell type table %s: %s",
                    self.cell_type_table, e,
                )

        # Query nucleus position
        soma_position_nm = None
        if self.nucleus_table:
            try:
                nuc_df = self.client.materialize.query_table(
                    self.nucleus_table,
                    filter_equal_dict={"pt_root_id": root_id},
                    select_columns=["pt_root_id", "pt_position"],
                )
                if not nuc_df.empty:
                    pos = nuc_df.iloc[0].get("pt_position")
                    if pos is not None:
                        if isinstance(pos, (list, tuple)) and len(pos) == 3:
                            soma_position_nm = tuple(float(x) for x in pos)
                        elif hasattr(pos, "tolist"):
                            coords = pos.tolist()
                            if len(coords) == 3:
                                soma_position_nm = tuple(float(x) for x in coords)
            except Exception as e:
                logger.warning("Failed to query nucleus position: %s", e)

        # Synapse counts
        n_pre = None
        n_post = None
        if self.synapse_table:
            try:
                pre_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"pre_pt_root_id": root_id},
                    select_columns=["pre_pt_root_id"],
                )
                n_pre = len(pre_df)
            except Exception as e:
                logger.warning("Failed to query presynaptic count: %s", e)

            try:
                post_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"post_pt_root_id": root_id},
                    select_columns=["post_pt_root_id"],
                )
                n_post = len(post_df)
            except Exception as e:
                logger.warning("Failed to query postsynaptic count: %s", e)

        # Get materialization version
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        result = {
            "neuron_id": root_id,
            "dataset": self.dataset_name,
            "cell_type": cell_type,
            "cell_class": cell_class,
            "soma_position_nm": soma_position_nm,
            "n_pre_synapses": n_pre,
            "n_post_synapses": n_post,
            "is_current": is_current,
            "materialization_version": mat_version,
            "neurotransmitter_type": None,
            "classification_hierarchy": None,
            "warnings": warnings,
        }

        # Dataset-specific enrichment hook
        return self._enrich_neuron_info(root_id, result)

    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        """Fetch all connectivity partners from CAVE.

        Parameters
        ----------
        neuron_id : int | str
            Root ID of the neuron.
        direction : str
            "upstream", "downstream", or "both".

        Returns
        -------
        dict
            Keys: neuron_id, dataset, is_current, materialization_version,
            warnings, partners_df (a pd.DataFrame with all partner rows).
        """
        root_id = int(neuron_id)
        logger.debug(
            "get_connectivity(%d, %s) on %s", root_id, direction, self.dataset_name
        )

        warnings: list[str] = []

        # Check root ID currency
        is_current = True
        try:
            is_latest = self.client.chunkedgraph.is_latest_roots([root_id])
            is_current = bool(is_latest[0]) if is_latest else True
            if not is_current:
                warnings.append(
                    f"Root ID {root_id} is outdated. "
                    f"Use `validate_root_ids()` to get current IDs."
                )
        except Exception as e:
            logger.warning("Failed to check root ID currency: %s", e)
            warnings.append(f"Could not verify root ID currency: {e}")

        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        rows: list[dict] = []

        # Upstream partners: neurons presynaptic to this neuron
        if direction in ("both", "upstream"):
            try:
                post_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"post_pt_root_id": root_id},
                    select_columns=["pre_pt_root_id"],
                )
                if not post_df.empty:
                    counts = post_df["pre_pt_root_id"].value_counts()
                    total_input = int(counts.sum())
                    for partner_id, n_syn in counts.items():
                        rows.append({
                            "partner_id": int(partner_id),
                            "direction": "upstream",
                            "n_synapses": int(n_syn),
                            "weight_normalized": (
                                float(n_syn) / total_input if total_input > 0 else 0.0
                            ),
                        })
            except Exception as e:
                logger.warning("Failed to query upstream partners: %s", e)
                warnings.append(f"Failed to query upstream partners: {e}")

        # Downstream partners: neurons postsynaptic to this neuron
        if direction in ("both", "downstream"):
            try:
                pre_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"pre_pt_root_id": root_id},
                    select_columns=["post_pt_root_id"],
                )
                if not pre_df.empty:
                    counts = pre_df["post_pt_root_id"].value_counts()
                    total_output = int(counts.sum())
                    for partner_id, n_syn in counts.items():
                        rows.append({
                            "partner_id": int(partner_id),
                            "direction": "downstream",
                            "n_synapses": int(n_syn),
                            "weight_normalized": (
                                float(n_syn) / total_output
                                if total_output > 0
                                else 0.0
                            ),
                        })
            except Exception as e:
                logger.warning("Failed to query downstream partners: %s", e)
                warnings.append(f"Failed to query downstream partners: {e}")

        partners_df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(
                columns=[
                    "partner_id", "direction", "n_synapses", "weight_normalized",
                ]
            )
        )

        # Batch lookup cell types for all partners
        if self.cell_type_table and not partners_df.empty:
            all_partner_ids = partners_df["partner_id"].unique().tolist()
            try:
                ct_df = self.client.materialize.query_table(
                    self.cell_type_table,
                    filter_in_dict={"pt_root_id": all_partner_ids},
                    select_columns=["pt_root_id", self.cell_type_column],
                )
                if not ct_df.empty:
                    ct_map = dict(
                        zip(ct_df["pt_root_id"], ct_df[self.cell_type_column])
                    )
                    partners_df["partner_type"] = partners_df["partner_id"].map(
                        ct_map
                    )
                else:
                    partners_df["partner_type"] = None
            except Exception as e:
                logger.warning(
                    "Failed to batch lookup partner cell types: %s", e
                )
                partners_df["partner_type"] = None
        elif not partners_df.empty:
            partners_df["partner_type"] = None

        # Add placeholder columns expected by the artifact schema
        if not partners_df.empty:
            if "partner_class" not in partners_df.columns:
                partners_df["partner_class"] = None
            if "partner_region" not in partners_df.columns:
                partners_df["partner_region"] = None
            if "neuroglancer_url" not in partners_df.columns:
                partners_df["neuroglancer_url"] = ""

        # Dataset-specific enrichment hook
        partners_df = self._enrich_connectivity(
            root_id, direction, partners_df, warnings
        )

        return {
            "neuron_id": root_id,
            "dataset": self.dataset_name,
            "is_current": is_current,
            "materialization_version": mat_version,
            "warnings": warnings,
            "partners_df": partners_df,
        }

    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        """Check whether CAVE root IDs are current.

        Parameters
        ----------
        root_ids : list[int]
            Root IDs to validate.

        Returns
        -------
        dict
            Keys: dataset, materialization_version, results (list of dicts),
            warnings.
        """
        logger.debug("validate_root_ids(%s) on %s", root_ids, self.dataset_name)

        warnings: list[str] = []
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        # Batch check currency
        is_latest = [True] * len(root_ids)
        try:
            is_latest = list(self.client.chunkedgraph.is_latest_roots(root_ids))
        except Exception as e:
            logger.warning("Failed to check root ID currency: %s", e)
            warnings.append(f"Could not verify root ID currency: {e}")

        results = []
        for root_id, current in zip(root_ids, is_latest):
            result: dict[str, Any] = {
                "root_id": int(root_id),
                "is_current": bool(current),
                "last_edit_timestamp": None,
                "suggested_current_id": None,
            }

            if not current:
                # Try to get the current replacement ID
                try:
                    latest = self.client.chunkedgraph.get_latest_roots(root_id)
                    if latest is not None and len(latest) > 0:
                        result["suggested_current_id"] = int(latest[0])
                except Exception as e:
                    logger.warning(
                        "Failed to get latest root for %d: %s", root_id, e
                    )

                # Try to get last edit timestamp
                try:
                    changelog_dict = (
                        self.client.chunkedgraph.get_tabular_change_log(
                            [root_id]
                        )
                    )
                    changelog = changelog_dict.get(root_id, pd.DataFrame())
                    if isinstance(changelog, pd.DataFrame) and not changelog.empty:
                        last_ts = changelog.iloc[-1].get("timestamp")
                        if last_ts is not None:
                            result["last_edit_timestamp"] = str(last_ts)
                except Exception as e:
                    logger.debug(
                        "Could not fetch changelog for %d: %s", root_id, e
                    )

            results.append(result)

        return {
            "dataset": self.dataset_name,
            "materialization_version": mat_version,
            "results": results,
            "warnings": warnings,
        }

    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        """Fetch proofreading status for a CAVE neuron.

        Parameters
        ----------
        neuron_id : int
            Root ID of the neuron.

        Returns
        -------
        dict
            Raw proofreading status dict for the formatter.
        """
        root_id = int(neuron_id)
        logger.debug(
            "get_proofreading_status(%d) on %s", root_id, self.dataset_name
        )

        warnings: list[str] = []

        # Check root ID currency
        is_current = True
        try:
            is_latest = self.client.chunkedgraph.is_latest_roots([root_id])
            is_current = bool(is_latest[0]) if is_latest else True
            if not is_current:
                warnings.append(
                    f"Root ID {root_id} is outdated. "
                    f"Use `validate_root_ids()` to get current IDs."
                )
        except Exception as e:
            logger.warning("Failed to check root ID currency: %s", e)
            warnings.append(f"Could not verify root ID currency: {e}")

        # Query proofreading table
        axon_proofread = None
        dendrite_proofread = None
        strategy_axon = None
        strategy_dendrite = None

        if self.proofreading_table:
            try:
                pr_df = self.client.materialize.query_table(
                    self.proofreading_table,
                    filter_equal_dict={"pt_root_id": root_id},
                )
                if not pr_df.empty:
                    row = pr_df.iloc[0]
                    interpreted = self._interpret_proofreading_row(row)
                    axon_proofread = interpreted["axon_proofread"]
                    dendrite_proofread = interpreted["dendrite_proofread"]
                    strategy_axon = interpreted["strategy_axon"]
                    strategy_dendrite = interpreted["strategy_dendrite"]
            except Exception as e:
                logger.warning(
                    "Failed to query proofreading table %s: %s",
                    self.proofreading_table, e,
                )
                warnings.append(f"Could not query proofreading table: {e}")
        else:
            warnings.append(
                f"No proofreading table configured for {self.dataset_name}"
            )

        # Get edit count and last edit timestamp
        n_edits = None
        last_edit_timestamp = None
        try:
            changelog_dict = self.client.chunkedgraph.get_tabular_change_log(
                [root_id]
            )
            changelog = changelog_dict.get(root_id, pd.DataFrame())
            if isinstance(changelog, pd.DataFrame) and not changelog.empty:
                n_edits = len(changelog)
                last_ts = changelog.iloc[-1].get("timestamp")
                if last_ts is not None:
                    last_edit_timestamp = str(last_ts)
        except Exception as e:
            logger.debug("Could not fetch changelog for %d: %s", root_id, e)

        return {
            "neuron_id": root_id,
            "dataset": self.dataset_name,
            "is_current": is_current,
            "axon_proofread": axon_proofread,
            "dendrite_proofread": dendrite_proofread,
            "strategy_axon": strategy_axon,
            "strategy_dendrite": strategy_dendrite,
            "n_edits": n_edits,
            "last_edit_timestamp": last_edit_timestamp,
            "warnings": warnings,
        }

    def resolve_nucleus_ids(
        self,
        nucleus_ids: list[int],
        materialization_version: int | None = None,
    ) -> dict[str, Any]:
        """Resolve nucleus IDs to current pt_root_ids.

        Parameters
        ----------
        nucleus_ids : list[int]
            Nucleus IDs to resolve.
        materialization_version : int, optional
            Materialization version to query at. Defaults to latest.

        Returns
        -------
        dict
            Raw resolution result with keys: dataset, materialization_version,
            resolutions, n_resolved, n_merge_conflicts, n_no_segment, warnings.
        """
        logger.debug(
            "resolve_nucleus_ids(%s) on %s", nucleus_ids, self.dataset_name
        )

        warnings: list[str] = []
        mat_version = materialization_version
        if mat_version is None:
            try:
                mat_version = self.client.materialize.version
            except Exception:
                mat_version = 0

        nuc_table = self.nucleus_table or "nucleus_detection_v0"

        # Query nucleus table for the given IDs
        nuc_df = pd.DataFrame()
        try:
            nuc_df = self.client.materialize.query_table(
                nuc_table,
                filter_in_dict={"id": nucleus_ids},
                select_columns=["id", "pt_root_id"],
            )
        except Exception as e:
            logger.warning("Failed to query nucleus table %s: %s", nuc_table, e)
            warnings.append(f"Failed to query nucleus table: {e}")

        # Build lookup: nucleus_id → pt_root_id
        nuc_to_root: dict[int, int | None] = {}
        if not nuc_df.empty:
            for _, row in nuc_df.iterrows():
                nid = int(row["id"])
                rid = row.get("pt_root_id")
                if rid is not None and pd.notna(rid):
                    nuc_to_root[nid] = int(rid)
                else:
                    nuc_to_root[nid] = None

        # Detect merge conflicts: group nucleus IDs by pt_root_id
        root_to_nucs: dict[int, list[int]] = {}
        for nid, rid in nuc_to_root.items():
            if rid is not None:
                root_to_nucs.setdefault(rid, []).append(nid)

        merge_conflict_nucs: set[int] = set()
        for rid, nids in root_to_nucs.items():
            if len(nids) > 1:
                merge_conflict_nucs.update(nids)

        # Build resolutions
        resolutions: list[dict[str, Any]] = []
        n_resolved = 0
        n_merge_conflicts = 0
        n_no_segment = 0

        for nid in nucleus_ids:
            rid = nuc_to_root.get(nid)

            if nid not in nuc_to_root or rid is None:
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": None,
                    "resolution_status": "no_segment",
                    "conflicting_nucleus_ids": [],
                    "materialization_version": mat_version,
                })
                n_no_segment += 1
            elif nid in merge_conflict_nucs:
                conflicting = [
                    other for other in root_to_nucs[rid] if other != nid
                ]
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": rid,
                    "resolution_status": "merge_conflict",
                    "conflicting_nucleus_ids": conflicting,
                    "materialization_version": mat_version,
                })
                n_merge_conflicts += 1
            else:
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": rid,
                    "resolution_status": "resolved",
                    "conflicting_nucleus_ids": [],
                    "materialization_version": mat_version,
                })
                n_resolved += 1

        return {
            "dataset": self.dataset_name,
            "materialization_version": mat_version,
            "resolutions": resolutions,
            "n_resolved": n_resolved,
            "n_merge_conflicts": n_merge_conflicts,
            "n_no_segment": n_no_segment,
            "warnings": warnings,
        }

    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        """Fetch region-to-region connectivity from CAVE.

        Parameters
        ----------
        source_region : str, optional
            Filter to connections from this region.
        target_region : str, optional
            Filter to connections to this region.

        Returns
        -------
        dict
            Keys: dataset, materialization_version, warnings, region_df.
        """
        logger.debug(
            "get_region_connectivity(source=%s, target=%s) on %s",
            source_region, target_region, self.dataset_name,
        )

        warnings: list[str] = []
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        empty_df = pd.DataFrame(
            columns=[
                "source_region", "target_region", "n_synapses",
                "n_neurons_pre", "n_neurons_post",
            ]
        )

        if not self.cell_type_table:
            warnings.append("No cell type table configured for this dataset")
            return {
                "dataset": self.dataset_name,
                "materialization_version": mat_version,
                "warnings": warnings,
                "region_df": empty_df,
            }

        # Query synapse table for pre/post root IDs
        try:
            syn_df = self.client.materialize.query_table(
                self.synapse_table,
                select_columns=["pre_pt_root_id", "post_pt_root_id"],
            )
        except Exception as e:
            logger.warning("Failed to query synapse table: %s", e)
            warnings.append(f"Failed to query synapse table: {e}")
            return {
                "dataset": self.dataset_name,
                "materialization_version": mat_version,
                "warnings": warnings,
                "region_df": empty_df,
            }

        if syn_df.empty:
            return {
                "dataset": self.dataset_name,
                "materialization_version": mat_version,
                "warnings": warnings,
                "region_df": empty_df,
            }

        # Get region annotations for all neurons
        all_ids = list(
            set(syn_df["pre_pt_root_id"].unique().tolist())
            | set(syn_df["post_pt_root_id"].unique().tolist())
        )
        try:
            ct_df = self.client.materialize.query_table(
                self.cell_type_table,
                filter_in_dict={"pt_root_id": all_ids},
                select_columns=["pt_root_id", self.cell_type_column],
            )
        except Exception as e:
            logger.warning(
                "Failed to query cell type table for regions: %s", e
            )
            warnings.append(f"Failed to query cell type table: {e}")
            return {
                "dataset": self.dataset_name,
                "materialization_version": mat_version,
                "warnings": warnings,
                "region_df": empty_df,
            }

        # Build region map from cell type table
        region_map: dict[int, str] = {}
        for _, row in ct_df.iterrows():
            rid = row.get("pt_root_id")
            region = (
                row.get("tag") or row.get("region") or row.get(self.cell_type_column)
            )
            if rid is not None and region is not None:
                region_map[int(rid)] = str(region)

        # Map regions onto synapses
        syn_df = syn_df.copy()
        syn_df["source_region"] = syn_df["pre_pt_root_id"].map(region_map)
        syn_df["target_region"] = syn_df["post_pt_root_id"].map(region_map)

        # Drop rows where region is unknown
        syn_df = syn_df.dropna(subset=["source_region", "target_region"])

        # Apply filters
        if source_region:
            syn_df = syn_df[
                syn_df["source_region"].str.contains(
                    source_region, case=False, na=False
                )
            ]
        if target_region:
            syn_df = syn_df[
                syn_df["target_region"].str.contains(
                    target_region, case=False, na=False
                )
            ]

        if syn_df.empty:
            return {
                "dataset": self.dataset_name,
                "materialization_version": mat_version,
                "warnings": warnings,
                "region_df": empty_df,
            }

        # Group by region pair
        grouped = syn_df.groupby(["source_region", "target_region"])
        region_rows: list[dict] = []
        for (src, tgt), group in grouped:
            region_rows.append({
                "source_region": src,
                "target_region": tgt,
                "n_synapses": len(group),
                "n_neurons_pre": group["pre_pt_root_id"].nunique(),
                "n_neurons_post": group["post_pt_root_id"].nunique(),
            })

        region_df = pd.DataFrame(region_rows)

        return {
            "dataset": self.dataset_name,
            "materialization_version": mat_version,
            "warnings": warnings,
            "region_df": region_df,
        }

    def query_annotation_table(
        self,
        table_name: str,
        filter_equal_dict: dict[str, Any] | None = None,
        filter_in_dict: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Query an arbitrary CAVE annotation table.

        Parameters
        ----------
        table_name : str
            Name of the annotation table to query.
        filter_equal_dict : dict, optional
            Equality filters passed to ``client.materialize.query_table()``.
        filter_in_dict : dict, optional
            Membership filters passed to ``client.materialize.query_table()``.

        Returns
        -------
        dict
            Keys: dataset, table_name, materialization_version, warnings,
            table_df, schema_description.
        """
        logger.debug(
            "query_annotation_table(%s) on %s", table_name, self.dataset_name
        )

        warnings: list[str] = []
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        query_kwargs: dict[str, Any] = {}
        if filter_equal_dict:
            query_kwargs["filter_equal_dict"] = filter_equal_dict
        if filter_in_dict:
            query_kwargs["filter_in_dict"] = filter_in_dict

        try:
            table_df = self.client.materialize.query_table(
                table_name, **query_kwargs
            )
        except Exception as e:
            logger.warning(
                "Failed to query annotation table %s: %s", table_name, e
            )
            warnings.append(f"Failed to query annotation table: {e}")
            table_df = pd.DataFrame()

        # Build schema description from DataFrame dtypes
        if not table_df.empty:
            col_descs = [
                f"{col}: {dtype}" for col, dtype in table_df.dtypes.items()
            ]
            schema_description = "; ".join(col_descs)
        else:
            schema_description = "Empty result"

        return {
            "dataset": self.dataset_name,
            "table_name": table_name,
            "materialization_version": mat_version,
            "warnings": warnings,
            "table_df": table_df,
            "schema_description": schema_description,
        }

    def get_edit_history(self, neuron_id: int) -> dict[str, Any]:
        """Fetch edit history for a CAVE neuron.

        Parameters
        ----------
        neuron_id : int
            Root ID of the neuron.

        Returns
        -------
        dict
            Keys: neuron_id, dataset, is_current, materialization_version,
            warnings, edits_df.
        """
        root_id = int(neuron_id)
        logger.debug("get_edit_history(%d) on %s", root_id, self.dataset_name)

        warnings: list[str] = []

        # Check root ID currency
        is_current = True
        try:
            is_latest = self.client.chunkedgraph.is_latest_roots([root_id])
            is_current = bool(is_latest[0]) if is_latest else True
            if not is_current:
                warnings.append(
                    f"Root ID {root_id} is outdated. "
                    f"Use `validate_root_ids()` to get current IDs."
                )
        except Exception as e:
            logger.warning("Failed to check root ID currency: %s", e)
            warnings.append(f"Could not verify root ID currency: {e}")

        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        # Fetch edit changelog
        edits_df = pd.DataFrame(
            columns=["operation_id", "timestamp", "operation_type", "user_id"]
        )
        try:
            changelog_dict = self.client.chunkedgraph.get_tabular_change_log(
                [root_id]
            )
            changelog = changelog_dict.get(root_id, pd.DataFrame())
            if isinstance(changelog, pd.DataFrame) and not changelog.empty:
                rows: list[dict] = []
                for idx, row in changelog.iterrows():
                    is_merge = row.get("is_merge", None)
                    if is_merge is True:
                        op_type = "merge"
                    elif is_merge is False:
                        op_type = "split"
                    else:
                        op_type = "unknown"
                    rows.append({
                        "operation_id": (
                            int(idx)
                            if isinstance(idx, (int, float))
                            else len(rows)
                        ),
                        "timestamp": str(row.get("timestamp", "")),
                        "operation_type": op_type,
                        "user_id": str(row.get("user_id", "")),
                    })
                edits_df = pd.DataFrame(rows)
        except Exception as e:
            logger.warning(
                "Failed to fetch edit changelog for %d: %s", root_id, e
            )
            warnings.append(f"Failed to fetch edit changelog: {e}")

        return {
            "neuron_id": root_id,
            "dataset": self.dataset_name,
            "is_current": is_current,
            "materialization_version": mat_version,
            "warnings": warnings,
            "edits_df": edits_df,
        }

    def fetch_cypher(self, query: str) -> dict[str, Any]:
        """Not applicable for CAVE datasets."""
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported(self.dataset_name, "neuprint")

    def get_synapse_compartments(
        self, neuron_id: int | str, direction: str = "input"
    ) -> dict[str, Any]:
        """Not applicable for CAVE datasets."""
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported(self.dataset_name, "neuprint")

    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        """Fetch neurons matching a cell type from CAVE.

        Parameters
        ----------
        cell_type : str
            Cell type annotation to search for.
        region : str, optional
            Brain region filter.

        Returns
        -------
        dict
            Keys: dataset, query_cell_type, query_region,
            materialization_version, warnings, neurons_df.
        """
        logger.debug(
            "get_neurons_by_type(%s, region=%s) on %s",
            cell_type, region, self.dataset_name,
        )

        warnings: list[str] = []
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        empty_neurons_df = pd.DataFrame(
            columns=[
                "neuron_id", "cell_type", "cell_class",
                "region", "n_pre_synapses", "n_post_synapses",
                "proofread",
            ]
        )

        if not self.cell_type_table:
            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": mat_version,
                "warnings": ["No cell type table configured for this dataset"],
                "neurons_df": empty_neurons_df,
            }

        # Query cell type table
        try:
            ct_df = self.client.materialize.query_table(
                self.cell_type_table,
                filter_equal_dict={self.cell_type_column: cell_type},
            )
        except Exception as e:
            logger.warning("Failed to query cell type table: %s", e)
            warnings.append(f"Failed to query cell type table: {e}")
            ct_df = pd.DataFrame()

        if ct_df.empty:
            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": mat_version,
                "warnings": warnings,
                "neurons_df": empty_neurons_df,
            }

        # Build result DataFrame
        rows = []
        for _, row in ct_df.iterrows():
            root_id = row.get("pt_root_id")
            if root_id is None:
                continue
            rows.append({
                "neuron_id": int(root_id),
                "cell_type": row.get(self.cell_type_column),
                "cell_class": (
                    row.get("classification_system") or row.get("cell_class")
                ),
                "region": row.get("tag", None) or row.get("region", None),
                "n_pre_synapses": None,
                "n_post_synapses": None,
                "proofread": None,
            })

        neurons_df = pd.DataFrame(rows)

        # Filter by region if requested
        if region and not neurons_df.empty:
            mask = neurons_df["region"].str.contains(
                region, case=False, na=False
            )
            neurons_df = neurons_df[mask].reset_index(drop=True)

        return {
            "dataset": self.dataset_name,
            "query_cell_type": cell_type,
            "query_region": region,
            "materialization_version": mat_version,
            "warnings": warnings,
            "neurons_df": neurons_df,
        }


# ---------------------------------------------------------------------------
# MICrONSBackend
# ---------------------------------------------------------------------------

class MICrONSBackend(CAVEBackend):
    """CAVE backend for MICrONS (minnie65) with nucleus enrichment."""

    dataset_name = "minnie65"
    datastack = "minnie65_public"
    cell_type_table = "aibs_metamodel_celltypes_v661"
    cell_type_column = "cell_type"
    synapse_table = "synapses_pni_2"
    nucleus_table = "nucleus_detection_v0"
    proofreading_table = "proofreading_status_and_strategy"

    def _enrich_connectivity(
        self,
        root_id: int,
        direction: str,
        partners_df: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Add partner_nucleus_id and partner_nucleus_conflict columns."""
        if partners_df.empty:
            return partners_df

        all_partner_ids = partners_df["partner_id"].unique().tolist()
        try:
            nuc_df = self.client.materialize.query_table(
                self.nucleus_table,
                filter_in_dict={"pt_root_id": all_partner_ids},
                select_columns=["id", "pt_root_id"],
            )
            if not nuc_df.empty:
                # Group nucleus IDs by pt_root_id
                nuc_groups = (
                    nuc_df.groupby("pt_root_id")["id"].apply(list).to_dict()
                )

                nuc_ids: list[int | None] = []
                conflicts: list[bool] = []
                for pid in partners_df["partner_id"]:
                    nids = nuc_groups.get(int(pid), [])
                    if len(nids) == 1:
                        nuc_ids.append(int(nids[0]))
                        conflicts.append(False)
                    elif len(nids) > 1:
                        nuc_ids.append(None)
                        conflicts.append(True)
                        logger.debug(
                            "Partner %d has multiple nucleus IDs: %s",
                            int(pid), nids,
                        )
                    else:
                        nuc_ids.append(None)
                        conflicts.append(False)

                partners_df = partners_df.copy()
                partners_df["partner_nucleus_id"] = nuc_ids
                partners_df["partner_nucleus_conflict"] = conflicts
            else:
                partners_df = partners_df.copy()
                partners_df["partner_nucleus_id"] = None
                partners_df["partner_nucleus_conflict"] = False
        except Exception as e:
            logger.warning(
                "Failed to enrich partners with nucleus IDs: %s", e
            )
            warnings.append(f"Failed to enrich partners with nucleus IDs: {e}")
            partners_df = partners_df.copy()
            partners_df["partner_nucleus_id"] = None
            partners_df["partner_nucleus_conflict"] = False

        return partners_df


# ---------------------------------------------------------------------------
# FlyWireBackend
# ---------------------------------------------------------------------------

# Neurotransmitter column names in FlyWire's synapses_nt_v1
NT_COLUMNS = ["gaba", "ach", "glut", "oct", "ser", "da"]

# Human-readable names for argmax NT prediction
NT_LABELS = {
    "gaba": "GABA",
    "ach": "acetylcholine",
    "glut": "glutamate",
    "oct": "octopamine",
    "ser": "serotonin",
    "da": "dopamine",
}

# Hierarchy levels in hierarchical_neuron_annotations
HIERARCHY_LEVELS = [
    "super_class", "cell_class", "cell_sub_class", "cell_type",
]

_HIERARCHY_CACHE_TTL = 600  # 10 minutes


class FlyWireBackend(CAVEBackend):
    """CAVE backend for FlyWire with hierarchy, NT prediction, and NT enrichment."""

    dataset_name = "flywire"
    datastack = "flywire_fafb_public"
    cell_type_table = "neuron_information_v2"
    cell_type_column = "tag"
    synapse_table = "synapses_nt_v1"
    nucleus_table = "nuclei_v1"
    proofreading_table = "proofread_neurons"

    def __init__(self) -> None:
        super().__init__()
        self._hierarchy_cache: tuple[pd.DataFrame, float] | None = None

    # -- FlyWire-specific helpers -------------------------------------------

    def _get_hierarchy_df(self) -> pd.DataFrame:
        """Fetch or return cached hierarchical_neuron_annotations table.

        The table cannot be filtered server-side by pt_root_id (returns
        500), so we cache the full table in memory with a 10-minute TTL.
        """
        now = time.monotonic()
        if self._hierarchy_cache is not None:
            df, cached_at = self._hierarchy_cache
            if now - cached_at < _HIERARCHY_CACHE_TTL:
                return df

        df = self.client.materialize.query_table(
            "hierarchical_neuron_annotations",
            select_columns=[
                "pt_root_id", "classification_system", "cell_type",
            ],
        )
        self._hierarchy_cache = (df, now)
        return df

    def _get_flywire_hierarchy(self, root_id: int) -> dict | None:
        """Look up FlyWire hierarchical classification for a neuron.

        Returns a dict like::

            {
                "super_class": "central",
                "cell_class": "visual_projection",
                "cell_sub_class": "LC",
                "cell_type": "LC10",
            }

        or None if no annotation exists.
        """
        try:
            hier_df = self._get_hierarchy_df()
        except Exception as e:
            logger.warning("Failed to fetch hierarchy table: %s", e)
            return None

        rows = hier_df[hier_df["pt_root_id"] == root_id]
        if rows.empty:
            return None

        result: dict[str, str | None] = {}
        for _, row in rows.iterrows():
            level = row.get("classification_system")
            ct = row.get("cell_type")
            if level and ct and pd.notna(level) and pd.notna(ct):
                result[str(level)] = str(ct)

        return result if result else None

    # -- template-method overrides ------------------------------------------

    def _enrich_neuron_info(
        self, root_id: int, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Add hierarchical classification and NT prediction."""
        # Hierarchical classification (replaces neuron_information_v2 lookup)
        hierarchy = self._get_flywire_hierarchy(root_id)
        if hierarchy:
            result["classification_hierarchy"] = hierarchy
            # Use finest available level as cell_type
            for level in reversed(HIERARCHY_LEVELS):
                if level in hierarchy:
                    result["cell_type"] = hierarchy[level]
                    break

        # Neurotransmitter prediction from output synapses
        n_pre = result.get("n_pre_synapses")
        if self.synapse_table and n_pre and n_pre > 0:
            try:
                nt_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"pre_pt_root_id": root_id},
                    select_columns=["pre_pt_root_id"] + NT_COLUMNS,
                )
                if not nt_df.empty:
                    nt_means = nt_df[NT_COLUMNS].mean()
                    best_nt = nt_means.idxmax()
                    result["neurotransmitter_type"] = NT_LABELS.get(
                        best_nt, best_nt
                    )
            except Exception as e:
                logger.warning("Failed to query NT predictions: %s", e)

        return result

    def _enrich_connectivity(
        self,
        root_id: int,
        direction: str,
        partners_df: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Add partner_nt_type and partner_nt_confidence columns."""
        if partners_df.empty:
            return partners_df

        try:
            nt_dfs = []
            if direction in ("both", "upstream"):
                up_nt_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"post_pt_root_id": root_id},
                    select_columns=["pre_pt_root_id"] + NT_COLUMNS,
                )
                if not up_nt_df.empty:
                    up_nt_df = up_nt_df.rename(
                        columns={"pre_pt_root_id": "partner_id"}
                    )
                    nt_dfs.append(up_nt_df)

            if direction in ("both", "downstream"):
                dn_nt_df = self.client.materialize.query_table(
                    self.synapse_table,
                    filter_equal_dict={"pre_pt_root_id": root_id},
                    select_columns=["post_pt_root_id"] + NT_COLUMNS,
                )
                if not dn_nt_df.empty:
                    dn_nt_df = dn_nt_df.rename(
                        columns={"post_pt_root_id": "partner_id"}
                    )
                    nt_dfs.append(dn_nt_df)

            if nt_dfs:
                all_nt = pd.concat(nt_dfs, ignore_index=True)
                grouped = all_nt.groupby("partner_id")[NT_COLUMNS].mean()
                grouped["nt_type"] = grouped.idxmax(axis=1)
                grouped["nt_confidence"] = grouped[NT_COLUMNS].max(axis=1)
                nt_map = grouped["nt_type"].map(NT_LABELS).to_dict()
                conf_map = grouped["nt_confidence"].to_dict()

                partners_df = partners_df.copy()
                partners_df["partner_nt_type"] = partners_df["partner_id"].map(
                    nt_map
                )
                partners_df["partner_nt_confidence"] = partners_df[
                    "partner_id"
                ].map(conf_map)
            else:
                partners_df = partners_df.copy()
                partners_df["partner_nt_type"] = None
                partners_df["partner_nt_confidence"] = None
        except Exception as e:
            logger.warning(
                "Failed to enrich partners with NT predictions: %s", e
            )
            warnings.append(
                f"Failed to enrich partners with NT predictions: {e}"
            )
            partners_df = partners_df.copy()
            partners_df["partner_nt_type"] = None
            partners_df["partner_nt_confidence"] = None

        return partners_df

    def _interpret_proofreading_row(
        self, row: pd.Series
    ) -> dict[str, Any]:
        """FlyWire: presence in proofread_neurons table = proofread."""
        return {
            "axon_proofread": True,
            "dendrite_proofread": True,
            "strategy_axon": None,
            "strategy_dendrite": None,
        }

    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        """Fetch neurons by type using FlyWire hierarchy cache.

        Full override — uses ``hierarchical_neuron_annotations`` instead
        of ``neuron_information_v2``.
        """
        logger.debug(
            "get_neurons_by_type(%s, region=%s) on %s",
            cell_type, region, self.dataset_name,
        )

        warnings: list[str] = []
        mat_version = None
        try:
            mat_version = self.client.materialize.version
        except Exception:
            pass

        empty_neurons_df = pd.DataFrame(
            columns=[
                "neuron_id", "cell_type", "cell_class",
                "region", "n_pre_synapses", "n_post_synapses",
                "proofread",
            ]
        )

        try:
            hier_df = self._get_hierarchy_df()
            matches = hier_df[
                (hier_df["classification_system"] == "cell_type")
                & (hier_df["cell_type"] == cell_type)
            ]
            if matches.empty:
                return {
                    "dataset": self.dataset_name,
                    "query_cell_type": cell_type,
                    "query_region": region,
                    "materialization_version": mat_version,
                    "warnings": warnings,
                    "neurons_df": empty_neurons_df,
                }

            rows: list[dict] = []
            for _, row in matches.iterrows():
                rid = row.get("pt_root_id")
                if rid is None:
                    continue
                hierarchy = self._get_flywire_hierarchy(int(rid))
                rows.append({
                    "neuron_id": int(rid),
                    "cell_type": cell_type,
                    "cell_class": (
                        hierarchy.get("cell_class") if hierarchy else None
                    ),
                    "region": (
                        hierarchy.get("super_class") if hierarchy else None
                    ),
                    "n_pre_synapses": None,
                    "n_post_synapses": None,
                    "proofread": None,
                })

            neurons_df = pd.DataFrame(rows)

            if region and not neurons_df.empty:
                mask = neurons_df["region"].str.contains(
                    region, case=False, na=False
                )
                neurons_df = neurons_df[mask].reset_index(drop=True)

            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": mat_version,
                "warnings": warnings,
                "neurons_df": neurons_df,
            }
        except Exception as e:
            logger.warning("Failed to query hierarchy table: %s", e)
            warnings.append(f"Failed to query hierarchy table: {e}")
            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": mat_version,
                "warnings": warnings,
                "neurons_df": empty_neurons_df,
            }


