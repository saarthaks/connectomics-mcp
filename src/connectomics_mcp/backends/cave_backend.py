"""CAVE backend adapter for MICrONS, FlyWire, and FANC datasets."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from connectomics_mcp.backends.base import ConnectomeBackend
from connectomics_mcp.exceptions import BackendConnectionError

logger = logging.getLogger(__name__)

# Dataset-specific table names for cell type annotations
CELL_TYPE_TABLES: dict[str, str] = {
    "minnie65": "aibs_metamodel_celltypes_v661",
    "flywire": "classification",
    "fanc": "cell_info",
}

SYNAPSE_TABLES: dict[str, str] = {
    "minnie65": "synapses_pni_2",
    "flywire": "synapses",
    "fanc": "synapses",
}

NUCLEUS_TABLES: dict[str, str] = {
    "minnie65": "nucleus_detection_v0",
    "flywire": "nuclei_v1",
    "fanc": "nuclei_v1",
}

PROOFREADING_TABLES: dict[str, str] = {
    "minnie65": "proofreading_status_public_release",
    "flywire": "proofreading_status_table",
    "fanc": "proofreading_status",
}


class CAVEBackend(ConnectomeBackend):
    """Backend adapter for CAVE-based connectomic datasets.

    Parameters
    ----------
    datastack : str
        The CAVE datastack name.
    dataset_name : str
        Human-readable dataset name for error messages.
    """

    def __init__(self, datastack: str, dataset_name: str) -> None:
        self.datastack = datastack
        self.dataset_name = dataset_name
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
        ct_table = CELL_TYPE_TABLES.get(self.dataset_name)
        if ct_table:
            try:
                ct_df = self.client.materialize.query_table(
                    ct_table, filter_equal_dict={"pt_root_id": root_id}
                )
                if len(ct_df) > 0:
                    row = ct_df.iloc[0]
                    cell_type = row.get("cell_type", None)
                    cell_class = row.get("classification_system", None) or row.get(
                        "cell_class", None
                    )
            except Exception as e:
                logger.warning("Failed to query cell type table %s: %s", ct_table, e)

        # Query nucleus position
        soma_position_nm = None
        nuc_table = NUCLEUS_TABLES.get(self.dataset_name)
        if nuc_table:
            try:
                nuc_df = self.client.materialize.query_table(
                    nuc_table, filter_equal_dict={"pt_root_id": root_id}
                )
                if len(nuc_df) > 0:
                    row = nuc_df.iloc[0]
                    pt_position = row.get("pt_position", None)
                    if pt_position is not None:
                        soma_position_nm = tuple(float(x) for x in pt_position)
            except Exception as e:
                logger.warning("Failed to query nucleus table %s: %s", nuc_table, e)

        # Query synapse counts
        n_pre = None
        n_post = None
        syn_table = SYNAPSE_TABLES.get(self.dataset_name)
        if syn_table:
            try:
                pre_df = self.client.materialize.query_table(
                    syn_table,
                    filter_equal_dict={"pre_pt_root_id": root_id},
                    select_columns=["id"],
                )
                n_pre = len(pre_df)
            except Exception as e:
                logger.warning("Failed to query presynaptic count: %s", e)

            try:
                post_df = self.client.materialize.query_table(
                    syn_table,
                    filter_equal_dict={"post_pt_root_id": root_id},
                    select_columns=["id"],
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

        return {
            "neuron_id": root_id,
            "dataset": self.dataset_name,
            "cell_type": cell_type,
            "cell_class": cell_class,
            "soma_position_nm": soma_position_nm,
            "n_pre_synapses": n_pre,
            "n_post_synapses": n_post,
            "is_current": is_current,
            "materialization_version": mat_version,
            "warnings": warnings,
        }

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
        logger.debug("get_connectivity(%d, %s) on %s", root_id, direction, self.dataset_name)

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

        syn_table = SYNAPSE_TABLES.get(self.dataset_name, "synapses")
        ct_table = CELL_TYPE_TABLES.get(self.dataset_name)
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
                    syn_table,
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
                            "weight_normalized": float(n_syn) / total_input if total_input > 0 else 0.0,
                        })
            except Exception as e:
                logger.warning("Failed to query upstream partners: %s", e)
                warnings.append(f"Failed to query upstream partners: {e}")

        # Downstream partners: neurons postsynaptic to this neuron
        if direction in ("both", "downstream"):
            try:
                pre_df = self.client.materialize.query_table(
                    syn_table,
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
                            "weight_normalized": float(n_syn) / total_output if total_output > 0 else 0.0,
                        })
            except Exception as e:
                logger.warning("Failed to query downstream partners: %s", e)
                warnings.append(f"Failed to query downstream partners: {e}")

        partners_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["partner_id", "direction", "n_synapses", "weight_normalized"]
        )

        # Batch lookup cell types for all partners
        if ct_table and not partners_df.empty:
            all_partner_ids = partners_df["partner_id"].unique().tolist()
            try:
                ct_df = self.client.materialize.query_table(
                    ct_table,
                    filter_in_dict={"pt_root_id": all_partner_ids},
                    select_columns=["pt_root_id", "cell_type"],
                )
                if not ct_df.empty:
                    ct_map = dict(zip(ct_df["pt_root_id"], ct_df["cell_type"]))
                    partners_df["partner_type"] = partners_df["partner_id"].map(ct_map)
                else:
                    partners_df["partner_type"] = None
            except Exception as e:
                logger.warning("Failed to batch lookup partner cell types: %s", e)
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
                    changelog = self.client.chunkedgraph.get_tabular_changelog(
                        root_id
                    )
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
        proofread_table = PROOFREADING_TABLES.get(self.dataset_name)
        axon_proofread = None
        dendrite_proofread = None
        strategy_axon = None
        strategy_dendrite = None

        if proofread_table:
            try:
                pr_df = self.client.materialize.query_table(
                    proofread_table,
                    filter_equal_dict={"pt_root_id": root_id},
                )
                if not pr_df.empty:
                    row = pr_df.iloc[0]
                    axon_proofread = row.get("status_axon") in (
                        True, "t", "True", "extended",
                    ) if "status_axon" in row.index else None
                    dendrite_proofread = row.get("status_dendrite") in (
                        True, "t", "True", "extended",
                    ) if "status_dendrite" in row.index else None
                    strategy_axon = row.get("strategy_axon")
                    strategy_dendrite = row.get("strategy_dendrite")
            except Exception as e:
                logger.warning(
                    "Failed to query proofreading table %s: %s",
                    proofread_table, e,
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
            changelog = self.client.chunkedgraph.get_tabular_changelog(root_id)
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

        syn_table = SYNAPSE_TABLES.get(self.dataset_name, "synapses")
        ct_table = CELL_TYPE_TABLES.get(self.dataset_name)

        empty_df = pd.DataFrame(
            columns=["source_region", "target_region", "n_synapses",
                     "n_neurons_pre", "n_neurons_post"]
        )

        if not ct_table:
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
                syn_table,
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
                ct_table,
                filter_in_dict={"pt_root_id": all_ids},
                select_columns=["pt_root_id", "cell_type"],
            )
        except Exception as e:
            logger.warning("Failed to query cell type table for regions: %s", e)
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
            region = row.get("tag") or row.get("region") or row.get("cell_type")
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

        ct_table = CELL_TYPE_TABLES.get(self.dataset_name)
        if not ct_table:
            return {
                "dataset": self.dataset_name,
                "query_cell_type": cell_type,
                "query_region": region,
                "materialization_version": mat_version,
                "warnings": ["No cell type table configured for this dataset"],
                "neurons_df": pd.DataFrame(
                    columns=["neuron_id", "cell_type", "cell_class",
                             "region", "n_pre_synapses", "n_post_synapses",
                             "proofread"]
                ),
            }

        # Query cell type table
        try:
            ct_df = self.client.materialize.query_table(
                ct_table, filter_equal_dict={"cell_type": cell_type}
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
                "neurons_df": pd.DataFrame(
                    columns=["neuron_id", "cell_type", "cell_class",
                             "region", "n_pre_synapses", "n_post_synapses",
                             "proofread"]
                ),
            }

        # Build result DataFrame
        rows: list[dict] = []
        for _, row in ct_df.iterrows():
            root_id = row.get("pt_root_id")
            if root_id is None:
                continue
            rows.append({
                "neuron_id": int(root_id),
                "cell_type": row.get("cell_type"),
                "cell_class": (
                    row.get("classification_system")
                    or row.get("cell_class")
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
