"""Abstract base class for connectome backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConnectomeBackend(ABC):
    """Abstract interface for connectome dataset backends.

    Each backend adapts a specific API (CAVE, neuPrint) to the
    universal tool interface. Methods return raw dicts; formatting
    to Pydantic schemas happens in ``output_contracts.formatters``.
    """

    @abstractmethod
    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        """Fetch basic neuron information.

        Parameters
        ----------
        neuron_id : int | str
            Root ID (CAVE) or body ID (neuPrint).

        Returns
        -------
        dict
            Raw neuron info dict for the formatter.
        """

    @abstractmethod
    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        """Fetch all connectivity partners.

        Parameters
        ----------
        neuron_id : int | str
            Root ID (CAVE) or body ID (neuPrint).
        direction : str
            "upstream", "downstream", or "both".

        Returns
        -------
        dict
            Raw connectivity dict with ``partners_df`` key for the formatter.
        """

    @abstractmethod
    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        """Fetch neurons matching a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type annotation to search for.
        region : str, optional
            Brain region filter.

        Returns
        -------
        dict
            Raw neuron list dict for the formatter.
        """

    @abstractmethod
    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        """Check whether root IDs are current.

        Parameters
        ----------
        root_ids : list[int]
            Root IDs (CAVE) or body IDs (neuPrint) to validate.

        Returns
        -------
        dict
            Raw validation results for the formatter.
        """

    @abstractmethod
    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        """Fetch region-to-region connectivity.

        Parameters
        ----------
        source_region : str, optional
            Filter to connections from this region.
        target_region : str, optional
            Filter to connections to this region.

        Returns
        -------
        dict
            Raw region connectivity dict with ``region_df`` key for the formatter.
        """

    @abstractmethod
    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        """Fetch proofreading status for a neuron.

        Parameters
        ----------
        neuron_id : int
            Root ID of the neuron.

        Returns
        -------
        dict
            Raw proofreading status for the formatter.
        """
