"""Custom exceptions for the Connectomics MCP server."""


class DatasetNotSupported(Exception):
    """Raised when a tool is called against an incompatible dataset.

    Parameters
    ----------
    dataset : str
        The dataset name that was requested.
    capability : str, optional
        The capability that the dataset lacks.
    """

    def __init__(self, dataset: str, capability: str | None = None) -> None:
        if capability:
            msg = (
                f"Dataset '{dataset}' does not support the '{capability}' capability. "
                f"Check TOOL_REGISTRY.md for dataset compatibility."
            )
        else:
            msg = f"Unknown dataset '{dataset}'. Supported: minnie65, flywire, fanc, hemibrain."
        super().__init__(msg)
        self.dataset = dataset
        self.capability = capability


class StaleRootIdError(Exception):
    """Raised when a CAVE root ID is outdated due to proofreading.

    Parameters
    ----------
    root_id : int
        The stale root ID.
    """

    def __init__(self, root_id: int) -> None:
        msg = (
            f"Root ID {root_id} is outdated. "
            f"Use `validate_root_ids()` to get current IDs."
        )
        super().__init__(msg)
        self.root_id = root_id


class BackendConnectionError(Exception):
    """Raised when a backend API server cannot be reached.

    Parameters
    ----------
    backend : str
        The backend name (e.g. "cave", "neuprint").
    detail : str
        Additional error detail.
    """

    def __init__(self, backend: str, detail: str = "") -> None:
        msg = f"Cannot connect to {backend} backend."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg)
        self.backend = backend
