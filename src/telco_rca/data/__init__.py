from .base import Case, DatasetAdapter, KPISpec, ReferenceProfile, Ticket, robust_profile

ADAPTERS = {}


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate a registered adapter by name (lazy imports keep optional deps optional)."""
    if name == "telecomts":
        from .telecomts import TelecomTSAdapter

        return TelecomTSAdapter(**kwargs)
    if name == "synthetic":
        from .synthetic import SyntheticAdapter

        return SyntheticAdapter(**kwargs)
    if name in ADAPTERS:
        return ADAPTERS[name](**kwargs)
    raise KeyError(f"unknown adapter {name!r}; register it in telco_rca.data.ADAPTERS")


__all__ = [
    "ADAPTERS",
    "Case",
    "DatasetAdapter",
    "KPISpec",
    "ReferenceProfile",
    "Ticket",
    "get_adapter",
    "robust_profile",
]
