"""Base adapters distributed with ECHO."""
from . import Source, Tent, EATA, ROID


def setup_adapter(model, cfg):
    return {"Source": Source, "Tent": Tent, "EATA": EATA, "ROID": ROID}[cfg.ADAPTER.NAME].setup(model, cfg)
