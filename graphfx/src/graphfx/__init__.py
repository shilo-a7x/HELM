"""graphfx: slim node/edge graph-feature computation.

Backends: plain NetworkX (CPU, always available; transparently GPU-accelerated
when nx-cugraph is installed and enabled), and a subprocess boundary to the
upstream `graph-measures` library for motif3/motif4 (see `graphfx.motif_client`).
"""

from graphfx.api import compute_node_features, compute_edge_features

__all__ = ["compute_node_features", "compute_edge_features"]
