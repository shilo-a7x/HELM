# Third-party notices

HELM is licensed under the GNU General Public License, version 3 or later (see `LICENSE`).
It uses the software and data listed here under their own terms.

## Code

| Component | License | How it is used |
|---|---|---|
| [graph-measures](https://github.com/louzounlab/graph-measures) (Louzoun lab) | GPL-3.0 | motif3 / motif4 features. Cloned and compiled by `scripts/setup.sh` into `graph-measures/`; it is not part of this repository. HELM runs it in a separate process (`graphfx/src/graphfx/motif_client.py`). |
| [SpringRank](https://github.com/LarremoreLab/SpringRank) `springrank` | GPL-3.0-or-later | `hierarchy_energy` feature, imported by graphfx. |
| [leidenalg](https://github.com/vtraag/leidenalg) | GPL-3.0-or-later | optional CPU fallback for Leiden community detection. |
| [python-igraph](https://github.com/igraph/python-igraph) | GPL-2.0-or-later | required by leidenalg. |
| [KACTL](https://github.com/kth-competitive-programming/kactl) `DirectedMST` | CC0 (stated in the file header; authors chilli, Takanori Maehara, Benq, Simon Lindholm) | `helm/arborescence/dmst_tarjan.py` is a Python port of `content/graph/DirectedMST.h` (Tarjan's O(E log V) algorithm, with the reconstruction by Benq). |
| [NetworkX](https://networkx.org) | BSD-3-Clause | graphs, most graph features, reference arborescence solver. |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org), [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org) | BSD-3-Clause | numerics, data handling, scaling, metrics. |
| [XGBoost](https://xgboost.ai) | Apache-2.0 | edge scorer. |
| [Optuna](https://optuna.org), optuna-integration | MIT | hyperparameter search. |
| [Matplotlib](https://matplotlib.org) | PSF-based Matplotlib license | ROC plots. |
| [PyTorch](https://pytorch.org) | BSD-3-Clause and others (see its license file) | optional GNN scorer. |
| [PyTorch Geometric](https://pyg.org) | MIT | optional GNN scorer. |
| [cuGraph, cuDF, nx-cugraph](https://github.com/rapidsai) (RAPIDS) | Apache-2.0 | optional GPU acceleration of the graph features. |

The licenses above were read from the installed packages' metadata (and, for RAPIDS, from the
projects' `LICENSE` files). Each package's own license file governs; the exact versions used are
pinned in `requirements.lock` and `graphfx/uv.lock`.

GPL-2.0-or-later (python-igraph) and Apache-2.0 (XGBoost, RAPIDS) are compatible with distributing
HELM under GPL-3.0-or-later.

## Data

The bundled graphs are derived from the sources below; see `data/README.md`. The data keeps the
terms of its sources, which are not the GPL.

- **Wikipedia**: article and category titles and link structure. Wikipedia text is available under
  CC BY-SA 4.0 and the GFDL; the bundled files contain no article text, only vertex IDs and links.
- **MemeTracker**: J. Leskovec, L. Backstrom and J. Kleinberg, "Meme-tracking and the Dynamics of the
  News Cycle", KDD 2009, via the [SNAP](https://snap.stanford.edu/data/memetracker9.html) data
  page. SNAP states no license for it and asks for citation of the paper.
- **Microbiome**: co-occurrence networks of the studies analysed in O. Shtossel et al., "Shifting
  from Abundance and Centrality-Fluctuating Microbes are Key Drivers of Long Lasting Microbiome
  Manipulation" (2025).
