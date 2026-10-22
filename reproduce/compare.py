"""Compare per-graph metrics of a run with the stored reference.

    python reproduce/compare.py RESULTS.csv EXPECTED.csv [--tol 1e-9]

Prints the mean of each metric in both files and the largest per-graph difference. Exits with
status 1 if any metric differs by more than --tol.
"""
import argparse
import sys

import pandas as pd

METRICS = ["directed_recall", "undirected_recall", "directed_precision", "root_accuracy", "root_distance"]

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("expected")
p.add_argument("--tol", type=float, default=1e-9, help="largest allowed per-graph difference [1e-9]")
a = p.parse_args()

new, ref = pd.read_csv(a.results), pd.read_csv(a.expected)
m = ref.merge(new, on="graph_id", suffixes=("_ref", "_new"))
if not (len(m) == len(ref) == len(new)):
    sys.exit(f"graph sets differ: expected {len(ref)}, got {len(new)}, in common {len(m)}")
worst = 0.0
print(f"{'metric':22s} {'expected':>9s} {'this run':>9s} {'max diff':>9s}")
for c in METRICS:
    d = float((m[c + "_ref"] - m[c + "_new"]).abs().max())
    worst = max(worst, d)
    print(f"{c:22s} {m[c + '_ref'].mean():9.4f} {m[c + '_new'].mean():9.4f} {d:9.3g}")
ok = worst <= a.tol
print("RESULT:", "match" if ok else f"differs (largest per-graph difference {worst:.3g} > tolerance {a.tol:g})")
sys.exit(0 if ok else 1)
