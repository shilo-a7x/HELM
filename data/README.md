# Data

Sixty graphs in three families, 20 each: Wikipedia category graphs, microbiome co-occurrence
networks and MemeTracker cascades. Every graph comes with its ground-truth hierarchy.
`manifests/<family>_train.json` and `manifests/<family>_test.json` hold the split into 10 + 10.

```
data/<family>/<graph_id>/G.edgelist     the graph
data/<family>/<graph_id>/T.edgelist     its ground-truth tree (parent -> child edges, directed)
```

| Family | Graphs | G | Vertices | Edges | T |
|---|---|---|---|---|---|
| `wiki` | 20 | directed article link graph induced by a category (graph ID = category name) | 1,987 - 10,135 | 18,453 - 105,827 | the category tree, built by breadth-first search from the root category down to sub-categories and articles |
| `microbiome` | 20 | undirected microbial co-occurrence network of one study | 110 | 853 - 854 | taxonomy-informed hierarchy of the species |
| `memetracker` | 20 | undirected hyperlink graph between news and blog sites within one meme cascade | 681 - 2,763 | 19,798 - 89,010 | the propagation tree of the meme on that cascade, one stochastic realisation of the spreading process |

Vertex IDs are integers. The original Wikipedia titles and the original taxonomy labels are not
included.

## File format

An edge list: one `source target` pair per line.

```
# undirected
# vertices 0 1 2 3 4
0 1
0 2
...
```

- The first line says `# directed` or `# undirected`. Without it a reader must be told which.
- The optional `# vertices ...` line lists the vertices in their original order and names isolated
  vertices. The files here include it so that graphs reload exactly as they were stored (vertex and
  edge order can affect tie-breaking).
- Ground-truth trees are always directed, parent first.

`python -m helm.cli convert` builds these files from plain `source,target` CSVs.

## Sources and terms

The data keeps the terms of its sources, which differ from the GPL that covers the code.

- **Wikipedia**: link and category structure from Wikipedia, whose text is under CC BY-SA 4.0. The
  bundled files contain no article text. Please credit Wikipedia if you redistribute them.
- **MemeTracker**: derived from the MemeTracker data of J. Leskovec, L. Backstrom and
  J. Kleinberg, "Meme-tracking and the Dynamics of the News Cycle", ACM SIGKDD 2009
  (<https://snap.stanford.edu/data/memetracker9.html>). Please cite that paper.
- **Microbiome**: co-occurrence networks of the studies analysed in O. Shtossel, A. Goldberg,
  S. Turjeman, E. Shtossel, S. Finkelstein, N. Shnerb, O. Koren and Y. Louzoun, "Shifting from
  Abundance and Centrality-Fluctuating Microbes are Key Drivers of Long Lasting Microbiome
  Manipulation" (2025).
