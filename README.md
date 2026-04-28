# Kodik Lab

Kodik Lab is an interactive Streamlit prototype for exploring weighted graphs through topology metrics, spectral descriptors, attack simulations, null models and approximate robustness diagnostics.

The app is intended for exploratory graph analysis: load an edge list, filter by confidence and weight, inspect summary metrics, run node/edge attacks, compare simple null models and compute sampled Ollivier-Ricci curvature when needed.

## Features

- Weighted undirected graph loading from CSV/Excel.
- Edge filtering by `confidence` and `weight`.
- Basic topology metrics: node/edge counts, density, connected components, LCC size, clustering and approximate diameter.
- Spectral descriptors such as weighted adjacency spectral radius and LCC algebraic connectivity.
- Entropy-style descriptors over degree, weight, confidence and related distributions.
- Node and edge attack simulations with LCC and metric history.
- Phase transition heuristic based on sharp LCC jump dynamics.
- Simple null models: Erdos-Renyi, configuration model and rewiring.
- Sampled Ollivier-Ricci curvature summaries for interactive use.
- Toy/proxy energy flow visualizations.

## Input Format

The normal edge-list format is:

```csv
src,dst,weight,confidence
1,2,1.0,100
2,3,0.8,90
```

`confidence` is expected in 0..100 scale. If it is missing, the default is `100.0`.

`weight` is expected to be finite and positive after preprocessing/filtering. Weight policy is applied in preprocessing/filtering, before graph construction. Graph construction only validates final weights.

The fixed connectome-like importer expects source and target in columns 1 and 2, confidence in column 9 and weight in column 10.

Interactive uploads are capped at 20 MB, 300,000 rows and 100 columns. Imported workspace JSON files are capped as well, including the decoded tables inside them.

## Examples

Sample files are in `examples/`:

- `examples/tiny_graph.csv`
- `examples/weighted_graph.csv`
- `examples/connectome_like_edges.csv`
- `examples/sample_graph_edges.csv`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Limitations

- Most heavy metrics are approximate or sample-based.
- Ricci curvature is computed on sampled edges in interactive mode.
- Phase transition detection is heuristic and based on LCC jump dynamics.
- Energy flow is a toy/proxy simulation, not a physical simulator.
- Attack strategies may be static or adaptive depending on mode.
- The app is intended for exploratory analysis, not final scientific inference without validation.

## Development Notes

- `src/preprocess.py` owns weight policy conversion.
- `src/graph_build.py` only validates final edge weights and confidence values.
- Tests live in `tests/`.
- Ruff and pytest settings are in `pyproject.toml`.
