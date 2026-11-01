# Notebooks

Store focused exploration and analysis notebooks here using descriptive
`snake_case.ipynb` names. Move reusable logic into `src/`.

Start JupyterLab from the repository root with `make notebooks`. Select the project
Python 3.14 environment. Before execution, inspect dependencies, data, compute, and
side effects. Follow the execution and authorization rules in `AGENTS.md`.

For lightweight CPU notebooks, restart the kernel and run all cells in order.
Retain useful outputs only when they are reproducible and contain no secrets,
private data, or excessive logs. Document skipped execution and potentially stale
outputs when a notebook cannot be rerun.
