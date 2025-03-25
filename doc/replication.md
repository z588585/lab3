
## Run Configuration Performance Tuning

> **Note:** All experiment results are already included in the repository. You don't need to rerun the tuning algorithms to view the visualizations.

To rerun the tuning process manually:

1. Open PowerShell or terminal.
2. Run:

```bash
py -3.10 main.py
```

- The program will complete tuning (excluding Bayesian) in about **1 hour**.
- **Bayesian Optimization** takes several hours — uncomment the corresponding line in `main.py` if you wish to run it.
- The default seed is `42`, which reproduces all results from the included research paper.

---

## Run Visualization

1. Open the project in VSCode.
2. Open and run `visualization.ipynb`.
3. View the results:
   - Visualizations will be saved in the `results_img/` folder.
   - Or directly view plots inside the notebook after execution.

---

## 100 comparisons between bestconfig_fast_search and base

This will provide a comparison chart of all systems

```bash
py -3.10 main.py -t
```
---