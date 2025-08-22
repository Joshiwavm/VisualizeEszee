# VisualizeEszee

Lightweight utilities to build, sample and visualize SZ cluster models against interferometric and single-dish data.

## Quick start

1. Install the package in editable mode (optional but useful during development):

```bash
pip install -e .
```

2. Minimal example (from `Notebooks/plot.ipynb`):

```python
from VisualizeEszee import Manager
pm = Manager(target='CL_J0459-4947')

# Add data (example adapted from notebook)
pm.add_data(name='Band3_12m', obstype='interferometer', band='band3', array='com12m', fields=['0'], spws=['5','7','9','11'], binvis='../output/com12m/output_band3_com12m.im.field-fid.spw-sid')

# Prepare model parameters
from VisualizeEszee.model import get_models
params = get_models('a10_up', ra=74.9229603, dec=-49.7818421, redshift=1.71, mass=2.5e14)

# Add a model
pm.add_model(name='0459_1', source_type='parameters', model_type=params['model']['type'], parameters=params)

# Match model to all data and inspect
pm.match_model()
pm.plot_map(model_name='0459_1', data_name='Band3_12m', types=['filtered','data','residual'])

# Create a JvM-style deconvolved product and plot
pm.JvM_clean(model_name='0459_1', data_name='Band3_12m')
pm.plot_map(model_name='0459_1', data_name='Band3_12m', types='deconvolved')
```

3. More examples

- See `Notebooks/plot.ipynb` for a runnable notebook demonstrating many common workflows (data registration, plotting radial distributions, adding multiple models).
- The markdown docs from the project have been moved into `VisualizeEszee/` for convenience:
  - `CLUSTER_PARAMETERS_GUIDE.md`
  - `MODEL_RESTRUCTURE_SUMMARY.md`

## Notes
- The notebook is a good starting point for practical usage; adapt paths and data names to your local setup.
- If any functions require external heavy dependencies (NUFFT backends), ensure those are installed in your environment.

## Where to go next
- Run the notebook `Notebooks/plot.ipynb` interactively to reproduce the examples and adapt them for your target cluster.
- Open `VisualizeEszee/CLUSTER_PARAMETERS_GUIDE.md` for details on `get_models()`.
