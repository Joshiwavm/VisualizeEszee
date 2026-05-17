# VisualizeEszee

Post-processing and visualization for `eszee` SZ cluster fits. Takes `eszee` outputs (binned UV data, FITS images, dynesty posteriors) and provides a `Manager` class for the full inspection loop.

**Requires `eszee` upstream** — VisualizeEszee does not run the sampler.

## Workflow

```
add_data → add_model → match_model → [apply_point_source_correction] → [JvM_clean] → plot_*
```

## Dependencies

- Python ≥ 3.10
- `jax` + `jax_finufft` — install first ([JAX install guide](https://jax.readthedocs.io/en/latest/installation.html))
- `numpy`, `matplotlib`, `astropy`, `scipy`, `corner`, `reproject`, `pyyaml` — installed via `pip`

## Install

```bash
pip install "jax[cpu]"
pip install jax_finufft

git clone https://github.com/<your-org>/VisualizeEszee.git
cd VisualizeEszee
pip install -e .
```

## Input files (per field / SPW, from `eszee`)

```
output_<band>_<array>.im.field-<fid>.spw-<sid>.data.npz    # binned visibilities
output_<band>_<array>.im.field-<fid>.spw-<sid>.image.fits   # CLEAN image
output_<band>_<array>.im.field-<fid>.spw-<sid>.pbeam.fits   # primary beam
```

## Quick start

```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

pm = Manager(target='My_Cluster')

pm.add_data(
    name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
    fields=['0', '1'], spws=['0', '1', '2', '3'],
    binvis='/path/to/eszee/output/output_band3_com12m.im.field-fid.spw-sid'
)

params = get_models('a10_up', custom_params={
    'ra': 83.822, 'dec': -5.372, 'redshift': 0.55, 'mass': 6.0e14,
})
pm.add_model(name='A10', source_type='parameters',
             model_type=params['model']['type'], parameters=params)

# Or load posterior median from an eszee pickle — registered as 'fit_q0.5'
pm.add_model(name='fit', source_type='pickle',
             filename='/path/to/dumps/my_run_pickle', quantiles=[0.5])

pm.match_model()

pm.plot_map(model_name='fit_q0.5', data_name='Band3_12m',
            types=['filtered', 'data', 'residual'],
            fov=120, center=(83.822, -5.372))
```

See **[TUTORIAL.md](TUTORIAL.md)** for the full workflow.
