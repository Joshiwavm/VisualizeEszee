
# VisualizeEszee

Lightweight utilities for constructing and evaluating SZ cluster models (pressure / ancillary components) against interferometric and single-dish data.

## Install
```bash
pip install -e .
```

## Minimal Example
```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

pm = Manager(target='CL_J0459-4947')

pm.add_data(name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
            fields=['0'], spws=['0','1','2','3'],
            binvis='output/com12m/output_band3_com12m.im.field-fid.spw-sid')

params = get_models('a10_up', custom_params={
    'ra': 74.92296, 'dec': -49.78184, 'redshift': 1.71, 'mass': 2.5e14
})
pm.add_model(name='pA10', source_type='parameters',
             model_type=params['model']['type'], parameters=params)

pm.match_model()
pm.plot_map(model_name='pA10', data_name='Band3_12m',
            types=['filtered', 'data', 'residual'], fov=150)
```

## Docs
See `TUTORIAL.md` for the full workflow: data types (interferometer + ACT), quantile/pickle models, point source subtraction, JvM deconvolution, and all plot methods.

## Notes
Requires `jax` + `jax_finufft` for Fourier operations. Adapt paths to your environment.
