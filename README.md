
# VisualizeEszee

Lightweight utilities for constructing and evaluating SZ cluster models (pressure / ancillary components) against interferometric and single‑dish data.

## Install
```bash
pip install -e .
```

## Minimal Example
```python
from VisualizeEszee import Manager
from VisualizeEszee.model import get_models

pm = Manager(target='CL_J0459-4947')
pm.add_data(name='B3_12m', obstype='interferometer', band='band3', array='com12m',
            fields=['0'], spws=['5','7','9','11'],
            binvis='output/com12m/output_band3_com12m.im.field-fid.spw-sid')
params = get_models('a10_up', ra=74.92296, dec=-49.78184, redshift=1.71, mass=2.5e14)
pm.add_model(name='pA10', source_type='parameters', model_type=params['model']['type'], parameters=params)
pm.match_model()
pm.plot_map(model_name='pA10', data_name='B3_12m', types=['data','model','residual'])
```

## Docs
See `TUTORIAL.md` for full workflow (parameter styles, quantiles, deconvolution, extensions). Local notebooks, if any, are examples only; the tutorial is canonical.

## Notes
Adapt paths to your environment. Optional acceleration libraries can be added as needed.
