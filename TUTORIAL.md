# VisualizeEszee Tutorial

This tutorial gives an end‑to‑end walkthrough of using VisualizeEszee to register data, construct SZ pressure (and ancillary) models, generate model maps, perform matching / cleaning, and inspect residuals. It is intentionally verbose so the `README.md` can stay short. Example notebooks you may have locally are purely illustrative and are **not** required; everything needed is described here.

---
## 1. Installation & Environment

Create / activate a Python environment (>=3.9 recommended) and install in editable mode:

```bash
pip install -e .
```

Optional (performance / extras): install any accelerated FFT / NUFFT backends your workflow relies on.

---
## 2. Core Concepts

| Concept | Object / Function | Purpose |
|---------|-------------------|---------|
| Manager / Plot Manager | `VisualizeEszee.Manager` (or `PlotManager` if exposed) | Orchestrates data + model registration and plotting operations |
| Data Registration | `add_data()` | Describe interferometric or single‑dish datasets (fields, spws, beam, etc.) |
| Model Registration | `add_model()` | Register either direct parameter models or posterior (pickle) derived models |
| Parameter Construction | `get_models()` / pickle loader | Build combined model parameter dict (YAML defaults + user inputs or posterior sample/quantile) |
| Matching | `match_model()` | Runs model → visibility / image domain matching & stores products |
| Cleaning | `JvM_clean()` | Produces deconvolved (JvM) map products |
| Plotting | `plot_map()`, others | Visual diagnostics (filtered, data, model, residual, deconvolved) |

Parameter assembly is unified: YAML hyperparameters and user‑provided source inputs (RA, Dec, redshift, mass, geometry, normalization) are merged into a single dictionary. Posterior products (quantiles or marginalized sampling) overwrite the free parameters extracted from the nested sampling file using the YAML ordering.

---
## 3. Building Model Parameters

You can construct parameters either (a) directly from a YAML model key plus source inputs, or (b) by loading a pickled nested‑sampling result and extracting quantiles / marginalizing.

`get_models(model_key, ...)` merges the YAML definition with your inputs; posterior loading performs the merge after substituting sampled values.

### Direct Arguments (recommended)
```python
from VisualizeEszee.model import get_models
params = get_models(
    'a10_up',              # YAML model key
    ra=74.92296,           # deg
    dec=-49.78184,         # deg
    redshift=1.71,
    mass=2.5e14,           # solar masses
    # Optional geometry overrides
    custom_params={'e': 0.15, 'angle': 35.0}
)
```

### Pure `custom_params`
```python
params = get_models('a10_up', custom_params={
    'ra': 74.92296,
    'dec': -49.78184,
    'redshift': 1.71,
    'mass': 2.5e14,
    'parameters': {  # override hyperparameters
        'alpha': 1.10
    }
})
```

### Mixed Style
```python
params = get_models('a10_up', ra=74.92296, dec=-49.78184,
                    custom_params={'redshift': 1.71, 'mass': 2.5e14})
```

The returned structure minimally contains:
```python
params['model']['type']         # e.g. 'A10Pressure'
params['model']['alpha']        # fixed hyperparameter
params['model']['ra'], ['dec']  # your inputs
params['model']['mass']         # linear mass (solar masses)
```

If you plan to combine multiple pressure components, call `get_models` per component and register each separately (naming convention tip: `clusterA_comp0`, `clusterA_comp1`).

---
## 4. Registering Data

Interferometric example:
```python
pm.add_data(
    name='B3_12m',
    obstype='interferometer',
    band='band3',
    array='com12m',
    fields=['0', '1'],            # field IDs as strings
    spws=['5','7','9','11'],  # spectral windows
    binvis='path/to/output_band3_com12m.im.field-fid.spw-sid'  # pattern; fid/sid replaced
)
```

Single-dish (example pattern):
```python
pm.add_data(
    name='B3_SD',
    obstype='single-dish',
    band='band3',
    array='com07m',
    fields=['0'],
    spws=['0'],
    image_root='output/aca/output_band3_aca.im.field-{field}.spw-{spw}'
)
```

Tips:
* Ensure the file pattern expands to actual `*.image.fits` and (if present) matching `*.pbeam.fits` files.
* Use consistent `name` values — they become keys in internal dictionaries.

---
## 5. Registering Models

Direct parameter model:
```python
pm.add_model(
    name='pA10',
    source_type='parameters',
    model_type=params['model']['type'],
    parameters=params
)
```

Posterior (pickle) model (quantile extraction):
```python
pm.add_model(
    name='pA10post',
    source_type='pickle',
    filename='dumps/CL_J0459-4946_05_wsz_m14_up_sph_1000_0.10_static_multi_rwalk_pickle',
    quantiles=[0.16, 0.5, 0.84]
)
```

Marginalized (full posterior sampling) form:
```python
pm.add_model(
    name='pA10marg',
    source_type='pickle',
    filename='dumps/CL_J0459-4946_05_wsz_m14_up_sph_1000_0.10_static_multi_rwalk_pickle',
    marginalized=True
)
```

Internally, quantile models are expanded into per‑component entries (`name_q0.5_c0`, etc.).

---
## 6. Matching & Map Generation

After data + model registration:
```python
pm.match_model()  # runs through all models & datasets
```

Plot maps:
```python
pm.plot_map(model_name='pA10', data_name='B3_12m', types=['data','model','residual'])
```

Deconvolution (JvM):
```python
pm.JvM_clean(model_name='pA10', data_name='B3_12m')
pm.plot_map(model_name='pA10', data_name='B3_12m', types='deconvolved')
```

If you provided multiple datasets (e.g. list of data names to a joint clean), the deconvolved key may be a concatenated identifier; `plot_map` should resolve it automatically if implemented accordingly.

---
## 7. Working With Quantiles & Posterior Products

When using `source_type='pickle'` with quantiles, the system:
1. Reads sampled chains / stored results.
2. Extracts the requested quantile values per free parameter.
3. Builds parameter dictionaries per (quantile, component) using the **YAML parameter order**.
4. Registers each with a disambiguated key.

You can then call, e.g.:
```python
pm.plot_map(model_name='pA10post_q0.5_c0', data_name='B3_12m', types='model')
```

---
## 8. Plotting Overview

VisualizeEszee exposes several plotting utilities beyond basic map inspection:

1. Map Panels (`plot_map`): show any combination of `data`, `model` (input), `filtered` (dirty model), `residual`, `deconvolved`.
2. Pressure Profiles (`plot_pressure_profile`): compare radial pressure profiles from one or more registered pressure models in physical units (kpc vs keV cm⁻³). Automatically marks r500 when mass & redshift are present.
3. Radial UV Distributions (methods in `PlotRadialDistributions` mixin):
    - Weighted real / imaginary visibilities binned by baseline length.
    - Model visibility slices along u or v (log-spaced kλ sampling) for diagnosing filtering.
4. Visibility Slices: extract 1D cuts through the model Fourier plane (supports phase‑centering and axis selection).
5. Sensitivity / Coverage (if implemented as a helper): produce radial baseline density or effective noise vs radius. (Add a convenience wrapper `plot_uv_sensitivity()` if not already present.)

Example pressure profile:
```python
pm.plot_pressure_profile(model_names=['pA10'])
```

Example radial uv distribution (band name matches your data key):
```python
pm.plot_radial_uv(band_name='B3_12m', nbins=25)
```

Example model visibility slice (pseudo‑API – adapt to actual method names if different):
```python
k, real, imag = pm._get_or_compute_model_slice('pA10', 'B3_12m', 'field0', 'spw5',
                                                             npts=60, r_min_k=0.5, r_max_k=30.0,
                                                             axis='v')
```

Deconvolved map panel:
```python
pm.plot_map(model_name='pA10', data_name='B3_12m', types=['deconvolved'])
```

If you add new plotting helpers, keep the naming consistent (`plot_*`).

---
## 9. Common Pitfalls & Validation

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing required key (ValueError) | Not provided RA/Dec/mass/redshift when needed | Supply as direct args or via `custom_params` |
| `KeyError` in builder | YAML parameter ordering mismatch | Ensure YAML reflects actual sampled order or update alias mapping |
| Blank / zero model map | Wrong center or units | Verify RA/Dec degrees & amplitude scale |
| Quantile key not found in plots | Used base name instead of expanded quantile key | List `pm.models.keys()` to inspect |

Diagnostic snippet:
```python
print(list(pm.models.keys()))
print(pm.models['pA10']['parameters']['model'])
```

---
## 10. Extending With New Models

1. Add hyperparameters to the appropriate YAML (ensure order is meaningful).
2. Update / implement a matching builder in `model/parameter_utils.py` (or equivalent) returning derived / normalized fields.
3. Add corresponding profile function if physics differs.
4. Re-run parameter construction to verify new keys appear.

Keep naming consistent; prefer lowerCamelCase or snake_case uniformly within a model definition.

---
## 11. Performance Notes

* Use coarser radial sampling first when iterating on model forms (`rs` grid length can dominate runtime).
* Cache intermediate FITS headers / WCS lookups if running many variants.
* Posterior marginalization paths may require thinning or subsampling large chains for interactive plotting.

---
## 12. Example End‑to‑End Script
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
pm.JvM_clean(model_name='pA10', data_name='B3_12m')
pm.plot_map(model_name='pA10', data_name='B3_12m', types='deconvolved')
```

---
## 13. Notebooks

If you have local notebooks, treat them strictly as *examples*; this tutorial supersedes them for canonical usage and is self‑contained. All operations shown there map to the documented API calls above.

---
## 14. Troubleshooting Checklist

* Verify environment: `python -c "import VisualizeEszee; print('ok')"`
* Confirm data paths: each constructed FITS path exists.
* Dump model keys: `print(pm.models.keys())`
* Inspect parameter dict: `print(pm.models['pA10']['parameters'])`
* Rebuild parameters if YAML changed.

---
## 15. Glossary

| Term | Meaning |
|------|---------|
| Hyperparameter | Fixed profile parameter defined in YAML |
| Source input | Source‑specific quantity (ra, dec, redshift, mass) |
| Quantile model | Parameter set constructed from a posterior quantile |
| Marginalized model | Map built by integrating / sampling across the full posterior |
| JvM clean | Deconvolution producing a deconvolved map using model guidance |

---
## 16. Feedback

Refinements welcome: clarify ambiguous steps, propose new examples, or contribute model definitions. Keeping the YAML clean and ordered is the simplest way to ensure reproducible parameter mapping.
