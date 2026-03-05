# VisualizeEszee Tutorial

End-to-end walkthrough for loading data, constructing SZ models, generating maps, cleaning, and producing diagnostic plots. The canonical reference — local notebooks are illustrative only.

---
## 1. Installation & Environment

```bash
pip install -e .
```

Optional performance/extras: `jax`, `jax_finufft` (required by `FourierManager`), `corner` (for `plot_corner` with `backend='corner'`).

---
## 2. Core Concepts

| Concept | Method / Object | Purpose |
|---------|-----------------|---------|
| Manager | `visualizeeszee.Manager` | Orchestrates all operations |
| Data registration | `add_data()` | Load interferometer or ACT datasets |
| Available models | `list_available_distributions()` | List YAML model keys |
| Model registration | `add_model()` | Register parameter or pickle-derived models |
| Matching | `match_model()` | Build model maps and Fourier products per dataset |
| Deconvolution | `JvM_clean()` | JvM-style deconvolved map |
| Plotting | `plot_*()` family | Diagnostics |
| Inspection | `summary()`, `dump_structure()` | Inspect loaded state |

**Data flow**: `add_data` → `add_model` → `match_model` → [`apply_point_source_correction`] → [`JvM_clean`] → `plot_*`

---
## 3. Building Model Parameters

```python
from visualizeeszee.model import get_models, list_available_distributions

# Show all available YAML model keys
print(list_available_distributions())
# ['m14_up', 'm14_cc', 'm14_nc', 'a10_up', 'a10_cc', 'a10_md',
#  'gnfw', 'g17_ex', 'g17_st', 'l15_00', 'l15_80', 'l15_85']
```

### Using `custom_params` (recommended)
```python
cluster_params = {
    'ra': 74.9229603, 'dec': -49.7818421,
    'redshift': 1.285, 'mass': 2.166e14,
    'alpha': 1.551, 'beta': 5.046, 'gamma': 0.0029,
}
params = get_models('a10_up', custom_params=cluster_params)
```

### Mixed style (keyword args + `custom_params`)
```python
params = get_models('a10_up', ra=74.92296, dec=-49.78184,
                    custom_params={'redshift': 1.71, 'mass': 2.5e14})
```

### gNFW model (direct parameters)
```python
cluster_params_gnfw = {
    'ra': 74.9229603, 'dec': -49.7818421,
    'p_norm': 2.35e-01, 'r_s': 2.98e-02,
    'alpha': 0.80, 'beta': 6.05, 'gamma': 0.126,
    'redshift': 1.71,
}
params = get_models('gnfw', custom_params=cluster_params_gnfw)
```

The returned dict has at minimum:
```python
params['model']['type']   # e.g. 'A10Pressure', 'gnfwPressure'
params['model']['ra']     # deg
params['model']['mass']   # solar masses (linear)
```

---
## 4. Registering Data

### Interferometer (ALMA 12m / 7m)
```python
from visualizeeszee import Manager

pm = Manager(target='CL_J0459-4947')

pm.add_data(
    name='Band3_12m',
    obstype='interferometer',
    band='band3',
    array='com12m',
    fields=['0', '1', '2', '3', '4', '5', '6'],   # field IDs as strings
    spws=['0', '1', '2', '3'],                      # flat list — applied to all fields
    binvis='output/lineremoved/com12m/output_band3_com12m.im.field-fid.spw-sid'
    # 'fid' and 'sid' in the pattern are replaced with actual field/spw IDs
)

pm.add_data(
    name='Band3_07m',
    obstype='interferometer',
    band='band3',
    array='com07m',
    fields=['0', '1', '2', '3', '4', '5', '6'],
    spws=['0', '1', '2', '3'],
    binvis='output/lineremoved/com07m/output_band3_com07m.im.field-fid.spw-sid'
)

pm.add_data(
    name='Band1_12m',
    obstype='interferometer',
    band='band1',
    array='com12m',
    fields=['0'],
    spws=['0', '1', '2', '3'],
    binvis='output/lineremoved/com12m/output_band1_com12m.im.field-fid.spw-sid'
)
```

The `binvis` pattern expands to `*.data.npz` files. Matching `*.image.fits` and `*.pbeam.fits` must exist alongside them.

`spws` can also be a nested list `[[spws_for_field0], [spws_for_field1], ...]` for per-field SPW control.

### ACT single-dish
```python
pm.add_data(
    name='ACT_DR6',
    obstype='ACT',
    fdir='/path/to/cutout_act/',
)
```

---
## 5. Registering Models

### Direct parameter model
```python
params = get_models('a10_up', custom_params=cluster_params)

pm.add_model(
    name='A10_up',
    source_type='parameters',
    model_type=params['model']['type'],   # optional; inferred from parameters if omitted
    parameters=params,
)
```

### Pickle model (quantile extraction)
```python
fname = 'dumps/CL_J0459-4946_002_wsz_a10_up_sph_1000_0.10_static_multi_rwalk_pickle'

pm.add_model(
    name='0459_loaded',
    source_type='pickle',
    filename=fname,
    quantiles=[0.5],          # one registered model per quantile
    marginalized=False,
)
# Single-component → key: '0459_loaded_q0.5'
# Multi-component  → keys: '0459_loaded_q0.5_c0', '0459_loaded_q0.5_c1', ...
```

Multiple quantiles at once:
```python
pm.add_model(name='0459_post', source_type='pickle', filename=fname,
             quantiles=[0.16, 0.5, 0.84])
# Keys: '0459_post_q0.16', '0459_post_q0.5', '0459_post_q0.84'
```

### Post-registration parameter tweaks
Individual parameters can be overridden before calling `match_model`:
```python
pm.models['0459_loaded_q0.5']['parameters']['model']['gamma'] = 0.01
pm.models['0459_loaded_q0.5']['parameters']['model']['mass']  = 3.39e14
```

---
## 6. Matching

```python
pm.match_model()
# Matches all registered models against all interferometer datasets.
# Restrict with: match_model(model_name='A10_up', data_name='Band3_12m')
```

Optional: save dirty maps and Compton-y FITS during matching:
```python
pm.match_model(save_output='output/VisualizeEszee/')
```

> **Note**: `match_model()` resets `resid_vis` for each model/data pair.
> Always re-run `match_model()` before re-applying `apply_point_source_correction`.

---
## 7. Inspection

```python
pm.summary()
# Prints: data list, all model parameters, point sources, matched pairs

pm.dump_structure(model_name='0459_loaded_q0.5', data_name='Band3_12m', depth=3)
# ASCII tree of matched_models[model][data] — useful for debugging internal keys
```

---
## 8. Point Source Workflow

Point sources fitted by eszee are stored in the pickle. Extract them and subtract from residual visibilities:

```python
# 1. Extract list of point-source dicts from pickle
ps_list = pm.get_point_sources_from_pickle(fname)
# Each dict: {ra, dec, amplitude [Jy], spec_type, spec_index, ref_freq [Hz]}

# 2. (Optional) inspect spectra and image-plane maps
pm.plot_point_source_spectra(
    ps_list,
    data_names=['Band1_12m', 'Band3_12m', 'Band3_07m'],
    model_name='0459_loaded_q0.5',
    plot_maps=True,
    aperture_arcsec=10.0,
)

# 3. Subtract from residual visibilities (in-place, per dataset)
pm.apply_point_source_correction('0459_loaded_q0.5', 'Band1_12m', ps_list)
pm.apply_point_source_correction('0459_loaded_q0.5', 'Band3_12m', ps_list)
pm.apply_point_source_correction('0459_loaded_q0.5', 'Band3_07m', ps_list)
```

> **Warning**: `apply_point_source_correction` subtracts in-place. Calling it twice
> double-subtracts. To redo: call `match_model()` first to reset `resid_vis`.

> **Note on amplitudes**: eszee applies the primary beam during forward modelling,
> so fitted amplitudes are intrinsic fluxes. `apply_point_source_correction` does
> *not* re-apply the PB (negligible for sources near phase centre where PB ≈ 1).

The `pm.point_sources` list (populated by `get_point_sources_from_pickle`) is shown by `summary()`.

---
## 9. Sensitivity & UV Distributions

### Fourier sensitivity / weight distributions
```python
pm.plot_weight_distributions(save_plots=True)
# Inverse-noise (σ in µJy) vs uv-distance [kλ] for all loaded datasets.
# ACT sensitivity ranges shown when ACT data is loaded.
```

### Radial UV distributions (data only)
```python
pm.plot_radial_distributions(
    nbins=[14, 82, 82],               # one entry per loaded interferometer dataset (add_data order)
    custom_phase_center=(74.9229603, -49.7818421),
    save_plots=True,
)
```

### With model visibility overlay
```python
pm.plot_radial_distributions(
    model_name='0459_loaded_q0.5',
    nbins=[16, 82, 82],
    custom_phase_center=(74.9229603, -49.7818421),
    save_plots=True,
)
```

---
## 10. Map Plots

### Available types

| Type | Description |
|------|-------------|
| `'input'` | Compton-y model map (PB-corrected) |
| `'filtered'` | Dirty image from model visibilities (mJy/beam) |
| `'residual'` | Dirty residual for a single field+SPW (mJy/beam) |
| `'data'` | CASA CLEAN image plane (mJy/beam) |
| `'joint_residual'` | MFS dirty residual, all fields+SPWs combined (mJy/beam) |
| `'deconvolved'` | JvM-cleaned image — requires prior `JvM_clean` |

```python
pm.plot_map(
    model_name='0459_loaded_q0.5',
    data_name='Band3_12m',
    types=['filtered', 'data', 'residual'],
    field='0',                             # field ID string or int index (default: first)
    fov=150,                               # crop to 150" × 150" square
    center=(74.9229603, -49.7818421),      # (ra_deg, dec_deg) crop centre
    save_plots=True,
)

# All fields:
for fid in ['0', '1']:
    pm.plot_map(model_name='0459_loaded_q0.5', data_name='Band3_12m',
                types=['filtered', 'data', 'residual'], field=fid,
                fov=150, save_plots=True)
```

---
## 11. Deconvolution (JvM Clean)

`JvM_clean` accepts a single dataset name or a list for joint cleaning across datasets.

```python
# Single dataset
pm.JvM_clean(
    model_name='0459_loaded_q0.5',
    data_name=['Band3_12m'],
    save_output='output/VisualizeEszee/',
    taper=6/2.355,           # Gaussian uv-taper: FWHM [arcsec] / 2.355
)
pm.plot_map(model_name='0459_loaded_q0.5', data_name=['Band3_12m'],
            types='deconvolved', fov=150, center=(74.9229603,-49.7818421), save_plots=True)
pm.plot_map(model_name='0459_loaded_q0.5', data_name=['Band3_12m'],
            types='joint_residual', fov=150, center=(74.9229603,-49.7818421), save_plots=True)

# Joint 07m + 12m cleaning
pm.JvM_clean(
    model_name='0459_loaded_q0.5',
    data_name=['Band3_07m', 'Band3_12m'],
    save_output='output/VisualizeEszee/',
    taper=6/2.355,
)
pm.plot_map(model_name='0459_loaded_q0.5', data_name=['Band3_07m', 'Band3_12m'],
            types='deconvolved', fov=150, center=(74.9229603,-49.7818421), save_plots=True)
```

When `data_name` is a list, `JvM_clean` uses the header/pixel scale of the highest-resolution (smallest-beam) dataset for the deconvolved output.

---
## 12. Pressure Profiles

```python
# All registered models
fig, ax = pm.plot_pressure_profile(return_fig=True)

# Specific models
pm.plot_pressure_profile(model_names=['A10_up', '0459_loaded_q0.5'], save_plots=True)
```

Profiles are in physical units (kpc vs keV cm⁻³). `r500` is marked when mass and redshift are available.

---
## 13. Corner Plots

```python
pm.plot_corner(fname)
# Auto-generates parameter labels from the pickle's vary structure + YAML ordering.
```

Options:
```python
pm.plot_corner(
    fname,
    backend='corner',           # 'corner' (pip install corner) or 'dynesty'
    unit_scale={3: 3600},       # multiply column 3 by 3600 (e.g. deg → arcsec)
    n_sigma_range=4.0,
    save_output='plots/corner/',
    return_fig=False,
)
```

---
## 14. Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError` on quantile key | Expected `name_q0.5_c0` but model is single-component | Use `name_q0.5`; `_c{i}` suffix only for multi-component |
| Blank/zero model map | Wrong RA/Dec or amplitude | Verify units: RA/Dec in degrees, mass in solar masses |
| Double PS subtraction | `apply_point_source_correction` called twice | Re-run `match_model()` to reset `resid_vis` |
| `KeyError` in pickle builder | YAML parameter order mismatch | Check `pm.get_param_order_from_yaml(model_type)` |
| `ValueError`: no models | `match_model` before `add_model` | Register at least one model first |
| Wrong beam in deconvolved | `data_name` is list but wrong header used | Expected behaviour: smallest-beam dataset wins |

Diagnostic snippets:
```python
print(list(pm.models.keys()))
print(list(pm.matched_models['0459_loaded_q0.5'].keys()))
pm.dump_structure(model_name='0459_loaded_q0.5', data_name='Band3_12m', depth=3)
pm.summary()
```

---
## 15. Full End-to-End Example

```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

# 0. Check available models
print(list_available_distributions())

# 1. Init
pm = Manager(target='CL_J0459-4947')

# 2. Load data
pm.add_data(name='Band3_07m', obstype='interferometer', band='band3', array='com07m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='output/lineremoved/com07m/output_band3_com07m.im.field-fid.spw-sid')
pm.add_data(name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='output/lineremoved/com12m/output_band3_com12m.im.field-fid.spw-sid')
pm.add_data(name='Band1_12m', obstype='interferometer', band='band1', array='com12m',
            fields=['0'], spws=['0','1','2','3'],
            binvis='output/lineremoved/com12m/output_band1_com12m.im.field-fid.spw-sid')

# 3. Sensitivity check
pm.plot_weight_distributions(save_plots=True)

# 4. UV distributions (data only)
pm.plot_radial_distributions(nbins=[14, 82, 82],
                             custom_phase_center=(74.9229603, -49.7818421))

# 5. Load pickle model
fname = 'dumps/CL_J0459-4946_002_wsz_a10_up_sph_1000_0.10_static_multi_rwalk_pickle'
pm.add_model(name='m02', source_type='pickle', filename=fname, quantiles=[0.5])
# Key: 'm02_q0.5'

# 6. Corner plot
pm.plot_corner(fname)

# 7. Pressure profile
pm.plot_pressure_profile(return_fig=True)

# 8. Match
pm.match_model()

# 9. Point source workflow
ps_list = pm.get_point_sources_from_pickle(fname)
pm.plot_point_source_spectra(ps_list, data_names=['Band1_12m','Band3_12m','Band3_07m'],
                             model_name='m02_q0.5', plot_maps=True)
pm.apply_point_source_correction('m02_q0.5', 'Band1_12m', ps_list)
pm.apply_point_source_correction('m02_q0.5', 'Band3_12m', ps_list)
pm.apply_point_source_correction('m02_q0.5', 'Band3_07m', ps_list)

# 10. Dirty maps
for fid in ['0', '1']:
    pm.plot_map(model_name='m02_q0.5', data_name='Band3_12m',
                types=['filtered','data','residual'], field=fid, fov=150, save_plots=True)

# 11. UV distributions with model overlay
pm.plot_radial_distributions(model_name='m02_q0.5', nbins=[16,82,82],
                             custom_phase_center=(74.9229603,-49.7818421), save_plots=True)

# 12. JvM clean (single and joint)
for dn in [['Band1_12m'], ['Band3_12m'], ['Band3_07m'], ['Band3_07m','Band3_12m']]:
    pm.JvM_clean(model_name='m02_q0.5', data_name=dn,
                 save_output='output/VisualizeEszee/', taper=6/2.355)
    pm.plot_map(model_name='m02_q0.5', data_name=dn, types='deconvolved',
                fov=150, center=(74.9229603,-49.7818421), save_plots=True)
    pm.plot_map(model_name='m02_q0.5', data_name=dn, types='joint_residual',
                fov=150, center=(74.9229603,-49.7818421), save_plots=True)

# 13. Summary
pm.summary()
```

---
## 16. Extending With New Models

1. Add the model key and parameters to `visualizeeszee/model/brightness_models.yml` (parameter order must match the MCMC sampling order).
2. Implement a profile function in `model/models.py` if the physics differs.
3. Add a builder in `model/parameter_utils.py` returning `{'model': {...}, 'spectrum': {...}}`.
4. Verify: `get_models('new_key', custom_params={...})` and `list_available_distributions()`.

---
## 17. Performance Notes

- `jax` + `jax_finufft` are required for `FourierManager.map_to_vis`.
- Coarse radial grids (`rs`) dominate runtime — start sparse when iterating on model forms.
- `JvM_clean` on a list of datasets resamples all maps to the highest-resolution header; ensure consistent pixel scales across datasets.

---
## 18. Glossary

| Term | Meaning |
|------|---------|
| Hyperparameter | Fixed profile shape parameter defined in YAML |
| Source input | Cluster-specific quantity (ra, dec, redshift, mass) |
| Quantile model | Parameter set at a posterior quantile (e.g. median q=0.5) |
| Marginalized model | Map integrated over the full posterior |
| JvM clean | Deconvolution combining model-guided clean components with noise-scaled residuals |
| `resid_vis` | Per-field/SPW residual visibilities = data − model; reset by `match_model()` |
