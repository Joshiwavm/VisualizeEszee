# VisualizeEszee Tutorial

## 1. Install

```bash
pip install "jax[cpu]"    # or "jax[cuda12]" for GPU
pip install jax_finufft
cd /path/to/VisualizeEszee
pip install -e .
```

## 2. Available models

```python
from visualizeeszee.model import list_available_distributions, get_models

print(list_available_distributions())
# m14_up, m14_cc, m14_nc   – Melin+14
# a10_up, a10_cc, a10_md   – Arnaud+10
# gnfw                      – free gNFW (p_norm, r_s, alpha, beta, gamma)
# g17_ex, g17_st            – Ghirardini+17
# l15_00, l15_80, l15_85   – Le Brun+15
```

## 3. Init and load data

```python
from visualizeeszee import Manager

pm = Manager(target='My_Cluster')

# Interferometer
pm.add_data(
    name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
    fields=['0', '1', '2'],       # field IDs as strings
    spws=['0', '1', '2', '3'],    # same SPWs for all fields; or pass nested list per field
    binvis='/path/to/output/output_band3_com12m.im.field-fid.spw-sid'
    # 'fid' and 'sid' are replaced automatically
)

# ACT (sensitivity comparison only — not passed through NUFFT)
pm.add_data(name='ACT_DR6', obstype='ACT', fdir='/path/to/act_cutouts/')
```

## 4. Build model parameters

```python
# Standard profile
params = get_models('a10_up', custom_params={
    'ra': 83.822, 'dec': -5.372, 'redshift': 0.55, 'mass': 6.0e14,
})

# Free gNFW
params = get_models('gnfw', custom_params={
    'ra': 83.822, 'dec': -5.372,
    'p_norm': 0.24, 'r_s': 0.030, 'alpha': 0.80, 'beta': 6.05, 'gamma': 0.13,
    'redshift': 0.55,
})
```

## 5. Register models

```python
# From parameters
pm.add_model(name='A10_up', source_type='parameters',
             model_type=params['model']['type'], parameters=params)

# From eszee dynesty pickle — registered as 'fit_q0.5'
pm.add_model(name='fit', source_type='pickle', filename=fname,
             quantiles=[0.5])           # [0.16, 0.5, 0.84] for ±1σ envelope

# Override a parameter before matching
pm.models['fit_q0.5']['parameters']['model']['gamma'] = 0.0
```

## 6. Match model to data

```python
pm.match_model()                                         # all models × all datasets
pm.match_model(model_name='A10_up', data_name='Band3_12m')  # specific pair
pm.match_model(save_output='/path/to/output/maps/')
```

> **Warning:** `match_model` resets `resid_vis`. Always re-run it before re-applying
> `apply_point_source_correction` — otherwise you double-subtract.

## 7. Inspect state

```python
pm.summary()
pm.dump_structure(model_name='fit_q0.5', data_name='Band3_12m', depth=3)
print(list(pm.models.keys()))
```

## 8. Sensitivity and UV distributions

```python
pm.plot_weight_distributions(save_plots=True)

# Data only
pm.plot_radial_distributions(nbins=[20, 80, 80],
                             custom_phase_center=(ra, dec), save_plots=True)

# With model overlay (after match_model)
pm.plot_radial_distributions(model_name='fit_q0.5', nbins=[20, 80, 80],
                             custom_phase_center=(ra, dec), save_plots=True)
```

## 9. Point source workflow

Call order: `match_model` → `apply_point_source_correction` → `JvM_clean`

```python
ps_list = pm.get_point_sources_from_pickle(fname)

# Inspect spectra and aperture fluxes (optional)
pm.plot_point_source_spectra(ps_list, data_names=['Band1_12m', 'Band3_12m', 'Band3_07m'],
                             model_name='fit_q0.5', plot_maps=True, aperture_arcsec=10.0)

# Subtract from uvdata (once) and all matched model resid_vis — same order as add_data
for dname in ['Band3_07m', 'Band3_12m', 'Band1_12m']:
    pm.apply_point_source_correction(dname, ps_list)
```

> **Warning:** `apply_point_source_correction` modifies `resid_vis` in-place.
> Calling it twice double-subtracts. To redo: `match_model()` first, then re-apply.

## 10. Map panels

| Type | Shows | When |
|------|-------|------|
| `'input'` | Compton-y model, PB-corrected | check model morphology |
| `'filtered'` | Dirty model visibilities | compare to data |
| `'residual'` | Dirty residual per field+SPW | per-field quality check |
| `'data'` | CASA CLEAN image | reference |
| `'joint_residual'` | MFS dirty residual, all fields+SPWs | best S/N residual |
| `'deconvolved'` | JvM-cleaned image (requires `JvM_clean` first) | publication maps |

```python
pm.plot_map(model_name='fit_q0.5', data_name='Band3_12m',
            types=['filtered', 'data', 'residual'],
            field='0', fov=120, center=(ra, dec), save_plots=True)
```

## 11. JvM deconvolution

```python
taper = 6 / 2.355    # smooth to ~6" FWHM; None for native resolution

# Single array
pm.JvM_clean(model_name='fit_q0.5', data_name=['Band3_12m'],
             save_output='/path/to/output/maps/', taper=taper)

# Joint (combines all residual vis before deconvolving — best for extended emission)
pm.JvM_clean(model_name='fit_q0.5', data_name=['Band3_07m', 'Band3_12m'],
             save_output='/path/to/output/maps/', taper=taper)

pm.plot_map(model_name='fit_q0.5', data_name=['Band3_07m', 'Band3_12m'],
            types='deconvolved', fov=120, center=(ra, dec), save_plots=True)
pm.plot_map(model_name='fit_q0.5', data_name=['Band3_07m', 'Band3_12m'],
            types='joint_residual', fov=120, center=(ra, dec), save_plots=True)
```

Multi-dataset output uses the header/pixel scale of the smallest-beam dataset.

## 12. Pressure profiles and corner plots

```python
pm.plot_pressure_profile(model_names=['A10_up', 'fit_q0.5'], save_plots=True)

pm.plot_corner(fname, unit_scale={2: 3600},   # e.g. r_s: deg → arcsec
               n_sigma_range=4.0, save_output='/path/to/plots/corner/')
```

## 13. Full example

```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

ra, dec = 83.822, -5.372
fname   = '/path/to/dumps/my_run_pickle'

pm = Manager(target='My_Cluster')

pm.add_data(name='Band3_07m', obstype='interferometer', band='band3', array='com07m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='/path/to/output/com07m/output_band3_com07m.im.field-fid.spw-sid')
pm.add_data(name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='/path/to/output/com12m/output_band3_com12m.im.field-fid.spw-sid')
pm.add_data(name='Band1_12m', obstype='interferometer', band='band1', array='com12m',
            fields=['0'], spws=['0','1','2','3'],
            binvis='/path/to/output/com12m/output_band1_com12m.im.field-fid.spw-sid')

pm.plot_weight_distributions(save_plots=True)
pm.plot_radial_distributions(nbins=[20, 80, 80], custom_phase_center=(ra, dec))

pm.add_model(name='fit', source_type='pickle', filename=fname, quantiles=[0.5])

pm.plot_corner(fname)
pm.plot_pressure_profile(return_fig=True)

pm.match_model()
pm.summary()

ps_list = pm.get_point_sources_from_pickle(fname)
pm.plot_point_source_spectra(ps_list, data_names=['Band1_12m','Band3_12m','Band3_07m'],
                             model_name='fit_q0.5', plot_maps=True)
for dname in ['Band3_07m', 'Band3_12m', 'Band1_12m']:   # same order as add_data
    pm.apply_point_source_correction(dname, ps_list)

for dname in ['Band3_07m', 'Band3_12m', 'Band1_12m']:
    for fid in ['0', '1']:
        pm.plot_map(model_name='fit_q0.5', data_name=dname,
                    types=['filtered','data','residual'],
                    field=fid, fov=120, save_plots=True)

pm.plot_radial_distributions(model_name='fit_q0.5', nbins=[20, 80, 80],
                             custom_phase_center=(ra, dec), save_plots=True)

taper = 6 / 2.355
for dn in [['Band1_12m'], ['Band3_12m'], ['Band3_07m'], ['Band3_07m','Band3_12m']]:
    pm.JvM_clean(model_name='fit_q0.5', data_name=dn,
                 save_output='/path/to/output/maps/', taper=taper)
    pm.plot_map(model_name='fit_q0.5', data_name=dn, types='deconvolved',
                fov=120, center=(ra, dec), save_plots=True)
    pm.plot_map(model_name='fit_q0.5', data_name=dn, types='joint_residual',
                fov=120, center=(ra, dec), save_plots=True)
```

## 14. Pitfalls

| Issue | Fix |
|-------|-----|
| `KeyError` on `'fit_q0.5_c0'` | Single-component model — use `'fit_q0.5'`; `_c{i}` only for multi-component |
| Blank model map | Check RA/Dec in decimal degrees, mass in solar masses |
| Double PS subtraction | Re-run `match_model()` to reset `resid_vis`, then re-apply once |
| JvM beam wrong size | Expected — smallest-beam dataset wins; verify with `pm.summary()` |
| Missing FITS on `add_data` | Check pattern: `binvis.replace('fid','0').replace('sid','0')` |

## 15. Adding new models

1. Add entry to `visualizeeszee/model/brightness_models.yml` — parameter order must match `eszee` MCMC order.
2. If new profile function: add to `model/models.py`.
3. Add builder in `model/parameter_utils.py` returning `{'model': {...}, 'spectrum': {...}}`.
4. Verify: `get_models('my_new_key', custom_params={...})` and `list_available_distributions()`.
