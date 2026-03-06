# VisualizeEszee Tutorial

This tutorial walks you through the complete VisualizeEszee workflow from loading data to producing publication-quality diagnostic plots. Read it top to bottom the first time; afterwards use it as a reference for individual steps.

---
## 1. Installation

Install JAX for your platform first — it is not on PyPI in a single universal wheel:

```bash
# CPU (recommended for first-time setup)
pip install "jax[cpu]"
pip install jax_finufft

# GPU (CUDA 12)
pip install "jax[cuda12]"
pip install jax_finufft
```

Then install VisualizeEszee in editable mode so changes to the source are picked up immediately:

```bash
cd /path/to/VisualizeEszee
pip install -e .
```

Test the install:
```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions
print(list_available_distributions())
```

---
## 2. How it works

VisualizeEszee sits on top of the `eszee` forward model. The core loop is:

1. **Data** — load binned UV visibilities and FITS images per field and spectral window.
2. **Model** — construct a pressure profile parameter dict (from YAML presets, custom values, or a posterior pickle).
3. **Match** — for each model/dataset pair, compute model visibilities via NUFFT, accumulate dirty maps, and store residual visibilities `resid_vis = data_vis − model_vis`.
4. **Point sources** — optionally subtract compact sources from `resid_vis` in the UV plane.
5. **Deconvolve** — run JvM cleaning on the residual visibilities to produce a noise-calibrated deconvolved map.
6. **Plot** — inspect maps, UV distributions, pressure profiles, and posterior corner plots.

Everything is coordinated through a single `Manager` instance:

```
add_data  →  add_model  →  match_model  →  [apply_point_source_correction]  →  [JvM_clean]  →  plot_*
```

---
## 3. Data format

VisualizeEszee reads pre-binned UV data produced by `eszee`. For each field `fid` and spectral window `sid` you need three files:

```
output_<band>_<array>.im.field-<fid>.spw-<sid>.data.npz   # binned visibilities
output_<band>_<array>.im.field-<fid>.spw-<sid>.image.fits  # CASA CLEAN image (for 'data' map panel)
output_<band>_<array>.im.field-<fid>.spw-<sid>.pbeam.fits  # primary beam response
```

The `.npz` file contains six arrays in order: `[uwave, vwave, uvreal, uvimag, suvwght, uvfreq]`.

The `binvis` argument to `add_data` is the path to this pattern with `fid` and `sid` as literal placeholders — they are replaced automatically for each field/SPW combination.

---
## 4. Initialising the Manager

```python
from visualizeeszee import Manager

pm = Manager(target='My_Cluster')  # target label used in plot titles and filenames
```

---
## 5. Loading data

### Interferometer data

Provide the list of field IDs and spectral windows that exist in your reduction. The `binvis` pattern must expand to files that actually exist on disk.

```python
pm.add_data(
    name='Band3_12m',            # arbitrary key — used to reference this dataset everywhere
    obstype='interferometer',
    band='band3',                # for bookkeeping only; does not affect file loading
    array='com12m',              # idem
    fields=['0', '1', '2'],      # field IDs as strings, matching those in the filenames
    spws=['0', '1', '2', '3'],   # flat list — the same SPWs are loaded for every field
    binvis='/path/to/output/output_band3_com12m.im.field-fid.spw-sid'
    # ^ 'fid' → actual field ID, 'sid' → actual SPW ID
)
```

If different fields have different SPWs, pass a nested list:
```python
spws=[['0','1'], ['0','1','2','3'], ['0','1','2','3']]  # per-field
```

Add as many datasets as you have — 7m, 12m, different bands all coexist in the same `Manager`:
```python
pm.add_data(name='Band3_07m', obstype='interferometer', band='band3', array='com07m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='/path/to/output/com07m/output_band3_com07m.im.field-fid.spw-sid')

pm.add_data(name='Band1_12m', obstype='interferometer', band='band1', array='com12m',
            fields=['0'], spws=['0','1','2','3'],
            binvis='/path/to/output/com12m/output_band1_com12m.im.field-fid.spw-sid')
```

### ACT data

```python
pm.add_data(
    name='ACT_DR6',
    obstype='ACT',
    fdir='/path/to/act_cutouts/',   # directory containing the ACT FITS cutouts
)
```

ACT data is used for sensitivity comparison in `plot_weight_distributions` but is not passed through the NUFFT matching pipeline.

---
## 6. Available pressure profile models

```python
from visualizeeszee.model import list_available_distributions, get_models

print(list_available_distributions())
# ['m14_up', 'm14_cc', 'm14_nc',   ← Melin+14 (universal, cool-core, non-cool-core)
#  'a10_up', 'a10_cc', 'a10_md',   ← Arnaud+10 (universal, cool-core, morphologically disturbed)
#  'gnfw',                          ← free gNFW (p_norm, r_s, alpha, beta, gamma)
#  'g17_ex', 'g17_st',             ← Ghirardini+17 (relaxed, disturbed)
#  'l15_00', 'l15_80', 'l15_85']   ← Le Brun+15 AGN feedback models
```

Each key maps to a set of YAML-defined hyperparameters (fixed profile shape). You then supply the cluster-specific parameters (`ra`, `dec`, `redshift`, `mass`, and optionally shape overrides).

---
## 7. Building model parameters

`get_models` merges YAML defaults with your cluster inputs and returns a parameter dict ready to pass to `add_model`.

### Standard pressure profile (recommended starting point)
```python
params = get_models('a10_up', custom_params={
    'ra':       83.822,    # cluster centre RA [deg]
    'dec':      -5.372,    # cluster centre Dec [deg]
    'redshift': 0.55,
    'mass':     6.0e14,    # M500 [solar masses]
})
```

You can override individual profile shape parameters as well:
```python
params = get_models('a10_up', custom_params={
    'ra': 83.822, 'dec': -5.372, 'redshift': 0.55, 'mass': 6.0e14,
    'alpha': 1.2, 'beta': 4.8, 'gamma': 0.1,   # override A10 shape params
})
```

### Free gNFW (all shape parameters explicit)
```python
params = get_models('gnfw', custom_params={
    'ra': 83.822, 'dec': -5.372,
    'p_norm': 0.24,     # central pressure normalisation [keV cm⁻³]
    'r_s':    0.030,    # scale radius [deg]
    'alpha':  0.80,
    'beta':   6.05,
    'gamma':  0.13,
    'redshift': 0.55,
})
```

The returned dict always has `params['model']['type']` — pass this directly to `add_model`.

---
## 8. Registering models

### From parameters
```python
pm.add_model(
    name='A10_up',                          # your label for this model
    source_type='parameters',
    model_type=params['model']['type'],     # can be omitted; inferred automatically
    parameters=params,
)
```

### From a dynesty posterior pickle (quantile extraction)

When you have a fitted posterior from `eszee`, load the median (or any quantile) directly:

```python
fname = '/path/to/dumps/my_run_pickle'

pm.add_model(
    name='fit',
    source_type='pickle',
    filename=fname,
    quantiles=[0.5],        # median; use [0.16, 0.5, 0.84] for ±1σ envelope
    marginalized=False,
)
# The registered key is:  'fit_q0.5'              (single-component model)
# Multi-component runs:   'fit_q0.5_c0', 'fit_q0.5_c1', ...
```

Inspect what was registered:
```python
print(list(pm.models.keys()))          # → ['fit_q0.5']
print(pm.models['fit_q0.5']['parameters']['model'])
```

### Tweaking parameters after loading

You can override individual parameters before running `match_model`:
```python
pm.models['fit_q0.5']['parameters']['model']['gamma'] = 0.0
```

This is useful for testing the effect of fixing a shape parameter to a specific value without re-loading the pickle.

---
## 9. Matching models to data

`match_model` is the core step. It computes model visibilities for every registered model against every interferometer dataset and stores:
- The dirty model map (`filtered` panel)
- The residual visibilities `resid_vis = data_vis − model_vis`
- The input Compton-y map (`input` panel)

```python
pm.match_model()
# Matches all models × all interferometer datasets.

# Or restrict to a specific pair:
pm.match_model(model_name='A10_up', data_name='Band3_12m')
```

Save dirty maps and Compton-y FITS to disk while matching:
```python
pm.match_model(save_output='/path/to/output/maps/')
```

> **Important**: `match_model` resets `resid_vis`. Always re-run it before
> re-applying `apply_point_source_correction` — otherwise you double-subtract.

---
## 10. Inspecting the loaded state

After adding data and models, use `summary()` to get a human-readable overview:

```python
pm.summary()
# Shows: datasets, model parameters (including derived gNFW equivalents),
#        point sources, and which model/data pairs have been matched.
```

For debugging the internal data structure after matching:
```python
pm.dump_structure(model_name='fit_q0.5', data_name='Band3_12m', depth=3)
# Prints an ASCII tree of matched_models[model][data]
```

---
## 11. UV sensitivity and weight distributions

Before diving into maps, it is useful to understand your data's sensitivity as a function of spatial scale:

```python
pm.plot_weight_distributions(save_plots=True)
```

This plots inverse-noise (σ in µJy) versus uv-distance [kλ] for every loaded dataset. The dashed horizontal line shows the point-source sensitivity (all baselines combined). If ACT data is loaded, its effective sensitivity is overplotted for direct comparison.

**What to look for**: large gaps in uv-coverage between arrays (e.g. 7m ends at ~30 kλ, 12m starts at ~15 kλ) tell you which spatial scales are well-constrained and which are filtered out.

---
## 12. Radial UV distributions

Radial UV distributions show the azimuthally averaged real and imaginary parts of the visibilities as a function of baseline length. The imaginary part should be consistent with zero for a symmetric source centred on the phase centre.

### Data only (before matching)
```python
pm.plot_radial_distributions(
    nbins=[20, 80, 80],        # one integer per loaded interferometer dataset, in add_data order
                               # more bins = finer baseline sampling but larger error bars
    custom_phase_center=(ra_cluster, dec_cluster),   # (ra_deg, dec_deg)
    save_plots=True,
)
```

**Choosing `nbins`**: shorter-baseline arrays (7m) have fewer long-baseline points so need fewer bins; longer-baseline arrays (12m) can support more bins. Start with ~20 per kλ of uv-coverage range as a guide.

### With model visibility overlay (after matching)
```python
pm.plot_radial_distributions(
    model_name='fit_q0.5',     # overlays the model visibility curve
    nbins=[20, 80, 80],
    custom_phase_center=(ra_cluster, dec_cluster),
    save_plots=True,
)
```

---
## 13. Point source workflow

If your `eszee` run included point-source components, extract them from the pickle and subtract their model visibilities from `resid_vis` so they do not appear as artefacts in your maps.

```python
fname = '/path/to/dumps/my_run_pickle'

# 1. Extract the frozen point-source list
ps_list = pm.get_point_sources_from_pickle(fname)
# Each entry: {ra [deg], dec [deg], amplitude [Jy], spec_type, spec_index, ref_freq [Hz]}

# 2. Inspect spectra and aperture fluxes across bands (optional but recommended)
pm.plot_point_source_spectra(
    ps_list,
    data_names=['Band1_12m', 'Band3_12m', 'Band3_07m'],
    model_name='fit_q0.5',
    plot_maps=True,          # also shows dirty maps with apertures
    aperture_arcsec=10.0,    # aperture radius for image-plane flux measurement
)

# 3. Subtract from residual visibilities, per dataset
pm.apply_point_source_correction('fit_q0.5', 'Band1_12m', ps_list)
pm.apply_point_source_correction('fit_q0.5', 'Band3_12m', ps_list)
pm.apply_point_source_correction('fit_q0.5', 'Band3_07m', ps_list)
```

> **Warning**: `apply_point_source_correction` modifies `resid_vis` in-place.
> Calling it a second time on the same dataset **double-subtracts**.
> To redo: call `match_model()` first to reset, then re-apply.

> **On amplitudes**: the `eszee` forward model multiplies point-source amplitudes by
> the primary beam response, so the stored amplitudes are *intrinsic* fluxes.
> `apply_point_source_correction` does not re-apply the PB correction (negligible
> for sources within a few arcsec of the phase centre where PB ≈ 1).

---
## 14. Map panels

After matching (and optionally subtracting point sources), plot dirty maps:

### Available panel types

| Type | What it shows | When to use |
|------|---------------|-------------|
| `'input'` | Compton-y model map, PB-corrected | Check model morphology |
| `'filtered'` | Dirty image of model visibilities | Compare to data in image plane |
| `'residual'` | Dirty residual (data − model), one field+SPW | Quick per-field quality check |
| `'data'` | CASA CLEAN image from `.image.fits` | Reference image |
| `'joint_residual'` | MFS dirty residual, all fields+SPWs combined | Best signal-to-noise residual view |
| `'deconvolved'` | JvM-cleaned image (requires prior `JvM_clean`) | Publication-quality map |

```python
pm.plot_map(
    model_name='fit_q0.5',
    data_name='Band3_12m',
    types=['filtered', 'data', 'residual'],   # or a single string
    field='0',                    # field ID to display (default: first field)
    fov=120,                      # crop to 120" × 120" — set to None for full image
    center=(ra_cluster, dec_cluster),  # (ra_deg, dec_deg) for the crop centre
    save_plots=True,
)
```

Loop over fields to check all of them:
```python
for fid in ['0', '1', '2']:
    pm.plot_map(model_name='fit_q0.5', data_name='Band3_12m',
                types=['filtered', 'data', 'residual'],
                field=fid, fov=120, save_plots=True)
```

---
## 15. Deconvolution — JvM clean

The JvM method produces a deconvolved map that has correct noise statistics in the residuals. It works on the stored `resid_vis` (which may already have point sources subtracted). You can clean a single dataset or combine multiple arrays jointly.

```python
# Single array
pm.JvM_clean(
    model_name='fit_q0.5',
    data_name=['Band3_12m'],
    save_output='/path/to/output/maps/',    # saves deconvolved FITS to disk
    taper=6/2.355,    # optional Gaussian uv-taper: FWHM [arcsec] / 2.355
                      # smooths to ~6" resolution; set to None for no taper
)
pm.plot_map(model_name='fit_q0.5', data_name=['Band3_12m'],
            types='deconvolved', fov=120, center=(ra_cluster, dec_cluster), save_plots=True)
pm.plot_map(model_name='fit_q0.5', data_name=['Band3_12m'],
            types='joint_residual', fov=120, center=(ra_cluster, dec_cluster), save_plots=True)
```

### Joint cleaning across arrays

Passing a list of dataset names combines all their residual visibilities before deconvolution. This is the best way to recover extended emission that is filtered by the 12m array alone:

```python
pm.JvM_clean(
    model_name='fit_q0.5',
    data_name=['Band3_07m', 'Band3_12m'],   # 7m + 12m jointly
    save_output='/path/to/output/maps/',
    taper=6/2.355,
)
pm.plot_map(model_name='fit_q0.5', data_name=['Band3_07m', 'Band3_12m'],
            types='deconvolved', fov=120, center=(ra_cluster, dec_cluster), save_plots=True)
```

When `data_name` is a list, the output map uses the header and pixel scale of the **highest-resolution (smallest-beam) dataset**.

**On the taper**: without a taper the deconvolved beam matches the native resolution of the data. A Gaussian taper (σ = FWHM/2.355) suppresses long baselines and smooths to a rounder beam — useful when comparing arrays with very different resolutions. A typical choice is 1–2× the 7m beam FWHM.

---
## 16. Pressure profiles

```python
# All registered models overlaid on the same axes
fig, ax = pm.plot_pressure_profile(return_fig=True)

# Specific models only
pm.plot_pressure_profile(
    model_names=['A10_up', 'fit_q0.5'],
    save_plots=True,
    output_dir='/path/to/plots/',
)
```

Profiles are plotted in physical units (radius in kpc, pressure in keV cm⁻³) using the redshift and mass from each model's parameter dict. `r500` is marked automatically.

---
## 17. Corner plots

Visualise the posterior from a dynesty run directly, with auto-generated parameter labels:

```python
pm.plot_corner('/path/to/dumps/my_run_pickle')
```

Options:
```python
pm.plot_corner(
    '/path/to/dumps/my_run_pickle',
    backend='corner',         # 'corner' (default, pip install corner) or 'dynesty'
    unit_scale={2: 3600},     # scale column 2 by 3600 (e.g. r_s from deg → arcsec)
    n_sigma_range=4.0,        # plot range: median ± n_sigma × half-IQR
    save_output='/path/to/plots/corner/',
)
```

Parameter labels are auto-generated from the pickle's `vary` structure and the YAML parameter ordering, so you rarely need to pass `labels` manually.

---
## 18. Full end-to-end example

```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

# Cluster-specific values — replace with your own
ra_cluster  = 83.822    # deg
dec_cluster = -5.372    # deg
phase_center = (ra_cluster, dec_cluster)
fname = '/path/to/dumps/my_run_pickle'

# ── 0. Available models ──────────────────────────────────────────────────────
print(list_available_distributions())

# ── 1. Init ──────────────────────────────────────────────────────────────────
pm = Manager(target='My_Cluster')

# ── 2. Load data ─────────────────────────────────────────────────────────────
pm.add_data(name='Band3_07m', obstype='interferometer', band='band3', array='com07m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='/path/to/output/com07m/output_band3_com07m.im.field-fid.spw-sid')
pm.add_data(name='Band3_12m', obstype='interferometer', band='band3', array='com12m',
            fields=['0','1','2','3','4','5','6'], spws=['0','1','2','3'],
            binvis='/path/to/output/com12m/output_band3_com12m.im.field-fid.spw-sid')
pm.add_data(name='Band1_12m', obstype='interferometer', band='band1', array='com12m',
            fields=['0'], spws=['0','1','2','3'],
            binvis='/path/to/output/com12m/output_band1_com12m.im.field-fid.spw-sid')

# ── 3. Sensitivity overview ───────────────────────────────────────────────────
pm.plot_weight_distributions(save_plots=True)

# ── 4. UV distributions (data only) ──────────────────────────────────────────
pm.plot_radial_distributions(nbins=[20, 80, 80], custom_phase_center=phase_center)

# ── 5. Register a posterior model ────────────────────────────────────────────
pm.add_model(name='fit', source_type='pickle', filename=fname, quantiles=[0.5])
# Key: 'fit_q0.5'

# ── 6. Explore the posterior ──────────────────────────────────────────────────
pm.plot_corner(fname)
pm.plot_pressure_profile(return_fig=True)

# ── 7. Match model to data ────────────────────────────────────────────────────
pm.match_model()
pm.summary()   # inspect matched state

# ── 8. Point source subtraction ───────────────────────────────────────────────
ps_list = pm.get_point_sources_from_pickle(fname)
pm.plot_point_source_spectra(ps_list, data_names=['Band1_12m','Band3_12m','Band3_07m'],
                             model_name='fit_q0.5', plot_maps=True)
for dname in ['Band1_12m', 'Band3_12m', 'Band3_07m']:
    pm.apply_point_source_correction('fit_q0.5', dname, ps_list)

# ── 9. Dirty map inspection ───────────────────────────────────────────────────
for dname in ['Band3_07m', 'Band3_12m', 'Band1_12m']:
    for fid in ['0', '1']:
        pm.plot_map(model_name='fit_q0.5', data_name=dname,
                    types=['filtered','data','residual'],
                    field=fid, fov=120, save_plots=True)

# ── 10. UV distributions with model overlay ───────────────────────────────────
pm.plot_radial_distributions(model_name='fit_q0.5', nbins=[20, 80, 80],
                             custom_phase_center=phase_center, save_plots=True)

# ── 11. JvM deconvolution ─────────────────────────────────────────────────────
taper = 6 / 2.355   # smooth to ~6" resolution

for dn in [['Band1_12m'], ['Band3_12m'], ['Band3_07m'], ['Band3_07m','Band3_12m']]:
    pm.JvM_clean(model_name='fit_q0.5', data_name=dn,
                 save_output='/path/to/output/maps/', taper=taper)
    pm.plot_map(model_name='fit_q0.5', data_name=dn, types='deconvolved',
                fov=120, center=phase_center, save_plots=True)
    pm.plot_map(model_name='fit_q0.5', data_name=dn, types='joint_residual',
                fov=120, center=phase_center, save_plots=True)
```

---
## 19. Common pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError` on `'fit_q0.5_c0'` | Model is single-component, no `_c{i}` suffix | Use `'fit_q0.5'`; `_c{i}` only appears for multi-component runs |
| Blank or zero model map | Wrong RA/Dec sign or units | Check: RA/Dec in decimal degrees, mass in solar masses |
| Double PS subtraction | `apply_point_source_correction` called twice | Re-run `match_model()` to reset `resid_vis`, then re-apply once |
| `KeyError` in pickle parameter builder | YAML parameter order does not match sampled order | Run `pm.get_param_order_from_yaml(model_type)` and compare to `results['vary']` |
| JvM deconvolved map has wrong beam | `data_name` list resolved to wrong dataset header | Expected — smallest-beam dataset wins; verify with `pm.summary()` |
| Missing FITS file on `add_data` | `binvis` pattern does not expand correctly | Print the expanded path manually: `binvis.replace('fid','0').replace('sid','0')` |

Diagnostic snippets:
```python
pm.summary()                                             # overview of everything
print(list(pm.models.keys()))                            # registered model keys
print(list(pm.matched_models['fit_q0.5'].keys()))        # matched dataset keys
pm.dump_structure(model_name='fit_q0.5', data_name='Band3_12m', depth=3)
```

---
## 20. Extending with new models

1. Open `visualizeeszee/model/brightness_models.yml`. Add a new entry following the existing pattern. **Parameter order must match the order used in the `eszee` MCMC sampler.**
2. If the profile function is new physics, add it to `model/models.py`.
3. Add a builder function in `model/parameter_utils.py` that takes `custom_params` and returns `{'model': {...}, 'spectrum': {...}}`.
4. Check: `get_models('my_new_key', custom_params={...})` and `list_available_distributions()`.

---
## 21. Glossary

| Term | Meaning |
|------|---------|
| `resid_vis` | Residual visibilities = data − model, stored per field/SPW, reset by `match_model()` |
| Quantile model | Parameter set extracted at a posterior quantile (e.g. `q=0.5` = median) |
| Hyperparameter | Fixed profile shape parameter defined in `brightness_models.yml` |
| JvM clean | Joint deconvolution method that normalises residuals by the JvM ε factor |
| `filtered` map | Dirty image of the model visibilities — what the model looks like through the interferometer |
| `joint_residual` | Multi-field, multi-SPW dirty residual; combines all baselines for best S/N |
| gNFW | Generalised Navarro-Frenk-White pressure profile |
| PB | Primary beam — the antenna power pattern; amplitudes in the pickle are intrinsic (PB-corrected) |
