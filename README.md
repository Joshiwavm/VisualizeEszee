# VisualizeEszee

A Python toolkit for forward-modelling and visualizing Sunyaev-Zel'dovich (SZ) cluster pressure profiles against interferometric (ALMA) and single-dish (ACT) data.

**VisualizeEszee is a companion package to [`eszee`](https://github.com/<your-org>/eszee)** and is designed to be used after an `eszee` run. `eszee` handles the MCMC/nested sampling fit; VisualizeEszee takes the outputs (binned UV data, FITS images, and dynesty posterior pickles) and provides a high-level `Manager` class for the full post-processing and inspection loop:

1. **Load** binned UV data and FITS images produced by `eszee`
2. **Register** a pressure profile model — from YAML defaults, custom parameters, or a dynesty posterior pickle written by `eszee`
3. **Match** — compute model visibilities via NUFFT and build dirty maps
4. **Subtract** point sources in the UV plane using fitted components from the `eszee` pickle
5. **Deconvolve** using the JvM method across one or more arrays jointly
6. **Inspect** — UV distributions, pressure profiles, posterior corner plots, map panels

---

## Dependencies

VisualizeEszee requires [`eszee`](https://github.com/<your-org>/eszee) to be installed and used upstream to produce the input data. It does not re-run the forward model or sampler — it reads the `eszee` outputs.

Other requirements:
- Python ≥ 3.10
- `jax` + `jax_finufft` for Fourier operations (install these first — see the [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html))
- All other dependencies (`numpy`, `matplotlib`, `astropy`, `scipy`, `corner`, `reproject`, `pyyaml`) are installed automatically via `pip`

---

## Install

```bash
# 1. Install and run eszee first to produce your UV data and posterior pickles.
#    See the eszee repository for instructions.

# 2. Install JAX for your platform (CPU example):
pip install "jax[cpu]"
pip install jax_finufft

# 3. Install VisualizeEszee in editable mode:
git clone https://github.com/<your-org>/VisualizeEszee.git
cd VisualizeEszee
pip install -e .
```

Verify:
```python
import visualizeeszee
print('ok')
```

---

## Data requirements

VisualizeEszee reads three file types produced by `eszee` per field and spectral window:

```
output_<band>_<array>.im.field-<fid>.spw-<sid>.data.npz   # binned visibilities (eszee output)
output_<band>_<array>.im.field-<fid>.spw-<sid>.image.fits  # CASA CLEAN image
output_<band>_<array>.im.field-<fid>.spw-<sid>.pbeam.fits  # primary beam
```

The `binvis` argument to `add_data` is the path pattern with `fid` and `sid` as literal placeholders — they are replaced automatically for each field/SPW combination.

Posterior pickles from `eszee` (dynesty `.npz` format) are read by `add_model` and `plot_corner`.

---

## Quick start

```python
from visualizeeszee import Manager
from visualizeeszee.model import get_models, list_available_distributions

# See all available pressure profile models
print(list_available_distributions())

# Initialise
pm = Manager(target='My_Cluster')

# Load eszee output data
pm.add_data(
    name='Band3_12m',
    obstype='interferometer',
    band='band3',
    array='com12m',
    fields=['0', '1'],           # field IDs present in your eszee reduction
    spws=['0', '1', '2', '3'],   # spectral windows
    binvis='/path/to/eszee/output/output_band3_com12m.im.field-fid.spw-sid'
)

# Build a model from YAML defaults + your cluster parameters
params = get_models('a10_up', custom_params={
    'ra': 83.822,      # deg
    'dec': -5.372,     # deg
    'redshift': 0.55,
    'mass': 6.0e14,    # solar masses
})
pm.add_model(name='A10', source_type='parameters',
             model_type=params['model']['type'], parameters=params)

# Or load the median posterior from an eszee dynesty pickle
pm.add_model(name='fit', source_type='pickle',
             filename='/path/to/eszee/dumps/my_run_pickle',
             quantiles=[0.5])
# Registered as 'fit_q0.5'

# Match model to data (builds dirty maps + residual visibilities)
pm.match_model()

# Plot data, filtered model, and residual side by side
pm.plot_map(model_name='fit_q0.5', data_name='Band3_12m',
            types=['filtered', 'data', 'residual'],
            fov=120,   # arcsec field of view
            center=(83.822, -5.372))
```

---

## Docs

See **[TUTORIAL.md](TUTORIAL.md)** for the complete workflow:
- All `add_data` options (interferometer, ACT)
- Building models from YAML, custom parameters, or `eszee` dynesty posteriors
- Point source extraction from `eszee` pickles and UV-plane subtraction
- UV sensitivity and radial distribution plots
- JvM deconvolution (single array and joint multi-array)
- Posterior corner plots and pressure profiles
- Troubleshooting and extending with new models
