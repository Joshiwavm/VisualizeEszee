"""Posterior map making utilities (deduplicated).

Provides a single `MapMaking` mixin with:
  - Radial grid construction
  - Model map generation from parameter dicts
  - Posterior marginalized map generation (weighted average over samples)

Notes / Assumptions:
  * Posterior npz structure expected keys: `samples`, `weights`, `vary`, `pars`.
  * The last entry in `vary` corresponds to calibration scales (ignored here).
  * Parameter ordering for variable parameters derived from YAML via `get_param_order_from_yaml` (provided by LoadPickles).
  * Multiple components are supported: variable parameters for each component are concatenated in sample rows.
  * Calibration handling intentionally skipped (user request) – scale parameters, if any, are ignored.
"""

from typing import Callable, Sequence, Any, Dict, List
import numpy as np
import scipy
from tqdm import tqdm
import os
import jax
import jax.numpy as jnp


from ..model.unitwrapper import TransformInput
from ..model.models import a10Profile, gnfwProfile, betaProfile
from ..utils import ysznorm, cosmo


class MapMaking:
    """Class for memory-efficient posterior map generation and marginalization."""

    # ------------------------------ Public API ------------------------------
    def get_map(self, model_info: Dict[str, Any], ra_map, dec_map, header):
        if not model_info.get('marginalized', False):
            return self.generate_model_from_parameters(
                model_info['type'], model_info['parameters'], ra_map, dec_map, header
            )
        return self.generate_marginalized_model(model_info, ra_map, dec_map, header)

    @staticmethod
    def make_radial_grid(ra_map, dec_map, model_params):
        ra_center = model_params.get('ra')
        dec_center = model_params.get('dec')
        angle = model_params.get('angle', 0)
        eccentricity = model_params.get('e', 0)
        cosy = np.cos(np.deg2rad(dec_center))
        cost = np.cos(np.deg2rad(angle))
        sint = np.sin(np.deg2rad(angle))
        modgrid_x = (-(ra_map - ra_center) * cosy * sint - (dec_map - dec_center) * cost)
        modgrid_y = ((ra_map - ra_center) * cosy * cost - (dec_map - dec_center) * sint)
        return np.sqrt(modgrid_x**2 + modgrid_y**2 / (1.0 - eccentricity)**2)

    @staticmethod
    def generate_model_from_parameters(model_type, parameters, ra_map, dec_map, header,
                                       rs=np.append(0.0, np.logspace(-5, 5, 100))):
        
        xform = TransformInput(parameters['model'], model_type)
        input_par = xform.run()
        rs_sample = rs[1:] if model_type == 'gnfwPressure' else rs

        if model_type == 'A10Pressure':
            profile = a10Profile(rs_sample,
                                 input_par.get('offset'),
                                 input_par['amp'],
                                 input_par.get('major'),
                                 input_par.get('e'),
                                 input_par['alpha'],
                                 input_par['beta'],
                                 input_par['gamma'],
                                 input_par['ap'],
                                 input_par['c500'],
                                 input_par['mass'])
        elif model_type == 'gnfwPressure':
            profile = gnfwProfile(rs_sample,
                                  input_par.get('offset'),
                                  input_par['amp'],
                                  input_par.get('major'),
                                  input_par.get('e'),
                                  input_par['alpha'],
                                  input_par['beta'],
                                  input_par['gamma'])
        elif model_type == 'betaPressure':
            profile = betaProfile(rs_sample,
                                  input_par.get('offset'),
                                  input_par['amp'],
                                  input_par.get('major'),
                                  input_par.get('e'),
                                  input_par['beta'])

        r_grid = MapMaking.make_radial_grid(ra_map, dec_map, parameters['model'])
        z = parameters['model'].get('redshift', parameters['model'].get('z'))
        r_phys_mpc = np.deg2rad(r_grid) * cosmo.angular_diameter_distance(z).to('Mpc').value
        coord = r_phys_mpc / input_par.get('major')
        model_map = np.interp(coord, rs_sample, profile, left=profile[0], right=profile[-1])
        model_map = model_map * ysznorm
        return model_map

    # -------------------------- Marganalize ---------------------------
    def generate_marginalized_model(self, model_info: Dict[str, Any], ra_map, dec_map, header) -> np.ndarray: 

        self.results = np.load(model_info['filename'], allow_pickle=True)

        logwt = np.asarray(self.results['samples']['logwt'])
        logz = np.asarray(self.results['samples']['logz'])
        raw_samples = np.asarray(self.results['samples']['samples'])

        # Stable normalization of weights
        norm = scipy.special.logsumexp(logwt - logz[-1])
        weights = np.exp(logwt - norm - logz[-1])

        # Mask tiny weights
        m = weights > 1e-6
        samples = raw_samples[m]
        weights = weights[m]
        weights = weights / weights.sum()

        # Convert coordinate grids to JAX arrays (no copy if already ndarray)
        ra_j = jnp.asarray(ra_map)
        dec_j = jnp.asarray(dec_map)

        fixed_list = self._read_fixedvalues()

        # Accumulator as JAX (float32 for astro precision)
        im = jnp.zeros(ra_j.shape, dtype=jnp.float32)

        for idx, (sample, w) in enumerate(tqdm(zip(samples, weights), total=len(weights), desc='marginalizing (samples)')):
            pds = self._build_param_dicts(sample.reshape(-1,1), fixed_list)[0]

            smap = jnp.zeros_like(im)
            for comp in pds:
                comp_map_np = self.generate_model_from_parameters(comp['model']['type'], comp, ra_j, dec_j, header)
                comp_map = jnp.asarray(np.asarray(comp_map_np, dtype=float))
                smap = smap + comp_map

            im = im + float(w) * smap

        return np.array(im)