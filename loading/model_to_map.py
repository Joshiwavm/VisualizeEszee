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

    def generate_marginalized_model(self, model_info: Dict[str, Any], ra_map, dec_map, header):
        """Weighted posterior average map for (possibly multi-component) model.

        model_info must contain 'filename'. Model type(s) inferred from posterior file.
        """
        filename = model_info['filename']

        def generate_map(param_dicts: Sequence[Dict[str, Any]]):
            return np.sum([
                self.generate_model_from_parameters(pd['model']['type'], pd, ra_map, dec_map, header)
                for pd in param_dicts[0]
            ], axis=0)
        
        return self._make_marginalized_map_internal(filename, generate_map, ra_map, dec_map)

    # -------------------------- Internal helpers ---------------------------

    def _make_marginalized_map_internal(self, filename: str, 
                                        generate_map: Callable, ra_map: np.ndarray, dec_map: np.ndarray) -> np.ndarray:
        
        self.results = np.load(filename, allow_pickle=True)
        samples = np.asarray(self.results['samples']['samples'])
        weights = self.results['samples']['logwt'] - scipy.special.logsumexp(self.results['samples']['logwt'] - self.results['samples']['logz'][-1])
        weights = np.exp(weights - self.results['samples']['logz'][-1])

        m = weights > 0.000001
        samples, weights = samples[m], weights[m]
        weights = weights / weights.sum()

        acc = jnp.zeros(ra_map.shape, dtype=jnp.float32)
        for sample_row, w in zip(samples, weights):
            param_dicts = self._build_param_dicts_single_sample(sample_row)
            sample_map = generate_map(param_dicts).astype(np.float32)
            acc = acc + float(w) * jnp.asarray(sample_map)

        result = np.asarray(acc)
        return result

    def _build_param_dicts_single_sample(self, sample: np.ndarray) -> List[Dict[str, Any]]:
        """Reconstruct per-component parameter dicts for one sample.

        Inspired by LoadPickles._read_fixedvalues + _build_param_dicts logic to ensure
        consistent handling of fixed vs varying parameters using YAML ordering.
        """
        fixed_list = self._read_fixedvalues()
        params = self._build_param_dicts(sample.reshape(-1,1), fixed_list)
        self.test_params = params
        return params
