from typing import Dict, Any
import numpy as np
from astropy import units as u

# Reuse shared cosmology tools
from ..utils import calculate_r500
from ..utils.utils import cosmo


class TransformInput:
    """
    Transform high-level model parameters to inputs expected by models.py
    functions, without forcing a specific global normalization.

    Supports A10Pressure and gNFW pressure models where possible.
    """

    def __init__(self, model_params: Dict[str, Any], model_type: str):
        self.p = model_params
        self.model_type = model_type

    def run(self) -> Dict[str, Any]:
        base = {
            'offset': float(self.p.get('offset', 0.0)),
            'e': float(self.p.get('e', self.p.get('eccentricity', 0.0))),
            'limdist': np.inf,
            'epsrel': 1.0e-6,
            'freeLS': None,
        }

        if self.model_type == 'A10Pressure':
            return self._a10_pressure(base)
        if self.model_type == 'gnfwPressure':
            return self._gnfw_pressure(base)

        # Fallback: return the base; callers can decide how to proceed
        return base

    def _a10_pressure(self, base: Dict[str, Any]) -> Dict[str, Any]:
        z = float(self.p.get('redshift', self.p.get('z', 0.5)))
        mass = float(self.p.get('mass', 2e14))
        c500 = float(self.p.get('concentration', self.p.get('c500', 1.18)))

        # Hyperparameters
        alpha = float(self.p.get('alpha'))
        beta = float(self.p.get('beta'))
        gamma = float(self.p.get('gamma'))
        ap = float(self.p.get('alpha_p', 0.12))

        # Characteristic scale r_s = R500 / c500 (in Mpc)
        r500_kpc = calculate_r500(mass, z)
        r_s_mpc = (r500_kpc / c500) / 1000.0

        params = {
            **base,
            'amp': 1.0,           # dimensionless profile amplitude (shape only)
            'major': r_s_mpc,     # physical scale used for building r_norm outside
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'ap': ap,
            'c500': c500,
            'mass': mass / 3e14,  # expected by a10Profile
        }
        return params

    def _gnfw_pressure(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Build gNFW parameters. If an angular scale is provided, convert to physical.
        Accepted angular keys (priority order):
          - 'Major' (deg), 'major_deg', 'theta_s_deg', 'theta_s_arcmin', 'major' (assumed deg)
        """
        z = float(self.p.get('redshift', self.p.get('z', 0.5)))

        # Try to get an angular scale in degrees
        ang_deg = None
        if 'Major' in self.p:
            ang_deg = float(self.p['Major'])
        elif 'major_deg' in self.p:
            ang_deg = float(self.p['major_deg'])
        elif 'theta_s_deg' in self.p:
            ang_deg = float(self.p['theta_s_deg'])
        elif 'theta_s_arcmin' in self.p:
            ang_deg = float(self.p['theta_s_arcmin']) / 60.0
        elif 'major' in self.p:
            # Heuristic: treat as degrees to mirror old code behavior
            ang_deg = float(self.p['major'])

        if ang_deg is not None:
            major_mpc = (np.deg2rad(ang_deg) * cosmo.angular_diameter_distance(z).to(u.Mpc)).value
        else:
            major_mpc = None

        alpha = float(self.p.get('alpha', 1.05))
        beta = float(self.p.get('beta', 5.49))
        gamma = float(self.p.get('gamma', 0.31))

        params = {
            **base,
            'amp': 1.0,
            'major': major_mpc,  # may be None if no angular size was supplied
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
        }
        return params
