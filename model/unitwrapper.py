from typing import Dict, Any
import numpy as np
from astropy import units as u
from astropy import constants as const

from ..utils import calculate_r500
from ..utils.utils import cosmo


class TransformInput:
    """Map high-level parameters to low-level model inputs.
    Less strict: if essential cluster params (redshift/mass) absent, returns shape-only normalization (amp=1).
    Aliases accepted: redshift|z, mass|log10_m500|log10, p_norm|p0, concentration|c500, alpha_p|ap.
    """

    def __init__(self, model_params: Dict[str, Any], model_type: str):
        self.p = model_params
        self.model_type = model_type

    def generate(self) -> Dict[str, Any]:  # preferred
        return self.run()

    def run(self) -> Dict[str, Any]:  # backward compatibility
        base = {
            'offset': float(self.p.get('offset', 0.0)),  # offset may default to 0
            'e': float(self.p.get('e', self.p.get('eccentricity', 0.0))),  # ellipticity may default to 0
            'limdist': np.inf,
            'epsrel': 1.0e-6,
            'freeLS': self.p.get('depth', None),
        }
        if self.model_type == 'A10Pressure':
            return self._a10_pressure(base)
        if self.model_type == 'gnfwPressure':
            return self._gnfw_pressure(base)
        return base

    # --- Helpers ---
    def _first(self, *keys):
        for k in keys:
            if k in self.p and self.p[k] is not None:
                return self.p[k]
        return None

    def _angular_scale_deg(self) -> float | None:
        for key in ('Major','major_deg','theta_s_deg','theta_s_arcmin','major'):
            if key in self.p and self.p[key] is not None:
                if key == 'theta_s_arcmin':
                    return float(self.p[key]) / 60.0
                return float(self.p[key])
        return None

    # --- Profile builders ---
    def _a10_pressure(self, base: Dict[str, Any]) -> Dict[str, Any]:
        
        z = self._first('redshift', 'z')
        log10_m = self._first('log10_m500', 'log10')
        mass = self._first('mass')
        if mass is None and log10_m is not None:
            mass = 10 ** float(log10_m)
        c500 = self._first('concentration', 'c500')
        alpha = self._first('alpha')
        beta = self._first('beta')
        gamma = self._first('gamma')
        ap = (self._first('alpha_p', 'ap') or 0.0)
        p0 = self._first('p_norm', 'p0')
        bias = float(self.p.get('bias', 0.0))

        z = float(z); mass = float(mass); c500 = float(c500)
        alpha = float(alpha); beta = float(beta); gamma = float(gamma); ap = float(ap); p0 = float(p0)

        r500_kpc = calculate_r500(mass, z)
        r_s_mpc = (r500_kpc / c500) / 1000.0

        Hz = cosmo.H(z)

        m_eff = (1.0 - bias) * mass
        m_eff_kg = (m_eff * u.solMass)

        fb = float(self.p.get('fb', 0.175))
        mu = float(self.p.get('mu', 0.590))
        mue = float(self.p.get('mue', 1.140))

        amp  = p0 * (3.0 / 8.0 / np.pi) * (fb * mu / mue)
        amp *= (((2.5e2 * Hz * Hz) ** 2.0) * m_eff_kg / 1e15 / (const.G ** 0.5)) ** (2.0 / 3.0)
        amp = amp.to(u.keV / u.cm ** 3)
        amp = amp.value * 1e10

        mass_input = m_eff / 3e14 *(cosmo.H0.value/70.00)

        return {**base, 'amp': amp, 'major': r_s_mpc, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'ap': ap, 'c500': c500, 'mass': mass_input, 'phys_norm': True}

    def _gnfw_pressure(self, base: Dict[str, Any]) -> Dict[str, Any]:
        z = self._first('redshift', 'z')
        alpha = self._first('alpha')
        beta = self._first('beta')
        gamma = self._first('gamma')
        r_s = self._first('r_s','theta_s_deg')
        p_amp = self._first('p_norm','p0','p_0','P_0')

        z, alpha, beta, gamma, r_s, p_amp = map(float, (z, alpha, beta, gamma, r_s, p_amp))
        
        r_s_mpc = (np.deg2rad(r_s) * cosmo.angular_diameter_distance(z).to(u.Mpc)).value

        return {**base, 'amp': p_amp, 'major': r_s_mpc, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'phys_norm': True}