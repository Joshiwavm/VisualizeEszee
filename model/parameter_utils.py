"""
Parameter utilities for brightness (pressure) profile distributions.
Updated to use brightness_models.yml with 'distributions' key and log10M as primary mass quantity.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from astropy.cosmology import Planck18 as cosmo
import numpy as np

# ------------------------------------------------------------------
# Loading configuration
# ------------------------------------------------------------------

def load_brightness_models() -> Dict[str, Any]:
    path = Path(__file__).parent / "brightness_models.yml"
    if not path.exists():
        raise FileNotFoundError(f"Brightness models config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_spectral_models() -> Dict[str, Any]:
    path = Path(__file__).parent / "spectral_models.yml"
    if not path.exists():
        raise FileNotFoundError(f"Spectral models config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

def get_models(dist_name: str, profgeom: str = 'sph',
               ra: Optional[float] = None, dec: Optional[float] = None,
               redshift: Optional[float] = None, mass: Optional[float] = None,
               log10M: Optional[float] = None,
               custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return merged model + spectrum parameter dictionaries for a distribution.
    Precedence: explicit args > custom_params > distribution parameters > cluster_defaults.
    log10M is the authoritative mass representation; mass is derived.
    """
    config = load_brightness_models()
    dists = config.get('distributions', {})
    if dist_name not in dists:
        available = list(dists.keys())
        raise ValueError(f"Unknown distribution '{dist_name}'. Available: {available}")

    cluster_defaults = config.get('cluster_defaults', {})
    dist_cfg = dists[dist_name]

    merged_cluster: Dict[str, Any] = dict(cluster_defaults)

    # Collect cluster params from custom_params (case-insensitive variants)
    if custom_params:
        for k_src, k_dst in (
            ('ra', 'ra'), ('RA', 'ra'),
            ('dec', 'dec'), ('Dec', 'dec'),
            ('redshift', 'redshift'), ('z', 'redshift'),
            ('log10M', 'log10M'), ('log10', 'log10M'), ('log10_m500', 'log10M'),
            ('mass', 'mass'), ('M500', 'mass'),
            ('bias', 'bias'),
            ('e', 'e'), ('ellipticity', 'e'),
            ('angle', 'angle'), ('Angle', 'angle'),
            ('offset', 'offset'), ('Offset', 'offset'),
            ('temperature', 'temperature'), ('Temperature', 'temperature'),
            ('depth', 'depth'),
        ):
            if k_src in custom_params:
                merged_cluster[k_dst] = custom_params[k_src]

    # Override with explicit arguments
    if ra is not None: merged_cluster['ra'] = ra
    if dec is not None: merged_cluster['dec'] = dec
    if redshift is not None: merged_cluster['redshift'] = redshift
    if log10M is not None: merged_cluster['log10M'] = float(log10M)
    if mass is not None:  # allow direct mass override
        merged_cluster['mass'] = float(mass)
        merged_cluster['log10M'] = np.log10(merged_cluster['mass'])

    # Harmonize mass/log10M (log10M authoritative)
    if 'log10M' in merged_cluster and merged_cluster['log10M'] is not None:
        merged_cluster['log10M'] = float(merged_cluster['log10M'])
        merged_cluster['mass'] = 10.0 ** merged_cluster['log10M']
    elif 'mass' in merged_cluster and merged_cluster['mass'] is not None:
        merged_cluster['mass'] = float(merged_cluster['mass'])
        merged_cluster['log10M'] = np.log10(merged_cluster['mass'])
    else:
        # Neither provided; leave both absent (shape-only usage downstream)
        pass

    # Build model + spectrum parameter dictionaries
    model_params = _build_model_parameters(dist_cfg, profgeom)
    spectrum_params = _build_spectrum_parameters(dist_cfg)

    # Attach cluster-level parameters
    model_params.update(merged_cluster)

    return { 'model': model_params, 'spectrum': spectrum_params }

# ------------------------------------------------------------------
# Internal builders
# ------------------------------------------------------------------

def _build_model_parameters(dist_cfg: Dict[str, Any], profgeom: str) -> Dict[str, Any]:
    params = dist_cfg.get('parameters', {})
    model_type = dist_cfg.get('model_type')

    if model_type == 'A10Pressure':
        return _build_a10_parameters(params)
    if model_type == 'G17Pressure':
        return _build_g17_parameters(params)
    if model_type == 'L15Pressure':
        return _build_l15_parameters(params)
    if model_type == 'gnfwPressure':
        return _build_gnfw_parameters(params)
    raise ValueError(f"Unknown model type: {model_type}")


def _extract_p_norm(params: Dict[str, Any]) -> float:
    p_norm = params.get('p_norm')
    if p_norm is None:
        p_norm = params.get('p0')  # legacy alias
    if p_norm is None:
        raise ValueError("Profile requires 'p_norm' (or 'p0') in parameters")
    return p_norm


def _build_a10_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    p_norm = _extract_p_norm(params)
    # Optionally evolve p_norm depending on alpha_p (placeholder – keep as-is)
    return {
        'concentration': params.get('c500'),
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'p_norm': p_norm,
        'alpha_p': params.get('alpha_p', 0.0),
        'type': 'A10Pressure'
    }


def _build_g17_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'concentration': params.get('c500'),
        'alpha': params['alpha'],
        'beta0': params['beta0'],
        'beta1': params['beta1'],
        'beta2': params['beta2'],
        'gamma0': params['gamma0'],
        'gamma1': params['gamma1'],
        'gamma2': params['gamma2'],
        'p_norm': params['p_norm'],
        'alpha_p': params.get('alpha_p', 0.0),
        'c_p': params['c_p'],
        'type': 'G17Pressure'
    }


def _build_l15_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'concentration': params.get('c500'),
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'delta': params['delta'],
        'epsilon': params['epsilon'],
        'p_norm': params['p_norm'],
        'type': 'L15Pressure'
    }


def _build_gnfw_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'p_norm': params['p_norm'],
        'r_s': params.get('r_s'),
        'type': 'gnfwPressure'
    }


def _build_spectrum_parameters(dist_cfg: Dict[str, Any]) -> Dict[str, Any]:
    spec_type = dist_cfg.get('spectrum_type')
    if spec_type == 'tSZ':
        return {'type': 'tSZ'}
    raise ValueError(f"Unknown spectrum type: {spec_type}")

# ------------------------------------------------------------------
# Listing helpers
# ------------------------------------------------------------------

def list_available_distributions() -> list:
    config = load_brightness_models()
    return list(config.get('distributions', {}).keys())


def get_distribution_info(name: str) -> Dict[str, Any]:
    config = load_brightness_models()
    d = config.get('distributions', {})
    if name not in d:
        raise ValueError(f"Unknown distribution '{name}'. Available: {list(d.keys())}")
    return d[name]
