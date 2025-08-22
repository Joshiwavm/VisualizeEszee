"""
Parameter utilities for brightness (pressure) profile distributions.
Updated to use brightness_models.yml with 'distributions' key and log10M as primary mass quantity.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np

from ..utils.utils import cosmo


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
    Precedence (highest last): cluster_defaults < distribution params < custom_params < explicit args.
    Mass handling: log10M authoritative; derive other if only one given.
    custom_params may include both cluster-level (ra, dec, redshift, mass, etc.) and model-level (p_norm, r_s, alpha, ...).
    """
    config = load_brightness_models()
    dists = config.get('distributions', {})
    if dist_name not in dists:
        raise ValueError(f"Unknown distribution '{dist_name}'. Available: {list(dists.keys())}")

    cluster_defaults = config.get('cluster_defaults', {})
    dist_cfg = dists[dist_name]

    # Start with cluster defaults
    merged_cluster: Dict[str, Any] = dict(cluster_defaults)

    # Normalize custom params (case-insensitive keys) ---------------------------------
    norm_cp: Dict[str, Any] = {}
    if custom_params:
        for k, v in custom_params.items():
            norm_cp[k.lower()] = v

    # Alias maps ----------------------------------------------------------------------
    cluster_alias = {
        'ra': 'ra', 'dec': 'dec', 'z': 'redshift', 'redshift': 'redshift',
        'log10m': 'log10M', 'log10_m500': 'log10M', 'log10': 'log10M',
        'mass': 'mass', 'm500': 'mass', 'bias': 'bias',
        'e': 'e', 'ellipticity': 'e', 'angle': 'angle', 'offset': 'offset',
        'temperature': 'temperature', 'depth': 'depth'
    }
    model_alias = {
        'p_norm': 'p_norm', 'p0': 'p_norm', 'p_0': 'p_norm', 'p0norm': 'p_norm', 'p0_': 'p_norm', 'p\u2080': 'p_norm',
        'r_s': 'r_s', 'theta_s_deg': 'r_s', 'theta_s': 'r_s', 'theta_s_arcmin': 'r_s_arcmin',
        'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma',
        'alpha_p': 'alpha_p', 'ap': 'alpha_p',
        'concentration': 'concentration', 'c500': 'concentration'
    }

    # Extract cluster-level from custom params
    for k, v in norm_cp.items():
        if k in cluster_alias:
            merged_cluster[cluster_alias[k]] = v

    # Explicit argument overrides (highest precedence for cluster-level)
    if ra is not None: merged_cluster['ra'] = ra
    if dec is not None: merged_cluster['dec'] = dec
    if redshift is not None: merged_cluster['redshift'] = redshift
    if log10M is not None: merged_cluster['log10M'] = float(log10M)
    if mass is not None:
        merged_cluster['mass'] = float(mass)
        merged_cluster['log10M'] = np.log10(merged_cluster['mass'])

    # Harmonize mass / log10M
    if merged_cluster.get('log10M') is not None:
        merged_cluster['log10M'] = float(merged_cluster['log10M'])
        merged_cluster['mass'] = 10.0 ** merged_cluster['log10M']
    elif merged_cluster.get('mass') is not None:
        merged_cluster['mass'] = float(merged_cluster['mass'])
        merged_cluster['log10M'] = np.log10(merged_cluster['mass'])

    # Build base model + spectrum params from distribution ---------------------------
    model_params = _build_model_parameters(dist_cfg, profgeom)
    spectrum_params = _build_spectrum_parameters(dist_cfg)

    # Merge cluster into model params
    model_params.update(merged_cluster)

    # Apply model-level overrides from custom params ---------------------------------
    for k, v in norm_cp.items():
        if k in model_alias:
            mk = model_alias[k]
            # Special conversion for angular arcmin scale
            if mk == 'r_s_arcmin':  # convert to degrees then store as r_s
                try:
                    v = float(v) / 60.0
                except Exception:
                    pass
                mk = 'r_s'
            if mk == 'concentration':
                # Map to expected internal field names used by builders if present
                model_params['concentration'] = v
            else:
                model_params[mk] = v

    return {'model': model_params, 'spectrum': spectrum_params}

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
    if spec_type is None: 
        raise ValueError("Distribution missing 'spectrum_type'")
    # Always return a standardized dict with at least the type.
    return {'type': spec_type}

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
