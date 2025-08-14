"""
Parameter utilities for pressure profile models.
Provides clean interface to get model parameters for different profile types.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from astropy.cosmology import Planck18 as cosmo
import numpy as np  # added


def load_pressure_profiles() -> Dict[str, Any]:
    """Load pressure profile definitions from YAML file."""
    config_path = Path(__file__).parent / "pressure_profiles.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Pressure profiles config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_component_models() -> Dict[str, Any]:
    """Load component model definitions (non-spectral)."""
    config_path = Path(__file__).parent / "component_models.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Component models config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_spectral_models() -> Dict[str, Any]:
    """Load spectral model definitions."""
    config_path = Path(__file__).parent / "spectral_models.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Spectral models config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_models(proftype: str, profgeom: str = 'sph', 
               ra: Optional[float] = None, dec: Optional[float] = None,
               redshift: Optional[float] = None, mass: Optional[float] = None,
               custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get model parameters for a specific pressure profile type with merged cluster defaults.
    Order of precedence (highest wins): explicit args > custom_params > profile params > cluster_defaults
    """
    config = load_pressure_profiles()
    if proftype not in config['profiles']:
        available = list(config['profiles'].keys())
        raise ValueError(f"Unknown profile type '{proftype}'. Available: {available}")

    cluster_defaults = config.get('cluster_defaults', {})
    profile_config = config['profiles'][proftype].copy()

    # Start building merged dict
    merged_cluster: Dict[str, Any] = {}
    merged_cluster.update(cluster_defaults)

    # Populate from profile-specific overrides if they exist (rare)
    # (e.g., bias or concentration differences already inside profile_config)
    # Concentration is handled separately below when building model parameters.

    # Collect cluster params from custom_params
    if custom_params:
        # Accept alternative case keys
        for k_src, k_dst in (
            ('ra', 'ra'), ('RA', 'ra'),
            ('dec', 'dec'), ('Dec', 'dec'),
            ('redshift', 'redshift'), ('z', 'redshift'),
            ('mass', 'mass'), ('M500', 'mass'),
            ('log10_m500', 'log10_m500'), ('log10', 'log10_m500'),
            ('bias', 'bias'), ('e', 'e'), ('ellipticity', 'e'),
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
    if mass is not None: merged_cluster['mass'] = mass

    # Derive / normalize naming (mass is single source of truth)
    if 'mass' in merged_cluster and merged_cluster['mass'] is not None:
        merged_cluster['mass'] = float(merged_cluster['mass'])
        merged_cluster['log10_m500'] = np.log10(merged_cluster['mass'])
    elif 'log10_m500' in merged_cluster:
        merged_cluster['log10_m500'] = float(merged_cluster['log10_m500'])
        merged_cluster['mass'] = 10 ** merged_cluster['log10_m500']
        # Recompute to avoid rounding drift
        merged_cluster['log10_m500'] = np.log10(merged_cluster['mass'])

    model_params = _build_model_parameters(profile_config, profgeom)
    spectrum_params = _build_spectrum_parameters(profile_config)

    # Attach cluster params
    model_params.update(merged_cluster)

    return {
        'model': model_params,
        'spectrum': spectrum_params
    }


def _merge_configs(base_config: Dict, custom_config: Dict) -> Dict:
    """Recursively merge custom configuration into base configuration."""
    merged = base_config.copy()
    
    for key, value in custom_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _build_model_parameters(profile_config: Dict, profgeom: str) -> Dict[str, Any]:
    """Build model parameter arrays for the specified profile.
    
    Returns only the fixed hyperparameters for the profile type.
    User must provide RA, Dec, redshift, and mass when building models.
    """
    
    # Get profile-specific parameters
    params = profile_config['parameters']
    concentration = profile_config['concentration']
    
    # Build parameter arrays based on model type
    model_type = profile_config['model_type']
    
    if model_type == 'A10Pressure':
        return _build_a10_parameters(concentration, params)
    elif model_type == 'G17Pressure':
        return _build_g17_parameters(concentration, params)
    elif model_type == 'L15Pressure':
        return _build_l15_parameters(concentration, params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_a10_parameters(concentration, params):
    """Build A10 pressure profile parameters - only fixed hyperparameters."""
    p_norm = params.get('p_norm', params.get('p0'))
    if p_norm is None:
        raise ValueError("A10 profile requires 'p_norm' (or alias 'p0') in YAML parameters")
    if 'alpha_p' in params and params['alpha_p'] > 0:  # Universal-like evolutionary scaling
        p_norm = p_norm * ((cosmo.H0.value / 70.0) ** (-3.0/2.0))
    return {
        'concentration': params.get('c500', concentration),  # allow c500 override inside parameters
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'p_norm': p_norm,
        'alpha_p': params.get('alpha_p', 0.0),
        'type': 'A10Pressure'
    }


def _build_g17_parameters(concentration, params):
    """Build G17 pressure profile parameters - only fixed hyperparameters."""
    
    return {
        'concentration': concentration,
        'alpha': params['alpha'],
        'beta0': params['beta0'],
        'beta1': params['beta1'],
        'beta2': params['beta2'],
        'gamma0': params['gamma0'],
        'gamma1': params['gamma1'],
        'gamma2': params['gamma2'],
        'p_norm': params['p_norm'],
        'alpha_p': params['alpha_p'],
        'c_p': params['c_p'],
        'type': 'G17Pressure'
    }


def _build_l15_parameters(concentration, params):
    """Build L15 pressure profile parameters - only fixed hyperparameters."""
    
    return {
        'concentration': concentration,
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'delta': params['delta'],
        'epsilon': params['epsilon'],
        'p_norm': params['p_norm'],
        'type': 'L15Pressure'
    }

def _build_spectrum_parameters(profile_config: Dict) -> Dict[str, Any]:
    """Build spectrum parameter arrays."""
    
    spectrum_type = profile_config['spectrum_type']
    
    # For tSZ, no additional parameters needed
    if spectrum_type == 'tSZ':
        return {
            'type': 'tSZ'
        }
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}")


def list_available_profiles() -> list:
    """List all available pressure profile types."""
    config = load_pressure_profiles()
    return list(config['profiles'].keys())


def get_profile_info(proftype: str) -> Dict[str, Any]:
    """Get information about a specific profile type."""
    config = load_pressure_profiles()
    
    if proftype not in config['profiles']:
        available = list(config['profiles'].keys())
        raise ValueError(f"Unknown profile type '{proftype}'. Available: {available}")
    
    return config['profiles'][proftype]


def get_model_info(model_name: str, spectral: bool = False) -> Dict[str, Any]:
    """
    Get information about a specific model type.
    
    Parameters
    ---------- 
    model_name : str
        Name of the model (e.g., 'pointSource', 'A10Pressure')
    spectral : bool, optional
        Whether to look in spectral models (True) or component models (False)
        
    Returns
    -------
    dict
        Model configuration dictionary
    """
    if spectral:
        models = load_spectral_models()
    else:
        models = load_component_models()
    
    if model_name not in models:
        available = list(models.keys())
        model_type = "spectral" if spectral else "component"
        raise ValueError(f"Unknown {model_type} model '{model_name}'. Available: {available}")
    
    return models[model_name]


def list_available_models(spectral: Optional[bool] = None) -> Dict[str, list]:
    """
    List all available model types.
    
    Parameters
    ----------
    spectral : bool, optional
        If True, return only spectral models
        If False, return only component models  
        If None, return both
        
    Returns
    -------
    dict
        Dictionary with 'component' and/or 'spectral' keys containing model lists
    """
    result = {}
    
    if spectral is None or not spectral:
        component_models = load_component_models()
        result['component'] = list(component_models.keys())
    
    if spectral is None or spectral:
        spectral_models = load_spectral_models()
        result['spectral'] = list(spectral_models.keys())
    
    return result
