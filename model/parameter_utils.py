"""
Parameter utilities for pressure profile models.
Provides clean interface to get model parameters for different profile types.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from astropy.cosmology import Planck18 as cosmo


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
    """
    Get model parameters for a specific pressure profile type.
    
    Parameters
    ----------
    proftype : str
        Profile type identifier (e.g., 'a10_up', 'm14_cc', 'g17_ex')
    profgeom : str, optional
        Profile geometry: 'sph' (spherical) or 'ell' (elliptical), default 'sph'
    ra : float, optional
        Right ascension in degrees
    dec : float, optional
        Declination in degrees
    redshift : float, optional
        Redshift of the cluster
    mass : float, optional
        Mass of the cluster in solar masses
    custom_params : dict, optional
        Additional custom parameters to override defaults
        (cluster params like ra/dec/redshift/mass can be provided here too)
        
    Returns
    -------
    dict
        Dictionary containing model and spectrum parameter dictionaries
        with fixed hyperparameters and any provided cluster parameters
        
    Examples
    --------
    >>> # Get only fixed hyperparameters
    >>> params = get_models('a10_up')
    
    >>> # Provide cluster-specific parameters
    >>> params = get_models('a10_up', ra=74.92, dec=-49.78, redshift=1.71, mass=2.5e14)
    
    >>> # Or via custom_params
    >>> params = get_models('a10_up', custom_params={'ra': 74.92, 'dec': -49.78, 'redshift': 1.71, 'mass': 2.5e14})
    
    >>> # Use with PlotManager
    >>> pm.add_model(source_type='parameters', **params)
    """
    
    # Load configuration
    config = load_pressure_profiles()
    
    if proftype not in config['profiles']:
        available = list(config['profiles'].keys())
        raise ValueError(f"Unknown profile type '{proftype}'. Available: {available}")
    
    profile_config = config['profiles'][proftype].copy()
    
    # Add cluster-specific parameters if provided (positional args take precedence)
    cluster_params: Dict[str, Any] = {}
    if ra is not None:
        cluster_params['ra'] = ra
    if dec is not None:
        cluster_params['dec'] = dec
    if redshift is not None:
        cluster_params['redshift'] = redshift
    if mass is not None:
        cluster_params['mass'] = mass

    # Apply custom parameters if provided
    if custom_params:
        # Extract cluster params from custom_params if not already set via explicit args
        for k_src, k_dst in (
            ('ra', 'ra'), ('RA', 'ra'),
            ('dec', 'dec'), ('Dec', 'dec'),
            ('redshift', 'redshift'), ('z', 'redshift'),
            ('mass', 'mass'),
        ):
            if k_dst not in cluster_params and k_src in custom_params:
                cluster_params[k_dst] = custom_params[k_src]
        
        # Merge any other overrides into the profile config (e.g., concentration/parameters)
        profile_config = _merge_configs(profile_config, custom_params)
    
    # Build parameter arrays
    model_params = _build_model_parameters(profile_config, profgeom)
    spectrum_params = _build_spectrum_parameters(profile_config)
    
    # Add cluster parameters to model_params if provided
    if cluster_params:
        model_params.update(cluster_params)
    
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
    
    # Apply cosmology scaling for A10 universal profile if needed
    p_norm = params['p_norm']
    if 'alpha_p' in params and params['alpha_p'] > 0:  # Universal profile
        p_norm = p_norm * ((cosmo.H0.value / 70.0) ** (-3.0/2.0))
    
    return {
        'concentration': concentration,
        'alpha': params['alpha'],
        'beta': params['beta'],
        'gamma': params['gamma'],
        'p_norm': p_norm,
        'alpha_p': params['alpha_p'],
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
