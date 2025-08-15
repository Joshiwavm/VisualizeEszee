from .models import *
from .parameter_utils import (
    get_models,
    list_available_distributions,
    get_distribution_info,
    load_brightness_models,
    load_spectral_models
)

# Backward compatibility aliases
list_available_profiles = list_available_distributions
get_profile_info = get_distribution_info
load_component_models = load_brightness_models  # legacy name; now returns brightness models

__all__ = [
    'get_models',
    'list_available_distributions', 'get_distribution_info',
    'list_available_profiles', 'get_profile_info',
    'load_brightness_models', 'load_component_models', 'load_spectral_models'
]
