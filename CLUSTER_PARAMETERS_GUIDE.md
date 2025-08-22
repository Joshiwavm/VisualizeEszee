# How to Provide Cluster-Specific Parameters

## Overview

The cleaned `get_models()` function now supports multiple ways to provide cluster-specific parameters (RA, Dec, redshift, mass) while keeping the fixed hyperparameters separate.

## Methods to Provide Cluster Parameters

### Method 1: Direct Parameters (Recommended)

```python
from VisualizeEszee.model import get_models

# Provide cluster parameters directly to get_models()
params = get_models('a10_up', 
                   ra=74.92,         # Right ascension in degrees
                   dec=-49.78,       # Declination in degrees  
                   redshift=1.71,    # Redshift
                   mass=2.5e14)      # Mass in solar masses

# Result includes both fixed hyperparameters AND cluster parameters
print(params['model']['type'])          # 'A10Pressure' (fixed)
print(params['model']['alpha'])         # 1.051 (fixed hyperparameter)
print(params['model']['ra'])            # 74.92 (your cluster parameter)
print(params['model']['mass'])          # 2.5e14 (your cluster parameter)
```

### Method 2: Using custom_params

```python
# Define cluster parameters in a dictionary
cluster_params = {
    'ra': 74.92,
    'dec': -49.78,
    'redshift': 1.71,
    'mass': 2.5e14,
    'parameters': {
        'alpha': 1.2  # Can also override hyperparameters
    }
}

params = get_models('a10_up', custom_params=cluster_params)
```

### Method 3: Mixed Approach

```python
# Some parameters direct, others via custom_params
params = get_models('a10_up', 
                   ra=74.92, dec=-49.78,  # Direct
                   custom_params={'mass': 2.5e14, 'redshift': 1.71})  # Custom
```

### Method 4: Get Fixed Parameters Only (Original Behavior)

```python
# Get only the fixed hyperparameters, provide cluster params elsewhere
params = get_models('a10_up')

# You handle cluster parameters separately in your code
cluster_ra = 74.92
cluster_dec = -49.78
cluster_z = 1.71
cluster_mass = 2.5e14
```

## Complete Example

```python
from VisualizeEszee import PlotManager
from VisualizeEszee.model import get_models

# Create PlotManager
pm = PlotManager(target='CL_J0459-4947')

# Get model with all parameters
params = get_models('a10_up', 
                   ra=74.92,      # RA of your cluster
                   dec=-49.78,    # Dec of your cluster
                   redshift=1.71, # Redshift of your cluster  
                   mass=2.5e14)   # Mass of your cluster

# Use with add_model (cluster params now included)
pm.add_model(
    name='my_cluster_model',
    source_type='parameters',
    band='band3',
    array='com12m',
    fields=['0'],
    spws=['5','7','9','11'],
    model_type=params['model']['type'],
    parameters=params
)
```

## Available Profile Types

Use any of these profile types with the methods above:

- `'a10_up'` - Arnaud et al. 2010 Universal Profile
- `'a10_cc'` - Arnaud et al. 2010 Cool Core  
- `'a10_md'` - Arnaud et al. 2010 Morphologically Disturbed
- `'m14_up'` - McDonald et al. 2014 Universal Profile
- `'m14_cc'` - McDonald et al. 2014 Cool Core
- `'m14_nc'` - McDonald et al. 2014 Non-Cool Core
- `'g17_ex'` - Ghirardini et al. 2017 Extended
- `'g17_st'` - Ghirardini et al. 2017 Standard  
- `'l15_00'` - Le Brun et al. 2015 (0% AGN feedback)
- `'l15_80'` - Le Brun et al. 2015 (80% AGN feedback)
- `'l15_85'` - Le Brun et al. 2015 (85% AGN feedback)

## What You Get

The returned `params` dictionary contains:

```python
{
    'model': {
        # Fixed hyperparameters from YAML
        'type': 'A10Pressure',
        'concentration': 1.177,
        'alpha': 1.051,
        'beta': 5.4905,
        'gamma': 0.3081,
        'p_norm': 8.403,
        'alpha_p': 0.12,
        
        # Your cluster-specific parameters (if provided)
        'ra': 74.92,
        'dec': -49.78,
        'redshift': 1.71,
        'mass': 2.5e14
    },
    'spectrum': {
        'type': 'tSZ'
    }
}
```

This gives you complete flexibility to provide cluster parameters when and how you need them!
