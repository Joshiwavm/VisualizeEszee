
# Cluster Parameter Usage Guide

This guide explains how to provide cluster-specific parameters (such as RA, Dec, redshift, and mass) to VisualizeEszee's model-building functions. The system is designed to keep fixed model hyperparameters separate from cluster properties, allowing flexible and clear parameter management.

## Supplying Cluster Parameters

VisualizeEszee supports several ways to supply cluster parameters to `get_models()`:

### 1. Direct Keyword Arguments (Recommended)

Supply cluster parameters directly as keyword arguments. This is the most transparent and robust method.

```python
from VisualizeEszee.model import get_models

params = get_models(
    'a10_up',
    ra=74.92,         # Right ascension (deg)
    dec=-49.78,       # Declination (deg)
    redshift=1.71,    # Redshift
    mass=2.5e14       # Mass (solar masses)
)

# The returned dictionary includes both fixed hyperparameters and your cluster parameters.
print(params['model']['type'])   # e.g., 'A10Pressure'
print(params['model']['ra'])     # 74.92
print(params['model']['mass'])   # 2.5e14
```

### 2. Using `custom_params` Dictionary

You can pass a dictionary containing cluster parameters and/or overrides for model hyperparameters.

```python
cluster_params = {
    'ra': 74.92,
    'dec': -49.78,
    'redshift': 1.71,
    'mass': 2.5e14,
    'parameters': {
        'alpha': 1.2  # Override a hyperparameter if needed
    }
}
params = get_models('a10_up', custom_params=cluster_params)
```

### 3. Mixed Approach

Combine direct arguments and `custom_params` for maximum flexibility.

```python
params = get_models(
    'a10_up',
    ra=74.92,
    dec=-49.78,
    custom_params={'mass': 2.5e14, 'redshift': 1.71}
)
```

### 4. Fixed Hyperparameters Only

If you only want the fixed model hyperparameters, call `get_models()` with just the model name. You can then handle cluster parameters elsewhere in your code.

```python
params = get_models('a10_up')
# Later, supply cluster parameters as needed
```

## Example Workflow

```python
from VisualizeEszee import Manager
from VisualizeEszee.model import get_models

# Initialize the manager for your target cluster
pm = Manager(target='CL_J0459-4947')

# Build model parameters with cluster properties
params = get_models(
    'a10_up',
    ra=74.92,
    dec=-49.78,
    redshift=1.71,
    mass=2.5e14
)

# Register the model
pm.add_model(
    name='0459_1',
    source_type='parameters',
    model_type=params['model']['type'],
    parameters=params
)

# Proceed with matching, plotting, and analysis as needed
```

## Tips
- All cluster parameters should be provided in physical units (degrees for RA/Dec, solar masses for mass).
- You can override any model hyperparameter using the `parameters` sub-dictionary in `custom_params`.
- The returned parameter dictionary is ready for use with model registration and map generation functions.

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
