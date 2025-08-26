
# VisualizeEszee

VisualizeEszee provides utilities to build, sample, and visualize Sunyaev-Zel'dovich (SZ) cluster models for both interferometric and single-dish data. It is designed for flexible model registration, parameter management, and map-based analysis workflows.

## Getting Started

1. **Install the package** (editable mode recommended for development):

    ```bash
    pip install -e .
    ```

2. **Basic Workflow**

   - **Initialize a manager for your target cluster:**
     ```python
     from VisualizeEszee import Manager
     pm = Manager(target='CL_J0459-4947')
     ```

   - **Register observational data:**
     ```python
     pm.add_data(
         name='Band3_12m',
         obstype='interferometer',
         band='band3',
         array='com12m',
         fields=['0'],
         spws=['5','7','9','11'],
         binvis='../output/com12m/output_band3_com12m.im.field-fid.spw-sid'
     )
     ```

   - **Build model parameters using cluster properties:**
     ```python
     from VisualizeEszee.model import get_models
     params = get_models('a10_up', ra=74.9229603, dec=-49.7818421, redshift=1.71, mass=2.5e14)
     ```

   - **Register a model for analysis:**
     ```python
     pm.add_model(
         name='0459_1',
         source_type='parameters',
         model_type=params['model']['type'],
         parameters=params
     )
     ```

   - **Match the model to data and visualize results:**
     ```python
     pm.match_model()
     pm.plot_map(model_name='0459_1', data_name='Band3_12m', types=['filtered','data','residual'])
     ```

   - **Perform deconvolution and plot deconvolved maps:**
     ```python
     pm.JvM_clean(model_name='0459_1', data_name='Band3_12m')
     pm.plot_map(model_name='0459_1', data_name='Band3_12m', types='deconvolved')
     ```

## Documentation & Guides

- **Parameter usage and customization:** See `CLUSTER_PARAMETERS_GUIDE.md` for details on how to provide cluster-specific parameters and customize model inputs.
- **Model system structure:** See `MODEL_RESTRUCTURE_SUMMARY.md` for a summary of the model YAML structure, refactoring, and coordinate grid construction.

## Notes
- Adapt data paths and names to your local setup as needed.
- Some functions may require additional dependencies (e.g., NUFFT backends); ensure these are installed in your environment.

## Next Steps
- Explore the guides in this directory for deeper usage patterns and advanced workflows.
