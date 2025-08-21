import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from ..model.parameter_utils import get_models
from ..model.models import *
from ..utils.style import setup_plot_style
from ..utils import calculate_r500, ysznorm
from ..model.unitwrapper import TransformInput  # renamed from transform


class PlotPressureProfiles:
    """
    Class for plotting radial pressure profiles of SZ cluster models.
    Generates profiles in physical units (keV/cm³ vs kpc) using the same
    model functions as the ModelHandler.
    
    This class is designed to be inherited by other plotting classes.
    """
        
    def plot_pressure_profile(self, model_names=None, r_range=(1, 2000), n_points=200, 
                              save_plots=False, output_dir='../plots/pressure_profiles/',
                              use_style=True, **plot_kwargs):
        # Setup plot style if requested
        if use_style:
            setup_plot_style()
            
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("No models have been added. Use add_model() first.")
        
        if model_names is None:
            model_names = list(self.models.keys())
        else:
            if isinstance(model_names, str):
                model_names = [model_names]
            available = list(self.models.keys())
            invalid = [name for name in model_names if name not in available]
            if invalid:
                raise ValueError(f"Models not found: {invalid}. Available: {available}")
        
        fig, ax = plt.subplots()
        
        for model_name in model_names:
            model_info = self.models[model_name]

            model_params = model_info['parameters']['model']
            model_type = model_params.get('type')
    
            z = model_params.get('redshift', model_params.get('z'))

            r_min, r_max = r_range
            r_kpc = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
            r_mpc = r_kpc / 1000.0

            xform = TransformInput(model_params, model_type)
            input_par = xform.generate()

            major = input_par.get('major')

            coord = r_mpc / major  # same dimensionless x as model_handler

            # base rs grid as in model_handler (append(0, logspace(-5,5,100)))
            rs_grid = np.append(0.0, np.logspace(-5, 5, 200))
            
            rs_sample = rs_grid[1:] if model_type == 'gnfwPressure' else rs_grid

            if model_type == 'A10Pressure':
                profile = a10Profile(
                    rs_sample, 
                    input_par['offset'],
                    input_par['amp'], 
                    input_par['major'],
                    input_par['e'],
                    input_par['alpha'], 
                    input_par['beta'], 
                    input_par['gamma'],
                    input_par['ap'], 
                    input_par['c500'], 
                    input_par['mass'],
                    radial=True
                )
            elif model_type == 'gnfwPressure':
                profile = gnfwProfile(
                    rs_sample, 
                    input_par['offset'],
                    input_par['amp'], 
                    input_par['major'],
                    input_par['e'],
                    input_par['alpha'], 
                    input_par['beta'], 
                    input_par['gamma'],
                    radial=True
                )
            else:  # betaPressure
                profile = betaProfile(
                    rs_sample, 
                    input_par['offset'],
                    input_par['amp'], 
                    input_par['major'],
                    input_par['e'],
                    input_par['beta'],
                    radial=True
                )
            
            pressure_interp = np.interp(coord, rs_sample, profile, left=profile[0], right=profile[-1])
            pressure_phys = pressure_interp

            label = f"{model_name} ({model_type})"
            line = ax.loglog(r_kpc, pressure_phys, label=label, **plot_kwargs)

            if model_type == 'A10Pressure' and model_params.get('mass') is not None and model_params.get('redshift') is not None:
                try:
                    r500_kpc = calculate_r500(model_params['mass'], model_params['redshift'])
                    ax.axvline(r500_kpc, color=line[0].get_color(), linestyle='--', alpha=0.5)
                except Exception:
                    pass
        
        ax.set_xlabel('Radius (kpc)', fontsize=10)
        ax.set_ylabel(r'Pressure (keV cm⁻³ $h_{70}$)' , fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(r_range)
        
        if save_plots:
            filename = 'pressure_profile_comparison.png' if len(model_names) > 1 else f'pressure_profile_{model_names[0]}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
        
        plt.show()