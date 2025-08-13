import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from ..model.parameter_utils import get_models
from ..model.models import *
from ..utils.style import setup_plot_style
from ..utils import calculate_r500
from ..model.transform import TransformInput


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
            if model_info['source'] != 'parameters':
                raise ValueError("Pressure profiles only supported for 'parameters' source type")
            
            model_params = model_info['parameters']['model']
            model_type = model_params.get('type')
            z = model_params.get('redshift', model_params.get('z', None))
            mass = model_params.get('mass', None)
            
            r_min, r_max = r_range
            r_kpc = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
            
            xform = TransformInput(model_params, model_type)
            input_par = xform.generate()
            
            if model_type == 'A10Pressure' and mass is not None:
                r_s_kpc = input_par['major'] * 1000.0
                r_norm = r_kpc / r_s_kpc
            elif model_type in ['gnfwPressure', 'betaPressure'] and input_par.get('major') is not None:
                r_s_kpc = input_par['major'] * 1000.0
                r_norm = r_kpc / r_s_kpc
            else:
                r_norm = r_kpc / 1.0
            
            if model_type == 'gnfwPressure':
                rs = np.logspace(np.log10(max(r_norm.min(), 1e-6)), np.log10(r_norm.max()*2), n_points)[1:]
            else:
                rs = np.logspace(np.log10(max(r_norm.min(), 1e-6)), np.log10(r_norm.max()*2), n_points)
            
            if model_type == 'A10Pressure':
                shape = a10Profile(rs, input_par.get('offset', 0.0), 1.0, 1.0, input_par.get('e', 0.0),
                                   input_par['alpha'], input_par['beta'], input_par['gamma'],
                                   input_par['ap'], input_par['c500'], input_par['mass'])
            elif model_type == 'gnfwPressure':
                shape = gnfwProfile(rs, input_par.get('offset', 0.0), 1.0, 1.0, input_par.get('e', 0.0),
                                    input_par['alpha'], input_par['beta'], input_par['gamma'])
            elif model_type == 'betaPressure':
                shape = betaProfile(rs, input_par.get('offset', 0.0), 1.0, 1.0, input_par.get('e', 0.0),
                                    input_par['beta'])
            else:
                raise ValueError(f"Unknown pressure profile type: {model_type}")
            
            pressure_norm = np.interp(r_norm, rs, shape)
            
            if mass is not None and z is not None and model_type == 'A10Pressure':
                Ez = cosmo.H(z) / cosmo.H0
                M500_norm = mass / 3e14
                P500 = 1.65e-3 * Ez.value**(8/3) * M500_norm**(2/3)
                p_norm = model_params.get('p_norm', 1.0)
                if 'alpha_p' in model_params:
                    alpha_p = model_params.get('alpha_p', 0.0)
                    if alpha_p > 0:
                        p_norm *= (cosmo.H0.value / 70.0)**(-3/2)
                pressure_kev = pressure_norm * P500 * p_norm
            else:
                pressure_kev = pressure_norm
            
            label = f"{model_name} ({model_type})"
            line = ax.loglog(r_kpc, pressure_kev, label=label, **plot_kwargs)
            
            if mass is not None and z is not None and model_type == 'A10Pressure':
                r500_kpc = calculate_r500(mass, z)
                ax.axvline(r500_kpc, color=line[0].get_color(), linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Radius (kpc)', fontsize=10)
        ax.set_ylabel('Pressure (keV cm⁻³)' if any(
            mi['parameters']['model'].get('mass') is not None for mi in self.models.values()
        ) else 'Normalized Pressure', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(r_range)
        
        if save_plots:
            filename = 'pressure_profile_comparison.png' if len(model_names) > 1 else f'pressure_profile_{model_names[0]}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
        
        plt.show()