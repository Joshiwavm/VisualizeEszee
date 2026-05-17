import re
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import units as u

from ..model.parameter_utils import get_models
from ..model.models import *
from ..utils.style import setup_plot_style
from ..utils import calculate_r500, ysznorm, cosmo
from ..model.unitwrapper import TransformInput


class PlotPressureProfiles:
    """
    Class for plotting radial pressure profiles of SZ cluster models.
    Generates profiles in physical units (keV/cm³ vs kpc) using the same
    model functions as the ModelHandler.

    This class is designed to be inherited by other plotting classes.
    """

    def _compute_pressure_profile_for_model(self, model_name, r_kpc):
        """Compute the total pressure profile for a single stored model.

        Parameters
        ----------
        model_name : str
            Key in self.models (e.g. 'a10_dynesty_q0.5').
        r_kpc : ndarray
            Radial grid in kpc.

        Returns
        -------
        pressure : ndarray or None
            Total pressure profile in keV cm⁻³. None if model is marginalized.
        r500_info : (mass, redshift) tuple or None
            Parameters for r500 vline, or None if not available.
        """
        model_info = self.models.get(model_name)
        if model_info is None or model_info.get('marginalized'):
            return None, None

        params_list = model_info['parameters']
        if not isinstance(params_list, list):
            params_list = [params_list]

        r_mpc = r_kpc / 1000.0
        total_profile = np.zeros(len(r_kpc))
        r500_info = None

        _GNFW_TYPES = {'gnfwPressure', 'gnfwEmulator'}

        for comp_params in params_list:
            model_params = comp_params['model']
            model_type = model_params.get('type')

            xform = TransformInput(model_params, model_type)
            input_par = xform.run()

            major = input_par.get('major')
            coord = r_mpc / major

            rs_grid = np.append(0.0, np.logspace(-5, 5, 200))
            rs_sample = rs_grid[1:] if model_type in _GNFW_TYPES else rs_grid

            if model_type == 'A10Pressure':
                profile = a10Profile(
                    rs_sample,
                    input_par['offset'], input_par['amp'], input_par['major'],
                    input_par['e'], input_par['alpha'], input_par['beta'],
                    input_par['gamma'], input_par['ap'], input_par['c500'],
                    input_par['mass'], radial=True
                )
                if (model_params.get('mass') is not None
                        and model_params.get('redshift') is not None):
                    r500_info = (model_params['mass'], model_params['redshift'])
            elif model_type in _GNFW_TYPES:
                profile = gnfwProfile(
                    rs_sample,
                    input_par['offset'], input_par['amp'], input_par['major'],
                    input_par['e'], input_par['alpha'], input_par['beta'],
                    input_par['gamma'], radial=True
                )
            else:
                profile = betaProfile(
                    rs_sample,
                    input_par['offset'], input_par['amp'], input_par['major'],
                    input_par['e'], input_par['beta'], radial=True
                )

            total_profile += np.interp(coord, rs_sample, profile,
                                       left=profile[0], right=profile[-1])

        return total_profile, r500_info

    def plot_pressure_profile(self, model_names=None, r_range=(1, 2000), n_points=200,
                              save_plots=False, output_dir=None,
                              use_style=True, return_fig: bool = False,
                              show_bands: bool = True, **plot_kwargs):
        if use_style:
            setup_plot_style()

        if save_plots:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/pressure_profiles/'
            os.makedirs(output_dir, exist_ok=True)

        if not hasattr(self, 'models') or not self.models:
            raise ValueError("No models have been added. Use add_model() first.")

        if model_names is None:
            model_names = list(self.models.keys())
        else:
            if isinstance(model_names, str):
                model_names = [model_names]
            # Expand base names to all matching quantile keys
            expanded = []
            available = list(self.models.keys())
            for mn in model_names:
                if mn in available:
                    expanded.append(mn)
                else:
                    # Treat as base name: collect all _q{val} variants
                    prefix = mn + '_q'
                    matches = [k for k in available if k.startswith(prefix)]
                    if matches:
                        expanded.extend(matches)
                    else:
                        raise ValueError(f"Model '{mn}' not found. Available: {available}")
            model_names = expanded

        r_kpc = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_points)

        fig, ax = plt.subplots(figsize=(5.5, 4))

        # Group model names by base name and quantile for band plotting
        grouped = {}  # {base_name: {q_float_or_None: model_name_str}}
        for mn in model_names:
            m = re.search(r'_q([\d.]+)$', mn)
            if m:
                base = mn[:m.start()]
                q = float(m.group(1))
            else:
                base = mn
                q = None
            grouped.setdefault(base, {})[q] = mn

        _GNFW_TYPES = {'gnfwPressure', 'gnfwEmulator'}
        colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        hatches = ['/', '\\', '|', '-', '+', 'x', 'o', '.']

        for ci, (base_name, q_dict) in enumerate(grouped.items()):
            color = colors[ci % len(colors)]
            hatch = hatches[ci % len(hatches)]

            # Identify median quantile
            _q_keys = [q for q in q_dict if q is not None]
            q_med = min(_q_keys, key=lambda q: abs(q - 0.5)) if _q_keys else None
            mn_med = q_dict[q_med] if q_med is not None else list(q_dict.values())[0]

            pressure_50, r500_info = self._compute_pressure_profile_for_model(mn_med, r_kpc)
            if pressure_50 is None:
                continue

            # Detect model type from stored params to set linestyle
            _minfo = self.models.get(mn_med, {})
            _plist = _minfo.get('parameters', [])
            if not isinstance(_plist, list):
                _plist = [_plist]
            _mtype = _plist[0].get('model', {}).get('type', '') if _plist else ''
            _ls = '--' if _mtype in _GNFW_TYPES else '-'

            label = base_name
            ax.loglog(r_kpc, pressure_50, label=label, color=color,
                      linestyle=_ls, **plot_kwargs)

            # r500 vline from median model
            if r500_info is not None:
                r500_kpc = calculate_r500(r500_info[0], r500_info[1])
                ax.axvline(r500_kpc, color=color, linestyle='--', alpha=0.5)

            # Quantile band
            if show_bands and 0.16 in q_dict and 0.84 in q_dict:
                pressure_16, _ = self._compute_pressure_profile_for_model(q_dict[0.16], r_kpc)
                pressure_84, _ = self._compute_pressure_profile_for_model(q_dict[0.84], r_kpc)
                if pressure_16 is not None and pressure_84 is not None:
                    ax.fill_between(r_kpc, pressure_16, pressure_84,
                                    alpha=0.15, facecolor=color,
                                    edgecolor=color, hatch=hatch, linewidth=0.5)

        ax.set_xlabel('Radius (kpc)', fontsize=10)
        ax.set_ylabel(r'Pressure (keV cm⁻³ $h_{70}$)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(r_range)

        if save_plots:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            _prefix = f"{_safe_target}_" if getattr(self, 'target', None) else ''
            filename = _prefix + ('pressure_profile_comparison.png' if len(grouped) > 1
                                  else f'pressure_profile_{list(grouped.keys())[0]}.png')
            plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')

        if return_fig:
            return fig, ax

        plt.show()
