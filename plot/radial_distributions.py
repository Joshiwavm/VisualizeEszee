import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

from ..utils.style import setup_plot_style
from ..utils.utils import arcsec_to_uvdist, uvdist_to_arcsec

class PlotRadialDistributions:

    # ----------------- Compute raidal uvprofiles ------------------------

    def _get_binned_uvdatapoints(self, band_name, nbins=20, custom_phase_center=None):
        """
        Bin UV data points radially after phase shifting all fields to central field.
        
        Parameters
        ----------
        band_name : str
            Name of the band to analyze
        nbins : int, optional
            Number of bins for radial averaging (default: 20)
        custom_phase_center : tuple, optional
            Custom phase center (RA, Dec) in degrees to shift all fields to.
            If None, uses automatically determined central field.
            
        Returns
        -------
        tuple
            Binned real, real errors, imaginary, imaginary errors, bin edges, bin centers
        """
        # Determine phase center to use
        if custom_phase_center is not None:
            central_phase_center = custom_phase_center
        else:
            # Find central field
            central_field = self._find_central_field(band_name)
            print(f"Using central field: {central_field}")
            central_phase_center = self.uvdata[band_name][central_field]['phase_center']
        
        # Collect phase-shifted data only for the central field
        all_uvreals = []
        all_uvimags = []
        all_uvdist = []
        all_uvwghts = []

        central_field = self._find_central_field(band_name)
        field_data = self.uvdata[band_name][central_field]

        # Calculate phase shift needed to move this field to the chosen central phase center
        field_phase_center = field_data['phase_center']
        dRA_rad =  np.deg2rad(central_phase_center[0] - field_phase_center[0])  # to arcsec
        dDec_rad = np.deg2rad(central_phase_center[1] - field_phase_center[1])  # to arcsec

        # Collect data from all SPWs in this (central) field
        for spw_name, spw_data in field_data.items():
            if spw_name == 'phase_center':
                continue

            # Apply phase shift
            shifted_data = self.phase_shift(spw_data.uvreal + 1j * spw_data.uvimag,
                                                  spw_data.uwave, spw_data.vwave,
                                                  dRA_rad, dDec_rad)

            # Calculate UV distance
            uvdist = np.sqrt(spw_data.uwave**2 + spw_data.vwave**2)

            all_uvreals.append(shifted_data.real)
            all_uvimags.append(shifted_data.imag)
            all_uvdist.append(uvdist)
            all_uvwghts.append(spw_data.suvwght)
        
        # Concatenate all data
        UVreals = np.concatenate(all_uvreals)
        UVimags = np.concatenate(all_uvimags)
        UVdist = np.concatenate(all_uvdist)
        UVwghts = np.concatenate(all_uvwghts)
        
        # Perform binning
        bins = np.linspace(0, len(UVdist[UVdist > 0]) - 1, nbins, dtype=int)
        UVdist_sorted = np.sort(UVdist[UVdist > 0])
            
        UVrealbinned = np.empty(nbins - 1)
        UVrealerrors = np.empty(nbins - 1)
        UVimagbinned = np.empty(nbins - 1)
        UVimagerrors = np.empty(nbins - 1)
        bin_cs = np.empty(nbins - 1)
            
        for i in range(len(bins) - 1):
            mask = (UVdist > UVdist_sorted[bins[i]]) & (UVdist <= UVdist_sorted[bins[i + 1]])        
            
            UVrealbinned[i], UVrealerrors[i] = np.average(UVreals[mask], weights=UVwghts[mask], returned=True) 
            UVimagbinned[i], UVimagerrors[i] = np.average(UVimags[mask], weights=UVwghts[mask], returned=True)

            UVrealerrors[i] = UVrealerrors[i]**(-0.5) 
            UVimagerrors[i] = UVimagerrors[i]**(-0.5) 
                
            bin_cs[i] = np.median(UVdist[mask])

        return UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, UVdist_sorted[bins], bin_cs

    def _get_or_compute_model_slice(self, model_name: str, data_name: str,
                                     field_key: str, spw_key: str,
                                     npts: int, r_min_k: float, r_max_k: float,
                                     axis: str = 'v', custom_phase_center = None):
        """Return (k_vals, real, imag) for a model's visibility slice.

        This implementation phase-shifts the entire uv-grid and then takes a
        slice, avoiding the interpolation of `sample_uv`.
        """
        # Determine phase center shift (in radians)
        field_phase_center = self.uvdata[data_name][field_key]['phase_center']
        if custom_phase_center is not None:
            central_phase_center_models = custom_phase_center
        else:
            central_phase_center_models = self.uvdata[data_name][field_key]['phase_center']

        # dRA/dDec in radians (difference central - field)

        print(model_name, data_name, central_phase_center_models, field_phase_center)

        dRA_rad = np.deg2rad(central_phase_center_models[0] - field_phase_center[0])
        dDec_rad = np.deg2rad(central_phase_center_models[1] - field_phase_center[1])

        # Access model map and grid parameter
        uv_entry = self.fft_map(model_name, data_name, field_key, spw_key)
        uv_grid = uv_entry['uv']
        du = uv_entry['du']

        # Build target k-values (k-lambda) and corresponding u/v sample coords (lambda)
        k_vals_final = np.logspace(np.log10(r_min_k), np.log10(r_max_k), npts)
        if axis.lower() == 'u':
            u_samples = k_vals_final * 1e3  # convert k-lambda -> lambda
            v_samples = np.zeros_like(u_samples)
        elif axis.lower() == 'v':
            v_samples = k_vals_final * 1e3
            u_samples = np.zeros_like(v_samples)
        else:
            raise ValueError("axis must be 'u' or 'v'")

        model_vis = self.sample_uv(uv_grid, u_samples, v_samples, du, dRA=0.0, dDec=0.0)
        shift_vis = self.phase_shift(model_vis.real + 1j * model_vis.imag,
                                     u_samples, v_samples,
                                     dRA_rad, dDec_rad)

        real_line = np.asarray(shift_vis).real
        imag_line = np.asarray(shift_vis).imag

        return k_vals_final, real_line, imag_line

    # ----------------- Plotting scripts ------------------------

    def _plot_single_radial_distribution(self, UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, 
                                       bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx,
                                       label_imag: bool = True, **kwargs):
        """
        Create a single radial distribution plot.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to matplotlib errorbar functions.
        """

        # Define colors for different datasets
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        color = colors[color_idx % len(colors)]

        # Convert bin edges to k-lambda units (assuming they're already in the right units)
        x = bin_centers / 1e3  # Convert to k-lambda if needed
        
        # Real part plot
        ax = axes[0]
        
        # Prepare errorbar kwargs, preserving required styling
        errorbar_kwargs = kwargs.copy()
        errorbar_kwargs.update({
            'ls': '',
            'c': color,
            'markeredgecolor': color,
            'markerfacecolor': 'white',
            'marker': 'D',
            'label': f'{name} (Real)',
            'alpha': errorbar_kwargs.get('alpha', 0.8)
        })
        
        # Handle markersize - use from kwargs if provided
        if 'markersize' not in errorbar_kwargs:
            errorbar_kwargs['markersize'] = 12
        
        ax.errorbar(x, UVrealbinned * 1e3, # Convert to mJy
                   xerr=[abs(bin_edges[:-1]/1e3 - x), abs(bin_edges[1:]/1e3 - x)], 
                   yerr=UVrealerrors * 1e3, 
                   **errorbar_kwargs)

        ax.axhline(0, ls='dashed', c='gray', alpha=0.7)
        ax.set_ylabel('Re(V) [mJy]')
        ax.set_xlabel('')
        
        # Add secondary axis for spatial scale (only once)
        if color_idx == 0:
            # add_secondary_uv_axis(ax)
            secax = ax.secondary_xaxis('top', functions=(arcsec_to_uvdist, uvdist_to_arcsec))
            secax.set_xlabel('Spatial scale ["]')
            ax.tick_params(axis='x', which='both', top=False)
        
        # Set axis limits
        ax.set_ylim(-2.1, 0.6)
        ax.set_xlim(1e0,1e1)

        ax.set_xscale('log')

        # Imaginary part plot
        ax = axes[1]
        
        # Prepare errorbar kwargs for imaginary part, preserving required styling
        errorbar_kwargs_imag = kwargs.copy()
        errorbar_kwargs_imag.update({
            'ls': '',
            'c': color,
            'markeredgecolor': color,
            'markerfacecolor': 'white',
            'marker': 'o',
            'label': f'{name} (Imag)' if label_imag else '__nolegend__',
            'alpha': errorbar_kwargs_imag.get('alpha', 0.8)
        })
        
        # Handle markersize - use from kwargs if provided
        if 'markersize' not in errorbar_kwargs_imag:
            errorbar_kwargs_imag['markersize'] = 10
        
        ax.errorbar(x, UVimagbinned * 1e3, # Convert to mJy
                   xerr=[abs(bin_edges[:-1]/1e3 - x), abs(bin_edges[1:]/1e3 - x)], 
                   yerr=UVimagerrors * 1e3, 
                   **errorbar_kwargs_imag)

        ax.axhline(0, ls='dashed', c='gray', alpha=0.7)
        ax.set_ylim(-0.4, 0.4)
        ax.set_xlim(1e0,1e1)

        ax.set_ylabel('Imag(V) [mJy]')
        ax.set_xlabel(r'uv distance [k$\lambda$]')
        ax.set_xscale('log')

        # Add target name if available (only once)
        if hasattr(self, 'target') and self.target and color_idx == 0:
            ax.text(0.03, 0.97, f'{self.target}', transform=axes[0].transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        return errorbar_kwargs.get('handle', ax)  # return axis for handle capture (we'll just return ax)

    # ----------------- Main plotting script ------------------------

    def plot_radial_distributions(self, nbins=20, save_plots=True, output_dir='../plots/uvplots/', 
                                custom_phase_center=None, use_style=True, data_name: str | None = None,
                                model_name: str | None = None,
                                n_model_pts: int = 500, r_min_k: float = 1, r_max_k: float = 20.0,
                                axis: str = 'v',
                                separate_legends: bool = True,
                                return_fig: bool = False,
                                **kwargs):
        """
        Plot radial UV distributions.

        If data_name is provided, only that dataset is plotted; otherwise all datasets.
        """
        # Setup plot style if requested
        if use_style:
            style_applied = setup_plot_style()
            
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Select datasets
        if data_name is not None:
            if data_name not in self.uvdata:
                raise ValueError(f"Dataset '{data_name}' not found. Available: {[k for k in self.uvdata if k!='metadata']}")
            dataset_names = [data_name]
        else:
            dataset_names = [name for name in self.uvdata if name != 'metadata']

        # Handle nbins as single value or list
        if isinstance(nbins, (int, float)):
            nbins_list = [int(nbins)] * len(dataset_names)
        elif isinstance(nbins, (list, tuple)):
            if len(nbins) != len(dataset_names):
                raise ValueError(f"Length of nbins list ({len(nbins)}) must match number of datasets ({len(dataset_names)})")
            nbins_list = [int(n) for n in nbins]
        else:
            raise ValueError("nbins must be an integer or a list of integers")

        fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})
        fig.set_figwidth(8)
        fig.set_figheight(6)

        # Data plotting -------------------------------------------------
        color_idx = 0
        data_handles = []
        for i, name in enumerate(dataset_names):
            current_nbins = nbins_list[i]
            UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_centers = self._get_binned_uvdatapoints(
                name, current_nbins, custom_phase_center
            )
            h_real = self._plot_single_radial_distribution(
                UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors,
                bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx,
                label_imag=False, **kwargs
            )
            data_handles.append(h_real)
            color_idx += 1

        # Model overlays -----------------------------------------------
        # This is not taking into account calibration parameter.
        model_linestyles_map = {}
        if hasattr(self, 'matched_models') and hasattr(self, 'fft_map') and hasattr(self, 'sample_uv'):
            base_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 1, 1, 1))]
            
            # Loop over datasets for model overlays (parallel to data plotting)
            for i, name in enumerate(dataset_names):
                central_field = self._find_central_field(name)
                dataset_color = f"C{i % 10}"

                # Collect and plot candidate models
                model_entries = []
                for m, mdat in getattr(self, 'matched_models', {}).items():
                    if name in mdat and central_field in mdat[name].get('maps', {}) and mdat[name]['maps'][central_field]:
                        if model_name is None or m == model_name:
                            first_spw = next(iter(mdat[name]['maps'][central_field].keys()))
                            model_entries.append((m, first_spw))

                if model_name is not None and not model_entries:
                    print(f"Model '{model_name}' not available for central field '{central_field}'; skipping model curves.")

                for mi, (mname, spw_used) in enumerate(model_entries):
                    # Assign linestyle if not already set
                    if mname not in model_linestyles_map:
                        style_idx = len(model_linestyles_map)
                        model_linestyles_map[mname] = base_linestyles[style_idx % len(base_linestyles)]
                    
                    # Get model data and plot
                    k_vals, real_line, imag_line = self._get_or_compute_model_slice(
                        mname, name, central_field, spw_used,
                        n_model_pts, r_min_k, r_max_k, axis,
                        custom_phase_center=custom_phase_center
                    )

                    axes[0].plot(k_vals, real_line * 1e3, lw=2.0, ls=model_linestyles_map[mname], c=dataset_color, label='__nolegend__')
                    axes[1].plot(k_vals, imag_line * 1e3, lw=2.0, ls=model_linestyles_map[mname], c=dataset_color, label='__nolegend__')
        else:
            if model_name is not None:
                print("Model plotting requested but required attributes (matched_models, fft_map, sample_uv) missing.")

        # Legends ------------------------------------------------------
        if separate_legends:
            data_labels = [f"{dn}" for dn in dataset_names]
            # Build proxy handles for real-part markers (match colors)
            proxy_handles = []
            for i, dn in enumerate(dataset_names):
                color = f"C{i % 10}"
                proxy_handles.append(Line2D([0], [0], ls='', marker='D', markerfacecolor='white',
                                            markeredgecolor=color, color=color, label=dn))
            leg_data = axes[0].legend(proxy_handles, data_labels, frameon=False, loc='lower right', fontsize=9, title='Data')
            if model_linestyles_map:
                model_legend_handles = [Line2D([0], [0], color='black', lw=2.0, linestyle=ls, label=mn)
                                        for mn, ls in model_linestyles_map.items()]
                leg_models = axes[0].legend(model_legend_handles,
                                            [h.get_label() for h in model_legend_handles],
                                            frameon=False, loc='lower left', fontsize=9, title='Models')
                # Preserve both legends on same axes
                axes[0].add_artist(leg_data)
        else:
            axes[0].legend(frameon=False, loc='lower right', fontsize=9)

        plt.tight_layout()

        if save_plots:
            filename = 'UVradial_data_combined.png' if data_name is None else f'UVradial_{data_name}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_dir}/{filename}")
        
        if return_fig:
            return fig, axes
        plt.show()