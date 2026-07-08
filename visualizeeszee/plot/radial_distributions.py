import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

from ..utils.style import setup_plot_style
from ..utils.utils import arcsec_to_uvdist, uvdist_to_arcsec


def _group_quantile_models(model_entries):
    """Group (model_name, spw) entries by base name and quantile.

    Returns {base_name: {q_float_or_None: (full_name, spw)}}.
    Entries without a _q{val} suffix are stored under key None.
    """
    groups = {}
    for mname, spw in model_entries:
        m = re.search(r'_q([\d.]+)$', mname)
        if m:
            base = mname[:m.start()]
            q = float(m.group(1))
        else:
            base = mname
            q = None
        groups.setdefault(base, {})[q] = (mname, spw)
    return groups


class PlotRadialDistributions:

    # ----------------- Compute raidal uvprofiles ------------------------

    def _get_binned_uvdatapoints(self, band_name, nbins=20, custom_phase_center=None, log_bins=False):
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
        central_field = self._find_central_field(band_name)
        if custom_phase_center is not None:
            # Flatten list/array elements to scalars (take first if multi-component)
            ra_c  = float(np.atleast_1d(custom_phase_center[0])[0])
            dec_c = float(np.atleast_1d(custom_phase_center[1])[0])
            central_phase_center = (ra_c, dec_c)
        else:
            print(f"Using central field: {central_field}")
            central_phase_center = self.uvdata[band_name][central_field]['phase_center']

        # Collect phase-shifted data only for the central field
        all_uvreals = []
        all_uvimags = []
        all_uvdist = []
        all_uvwghts = []

        field_data = self.uvdata[band_name][central_field]

        # Calculate phase shift needed to move this field to the chosen central phase center
        field_phase_center = field_data['phase_center']
        dRA_rad =  np.deg2rad(central_phase_center[0] - field_phase_center[0]) *\
                   np.cos(np.deg2rad(0.5 * (central_phase_center[1] + field_phase_center[1])))
        dDec_rad = np.deg2rad(central_phase_center[1] - field_phase_center[1])

        # Collect data from all SPWs in this (central) field
        for spw_name, spw_data in field_data.items():
            if spw_name == 'phase_center':
                continue

            raw_vis = spw_data.uvreal + 1j * spw_data.uvimag

            # Apply phase shift
            shifted_data = self.phase_shift(raw_vis,
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
        
        # Build bin edges
        UVdist_pos = UVdist[UVdist > 0]
        if log_bins:
            bin_edges = np.logspace(np.log10(UVdist_pos.min()), np.log10(UVdist_pos.max()), nbins)
        else:
            UVdist_sorted = np.sort(UVdist_pos)
            idx = np.linspace(0, len(UVdist_sorted) - 1, nbins, dtype=int)
            bin_edges = UVdist_sorted[idx]

        n_bins = nbins - 1
        UVrealbinned = np.full(n_bins, np.nan)
        UVrealerrors = np.full(n_bins, np.nan)
        UVimagbinned = np.full(n_bins, np.nan)
        UVimagerrors = np.full(n_bins, np.nan)
        bin_cs = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (UVdist > bin_edges[i]) & (UVdist <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            UVrealbinned[i], sum_w_re = np.average(UVreals[mask], weights=UVwghts[mask], returned=True)
            UVimagbinned[i], sum_w_im = np.average(UVimags[mask], weights=UVwghts[mask], returned=True)
            # σ = 1/sqrt(Σw_i) for inverse-variance weighted mean
            UVrealerrors[i] = sum_w_re**(-0.5)
            UVimagerrors[i] = sum_w_im**(-0.5)
            bin_cs[i] = np.median(UVdist[mask])

        return UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_cs

    def _get_or_compute_model_slice(self, model_name: str, data_name: str,
                                     field_key: str, spw_key: str,
                                     npts: int, r_min_k: float, r_max_k: float,
                                     axis: str = 'v', custom_phase_center=None):
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
        dRA_rad = np.deg2rad(central_phase_center_models[0] - field_phase_center[0]) *\
                  np.cos(np.deg2rad(0.5 * (central_phase_center_models[1] + field_phase_center[1])))
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

        # Apply calibration scale stored by sample_fft for this model/dataset/field/spw
        calib = (self.matched_models.get(model_name, {})
                 .get(data_name, {})
                 .get('sampled_model', {})
                 .get(field_key, {})
                 .get(spw_key, {})
                 .get('calib', 1.0))
        model_vis = model_vis * calib

        shift_vis = self.phase_shift(model_vis.real + 1j * model_vis.imag,
                                     u_samples, v_samples,
                                     dRA_rad, dDec_rad)

        real_line = np.asarray(shift_vis).real
        imag_line = np.asarray(shift_vis).imag

        return k_vals_final, real_line, imag_line

    def _get_binned_residuals(self, band_name, model_name, nbins=20,
                              custom_phase_center=None, log_bins=False):
        """Bin (data − model) residual visibilities radially for the central field.

        Uses sampled_model[field][spw]['resid_vis'] which is calibration-aware
        (the same residuals fed into JvM_clean).
        """
        model_name = self._resolve_model_name(model_name)
        central_field = self._find_central_field(band_name)
        if custom_phase_center is not None:
            ra_c  = float(np.atleast_1d(custom_phase_center[0])[0])
            dec_c = float(np.atleast_1d(custom_phase_center[1])[0])
            central_phase_center = (ra_c, dec_c)
        else:
            central_phase_center = self.uvdata[band_name][central_field]['phase_center']

        sampled = (self.matched_models
                   .get(model_name, {})
                   .get(band_name, {})
                   .get('sampled_model', {}))
        if not sampled or central_field not in sampled:
            raise ValueError(
                f"No sampled_model for {model_name}/{band_name}/{central_field}. "
                "Run match_model() first."
            )

        field_phase_center = self.uvdata[band_name][central_field]['phase_center']
        dRA_rad  = (np.deg2rad(central_phase_center[0] - field_phase_center[0])
                    * np.cos(np.deg2rad(0.5 * (central_phase_center[1] + field_phase_center[1]))))
        dDec_rad = np.deg2rad(central_phase_center[1] - field_phase_center[1])

        all_re, all_im, all_dist, all_w = [], [], [], []
        for spw_name, spw_s in sampled[central_field].items():
            resid = np.asarray(spw_s.get('resid_vis', []))
            u     = np.asarray(spw_s.get('u', []))
            v     = np.asarray(spw_s.get('v', []))
            w     = np.asarray(spw_s.get('weights', []))
            if resid.size == 0:
                continue
            shifted = self.phase_shift(resid, u, v, dRA_rad, dDec_rad)
            uvdist  = np.sqrt(u**2 + v**2)
            all_re.append(shifted.real);  all_im.append(shifted.imag)
            all_dist.append(uvdist);      all_w.append(w)

        UVreals  = np.concatenate(all_re);   UVimags = np.concatenate(all_im)
        UVdist   = np.concatenate(all_dist); UVwghts = np.concatenate(all_w)

        UVdist_pos = UVdist[UVdist > 0]
        if log_bins:
            bin_edges = np.logspace(np.log10(UVdist_pos.min()), np.log10(UVdist_pos.max()), nbins)
        else:
            UVdist_sorted = np.sort(UVdist_pos)
            idx = np.linspace(0, len(UVdist_sorted) - 1, nbins, dtype=int)
            bin_edges = UVdist_sorted[idx]

        n_bins = nbins - 1
        UVrealbinned = np.full(n_bins, np.nan); UVrealerrors = np.full(n_bins, np.nan)
        UVimagbinned = np.full(n_bins, np.nan); UVimagerrors = np.full(n_bins, np.nan)
        bin_cs = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = (UVdist > bin_edges[i]) & (UVdist <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            UVrealbinned[i], sw_re = np.average(UVreals[mask], weights=UVwghts[mask], returned=True)
            UVimagbinned[i], sw_im = np.average(UVimags[mask], weights=UVwghts[mask], returned=True)
            UVrealerrors[i] = sw_re**(-0.5)
            UVimagerrors[i] = sw_im**(-0.5)
            bin_cs[i] = np.median(UVdist[mask])

        return UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_cs

    # ----------------- Plotting scripts ------------------------

    def _plot_single_radial_distribution(self, UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors,
                                       bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx,
                                       label_imag: bool = True, show_label: bool = True,
                                       ylim_real=(-4, 0.6), ylim_imag=(-0.4, 0.4), **kwargs):
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
        ax.set_ylabel('Re(V) [mJy]', fontsize=9)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=9)

        # Add secondary axis for spatial scale (only once)
        if color_idx == 0:
            secax = ax.secondary_xaxis('top', functions=(arcsec_to_uvdist, uvdist_to_arcsec))
            secax.set_xlabel('Spatial scale ["]', fontsize=9)
            secax.tick_params(labelsize=9)
            ax.tick_params(axis='x', which='both', top=False)

        # Set axis limits
        if ylim_real is not None:
            ax.set_ylim(*ylim_real)
        ax.set_xlim(1e0, 2e1)

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
        if ylim_imag is not None:
            ax.set_ylim(*ylim_imag)
        ax.set_xlim(1e0, 2e1)

        ax.set_ylabel('Imag(V) [mJy]', fontsize=9)
        ax.set_xlabel(r'uv-distance [k$\lambda$]', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.set_xscale('log')

        # Add target name if available (only once)
        if show_label and hasattr(self, 'target') and self.target and color_idx == 0:
            ax.text(0.03, 0.97, f'{self.target}', transform=axes[0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        return errorbar_kwargs.get('handle', ax)  # return axis for handle capture (we'll just return ax)

    # ----------------- Main plotting script ------------------------

    def plot_radial_distributions(self, nbins=20, save_plots=True, output_dir=None,
                                custom_phase_center=None, use_style=True, data_name: str | None = None,
                                model_name: str | list[str] | None = None,
                                n_model_pts: int = 1500, r_min_k: float = 0.1, r_max_k: float = 30.0,
                                axis: str = 'v',
                                separate_legends: bool = True,
                                show_legends: bool = True,
                                legend_layout: str | None = None,
                                return_fig: bool = False,
                                show_label: bool = True,
                                log_bins: bool = False,
                                residual_model: str | None = None,
                                show_bands: bool = True,
                                data_zorder: dict | None = None,
                                ylim_real: tuple | None = None,
                                ylim_imag: tuple | None = None,
                                **kwargs):
        """
        Plot radial UV distributions.

        If data_name is provided, only that dataset is plotted; otherwise all datasets.

        Parameters
        ----------
        residual_model : str or None
            When set, plot (data − model) residuals in uv-space instead of raw data.
            The named model is subtracted from the data using the calibration-aware
            residual visibilities already stored by match_model().
            All other model curves are also shown relative to this reference
            (each curve − reference_curve), so the reference model appears as a
            zero line and deviations highlight differences between models.
        """
        if model_name is not None:
            if isinstance(model_name, str):
                model_name = [self._resolve_model_name(model_name)]
            else:
                model_name = [self._resolve_model_name(n) for n in model_name]
            # Also include base names so quantile-suffixed keys match
            _model_name_set = set(model_name) | {re.sub(r'_q[\d.]+$', '', n) for n in model_name}
        else:
            _model_name_set = None
        if residual_model is not None:
            residual_model = self._resolve_model_name(residual_model)

        # Setup plot style if requested
        if use_style:
            style_applied = setup_plot_style()

        if save_plots:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/uvplots/'
            os.makedirs(output_dir, exist_ok=True)

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

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 5),
                                 gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})

        ylim_real_arg = ylim_real if ylim_real is not None else ((-4, 0.6) if residual_model is None else (-1.3, 1))
        ylim_imag_arg = ylim_imag if ylim_imag is not None else ((-0.4, 0.4) if residual_model is None else None)

        # Data plotting -------------------------------------------------
        color_idx = 0
        data_handles = []
        for i, name in enumerate(dataset_names):
            current_nbins = nbins_list[i]
            if residual_model is not None:
                UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_centers = \
                    self._get_binned_residuals(name, residual_model, current_nbins,
                                              custom_phase_center, log_bins=log_bins)
            else:
                UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_centers = \
                    self._get_binned_uvdatapoints(name, current_nbins, custom_phase_center, log_bins=log_bins)
            _zorder = (data_zorder.get(name, i + 2) if data_zorder is not None else i + 2)
            h_real = self._plot_single_radial_distribution(
                UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors,
                bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx,
                label_imag=False, show_label=show_label,
                ylim_real=ylim_real_arg, ylim_imag=ylim_imag_arg,
                zorder=_zorder, **kwargs
            )
            data_handles.append(h_real)
            color_idx += 1

        # Update y-labels to show residual context
        if residual_model is not None:
            # In math mode plain spaces collapse, so use '~' to keep a visible gap.
            _ref_short = (residual_model.split('_q')[0] if '_q' in residual_model else residual_model).replace('_', '~')
            axes[0].set_ylabel(fr'Re(V $-$ V$_{{\rm {_ref_short}}}$) [mJy]', fontsize=9)
            axes[1].set_ylabel(fr'Im(V $-$ V$_{{\rm {_ref_short}}}$) [mJy]', fontsize=9)

        # Model overlays -----------------------------------------------
        model_linestyles_map = {}
        model_hatches_map = {}
        if hasattr(self, 'matched_models') and hasattr(self, 'fft_map') and hasattr(self, 'sample_uv'):
            base_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 1, 1, 1))]
            base_hatches    = ['/', '\\', '|', '-', '+', 'x', 'o', '.']

            for i, name in enumerate(dataset_names):
                central_field = self._find_central_field(name)
                dataset_color = f"C{i % 10}"

                # Collect candidate models; match by full name or base name
                model_entries = []
                for m, mdat in getattr(self, 'matched_models', {}).items():
                    if name in mdat and central_field in mdat[name].get('maps', {}) and mdat[name]['maps'][central_field]:
                        _base_m = re.sub(r'_q[\d.]+$', '', m)
                        if _model_name_set is None or m in _model_name_set or _base_m in _model_name_set:
                            first_spw = next(iter(mdat[name]['maps'][central_field].keys()))
                            model_entries.append((m, first_spw))

                if _model_name_set is not None and not model_entries:
                    print(f"Models {list(_model_name_set)} not available for central field '{central_field}'; skipping model curves.")

                # Pre-compute reference q50 slice for residual mode
                ref_real = ref_imag = None
                if residual_model is not None:
                    _ref_maps = (self.matched_models.get(residual_model, {})
                                 .get(name, {}).get('maps', {}))
                    if central_field in _ref_maps and _ref_maps[central_field]:
                        _ref_spw = next(iter(_ref_maps[central_field].keys()))
                        _, ref_real, ref_imag = self._get_or_compute_model_slice(
                            residual_model, name, central_field, _ref_spw,
                            n_model_pts, r_min_k, r_max_k, axis,
                            custom_phase_center=custom_phase_center,
                        )
                    else:
                        print(f"residual_model '{residual_model}' not found for {name}/{central_field}; "
                              "showing absolute model curves.")

                # Pre-compute reference q16/q84 slices for error propagation
                ref_slices = {}
                if residual_model is not None and ref_real is not None:
                    _ref_base = re.sub(r'_q[\d.]+$', '', residual_model)
                    for _q_cand in [0.16, 0.84]:
                        _ref_qname = f'{_ref_base}_q{_q_cand}'
                        _rqmaps = (self.matched_models.get(_ref_qname, {})
                                   .get(name, {}).get('maps', {}))
                        if central_field in _rqmaps and _rqmaps[central_field]:
                            _rqspw = next(iter(_rqmaps[central_field].keys()))
                            _, _rr, _ri = self._get_or_compute_model_slice(
                                _ref_qname, name, central_field, _rqspw,
                                n_model_pts, r_min_k, r_max_k, axis,
                                custom_phase_center=custom_phase_center,
                            )
                            ref_slices[_q_cand] = (_rr, _ri)
                    ref_slices[0.5] = (ref_real, ref_imag)

                # Model zorder always below the lowest data zorder
                _model_zorder = (min(data_zorder.values()) - 1 if data_zorder else 0)

                # Group entries by base model name and plot
                grouped = _group_quantile_models(model_entries)
                _ref_base_name = re.sub(r'_q[\d.]+$', '', residual_model) if residual_model else None

                for base_name, q_dict in grouped.items():
                    # Assign linestyle and hatch per base name
                    if base_name not in model_linestyles_map:
                        idx = len(model_linestyles_map)
                        model_linestyles_map[base_name] = base_linestyles[idx % len(base_linestyles)]
                        model_hatches_map[base_name]    = base_hatches[idx % len(base_hatches)]
                    ls    = model_linestyles_map[base_name]
                    hatch = model_hatches_map[base_name]

                    # Identify median quantile key
                    _q_keys = [q for q in q_dict if q is not None]
                    q_med = min(_q_keys, key=lambda q: abs(q - 0.5)) if _q_keys else None
                    mname_med, spw_med = q_dict[q_med] if q_med is not None else list(q_dict.values())[0]

                    # Compute median slice
                    k_vals, real_50, imag_50 = self._get_or_compute_model_slice(
                        mname_med, name, central_field, spw_med,
                        n_model_pts, r_min_k, r_max_k, axis,
                        custom_phase_center=custom_phase_center,
                    )

                    # Subtract reference (residual mode)
                    if residual_model is not None and ref_real is not None:
                        real_plot = real_50 - ref_real
                        imag_plot = imag_50 - ref_imag
                    else:
                        real_plot = real_50
                        imag_plot = imag_50

                    axes[0].plot(k_vals, real_plot * 1e3, lw=1.5, ls=ls, c=dataset_color, label='__nolegend__', zorder=_model_zorder)
                    axes[1].plot(k_vals, imag_plot * 1e3, lw=1.5, ls=ls, c=dataset_color, label='__nolegend__', zorder=_model_zorder)

                    # Quantile band (requires both q16 and q84)
                    if 0.16 in q_dict and 0.84 in q_dict:
                        _, real_16, imag_16 = self._get_or_compute_model_slice(
                            q_dict[0.16][0], name, central_field, q_dict[0.16][1],
                            n_model_pts, r_min_k, r_max_k, axis,
                            custom_phase_center=custom_phase_center,
                        )
                        _, real_84, imag_84 = self._get_or_compute_model_slice(
                            q_dict[0.84][0], name, central_field, q_dict[0.84][1],
                            n_model_pts, r_min_k, r_max_k, axis,
                            custom_phase_center=custom_phase_center,
                        )

                        _fb_kw = dict(alpha=0.15, facecolor=dataset_color,
                                      edgecolor=dataset_color, hatch=hatch, linewidth=0.5,
                                      zorder=_model_zorder)
                        if show_bands:
                            if residual_model is None or ref_real is None:
                                # Absolute mode: direct quantile band
                                axes[0].fill_between(k_vals, real_16 * 1e3, real_84 * 1e3, **_fb_kw)
                                axes[1].fill_between(k_vals, imag_16 * 1e3, imag_84 * 1e3, **_fb_kw)
                            else:
                                # Residual mode: shift each model's q16/q84 by ref q50 only.
                                # No quadrature error propagation — reduces scatter.
                                axes[0].fill_between(k_vals,
                                                     (real_16 - ref_real) * 1e3,
                                                     (real_84 - ref_real) * 1e3,
                                                     **_fb_kw)
                                axes[1].fill_between(k_vals,
                                                     (imag_16 - ref_imag) * 1e3,
                                                     (imag_84 - ref_imag) * 1e3,
                                                     **_fb_kw)
        else:
            if model_name is not None:
                print("Model plotting requested but required attributes (matched_models, fft_map, sample_uv) missing.")

        # Legends ------------------------------------------------------
        # legend_layout overrides separate_legends when provided:
        #   'split'         -> data lower-right, models lower-left (default behaviour)
        #   'stacked_right' -> both lower-right, models stacked above data
        #   'none'          -> no legends
        if legend_layout is None:
            legend_layout = 'split' if separate_legends else 'single'
        if not show_legends:
            legend_layout = 'none'

        if legend_layout == 'none':
            pass
        elif legend_layout in ('split', 'stacked_right'):
            data_labels = [dn.replace('_', ' ') for dn in dataset_names]
            # Build proxy handles for real-part markers (match colors)
            proxy_handles = []
            for i, dn in enumerate(dataset_names):
                color = f"C{i % 10}"
                proxy_handles.append(Line2D([0], [0], ls='', marker='D', markerfacecolor='white',
                                            markeredgecolor=color, color=color, label=dn.replace('_', ' ')))
            model_legend_handles = [Line2D([0], [0], color='black', lw=1.5, linestyle=ls, label=mn.replace('_', ' '))
                                    for mn, ls in model_linestyles_map.items()] if model_linestyles_map else []
            if legend_layout == 'stacked_right' and model_legend_handles:
                # Models at lower-right; Data stacked above it.
                # Right-align entries + titles (markers after labels, flush right).
                leg_models = axes[0].legend(model_legend_handles,
                                            [h.get_label() for h in model_legend_handles],
                                            frameon=False, loc='lower right',
                                            fontsize=7, title='Models', title_fontsize=9,
                                            alignment='right', markerfirst=False)
                fig.canvas.draw()
                inv = axes[0].transAxes.inverted()
                y_top = leg_models.get_window_extent().transformed(inv).y1
                leg_data = axes[0].legend(proxy_handles, data_labels, frameon=False,
                                          loc='lower right', bbox_to_anchor=(1.0, y_top),
                                          fontsize=7, title='Data', title_fontsize=9,
                                          alignment='right', markerfirst=False)
                axes[0].add_artist(leg_models)
            else:
                leg_data = axes[0].legend(proxy_handles, data_labels, frameon=False, loc='lower right', fontsize=7, title='Data', title_fontsize=9)
                if model_legend_handles:
                    leg_models = axes[0].legend(model_legend_handles,
                                                [h.get_label() for h in model_legend_handles],
                                                frameon=False, loc='lower left', fontsize=7, title='Models', title_fontsize=9)
                    # Preserve both legends on same axes
                    axes[0].add_artist(leg_data)
        else:
            axes[0].legend(frameon=False, loc='lower right', fontsize=7)

        plt.tight_layout()

        if save_plots:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            _prefix = f"{_safe_target}_" if getattr(self, 'target', None) else ''
            _resid_suffix = '_residual' if residual_model is not None else ''
            _base = 'UVradial_data_combined' if data_name is None else f'UVradial_{data_name}'
            filename = _prefix + _base + _resid_suffix + '.png'
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {out_path}")
        
        if return_fig:
            return fig, axes
        plt.show()