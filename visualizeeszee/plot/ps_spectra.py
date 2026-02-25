import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from scipy.interpolate import RectBivariateSpline

from ..utils.style import setup_plot_style
from ..utils.utils import get_map_beam_and_pix, extract_plane, circle_mask


class PlotPointSourceSpectra:

    def _make_spw_dirty_map(self, model_name, data_name, spw_key):
        sampled = self.matched_models.get(model_name, {}).get(data_name, {}).get('sampled_model', {})
        maps_mm = self.matched_models.get(model_name, {}).get(data_name, {}).get('maps', {})

        pixel_scale_deg = npix = None
        for fk in sampled:
            pb_entry = maps_mm.get(fk, {}).get(spw_key, {})
            hdr = pb_entry.get('header', {})
            if hdr:
                _, _, pix_i, _ = get_map_beam_and_pix(hdr)
                pixel_scale_deg = abs(float(pix_i))
                md = pb_entry.get('model_data')
                if md is not None:
                    npix = extract_plane(md).shape[0]
                break

        if pixel_scale_deg is None or npix is None:
            return None, None, None, None, None

        field_phases = {
            fk: self.uvdata[data_name].get(fk, {}).get('phase_center', (0.0, 0.0))
            for fk, spw_map in sampled.items() if spw_key in spw_map
        }
        if not field_phases:
            return None, None, None, None, None

        center_ra  = float(np.mean([pc[0] for pc in field_phases.values()]))
        center_dec = float(np.mean([pc[1] for pc in field_phases.values()]))

        field_weights = {fk: float(np.sum(sampled[fk][spw_key]['weights'])) for fk in field_phases}

        field_dicts = [
            {
                'u': sampled[fk][spw_key]['u'],
                'v': sampled[fk][spw_key]['v'],
                'vis': sampled[fk][spw_key]['resid_vis'],
                'weights': sampled[fk][spw_key]['weights'],
                'dRA':  np.deg2rad(center_ra - pc[0]) * np.cos(np.deg2rad(0.5 * (center_ra + pc[0]))),
                'dDec': np.deg2rad(center_dec - pc[1]),
            }
            for fk, pc in field_phases.items()
        ]
        dirty_map = self._multivis_fields_to_image(field_dicts, npix=npix, pixel_scale_deg=pixel_scale_deg)
        return dirty_map, center_ra, center_dec, pixel_scale_deg, field_weights

    def _make_joint_dirty_map(self, model_name, data_name):
        """Build a single dirty map from residual visibilities of all SPWs combined."""
        sampled = self.matched_models.get(model_name, {}).get(data_name, {}).get('sampled_model', {})
        maps_mm = self.matched_models.get(model_name, {}).get(data_name, {}).get('maps', {})

        pixel_scale_deg = npix = None
        for fk in sampled:
            for spw_key in sampled[fk]:
                pb_entry = maps_mm.get(fk, {}).get(spw_key, {})
                hdr = pb_entry.get('header', {})
                if hdr:
                    _, _, pix_i, _ = get_map_beam_and_pix(hdr)
                    pixel_scale_deg = abs(float(pix_i))
                    md = pb_entry.get('model_data')
                    if md is not None:
                        npix = extract_plane(md).shape[0]
                    break
            if pixel_scale_deg is not None:
                break

        if pixel_scale_deg is None or npix is None:
            return None, None, None, None

        field_phases = {
            fk: self.uvdata[data_name].get(fk, {}).get('phase_center', (0.0, 0.0))
            for fk in sampled if sampled[fk]
        }
        if not field_phases:
            return None, None, None, None

        center_ra  = float(np.mean([pc[0] for pc in field_phases.values()]))
        center_dec = float(np.mean([pc[1] for pc in field_phases.values()]))

        field_dicts = []
        for fk, pc in field_phases.items():
            u_all   = np.concatenate([sampled[fk][s]['u']         for s in sampled[fk]])
            v_all   = np.concatenate([sampled[fk][s]['v']         for s in sampled[fk]])
            vis_all = np.concatenate([sampled[fk][s]['resid_vis'] for s in sampled[fk]])
            wt_all  = np.concatenate([sampled[fk][s]['weights']   for s in sampled[fk]])
            field_dicts.append({
                'u': u_all, 'v': v_all, 'vis': vis_all, 'weights': wt_all,
                'dRA':  np.deg2rad(center_ra - pc[0]) * np.cos(np.deg2rad(0.5 * (center_ra + pc[0]))),
                'dDec': np.deg2rad(center_dec - pc[1]),
            })

        dirty_map = self._multivis_fields_to_image(field_dicts, npix=npix, pixel_scale_deg=pixel_scale_deg)
        return dirty_map, center_ra, center_dec, pixel_scale_deg

    @staticmethod
    def _ps_pixel_position(ps_ra, ps_dec, center_ra, center_dec, pixel_scale_deg, npix):
        dRA_pix  = (ps_ra  - center_ra)  * np.cos(np.deg2rad(center_dec)) / pixel_scale_deg
        dDec_pix = (ps_dec - center_dec) / pixel_scale_deg
        col = int(round(npix / 2 - dRA_pix))   # RA increases to left
        row = int(round(npix / 2 + dDec_pix))  # Dec increases upward in array
        return row, col

    def _pb_at_ps_position(self, model_name, data_name, spw_key, ps, field_weights):
        maps_mm = self.matched_models.get(model_name, {}).get(data_name, {}).get('maps', {})
        total_w = sum(field_weights.values()) or 1.0
        pb_sum = 0.0
        for fk, wsum in field_weights.items():
            pb_entry = maps_mm.get(fk, {}).get(spw_key, {})
            pb_val = 1.0
            if 'pbeam_data' in pb_entry and 'header' in pb_entry:
                hdr  = pb_entry['header']
                pb2d = np.nan_to_num(np.squeeze(pb_entry['pbeam_data']), nan=0.0)
                ny, nx = pb2d.shape
                spl  = RectBivariateSpline(np.linspace(0, ny, ny), np.linspace(0, nx, nx), pb2d, kx=1, ky=1, s=0)
                xs   = hdr['CRPIX1'] + (ps['ra']  - hdr['CRVAL1']) * np.cos(np.deg2rad(hdr['CRVAL2'])) / hdr['CDELT1']
                ys   = hdr['CRPIX2'] + (ps['dec'] - hdr['CRVAL2']) / hdr['CDELT2']
                pb_val = float(spl.ev(ys, xs))
            pb_sum += pb_val * wsum
        return pb_sum / total_w

    def _measure_ps_flux_image_plane(self, model_name, data_name, ps_list,
                                     aperture_arcsec=15.0):
        sampled  = self.matched_models.get(model_name, {}).get(data_name, {}).get('sampled_model', {})
        all_spws = sorted({spw for spw_map in sampled.values() for spw in spw_map})
        results  = [[] for _ in ps_list]

        for spw_key in all_spws:
            dirty_map, center_ra, center_dec, pixel_scale_deg, field_weights = \
                self._make_spw_dirty_map(model_name, data_name, spw_key)
            if dirty_map is None:
                continue

            npix         = dirty_map.shape[0]
            aperture_pix = aperture_arcsec / 3600.0 / pixel_scale_deg

            rep_freq_ghz = next(
                (float(np.mean(spw_map[spw_key].get('uvfreq', [1e11]))) / 1e9
                 for fk, spw_map in sampled.items() if spw_key in spw_map),
                None,
            )
            if rep_freq_ghz is None:
                continue

            ps_positions = [
                self._ps_pixel_position(ps['ra'], ps['dec'], center_ra, center_dec, pixel_scale_deg, npix)
                for ps in ps_list
            ]
            ps_masks = [circle_mask(dirty_map, col, row, aperture_pix) for row, col in ps_positions]

            combined_mask = np.zeros((npix, npix), dtype=bool)
            for m in ps_masks:
                combined_mask |= m
            noise = float(np.std(dirty_map[~combined_mask]))

            for pi, (ps, mask) in enumerate(zip(ps_list, ps_masks)):
                if not np.any(mask):
                    continue
                peak_apparent   = float(np.max(dirty_map[mask]))
                pb_val          = self._pb_at_ps_position(model_name, data_name, spw_key, ps, field_weights)
                flux_intrinsic  = peak_apparent / pb_val if pb_val > 0 else np.nan
                noise_intrinsic = noise         / pb_val if pb_val > 0 else np.nan
                results[pi].append((rep_freq_ghz, flux_intrinsic, noise_intrinsic))

        for r in results:
            r.sort(key=lambda x: x[0])
        return results

    @staticmethod
    def _compute_ps_model_spectrum(ps, freq_hz):
        freq_hz = np.asarray(freq_hz, dtype=float)
        if ps.get('spec_type', 'powerLaw') == 'doublePowerLaw':
            return (ps.get('amp1', 0.0) * (freq_hz / ps.get('ref_freq',  1e11)) ** ps.get('spec_index',  0.0)
                  + ps.get('amp2', 0.0) * (freq_hz / ps.get('ref_freq2', 4e10)) ** ps.get('spec_index2', 0.0))
        return ps.get('amplitude', 0.0) * (freq_hz / ps.get('ref_freq', 1e11)) ** ps.get('spec_index', 0.0)

    def _plot_extraction_maps(self, ps_list, data_names, model_name, aperture_arcsec):
        """Joint dirty map per dataset with all PS aperture circles overlaid.

        Layout: 1 row × len(data_names) columns. All SPWs are combined into a single
        multi-frequency synthesis map per dataset to maximise continuum sensitivity.

        Parameters
        ----------
        ps_list : list of dict
        data_names : list of str
        model_name : str
        aperture_arcsec : float
            Radius of the search aperture drawn around each PS.
        """
        n_cols = len(data_names)
        if n_cols == 0:
            return

        ps_colors = [f'C{i}' for i in range(len(ps_list))]

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4.5 * n_cols, 4.5),
            squeeze=False,
        )

        for ci, dn in enumerate(data_names):
            ax = axes[0, ci]
            dirty_map, center_ra, center_dec, pixel_scale_deg = \
                self._make_joint_dirty_map(model_name, dn)

            if dirty_map is None:
                ax.set_visible(False)
                continue

            pixel_scale_arcsec = pixel_scale_deg * 3600.0
            npix     = dirty_map.shape[0]
            half_fov = 0.5 * npix * pixel_scale_arcsec

            # Extent: RA increases to the left (east), Dec increases upward (north)
            extent = [half_fov, -half_fov, -half_fov, half_fov]

            vmax = float(np.nanmax(np.abs(dirty_map))) or 1.0
            im = ax.imshow(
                dirty_map, origin='lower', cmap='RdBu_r',
                vmin=-vmax, vmax=vmax,
                extent=extent,
            )

            for pi, ps in enumerate(ps_list):
                row, col = self._ps_pixel_position(
                    ps['ra'], ps['dec'], center_ra, center_dec, pixel_scale_deg, npix)
                x_arcsec = (npix / 2 - col) * pixel_scale_arcsec  # positive = east
                y_arcsec = (row - npix / 2) * pixel_scale_arcsec  # positive = north
                ax.add_patch(mpatches.Circle(
                    (x_arcsec, y_arcsec), radius=aperture_arcsec,
                    fill=False, edgecolor=ps_colors[pi], linewidth=1.2, zorder=5,
                ))

            ax.set_title(dn, fontsize=8)
            ax.set_xlabel('ΔRA ["]', fontsize=6)
            ax.set_ylabel('ΔDec ["]', fontsize=6)
            ax.tick_params(labelsize=5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Jy/bm', fontsize=5)

        # PS colour legend in the first subplot
        if ps_list:
            legend_handles = [
                mpatches.Patch(facecolor='none', edgecolor=f'C{i}', label=f'PS {i + 1}')
                for i in range(len(ps_list))
            ]
            axes[0, 0].legend(handles=legend_handles, fontsize=6, frameon=True,
                              loc='upper right', framealpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_point_source_spectra(self, ps_list, data_names, model_name,
                                  aperture_arcsec=10.0, save_plots=False,
                                  output_dir='../plots/ps_spectra/',
                                  use_style=True, return_fig=False,
                                  n_model_pts=300, log_log=True,
                                  plot_maps=False,
                                  **kwargs):
        if use_style:
            setup_plot_style()

        if isinstance(data_names, str):
            data_names = [data_names]

        results_by_dn = {}
        for dn in data_names:
            results_by_dn[dn] = self._measure_ps_flux_image_plane(
                model_name, dn, ps_list, aperture_arcsec)

        fig, axes = plt.subplots(1, len(ps_list), figsize=(4.5 * len(ps_list), 4.0), squeeze=False)

        for pi, ps in enumerate(ps_list):
            ax = axes[0, pi]

            all_freqs, all_fluxes, all_noises, all_labels = [], [], [], []
            for dn, per_ps in results_by_dn.items():
                for freq_ghz, flux, noise in per_ps[pi]:
                    all_freqs.append(freq_ghz); all_fluxes.append(flux)
                    all_noises.append(noise);   all_labels.append(dn)

            if not all_freqs:
                ax.set_title(f'PS {pi + 1}: no data')
                continue

            color_map = {dn: f'C{i}' for i, dn in enumerate(dict.fromkeys(all_labels))}
            plotted   = set()
            for freq_ghz, flux, noise, dn in zip(all_freqs, all_fluxes, all_noises, all_labels):
                if log_log and flux <= 0:
                    continue
                lbl = dn if dn not in plotted else '__nolegend__'
                plotted.add(dn)
                ax.errorbar(freq_ghz, flux * 1e3, yerr=noise * 1e3,
                            fmt='o', color=color_map[dn], label=lbl,
                            markersize=6, capsize=3, elinewidth=1.0)

            f_min, f_max = min(all_freqs) * 0.8, max(all_freqs) * 1.2
            freq_model_ghz = (np.logspace(np.log10(f_min), np.log10(f_max), n_model_pts)
                              if log_log else np.linspace(f_min, f_max, n_model_pts))
            ax.plot(freq_model_ghz, self._compute_ps_model_spectrum(ps, freq_model_ghz * 1e9) * 1e3,
                    'k-', lw=1.5, label='Model', zorder=5)

            if log_log:
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_ylim(1e-3, 1e0)
            else:
                ax.axhline(0, ls='--', c='gray', alpha=0.5, lw=0.8)
            ax.set_xlabel('Frequency [GHz]')
            ax.set_ylabel('Flux [mJy]')
            ax.set_title(f'PS {pi + 1}  ({ps["ra"]:.4f}°, {ps["dec"]:.4f}°)')
            ax.legend(fontsize=8, frameon=False)

        plt.tight_layout()

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, 'ps_spectra.png')
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved: {fname}")

        if plot_maps:
            self._plot_extraction_maps(ps_list, data_names, model_name, aperture_arcsec)

        if return_fig:
            return fig, axes
        plt.show()
