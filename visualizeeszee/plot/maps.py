from __future__ import annotations

import os
from typing import Optional, Tuple, List, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils.utils import get_map_beam_and_pix, extract_plane
from ..utils.style import setup_plot_style


class PlotMaps:
    """Mixin that supplies map plotting methods."""

    @staticmethod
    def _resolve_key(keys: List[str], selection: Optional[Any], prefix: str) -> str:
        """Resolve a dictionary key from a user selection.

        - None       → first key
        - int        → index into keys (clamped to valid range)
        - str match  → itself; otherwise try prefix+selection, fallback to first
        """
        if not keys:
            raise KeyError("No keys available while resolving selection.")
        if selection is None:
            return keys[0]
        if isinstance(selection, int):
            return keys[selection] if 0 <= selection < len(keys) else keys[0]
        if selection in keys:
            return selection
        guess = f"{prefix}{selection}"
        return guess if guess in keys else keys[0]

    def plot_map(self, model_name: str, data_name: str, types, *, field=None, spw=None,
                 cmap: str = 'RdBu_r', save_plots: bool = False, return_fig: bool = False,
                 output_dir: str = '../plots/maps/', use_style: bool = True,
                 filename: Optional[str] = None, taper: float | None = None,
                 aspect: str = 'equal',
                 fov: float | None = None,
                 center: Tuple[float, float] | None = None,
                 **imshow_kwargs):
        """Plot one or multiple map types in a single row.

        Parameters
        ----------
        types : str or list of {'input', 'filtered', 'residual', 'data', 'deconvolved', 'joint_residual'}
            input           : Compton-y model map (primary-beam corrected)
            filtered        : Dirty image from model visibilities (mJy/beam)
            residual        : Dirty residual image, data - model, single field+SPW (mJy/beam)
            data            : CASA CLEAN image plane (mJy/beam)
            deconvolved     : JvM-cleaned image (Jy/beam)
            joint_residual  : MFS dirty residual combining all fields and SPWs (mJy/beam)
        fov : float or None
            Crop each panel to this square field of view in arcsec.
        center : (ra_deg, dec_deg) or None
            Sky coordinate at the centre of the cropped view.
            Defaults to the image centre (CRVAL1/2) when fov is set.
        """

        def _add_beam(ax, header):
            bmaj = header.get('BMAJ')
            bmin = header.get('BMIN')
            if not bmaj or not bmin:
                return
            w_pix = bmaj / ipix_deg * 2
            h_pix = bmin / jpix_deg * 2
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            pad = max(4, 0.05 * abs(xlim[1] - xlim[0]))
            cx = xlim[1] - pad - 0.5 * w_pix
            cy = ylim[0] + pad + 0.5 * h_pix
            ax.add_patch(Ellipse((cx, cy), width=w_pix, height=h_pix,
                                 angle=header.get('BPA', 0.0),
                                 facecolor='none', edgecolor='k', lw=1.0, alpha=0.8))

        def _fov_limits(arr, header):
            fov_pix = fov / 3600.0 / abs(ipix_deg)
            if center is not None and header:
                crpix1 = header.get('CRPIX1', arr.shape[1] / 2.0 + 1)
                crpix2 = header.get('CRPIX2', arr.shape[0] / 2.0 + 1)
                crval1 = header.get('CRVAL1', 0.0)
                crval2 = header.get('CRVAL2', 0.0)
                cdelt1 = header.get('CDELT1', -abs(ipix_deg))
                cdelt2 = header.get('CDELT2',  abs(ipix_deg))
                x_cen = (crpix1 - 1) + (center[0] - crval1) / cdelt1
                y_cen = (crpix2 - 1) + (center[1] - crval2) / cdelt2
            else:
                x_cen = arr.shape[1] / 2.0
                y_cen = arr.shape[0] / 2.0
            return (x_cen - fov_pix / 2, x_cen + fov_pix / 2,
                    y_cen - fov_pix / 2, y_cen + fov_pix / 2)

        if use_style:
            setup_plot_style()

        map_types = types if isinstance(types, (list, tuple)) else [types]
        allowed_types = {'input', 'filtered', 'residual', 'data', 'deconvolved', 'joint_residual'}
        invalid = [t for t in map_types if t not in allowed_types]
        if invalid:
            raise ValueError(f"Unsupported map type(s): {invalid}. Allowed: {sorted(allowed_types)}")

        # Resolve matched_models entry (concatenated key for list data_name)
        if isinstance(data_name, (list, tuple)):
            concat_dn = "+".join(data_name)
            mm_entry = self.matched_models[model_name].get(concat_dn, {})
            first_dn = data_name[0] if data_name else None
            ds_entry = self.model_maps.get(model_name, {}).get(first_dn)
        else:
            mm_entry = self.matched_models[model_name].get(data_name, {})
            ds_entry = self.model_maps.get(model_name, {}).get(data_name)

        # Defaults; populated when ds_entry is available
        fkey = skey = ''
        header = {}
        ipix_deg = jpix_deg = None
        y_map = image_plane_jy = dirty_model = dirty_resid = None

        # For deconvolved images built from multiple datasets, prefer the header stored
        # by JvM_clean (smallest-beam pixel scale) over the first dataset's header.
        if 'header' in mm_entry:
            header = mm_entry['header']
            _, _, ipix_deg, jpix_deg = get_map_beam_and_pix(header)

        if ds_entry is not None:
            fkey = self._resolve_key(list(ds_entry.keys()), field, prefix='field')
            skey = self._resolve_key(list(ds_entry[fkey].keys()), spw, prefix='spw')
            entry = ds_entry[fkey][skey]

            if not header:  # only use ds_entry header if JvM_clean didn't provide one
                header = entry.get('header', {})
                _, _, ipix_deg, jpix_deg = get_map_beam_and_pix(header)
            y_map = extract_plane(entry['model_data'])
            if entry.get('image_data') is not None:
                image_plane_jy = extract_plane(entry['image_data'])

            if any(t in ('filtered', 'residual') for t in map_types):
                dirty_model = entry.get('dirty_model')
                dirty_resid = entry.get('dirty_resid')
                if dirty_model is None or dirty_resid is None:
                    sm = self.matched_models[model_name][data_name]['sampled_model'][fkey][skey]
                    npix = y_map.shape[0]
                    pix_deg = abs(header.get('CDELT1') or header.get('CD1_1'))
                    dirty_model = self.vis_to_image(
                        sm['u'], sm['v'], sm['model_vis'], weights=sm['weights'],
                        npix=npix, pixel_scale_deg=pix_deg, normalize=True, taper=taper)
                    dirty_resid = self.vis_to_image(
                        sm['u'], sm['v'], sm['resid_vis'], weights=sm['weights'],
                        npix=npix, pixel_scale_deg=pix_deg, normalize=True, taper=taper)
                    entry['dirty_model'] = dirty_model
                    entry['dirty_resid'] = dirty_resid

        # Joint residual: combine all fields and SPWs into a single MFS dirty map
        joint_resid_map = None
        if 'joint_residual' in map_types:
            dn_str = data_name[0] if isinstance(data_name, (list, tuple)) else data_name
            jdm, jcra, jcdec, jpix = self._make_joint_dirty_map(model_name, dn_str)
            if jdm is not None:
                joint_resid_map = jdm
                npix_j = jdm.shape[0]
                joint_header = {
                    'CRPIX1': npix_j / 2 + 1,
                    'CRPIX2': npix_j / 2 + 1,
                    'CRVAL1': jcra,
                    'CRVAL2': jcdec,
                    'CDELT1': -jpix,
                    'CDELT2':  jpix,
                }
                if ipix_deg is None:
                    ipix_deg = jpix_deg = jpix
                if not header:
                    header = joint_header

        # Build panel list
        panels = []
        for t in map_types:
            if t == 'input' and y_map is not None:
                panels.append(('input', y_map / 1e-4, 'y (1e-4)'))
            elif t == 'filtered' and dirty_model is not None:
                panels.append(('filtered', extract_plane(dirty_model) * 1e3, 'mJy/beam'))
            elif t == 'residual' and dirty_resid is not None:
                panels.append(('residual', extract_plane(dirty_resid) * 1e3, 'mJy/beam'))
            elif t == 'data' and image_plane_jy is not None:
                panels.append(('data', image_plane_jy * 1e3, 'mJy/beam'))
            elif t == 'deconvolved':
                deconv = mm_entry.get('deconvolved')
                if deconv is None:
                    raise ValueError(f"No deconvolved image found for {model_name}/{data_name}")
                panels.append(('deconvolved', extract_plane(deconv), 'Jy/beam'))
            elif t == 'joint_residual' and joint_resid_map is not None:
                panels.append(('joint_residual', extract_plane(joint_resid_map) * 1e3, 'mJy/beam'))

        # Shared symmetric colour scale for all non-input panels
        amp_arrays = [arr for name, arr, _ in panels if name != 'input']
        vmin = vmax = None
        if amp_arrays:
            vmin = min(np.nanmin(a) for a in amp_arrays) * 0.9
            vmax = -vmin

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        axes = axes[0]

        for ax, (name, arr, unit) in zip(axes, panels):
            im_kwargs = dict(origin='lower', cmap=cmap)
            if name != 'input' and vmin is not None:
                im_kwargs.update(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, **im_kwargs, **imshow_kwargs)
            ax.set_title(name)
            try:
                ax.set_aspect(aspect)
            except Exception:
                pass

            if fov is not None and ipix_deg is not None:
                x0, x1, y0, y1 = _fov_limits(arr, header)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, top=False, right=False)

            if unit.endswith('Jy/beam') or unit.endswith('mJy/beam'):
                _add_beam(ax, header)

            div = make_axes_locatable(ax)
            cax = div.append_axes('bottom', size='5%', pad=0.1)
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.set_label(unit)
            cb.ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            out = filename or f"maps_{model_name}_{data_name}_{fkey}_{skey}_{'_'.join(map_types)}.png"
            fig.savefig(os.path.join(output_dir, out), dpi=300, bbox_inches='tight')
        if return_fig:
            return fig, axes
