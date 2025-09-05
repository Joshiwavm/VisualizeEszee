from __future__ import annotations

# Standard library
import os
from typing import Optional, Tuple, Dict, Any, List

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve  # retained if future smoothing needed
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local
from ..utils.utils import JyBeamToJyPix, get_map_beam_and_pix, extract_plane
from ..utils.style import setup_plot_style

class PlotMaps:
    """Mixin that supplies map plotting methods."""

    # ------------------------------------------------------------------
    # Array / header utilities
    # ------------------------------------------------------------------
    # reuse shared helpers from plot.base: extract_plane, pixel_scale_deg

    # ------------------------------------------------------------------
    # Internal helpers for field/spw selection & plotting
    # ------------------------------------------------------------------
    # (Multi-field selection logic removed; legacy normalization helper dropped)

    # ------------------------------------------------------------------
    # Key resolution helper (single field/spw only now)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_key(keys: List[str], selection: Optional[Any], prefix: str) -> str:
        """Resolve a dictionary key from a user selection.

        Rules:
          - selection None  -> first key
          - int             -> index into keys (clamped)
          - exact match     -> itself
          - else try prefix+selection (e.g. 'field2'); fallback to first
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


    # ------------------------------------------------------------------
    # Unified map plotting (input / filtered / residual / data)
    # ------------------------------------------------------------------
    def plot_map(self, model_name: str, data_name: str, types, *, field=None, spw=None,
                 cmap: str = 'RdBu_r', save_plots: bool = False, return_fig: bool = False,
                 output_dir: str = '../plots/maps/', use_style: bool = True,
                 filename: Optional[str] = None,
                 aspect: str = 'equal', **imshow_kwargs):
        
        """Plot one or multiple map types in a single row.

        types: str or list of {'input','filtered','residual','data'}
            input    : Compton-y (primary-beam corrected) model map
            filtered : Dirty image from model visibilities (Jy/beam -> mJy/beam)
            residual : Dirty residual image (data - model) (mJy/beam)
            data     : CLEAN / image_data plane (mJy/beam)
        (WCS axis support removed – axes are simple pixel coordinates.)
        """

        def _add_beam(ax, header, shape):
            bmaj = header.get('BMAJ'); bmin = header.get('BMIN')
            if not bmaj or not bmin:
                return
            ipd, jpd = ipix_deg, jpix_deg
            w_pix = bmaj / ipd * 2
            h_pix = bmin / jpd * 2
            ny, nx = shape
            pad = 8
            cx = nx - pad - 0.5*w_pix
            cy = pad + 0.5*h_pix
            ax.add_patch(Ellipse((cx, cy), width=w_pix, height=h_pix, angle=header.get('BPA',0.0),
                                 facecolor='none', edgecolor='k', lw=1.0, alpha=0.8))
        
        if use_style:
            setup_plot_style()

        map_types = types if isinstance(types, (list, tuple)) else [types]
        allowed_types = {'input','filtered','residual','data','deconvolved'}
        invalid = [t for t in map_types if t not in allowed_types]
        if invalid:
            raise ValueError(f"Unsupported map type(s): {invalid}. Allowed types: {sorted(allowed_types)}")
        # Access top-level matched_models entry for potential 'deconvolved' image
        # Allow `data_name` to be a list/tuple (concatenated storage key used by JvM_clean)
        
        if isinstance(data_name, (list, tuple)):
            concat_dn = "+".join(data_name)
            mm_entry = self.matched_models[model_name].get(concat_dn, {})
        else:
            mm_entry = self.matched_models[model_name].get(data_name, {})

        # Try to populate per-field/spw variables only if maps exist for this dataset
        # If user passed a list/tuple for data_name, use the first dataset for
        # per-field/spw selection (deconvolved maps are looked up via concatenated key above).
        fkey = ''
        skey = ''
        if isinstance(data_name, (list, tuple)):
            first_dn = data_name[0] if data_name else None
            ds_entry = self.model_maps.get(model_name, {}).get(first_dn)
        else:
            ds_entry = self.model_maps.get(model_name, {}).get(data_name)
            
        header = {}
        ipix_deg = None
        jpix_deg = None
        y_map = None
        image_plane_jy = None
        dirty_model = None
        dirty_resid = None

        if ds_entry is not None:
            # pick a field/spw via helper
            fkey = self._resolve_key(list(ds_entry.keys()), field, prefix='field')
            spw_dict = ds_entry[fkey]
            skey = self._resolve_key(list(spw_dict.keys()), spw, prefix='spw')
            entry = spw_dict[skey]

            # Proceed with single field/spw plotting (restructured & robust)
            header = entry.get('header', {})
            bmaj, bmin, ipix_deg, jpix_deg = get_map_beam_and_pix(header)

            y_map = extract_plane(entry['model_data'])  # y * PB
            image_plane_jy = extract_plane(entry.get('image_data')) if entry.get('image_data') is not None else None

            want_filtered = any(t in ('filtered','residual') for t in map_types)
            dirty_model = entry.get('dirty_model')
            dirty_resid = entry.get('dirty_resid')
            if want_filtered and (dirty_model is None or dirty_resid is None):
                sm_entry = self.matched_models[model_name][data_name]['sampled_model'][fkey][skey]
                u = sm_entry['u']; v = sm_entry['v']; w = sm_entry['weights']
                model_vis = sm_entry['model_vis']; resid_vis = sm_entry['resid_vis']
                plane_pb = extract_plane(entry['model_data'])  # ensure defined
                npix = plane_pb.shape[0]
                pix_deg = abs(header.get('CDELT1') or header.get('CD1_1'))
                dirty_model = self.vis_to_image(u, v, model_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)
                dirty_resid = self.vis_to_image(u, v, resid_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)
                entry['dirty_model'] = dirty_model
                entry['dirty_resid'] = dirty_resid
        
        panels = [] 
        for t in map_types:
            if t == 'input' and y_map is not None:
                panels.append(('input', extract_plane(y_map)/1e-4, 'y (1e-4)'))
            elif t == 'filtered' and dirty_model is not None:
                panels.append(('filtered', extract_plane(dirty_model)*1e3, 'mJy/beam'))
            elif t == 'residual' and dirty_resid is not None:
                panels.append(('residual', extract_plane(dirty_resid)*1e3, 'mJy/beam'))
            elif t == 'data' and image_plane_jy is not None:
                panels.append(('data', extract_plane(image_plane_jy)*1e3, 'mJy/beam'))
            elif t == 'deconvolved':
                # If mm_entry is empty (e.g. data_name given as list), try concatenated key
                deconv = mm_entry.get('deconvolved')
                if deconv is None and isinstance(data_name, (list, tuple)):
                    concat_dn = "+".join(data_name)
                    deconv = self.matched_models[model_name].get(concat_dn, {}).get('deconvolved')
                if deconv is None:
                    raise ValueError(f"No deconvolved image found for {model_name}/{data_name}")
                panels.append(('deconvolved', extract_plane(deconv), 'Jy/beam'))
        
        # Collect arrays for all non-'input' panels (include 'deconvolved' here)
        amp_arrays = [arr for name, arr, unit in panels if name != 'input']
        if amp_arrays:
            vmin = min(np.nanmin(a) for a in amp_arrays) * 0.9
            vmax = -vmin
        
        n = len(panels)
        fig = plt.figure(figsize=(4*n, 4))
        axes = []
        
        for i_panel in range(n):
            ax = fig.add_subplot(1, n, i_panel+1)
            axes.append(ax)
        
        for idx_col, (ax, (name, arr, unit)) in enumerate(zip(axes, panels)):
            im_kwargs = dict(origin='lower', cmap=cmap)
            # apply symmetric vmin/vmax to every non-'input' panel
            if name != 'input' and vmin is not None:
                im_kwargs.update(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, **im_kwargs, **imshow_kwargs)
            ax.set_title(f"{name}")
            # Aspect management (try/except in case backend/version mismatch)
            try:
                ax.set_aspect(aspect)
            except Exception:
                pass
            # Axis tick handling: remove ALL y ticks & ticklabels (user preference)
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.tick_params(left=False)
            # Remove all x-axis ticks/labels (user preference) & top/right ticks
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.tick_params(bottom=False, top=False, right=False)
            if unit.endswith('Jy/beam') or unit.endswith('mJy/beam'):
                _add_beam(ax, header, arr.shape)
            # Colorbar handling (single approach)
            div = make_axes_locatable(ax)
            cax = div.append_axes('bottom', size='5%', pad=0.1)
            cb = plt.colorbar(im, cax=cax, orientation='horizontal')
            cb.set_label(unit)
            
            # Hard-disable y-axis ticks/labels for horizontal colorbars
            # try:
            cb.ax.set_ylabel('')
            cb.ax.get_yaxis().set_visible(False)
            cb.ax.set_yticks([])
            cb.ax.yaxis.set_ticklabels([])
            cb.ax.tick_params(axis='y', length=0)

            mid = len(axes)//2
            ax.set_ylabel('')
            ax.set_xlabel('')
                        
        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            out = filename or f"maps_{model_name}_{data_name}_{fkey}_{skey}_{'_'.join(map_types)}.png"
            fig.savefig(os.path.join(output_dir, out), dpi=300, bbox_inches='tight')
        if return_fig:
            return fig, axes

