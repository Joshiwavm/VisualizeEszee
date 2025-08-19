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
from ..utils.utils import JyBeamToJyPix
from ..utils.style import setup_plot_style


class PlotMaps:
    """Mixin that supplies map plotting methods."""

    # ------------------------------------------------------------------
    # Array / header utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_plane(arr: np.ndarray) -> np.ndarray:
        """Return a 2D spatial plane from model/image array of shape:
        (npol,nchan,ny,nx) or (nchan,ny,nx) or already (ny,nx).
        Currently selects the first polarization / channel.
        """
        a = np.asarray(arr)
        if a.ndim == 4:
            return a[0, 0]
        if a.ndim == 3:
            return a[0]
        return a

    @staticmethod
    def _pixel_scale_deg(header) -> Tuple[float, float]:
        """Return absolute pixel scale (deg) along RA/Dec axes."""
        cd1 = header.get("CDELT1") or header.get("CD1_1")
        cd2 = header.get("CDELT2") or header.get("CD2_2")
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale not found (CDELT1/2 or CD*_*).")
        return abs(cd1), abs(cd2)

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
                 cmap: str = 'RdBu_r', save_plots: bool = False,
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
        allowed_types = {'input','filtered','residual','data'}
        invalid = [t for t in map_types if t not in allowed_types]
        if invalid:
            raise ValueError(f"Unsupported map type(s): {invalid}. Allowed types: {sorted(allowed_types)}")
        
        # Locate dataset entry and pick single field/spw via helper
        ds_entry = self.model_maps[model_name][data_name]
        fkey = self._resolve_key(list(ds_entry.keys()), field, prefix='field')
        spw_dict = ds_entry[fkey]
        skey = self._resolve_key(list(spw_dict.keys()), spw, prefix='spw')
        entry = spw_dict[skey]
        
        # Proceed with single field/spw plotting (restructured & robust)
        header = entry['header']
        ipix_deg, jpix_deg = self._pixel_scale_deg(header)
        pb_map_full = entry.get('pb_map')

        plane_pb = self._extract_plane(entry['model_data'])  # y * PB
        pb_plane = self._extract_plane(pb_map_full)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_map = np.where(pb_plane > 0, plane_pb / pb_plane, 0.0) if pb_plane is not None else plane_pb
        
        image_plane_jy = self._extract_plane(entry.get('image_data')) if entry.get('image_data') is not None else None
        want_filtered = any(t in ('filtered','residual') for t in map_types)
        dirty_model = dirty_resid = None
        if want_filtered:
            sm_entry = self.matched_models[model_name][data_name]['sampled_model'][fkey][skey]
            u = sm_entry['u']; v = sm_entry['v']; w = sm_entry['weights']
            model_vis = sm_entry['model_vis']; resid_vis = sm_entry['resid_vis']
            npix = plane_pb.shape[0]
            pix_deg = abs(header.get('CDELT1') or header.get('CD1_1'))
            dirty_model = self.vis_to_image(u, v, model_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)
            dirty_resid = self.vis_to_image(u, v, resid_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)
        
        bmaj = header.get('BMAJ'); bmin = header.get('BMIN')
        _ = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin) if (bmaj and bmin) else 1.0  # placeholder if needed later
        
        panels = [] 
        for t in map_types:
            if t == 'input':
                panels.append(('input', self._extract_plane(y_map)/1e-4, 'y (1e-4)'))
            elif t == 'filtered' and dirty_model is not None:
                panels.append(('filtered', self._extract_plane(dirty_model)*1e3, 'mJy/beam'))
            elif t == 'residual' and dirty_resid is not None:
                panels.append(('residual', self._extract_plane(dirty_resid)*1e3, 'mJy/beam'))
            elif t == 'data' and image_plane_jy is not None:
                panels.append(('data', self._extract_plane(image_plane_jy)*1e3, 'mJy/beam'))
        
        amp_arrays = [arr for name, arr, unit in panels if unit.startswith('mJy')]
        if amp_arrays:
            vmin = min(np.nanmin(a) for a in amp_arrays)
            vmax = -vmin
        
        n = len(panels)
        fig = plt.figure(figsize=(4*n, 4))
        axes = []
        
        for i_panel in range(n):
            ax = fig.add_subplot(1, n, i_panel+1)
            axes.append(ax)
        
        for idx_col, (ax, (name, arr, unit)) in enumerate(zip(axes, panels)):
            im_kwargs = dict(origin='lower', cmap=cmap)
            if unit.startswith('mJy') and vmin is not None:
                im_kwargs.update(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, **im_kwargs, **imshow_kwargs)
            ax.set_title(f"{name} {fkey}:{skey}")
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
        plt.show()

