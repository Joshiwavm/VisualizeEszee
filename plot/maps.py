from __future__ import annotations

# Standard library
import os
import warnings
from typing import Optional, Tuple, Dict, Any

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astropy.wcs import WCS, FITSFixedWarning
from astropy.convolution import Gaussian2DKernel, convolve

# Local
from ..utils.utils import ytszToJyPix, JyBeamToJyPix
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
    def _freq_from_header(header) -> float:
        """Extract observing frequency in Hz from FITS header."""
        for key in ("RESTFRQ", "CRVAL3"):
            if key in header and header[key] not in (0, None):
                return header[key]
        raise ValueError("Frequency not found (need RESTFRQ or CRVAL3).")

    @staticmethod
    def _pixel_scale_deg(header) -> Tuple[float, float]:
        """Return absolute pixel scale (deg) along RA/Dec axes."""
        cd1 = header.get("CDELT1") or header.get("CD1_1")
        cd2 = header.get("CDELT2") or header.get("CD2_2")
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale not found (CDELT1/2 or CD*_*).")
        return abs(cd1), abs(cd2)

    @classmethod
    def _y_to_jy_per_beam(cls, y_map: np.ndarray, header, freq_hz: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert a Compton-y 2D map to Jy/beam (or Jy/pixel if beam absent)."""
        if freq_hz is None:
            freq_hz = cls._freq_from_header(header)
        ipix_deg, jpix_deg = cls._pixel_scale_deg(header)
        jy_pix = y_map * ytszToJyPix(freq_hz, ipix_deg, jpix_deg)
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        if not bmaj or not bmin:
            return jy_pix, {"unit": "Jy/pixel"}
        conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)
        jy_beam = jy_pix / conv
        return jy_beam, {"unit": "Jy/beam", "pix_per_beam_factor": 1 / conv}

    def _resolve_map_selection(self, model_name: Optional[str]):
        if not hasattr(self, "model_maps") or not self.model_maps:
            raise ValueError("No model maps available. Use match_model() to generate them.")
        model_name = model_name or next(iter(self.model_maps))
        if model_name not in self.model_maps:
            raise ValueError(f"Model name '{model_name}' not found. Available: {list(self.model_maps)}")
        model_entry = self.model_maps[model_name]
        if not model_entry:
            raise ValueError(f"Model '{model_name}' has no map entries. Call match_model().")
        dataset_name = next(iter(model_entry))
        ds_entry = model_entry[dataset_name]
        field_key = next((k for k in ds_entry if k.startswith('field')), None)
        if field_key is None:
            raise ValueError(f"No field entries for model '{model_name}' dataset '{dataset_name}'.")
        spw_dict = ds_entry[field_key]
        spw_key = next((k for k in spw_dict if k.startswith('spw')), None)
        if spw_key is None:
            raise ValueError(f"No spw entries for model '{model_name}' dataset '{dataset_name}' field '{field_key}'.")
        return spw_dict[spw_key], model_name, dataset_name

    @staticmethod
    def _safe_wcs(header, suppress: bool = True):
        """Create a WCS object suppressing Astropy FITSFixedWarning if requested."""
        if suppress:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                return WCS(header)[0, 0]
        return WCS(header)[0, 0]

    @staticmethod
    def _add_beam(ax, header, shape: Tuple[int, int], ipix_deg: float, jpix_deg: float, show: bool = True) -> None:
        """Overlay a simple ellipse showing the restoring beam (FWHM) in the lower-right corner."""
        if not show:
            return
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")

        w_pix = bmaj / ipix_deg * 2
        h_pix = bmin / jpix_deg * 2
        ny, nx = shape
        pad = 20
        
        cx = nx - pad - 0.5 * w_pix
        cy = pad + 0.5 * h_pix
        angle = header.get("BPA", 0.0)
        pix_transform = ax.get_transform('pixel') if hasattr(ax, 'get_transform') else ax.transData
        ax.add_patch(
            Ellipse((cx, cy), width=w_pix, height=h_pix, angle=angle,
                    facecolor="gray", edgecolor="gray", lw=0, alpha=0.3, zorder=9, transform=pix_transform)
        )
        ax.add_patch(
            Ellipse((cx, cy), width=w_pix, height=h_pix, angle=angle,
                    facecolor="none", edgecolor="black", lw=1.2, zorder=10, transform=pix_transform)
        )

    # ------------------------------------------------------------------
    # Internal helpers for field/spw selection & plotting
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_sel(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return [str(value)]

    def _collect_field_spw(self, ds_entry, fields_sel, spws_sel):
        """Return list of (fkey, skey, entry) respecting selections.

        Rules:
          - If neither field nor spw specified: return first field & its first spw.
          - If only field(s) specified: for each requested field return ONLY its first spw.
          - If only spw(s) specified: use first field, selecting requested spws (id match or index); unspecified -> ignore; if none valid -> first spw.
          - If both specified: for each requested field pair with requested spws; spw tokens interpreted as id OR zero-based index within that field's spw list.
          - If a requested field id does not exist, it is ignored; if all ignored -> fallback to first field & first spw.
          - If requested spws do not match any for a field, fallback to that field's first spw.
        """
        # Build mapping: field_id -> list of (spw_id, fkey, skey, entry)
        field_map = {}
        for fkey, spw_dict in ds_entry.items():
            if not fkey.startswith('field'):
                continue
            fid = fkey.replace('field', '')
            spw_list = []
            for skey, entry in spw_dict.items():
                if not skey.startswith('spw'):
                    continue
                sid = skey.replace('spw', '')
                spw_list.append((sid, fkey, skey, entry))
            if spw_list:
                # Preserve original ordering
                field_map[fid] = spw_list
        if not field_map:
            return []
        # Determine fields to use
        if fields_sel:
            chosen_fields = [fid for fid in fields_sel if fid in field_map]
            if not chosen_fields:
                chosen_fields = [next(iter(field_map))]
        else:
            chosen_fields = [next(iter(field_map))]
        results = []
        # Helper to interpret spw tokens for a given spw_list
        def resolve_spw_tokens(spw_tokens, spw_list):
            if not spw_tokens:
                return [spw_list[0]]  # default first
            chosen = []
            ids = [sid for sid, *_ in spw_list]
            for tok in spw_tokens:
                # direct id match
                if tok in ids:
                    idx = ids.index(tok)
                    chosen.append(spw_list[idx])
                    continue
                # index interpretation
                if tok.isdigit():
                    idx_int = int(tok)
                    if 0 <= idx_int < len(spw_list):
                        chosen.append(spw_list[idx_int])
                        continue
            if not chosen:
                chosen = [spw_list[0]]  # fallback
            return chosen
        for fid in chosen_fields:
            spw_list = field_map[fid]
            # Determine spws to use based on whether user supplied spws_sel and/or fields_sel
            if fields_sel and not spws_sel:
                # Only fields specified => default first spw per field
                chosen_spws = [spw_list[0]]
            elif spws_sel and not fields_sel:
                # Only spws specified => apply to first field only (already chosen_fields will just be first field)
                chosen_spws = resolve_spw_tokens(spws_sel, spw_list)
            else:
                # Both specified OR both unspecified -> resolve normally (unspecified case handled by tokens None)
                chosen_spws = resolve_spw_tokens(spws_sel, spw_list)
            for sid, fkey, skey, entry in chosen_spws:
                results.append((fkey, skey, entry))
        return results

    def _plot_single_y(self, m_name, dset, fkey, skey, entry, convert_to_jy_beam, freq_hz, cmap, save_plots, output_dir, filename, imshow_kwargs):
        header = entry['header']
        model_plane = self._extract_plane(entry['model_data'])
        data_arr = model_plane
        meta = None
        if convert_to_jy_beam:
            data_arr, meta = self._y_to_jy_per_beam(data_arr, header, freq_hz=freq_hz)
        wcs = self._safe_wcs(header)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={"projection": wcs})
        im = ax.imshow(data_arr, origin="lower", cmap=cmap, **imshow_kwargs)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
        cbar.set_label(meta['unit'] if convert_to_jy_beam else 'Compton y')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Dec (J2000)')
        ax.set_title(f"{m_name} [{dset}] {fkey}:{skey}")
        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                suffix = '_jybeam' if convert_to_jy_beam else ''
                outname = f"y_map_{m_name}_{dset}_{fkey}_{skey}{suffix}.png"
            else:
                outname = filename
            fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_single_model_image(self, m_name, dset, fkey, skey, entry, cmap, smooth, save_plots, output_dir, filename, imshow_kwargs):
        header = entry['header']
        model_plane_y = self._extract_plane(entry['model_data'])
        image_plane_jy = self._extract_plane(entry['image_data']) if 'image_data' in entry else None
        if image_plane_jy is None:
            warnings.warn(f"No image_data for '{m_name}' [{dset}] {fkey}:{skey} — skipping.")
            return
        freq_hz = self._freq_from_header(header)
        ipix_deg, jpix_deg = self._pixel_scale_deg(header)
        bmaj = header.get('BMAJ')
        bmin = header.get('BMIN')
        if not bmaj or not bmin:
            raise ValueError('Beam (BMAJ/BMIN) missing from header.')
        model_jy_pix = model_plane_y * ytszToJyPix(freq_hz, ipix_deg, jpix_deg)
        beam_conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)
        if smooth:
            fwhm_maj_pix = bmaj / ipix_deg
            fwhm_min_pix = bmin / jpix_deg
            sigma_maj_pix = fwhm_maj_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma_min_pix = fwhm_min_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            theta = np.deg2rad(header.get('BPA', 0.0))
            kernel = Gaussian2DKernel(x_stddev=sigma_maj_pix, y_stddev=sigma_min_pix, theta=theta)
            model_jy_pix = convolve(model_jy_pix, kernel, normalize_kernel=True)
        model_jy_beam = model_jy_pix / beam_conv
        model_mjy = model_jy_beam * 1e3
        image_mjy = image_plane_jy * 1e3
        vabs = np.nanmax(np.abs(image_mjy))
        vmin, vmax = -vabs, vabs
        wcs = self._safe_wcs(header)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': wcs}, gridspec_kw={'wspace': 0})
        im0 = axes[0].imshow(model_mjy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        im1 = axes[1].imshow(image_mjy, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        self._add_beam(axes[1], header, image_mjy.shape, ipix_deg, jpix_deg, show=True)
        self._add_beam(axes[0], header, model_mjy.shape, ipix_deg, jpix_deg, show=smooth)
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.05, fraction=0.046)
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.05, fraction=0.046)
        cbar0.set_label('mJy/beam' + (' (smoothed)' if smooth else ''))
        cbar1.set_label('mJy/beam')
        for ax in axes:
            ax.set_xlabel('Right Ascension (deg)')
            ax.coords[0].set_axislabel('RA (J2000)')
            ax.coords[1].set_axislabel('Dec (J2000)')
            ax.set_aspect('equal')
        axes[0].set_title(f"{m_name} [{dset}] {fkey}:{skey} model")
        axes[1].set_title(f"{m_name} [{dset}] {fkey}:{skey} image")
        axes[1].set_yticklabels([])
        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                suffix = '_smoothed' if smooth else ''
                outname = f"model_image_{m_name}_{dset}_{fkey}_{skey}_mJybeam{suffix}.png"
            else:
                outname = filename
            fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches='tight')
        plt.show()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_y_map(self, model_name: Optional[str] = None, data_name: Optional[str] = None,
                   convert_to_jy_beam: bool = False, freq_hz: Optional[float] = None,
                   cmap: str = "viridis", save_plots: bool = False,
                   output_dir: str = "../plots/maps/", use_style: bool = True,
                   filename: Optional[str] = None, field: Optional[Any] = None,
                   spw: Optional[Any] = None, **imshow_kwargs) -> None:
        """Plot Compton-y (optionally Jy/beam) maps.

        Field / SPW selection semantics:
          field=None, spw=None: first field & first spw
          field='1', spw=None: field 1 & its first spw
          field=None, spw='0': first field & first spw (index/id 0)
          field='1', spw='0': field 1 & its spw index/id 0
          Multiple values (lists) allowed; defaults per field to its first spw when spw not supplied.
        """
        if use_style:
            setup_plot_style()
        fields_sel = self._normalize_sel(field)
        spws_sel = self._normalize_sel(spw)
        # Build list of (model_name, data_name) pairs to plot
        pairs = []
        matched = getattr(self, 'matched_models', {})
        if model_name and data_name:
            pairs = [(model_name, data_name)]
        elif model_name:
            datas = list(matched.get(model_name, {}).keys())
            if not datas:
                raise ValueError(f"Model '{model_name}' has not been matched. Call match_model().")
            pairs = [(model_name, d) for d in datas]
        elif data_name:
            models = [m for m, ds in matched.items() if data_name in ds]
            if not models:
                raise ValueError(f"No models matched to data set '{data_name}'.")
            pairs = [(m, data_name) for m in models]
        else:
            for m, ds in matched.items():
                for d in ds.keys():
                    pairs.append((m, d))
            if not pairs:
                raise ValueError("No matched model/data pairs. Use match_model().")
        # Iterate over pairs
        for m_name, dset in pairs:
            if m_name not in self.model_maps or dset not in self.model_maps[m_name]:
                # Fallback selection
                entry, _, _ = self._resolve_map_selection(m_name)
                header = entry['header']
                model_plane = self._extract_plane(entry['model_data'])
                data_arr = model_plane
                meta = None
                if convert_to_jy_beam:
                    data_arr, meta = self._y_to_jy_per_beam(data_arr, header, freq_hz=freq_hz)
                wcs = self._safe_wcs(header)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={"projection": wcs})
                im = ax.imshow(data_arr, origin="lower", cmap=cmap, **imshow_kwargs)
                cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
                cbar.set_label(meta['unit'] if convert_to_jy_beam else 'Compton y')
                ax.set_xlabel('RA (J2000)')
                ax.set_ylabel('Dec (J2000)')
                ax.set_title(f"{m_name} [{dset}] (fallback)")
                plt.tight_layout()
                if save_plots:
                    os.makedirs(output_dir, exist_ok=True)
                    suffix = '_jybeam' if convert_to_jy_beam else ''
                    outname = filename or f"y_map_{m_name}_{dset}{suffix}.png"
                    fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches='tight')
                plt.show()
                continue
            ds_entry = self.model_maps[m_name][dset]
            selections = self._collect_field_spw(ds_entry, fields_sel, spws_sel)
            for fkey, skey, entry in selections:
                self._plot_single_y(m_name, dset, fkey, skey, entry, convert_to_jy_beam,
                                    freq_hz, cmap, save_plots, output_dir, filename, imshow_kwargs)

    def plot_model_image_comparison(self, model_name: Optional[str] = None, data_name: Optional[str] = None,
                                    cmap: str = 'RdBu_r', smooth: bool = True,
                                    save_plots: bool = False, output_dir: str = '../plots/maps/',
                                    use_style: bool = True, filename: Optional[str] = None,
                                    field: Optional[Any] = None, spw: Optional[Any] = None,
                                    **imshow_kwargs) -> None:
        """Plot model vs image panels.

        Selection semantics identical to plot_y_map (see its docstring)."""
        if use_style:
            setup_plot_style()
        fields_sel = self._normalize_sel(field)
        spws_sel = self._normalize_sel(spw)
        # Determine pairs
        pairs = []
        matched = getattr(self, 'matched_models', {})
        if model_name and data_name:
            pairs = [(model_name, data_name)]
        elif model_name:
            datas = list(matched.get(model_name, {}).keys())
            if not datas:
                raise ValueError(f"Model '{model_name}' has not been matched. Call match_model().")
            pairs = [(model_name, d) for d in datas]
        elif data_name:
            models = [m for m, ds in matched.items() if data_name in ds]
            if not models:
                raise ValueError(f"No models matched to data set '{data_name}'.")
            pairs = [(m, data_name) for m in models]
        else:
            for m, ds in matched.items():
                for d in ds.keys():
                    pairs.append((m, d))
            if not pairs:
                raise ValueError("No matched model/data pairs. Use match_model() first or specify model_name/data_name.")
        for m_name, dset in pairs:
            if m_name not in self.model_maps or dset not in self.model_maps[m_name]:
                entry, _, _ = self._resolve_map_selection(m_name)
                header = entry['header']
                # attempt fallback plot (single)
                ds_entry = { 'field0': { 'spw0': entry } }
            else:
                ds_entry = self.model_maps[m_name][dset]
            selections = self._collect_field_spw(ds_entry, fields_sel, spws_sel)
            for fkey, skey, entry in selections:
                self._plot_single_model_image(m_name, dset, fkey, skey, entry, cmap, smooth,
                                              save_plots, output_dir, filename, imshow_kwargs)