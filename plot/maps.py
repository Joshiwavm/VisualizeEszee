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
    # Public API
    # ------------------------------------------------------------------
    def plot_y_map(
        self,
        model_name: Optional[str] = None,
        data_name: Optional[str] = None,
        convert_to_jy_beam: bool = False,
        freq_hz: Optional[float] = None,
        cmap: str = "viridis",
        save_plots: bool = False,
        output_dir: str = "../plots/maps/",
        use_style: bool = True,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ) -> None:
        """Plot one or multiple model Compton-y maps.

        Supports iterating over matched model/data combinations established via
        match_model(). Selection logic:
          - model_name & data_name provided: plot that pair
          - model_name provided only: plot for all data sets matched to that model
          - data_name provided only: plot all models matched to that data set
          - neither provided: plot all matched model/data pairs
        """
        if use_style:
            setup_plot_style()
        # Build list of (model_name, data_name) pairs to plot
        pairs = []
        matched = getattr(self, '_matched_models', {})
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
            # Select entry: ensure we use specific dataset when available
            entry = None
            if m_name in self.model_maps and dset in self.model_maps[m_name]:
                # Use dataset-specific maps
                ds_entry = self.model_maps[m_name][dset]
                field_key = next((k for k in ds_entry if k.startswith('field')), None)
                if field_key is None:
                    continue
                spw_dict = ds_entry[field_key]
                spw_key = next((k for k in spw_dict if k.startswith('spw')), None)
                if spw_key is None:
                    continue
                entry = spw_dict[spw_key]
            else:
                # Fallback to first available
                entry, _, _ = self._resolve_map_selection(m_name)
                dset = list(self.model_maps[m_name].keys())[0]
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
            ax.set_title(f"{m_name} [{dset}]")
            plt.tight_layout()
            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                if filename is None:
                    suffix = '_jybeam' if convert_to_jy_beam else ''
                    outname = f"y_map_{m_name}_{dset}{suffix}.png"
                else:
                    outname = filename
                fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches='tight')
            plt.show()

    def plot_model_image_comparison(
        self,
        model_name: Optional[str] = None,
        data_name: Optional[str] = None,
        cmap: str = 'RdBu_r',
        smooth: bool = True,
        save_plots: bool = False,
        output_dir: str = '../plots/maps/',
        use_style: bool = True,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ) -> None:
        """Plot model vs image for one or multiple matched model/data pairs.

        Iteration rules identical to plot_y_map (see its docstring)."""
        if use_style:
            setup_plot_style()
        # Determine pairs
        pairs = []
        matched = getattr(self, '_matched_models', {})
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
            entry = None
            if m_name in self.model_maps and dset in self.model_maps[m_name]:
                ds_entry = self.model_maps[m_name][dset]
                field_key = next((k for k in ds_entry if k.startswith('field')), None)
                if field_key is None:
                    continue
                spw_dict = ds_entry[field_key]
                spw_key = next((k for k in spw_dict if k.startswith('spw')), None)
                if spw_key is None:
                    continue
                entry = spw_dict[spw_key]
            else:
                entry, _, _ = self._resolve_map_selection(m_name)
                dset = list(self.model_maps[m_name].keys())[0]
            header = entry['header']
            model_plane_y = self._extract_plane(entry['model_data'])
            image_plane_jy = self._extract_plane(entry['image_data']) if 'image_data' in entry else None
            if image_plane_jy is None:
                warnings.warn(f"No image_data for '{m_name}' [{dset}] — skipping.")
                continue
            # Convert model (Compton-y) to Jy/beam then optionally smooth
            freq_hz = self._freq_from_header(header)
            ipix_deg, jpix_deg = self._pixel_scale_deg(header)
            bmaj = header.get('BMAJ')
            bmin = header.get('BMIN')
            if not bmaj or not bmin:
                raise ValueError('Beam (BMAJ/BMIN) missing from header.')
            # y -> Jy/pixel
            model_jy_pix = model_plane_y * ytszToJyPix(freq_hz, ipix_deg, jpix_deg)
            # Jy/pixel -> Jy/beam factor
            beam_conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)  # Jy/beam -> Jy/pix
            if smooth:
                # Build Gaussian kernel in pixel units
                fwhm_maj_pix = bmaj / ipix_deg
                fwhm_min_pix = bmin / jpix_deg
                sigma_maj_pix = fwhm_maj_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                sigma_min_pix = fwhm_min_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                theta = np.deg2rad(header.get('BPA', 0.0))
                kernel = Gaussian2DKernel(x_stddev=sigma_maj_pix, y_stddev=sigma_min_pix, theta=theta)
                model_jy_pix = convolve(model_jy_pix, kernel, normalize_kernel=True)
            # Convert to Jy/beam
            model_jy_beam = model_jy_pix / beam_conv
            # Prepare image (assumed Jy/beam) and convert to mJy/beam for plotting
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
            axes[0].set_title(f"{m_name} [{dset}] model")
            axes[1].set_title(f"{m_name} [{dset}] image")
            axes[1].set_yticklabels([])
            plt.tight_layout()
            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                if filename is None:
                    suffix = '_smoothed' if smooth else ''
                    outname = f"model_image_{m_name}_{dset}_mJybeam{suffix}.png"
                else:
                    outname = filename
                fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches='tight')
            plt.show()