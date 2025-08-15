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

    def _resolve_map_selection(self, name: Optional[str], field: Optional[str], spw: Optional[str]):
        """Resolve (name, field, spw) selection with sensible defaults and return (entry, name, field, spw)."""
        if not hasattr(self, "model_maps") or not self.model_maps:
            raise ValueError("No model maps available. Generate models first.")
        name = name or next(iter(self.model_maps))
        if name not in self.model_maps:
            raise ValueError(f"Model name '{name}' not found. Available: {list(self.model_maps)}")
        model_entry = self.model_maps[name]
        field = field or next(k for k in model_entry if k.startswith("field"))
        if field not in model_entry:
            raise ValueError(f"Field '{field}' not in model '{name}'. Available: {list(model_entry)}")
        field_entry = model_entry[field]
        spw = spw or next(k for k in field_entry if k.startswith("spw"))
        if spw not in field_entry:
            raise ValueError(f"SPW '{spw}' not in {name}/{field}. Available: {list(field_entry)}")
        return field_entry[spw], name, field, spw

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
        name: Optional[str] = None,
        field: Optional[str] = None,
        spw: Optional[str] = None,
        convert_to_jy_beam: bool = False,
        freq_hz: Optional[float] = None,
        cmap: str = "viridis",
        save_plots: bool = False,
        output_dir: str = "../plots/maps/",
        use_style: bool = True,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ) -> None:
        """Plot a single model Compton-y map (optionally converted to Jy/beam).

        Parameters mirror other plotters for consistency.
        """
        if use_style:
            setup_plot_style()
        d, name, field, spw = self._resolve_map_selection(name, field, spw)
        header = d["header"]
        model_plane = self._extract_plane(d["model_data"])
        data = model_plane
        meta = None
        if convert_to_jy_beam:
            data, meta = self._y_to_jy_per_beam(data, header, freq_hz=freq_hz)
        wcs = self._safe_wcs(header)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={"projection": wcs})
        im = ax.imshow(data, origin="lower", cmap=cmap, **imshow_kwargs)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
        cbar.set_label(meta["unit"] if convert_to_jy_beam else "Compton y")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
        ax.set_title(f"{name} {field} {spw}")
        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                suffix = "_jybeam" if convert_to_jy_beam else ""
                filename = f"y_map_{name}_{field}_{spw}{suffix}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.show()

    def plot_model_image_comparison(
        self,
        name: Optional[str] = None,
        field: Optional[str] = None,
        spw: Optional[str] = None,
        cmap: str = "RdBu_r",
        smooth: bool = True,
        save_plots: bool = False,
        output_dir: str = "../plots/maps/",
        use_style: bool = True,
        filename: Optional[str] = None,
        **imshow_kwargs,
    ) -> None:
        """Plot model vs image side-by-side in mJy/beam.

        The model is always converted from Compton-y. If ``smooth`` is True the model
        is convolved with the restoring beam described by (BMAJ, BMIN, BPA) before
        conversion to Jy/beam.
        """
        if use_style:
            setup_plot_style()
        d, name, field, spw = self._resolve_map_selection(name, field, spw)
        header = d["header"]
        model_plane = self._extract_plane(d["model_data"])
        image_plane = self._extract_plane(d["image_data"])

        # --- Model: smoothing & conversion ---
        if smooth:
            freq_hz = self._freq_from_header(header)
            ipix_deg, jpix_deg = self._pixel_scale_deg(header)
            bmaj = header.get("BMAJ")
            bmin = header.get("BMIN")
            if not bmaj or not bmin:
                raise ValueError("Cannot smooth: beam (BMAJ/BMIN) missing in header.")
            # y -> Jy/pixel
            jy_pix = model_plane * ytszToJyPix(freq_hz, ipix_deg, jpix_deg)
            # Beam in pixel units
            fwhm_maj_pix = bmaj / ipix_deg
            fwhm_min_pix = bmin / jpix_deg
            sigma_maj_pix = fwhm_maj_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma_min_pix = fwhm_min_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            theta = np.deg2rad(header.get("BPA", 0.0))
            kernel = Gaussian2DKernel(x_stddev=sigma_maj_pix, y_stddev=sigma_min_pix, theta=theta)
            jy_pix_smoothed = convolve(jy_pix, kernel, normalize_kernel=True)
            conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)
            model_jy_beam = jy_pix_smoothed / conv
        else:
            model_jy_beam, _ = self._y_to_jy_per_beam(model_plane, header, freq_hz=None)

        # Image assumed already Jy/beam; convert both to mJy/beam
        model_mjy = model_jy_beam * 1e3
        image_mjy = image_plane * 1e3

        # Symmetric color scale from image panel
        vmin = np.nanmin(image_mjy)
        vmax = -vmin

        ipix_deg, jpix_deg = self._pixel_scale_deg(header)
        wcs = self._safe_wcs(header)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": wcs}, gridspec_kw={"wspace": 0})

        im0 = axes[0].imshow(model_mjy, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        im1 = axes[1].imshow(image_mjy, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)

        # Beam overlays (always right; left only if smoothed)
        self._add_beam(axes[1], header, image_mjy.shape, ipix_deg, jpix_deg, show=True)
        self._add_beam(axes[0], header, model_mjy.shape, ipix_deg, jpix_deg, show=smooth)

        cbar0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.05, fraction=0.046)
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.05, fraction=0.046)
        cbar0.set_label("mJy/beam" + (" (smoothed)" if smooth else ""))
        cbar1.set_label("mJy/beam")

        for ax in axes:
            ax.set_xlabel("Right Ascension (deg)")
            ax.coords[0].set_axislabel("RA (J2000)")
            ax.coords[1].set_axislabel("Dec (J2000)")
            ax.set_aspect("equal")
            ax.set_title("")
        axes[0].set_title(f"{name} {field} {spw}")
        axes[1].set_yticklabels([])

        plt.tight_layout()
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                suffix = "_smoothed" if smooth else ""
                filename = f"model_image_{name}_{field}_{spw}_mJybeam{suffix}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.show()