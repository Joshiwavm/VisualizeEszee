"""Map plotting utilities as a mixin class.

Provides:
- plot_y_map: single-panel Compton-y (or converted Jy/beam) model map.
- plot_model_image_comparison: two-panel model (y or Jy/beam) vs image data.
"""
from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from astropy.wcs import WCS
from ..utils.utils import ytszToJyPix

class PlotMaps:
    # ----------------- internal helpers -----------------
    @staticmethod
    def _extract_plane(arr):
        arr = np.asarray(arr)
        if arr.ndim == 4:
            return arr[0,0]
        if arr.ndim == 3:
            return arr[0]
        return arr

    @staticmethod
    def _freq_from_header(header):
        for k in ('RESTFRQ','CRVAL3'):
            if k in header and header[k] not in (0, None):
                return header[k]
        raise ValueError("Frequency not found in header (need RESTFRQ or CRVAL3) or pass freq_hz")

    @staticmethod
    def _pixel_scale_deg(header):
        cd1 = header.get('CDELT1') or header.get('CD1_1')
        cd2 = header.get('CDELT2') or header.get('CD2_2')
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale not found (CDELT1/2)")
        return abs(cd1), abs(cd2)

    @classmethod
    def _y_to_jy_per_beam(cls, y_map, header, freq_hz=None):
        if freq_hz is None:
            freq_hz = cls._freq_from_header(header)
        ipix_deg, jpix_deg = cls._pixel_scale_deg(header)
        jy_per_pix_factor = ytszToJyPix(freq_hz, ipix_deg, jpix_deg)
        jy_pix = y_map * jy_per_pix_factor
        bmaj = header.get('BMAJ')
        bmin = header.get('BMIN')
        if not bmaj or not bmin:
            return jy_pix, {'unit': 'Jy/pixel'}
        pix_area_sr = (ipix_deg*np.pi/180.0)*(jpix_deg*np.pi/180.0)
        beam_area_sr = 2.0*np.pi*(bmaj*np.pi/180.0)*(bmin*np.pi/180.0)/(8.0*np.log(2.0))
        beam_area_pix = beam_area_sr / pix_area_sr
        return jy_pix * beam_area_pix, {'unit': 'Jy/beam', 'beam_area_pix': beam_area_pix}

    # ----------------- public API -----------------
    def plot_y_map(self, d, convert_to_jy_beam=False, freq_hz=None, cmap='viridis'):
        header = d['header']
        model_plane = self._extract_plane(d['model_data'])
        data = model_plane
        meta = None
        if convert_to_jy_beam:
            data, meta = self._y_to_jy_per_beam(data, header, freq_hz=freq_hz)
        wcs = WCS(header)[0,0]
        fig, ax = plt.subplots(1,1, figsize=(6,5), subplot_kw={'projection': wcs})
        im = ax.imshow(data, origin='lower', cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label(meta['unit'] if convert_to_jy_beam else 'Compton y')
        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Dec (J2000)')
        ax.set_title('Model')
        plt.tight_layout()
        return fig, ax

    def plot_model_image_comparison(self, d, convert_to_jy_beam=False, freq_hz=None, cmap='viridis'):
        header = d['header']
        model_plane = self._extract_plane(d['model_data'])
        image_plane = self._extract_plane(d['image_data'])
        if convert_to_jy_beam:
            model_display, meta = self._y_to_jy_per_beam(model_plane, header, freq_hz=freq_hz)
            left_label = meta['unit']
        else:
            model_display = model_plane
            left_label = 'Compton y'
        wcs = WCS(header)[0,0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': wcs}, gridspec_kw={'wspace': 0})
        im0 = axes[0].imshow(model_display, origin='lower', cmap=cmap)
        im1 = axes[1].imshow(image_plane, origin='lower', cmap=cmap)
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.05, fraction=0.046)
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.05, fraction=0.046)
        cbar0.set_label(left_label)
        cbar1.set_label('Jy/beam')
        for ax in axes:
            ax.set_xlabel('Right Ascension (deg)')
            ax.coords[0].set_axislabel('RA (J2000)')
            ax.coords[1].set_axislabel('Dec (J2000)')
            ax.set_aspect('equal')
            ax.set_title('')
        axes[1].set_yticklabels([])
        plt.tight_layout()
        return fig, axes
