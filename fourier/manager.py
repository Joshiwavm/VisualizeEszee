"""FourierManager skeleton (steps 1-2).

Implements:
  * map_to_uvgrid: FFT of image (optionally PB * conversion already applied)
  * sample_uv: interpolate uv grid at arbitrary (u,v) baselines (sampleImage-like)

Design:
  - Implemented as a mixin class (to be inherited by PlotManager).
  - Stateless aside from optional caches (added later).
  - Uses rfft2 along one axis and linear spline interpolation over amplitude & phase
    (matching Veszee sampleImage logic) for speed.

Assumptions:
  * Input image is square (N x N) with even (WCS) pixel scale from FITS header.
  * Pixel scale (deg/pixel) provided explicitly to avoid passed header coupling.
  * Image already includes PB & unit conversion (Compton-y * PB converted if needed) upstream.

Pending (future steps): imaging, residual ops, multi-field concatenation, caching, JAX backend.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Sequence, Dict, Any, Optional

class FourierManager:
    """Mixin offering Fourier (visibility) utilities (initial subset)."""

    # ----------------------------
    # Step 1: Map -> uv grid
    # ----------------------------
    @staticmethod
    def map_to_uvgrid(image: np.ndarray, pixel_scale_deg: float) -> Tuple[np.ndarray, float]:
        """Compute rfft2 uv grid of an image.

        Parameters:
            image : 2D array (N,N) real.
            pixel_scale_deg : absolute pixel scale (deg per pixel).
        Returns:
            uv_rfft_shifted : complex array with shape (N, N//2+1) (fftshift along axis 0).
            du : uv spacing (wavelength units, dimensionless) = 1 / (N * pixel_scale_rad).
        """
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("image must be square 2D array")
        N = image.shape[0]
        delt_rad = np.deg2rad(abs(pixel_scale_deg))
        du = 1.0 / (N * delt_rad)
        uv_rfft_shifted = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(image)), axes=0)
        return uv_rfft_shifted, du

    # ----------------------------
    # Step 2: Sample uv coverage
    # ----------------------------
    @staticmethod
    def sample_uv(uv_rfft_shifted: np.ndarray, u: Sequence[float], v: Sequence[float], du: float,
                  dRA: float = 0.0, dDec: float = 0.0, PA: float = 0.0, origin: str = 'upper') -> np.ndarray:
        """Sample the provided uv_rfft_shifted grid at arbitrary (u,v) baselines.

        Parameters:
            uv_rfft_shifted : output of map_to_uvgrid (N, N//2+1)
            u, v : arrays of baseline coords (wavelength units)
            du   : grid spacing returned by map_to_uvgrid
            dRA, dDec : phase offsets (radians) (RA already cos(dec) corrected upstream)
            PA   : position angle rotation (radians)
            origin : 'upper' or 'lower' for sign convention (matches Veszee)
        Returns:
            complex visibilities array matching shape of u.
        """
        u = np.asarray(u)
        v = np.asarray(v)
        if u.shape != v.shape:
            raise ValueError("u and v must have same shape")
        if origin not in ('upper','lower'):
            raise ValueError("origin must be 'upper' or 'lower'")
        v_origin = 1.0 if origin == 'upper' else -1.0
        nxy = uv_rfft_shifted.shape[0]
        if uv_rfft_shifted.shape[1] != nxy//2 + 1:
            raise ValueError("uv_rfft_shifted second dimension inconsistent with first")
        # Rotation
        cos_PA = np.cos(PA)
        sin_PA = np.sin(PA)
        urot = u * cos_PA - v * sin_PA
        vrot = u * sin_PA + v * cos_PA
        dRArot = dRA * cos_PA - dDec * sin_PA
        dDecrot = dRA * sin_PA + dDec * cos_PA
        # Interpolation indices
        uroti = np.abs(urot) / du
        vroti = nxy/2.0 + v_origin * vrot / du
        uneg = urot < 0.0
        vroti[uneg] = nxy/2.0 - v_origin * vrot[uneg]/du
        # Axes
        u_axis = np.linspace(0.0, nxy//2, nxy//2 + 1)
        v_axis = np.linspace(0.0, nxy - 1, nxy)
        # Interpolate real/imag (linear)
        f_re = RectBivariateSpline(v_axis, u_axis, uv_rfft_shifted.real, kx=1, ky=1, s=0)
        f_im = RectBivariateSpline(v_axis, u_axis, uv_rfft_shifted.imag, kx=1, ky=1, s=0)
        f_amp = RectBivariateSpline(v_axis, u_axis, np.abs(uv_rfft_shifted), kx=1, ky=1, s=0)
        ReInt = f_re.ev(vroti, uroti)
        ImInt = f_im.ev(vroti, uroti)
        AmpInt = f_amp.ev(vroti, uroti)
        ImInt[uneg] *= -1.0
        PhaseInt = np.angle(ReInt + 1j * ImInt)
        theta = urot * dRArot + vrot * dDecrot
        vis = AmpInt * (np.cos(theta + PhaseInt) + 1j * np.sin(theta + PhaseInt))
        return vis

    # ----------------------------
    # Imaging (steps 3-5)
    # ----------------------------
    @staticmethod
    def vis_to_image(u: Sequence[float], v: Sequence[float], vis: Sequence[complex],
                     weights: Optional[Sequence[float]] = None,
                     npix: int | None = None,
                     pixel_scale_deg: float | None = None,
                     normalize: bool = True) -> np.ndarray:
        """Grid visibilities to a dirty image (nearest-neighbour).

        Parameters:
            u, v        : baseline coords (wavelength units)
            vis         : complex visibilities
            weights     : optional weights (defaults to ones)
            npix        : image size (required)
            pixel_scale_deg : pixel size in degrees (required for uv scaling)
            normalize   : divide by sum(weights) if True
        Returns:
            2D real dirty image array (npix, npix)
        """
        u = np.asarray(u)
        v = np.asarray(v)
        vis = np.asarray(vis)
        if u.shape != v.shape or u.shape != vis.shape:
            raise ValueError("u, v, vis must share shape")
        if npix is None or pixel_scale_deg is None:
            raise ValueError("npix and pixel_scale_deg required")
        if weights is None:
            weights = np.ones_like(u, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
        if weights.shape != u.shape:
            raise ValueError("weights shape mismatch")
        delt_rad = np.deg2rad(abs(pixel_scale_deg))
        du = 1.0 / (npix * delt_rad)
        grid = np.zeros((npix, npix), dtype=np.complex128)
        wgrid = np.zeros((npix, npix), dtype=float)
        center = (npix - 1) / 2.0
        iu = np.rint(u / du + center).astype(int)
        iv = np.rint(v / du + center).astype(int)
        mask = (iu >= 0) & (iu < npix) & (iv >= 0) & (iv < npix)
        for idx in np.where(mask)[0]:
            grid[iv[idx], iu[idx]] += vis[idx] * weights[idx]
            wgrid[iv[idx], iu[idx]] += weights[idx]
        nz = wgrid > 0
        grid[nz] /= wgrid[nz]
        image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
        if normalize:
            wsum = weights.sum()
            if wsum > 0:
                image /= wsum
        return image

    @staticmethod
    def multi_field_dirty(fields: Sequence[Dict[str, Any]], npix: int, pixel_scale_deg: float,
                          align: bool = True, normalize: bool = True) -> np.ndarray:
        """Joint dirty image from multiple field visibility sets.

        Each field dict must contain keys: 'u','v','vis' and optionally 'weights', 'dRA', 'dDec'.
        If align=True, phase shifts are applied to move each field to common center by
        multiplying by exp(2πi(u*dRA + v*dDec)). Assume dRA,dDec already in radians (RA corrected).
        """
        accum_u = []
        accum_v = []
        accum_vis = []
        accum_w = []
        for fd in fields:
            u = np.asarray(fd['u'])
            v = np.asarray(fd['v'])
            vis = np.asarray(fd['vis'])
            if align and ('dRA' in fd or 'dDec' in fd):
                dRA = fd.get('dRA', 0.0)
                dDec = fd.get('dDec', 0.0)
                phase = np.exp(2.0 * np.pi * 1j * (u * dRA + v * dDec))
                vis = vis * phase
            w = np.asarray(fd.get('weights', np.ones_like(u)))
            accum_u.append(u); accum_v.append(v); accum_vis.append(vis); accum_w.append(w)
        u_all = np.concatenate(accum_u) if accum_u else np.array([])
        v_all = np.concatenate(accum_v) if accum_v else np.array([])
        vis_all = np.concatenate(accum_vis) if accum_vis else np.array([])
        w_all = np.concatenate(accum_w) if accum_w else np.array([])
        if u_all.size == 0:
            return np.zeros((npix, npix))
        return FourierManager.vis_to_image(u_all, v_all, vis_all, weights=w_all,
                                           npix=npix, pixel_scale_deg=pixel_scale_deg,
                                           normalize=normalize)

    # ----------------------------
    # Visibility operations
    # ----------------------------
    @staticmethod
    def subtract_vis(data_vis: Sequence[complex], model_vis: Sequence[complex]) -> np.ndarray:
        data_vis = np.asarray(data_vis)
        model_vis = np.asarray(model_vis)
        if data_vis.shape != model_vis.shape:
            raise ValueError("data_vis and model_vis shape mismatch")
        return data_vis - model_vis

    @staticmethod
    def scale_vis(vis: Sequence[complex], factor: float) -> np.ndarray:
        return np.asarray(vis) * factor

    @staticmethod
    def phase_shift(vis: Sequence[complex], u: Sequence[float], v: Sequence[float],
                    dRA: float = 0.0, dDec: float = 0.0) -> np.ndarray:
        vis = np.asarray(vis)
        u = np.asarray(u); v = np.asarray(v)
        if vis.shape != u.shape:
            raise ValueError("vis and u/v shapes mismatch")
        phase = np.exp(2.0 * np.pi * 1j * (u * dRA + v * dDec))
        return vis * phase

    @staticmethod
    def residual_vis(data_vis: Sequence[complex], model_vis: Sequence[complex]) -> np.ndarray:
        return FourierManager.subtract_vis(data_vis, model_vis)

    # ------------------------------------------------------------------
    # High-level helpers integrating with model_maps / uvdata (moved from PlotManager)
    # ------------------------------------------------------------------
    def _get_uv_struct(self, data_name, field_key, spw_key):
        if data_name not in getattr(self, 'uvdata', {}):
            raise ValueError(f"Dataset '{data_name}' not loaded.")
        ds = self.uvdata[data_name]
        if field_key not in ds:
            raise ValueError(f"Field '{field_key}' missing in uvdata[{data_name}].")
        if spw_key not in ds[field_key]:
            raise ValueError(f"SPW '{spw_key}' missing in uvdata[{data_name}][{field_key}].")
        uvrec = ds[field_key][spw_key]
        if not isinstance(uvrec, tuple) or len(uvrec) < 6:
            raise ValueError(f"uv record malformed for {data_name}/{field_key}/{spw_key}")
        return uvrec

    def _convert_model_map_to_jybeam(self, entry):
        from ..utils.utils import ytszToJyPix, JyBeamToJyPix
        header = entry['header']
        model_plane = entry['model_data']
        freq = None
        for k in ("RESTFRQ", "CRVAL3"):
            if k in header and header[k]:
                freq = header[k]; break
        if freq is None:
            raise ValueError("Frequency not found in header (RESTFRQ/CRVAL3)")
        cd1 = header.get('CDELT1') or header.get('CD1_1')
        cd2 = header.get('CDELT2') or header.get('CD2_2')
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale missing (CDELT1/2 or CD*_*).")
        ipix_deg, jpix_deg = abs(cd1), abs(cd2)
        jy_pix = model_plane * ytszToJyPix(freq, ipix_deg, jpix_deg)
        bmaj = header.get('BMAJ'); bmin = header.get('BMIN')
        if not bmaj or not bmin:
            raise ValueError('Beam parameters (BMAJ/BMIN) required for Jy/beam conversion.')
        beam_conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)
        jy_beam = jy_pix / beam_conv
        return jy_beam, ipix_deg

    def _build_model_uv(self, model_name, data_name, field_key, spw_key, recompute=False):
        storage = getattr(self, 'model_uvgrids', None)
        if storage is None:
            self.model_uvgrids = {}
            storage = self.model_uvgrids
        mm = storage.setdefault(model_name, {}).setdefault(data_name, {}).setdefault(field_key, {})
        if not recompute and spw_key in mm:
            return
        entry = self.model_maps[model_name][data_name][field_key][spw_key]
        jy_beam_image, pix_deg = self._convert_model_map_to_jybeam(entry)
        uv_grid, du = self.map_to_uvgrid(jy_beam_image, pix_deg)
        mm[spw_key] = {'uv': uv_grid, 'du': du, 'pixscale_deg': pix_deg}

    def _sample_model_uv(self, model_name, data_name, field_key, spw_key, recompute=False):
        sm_store = getattr(self, 'sampled_model_vis', None)
        if sm_store is None:
            self.sampled_model_vis = {}
            sm_store = self.sampled_model_vis
        sm = sm_store.setdefault(model_name, {}).setdefault(data_name, {}).setdefault(field_key, {})
        if not recompute and spw_key in sm:
            return
        uvrec = self._get_uv_struct(data_name, field_key, spw_key)
        u, v, real, imag, wgt, freq = uvrec.uwave, uvrec.vwave, uvrec.uvreal, uvrec.uvimag, uvrec.suvwght, uvrec.uvfreq
        self._build_model_uv(model_name, data_name, field_key, spw_key)
        uv_entry = self.model_uvgrids[model_name][data_name][field_key][spw_key]
        model_vis = self.sample_uv(uv_entry['uv'], u, v, uv_entry['du'])
        sm[spw_key] = {'model_vis': model_vis, 'u': u, 'v': v, 'weights': wgt, 'data_vis': real + 1j * imag}
        # Residuals always
        res_store = getattr(self, 'residual_vis', None)
        if res_store is None:
            self.residual_vis = {}
            res_store = self.residual_vis
        rv = res_store.setdefault(model_name, {}).setdefault(data_name, {}).setdefault(field_key, {})
        rv[spw_key] = sm[spw_key]['data_vis'] - sm[spw_key]['model_vis']

    @staticmethod
    def uvpoint(u, v, dx=0.0, dy=0.0, amp=1.0, offset=0.0):
        """Analytic point source visibility: offset + amp * exp(2πi(u dx + v dy)).
        dx, dy in radians (RA already cos(dec) corrected). u, v in wavelengths.
        Broadcasts over numpy arrays.
        """
        u = np.asarray(u); v = np.asarray(v)
        return offset + amp * np.exp(2j * np.pi * (u * dx + v * dy))

    @staticmethod
    def uvgauss(u, v, dx=0.0, dy=0.0, amp=1.0, offset=0.0, fwhm=1.0, e=0.0, theta=0.0):
        """Analytic circular/elliptical Gaussian component in uv plane.
        Parameters:
          u,v    : baselines (wavelengths)
          dx,dy  : center offsets (radians)
          amp    : peak amplitude (Jy)
          offset : additive constant term (Jy)
          fwhm   : major-axis FWHM in arcsec
          e      : ellipticity (1 - b/a); 0 -> circular
          theta  : position angle (deg, E of N) of major axis
        """
        u = np.asarray(u); v = np.asarray(v)
        sigma_rad = np.deg2rad(fwhm / 3600.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        b_over_a = 1.0 - e
        th = np.deg2rad(theta)
        sint = np.sin(th); cost = np.cos(th)
        ur =  ( u * cost + v * sint)
        vr = (-u * sint + v * cost)
        # Gaussian FT: exp[-2 (π σ)^2 (ur^2 + (vr * b/a)^2)]
        fac = np.exp(-2.0 * (np.pi * sigma_rad)**2 * (ur**2 + (vr * b_over_a)**2))
        return offset + amp * fac * np.exp(2j * np.pi * (u * dx + v * dy))
