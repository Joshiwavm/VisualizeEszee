import jax; jax.config.update('jax_enable_x64',True)
import jax_finufft
import numpy as np
import scipy.integrate

try:
    import szmodel  # optional, only needed for tSZ spectral conversion in getspec
except ImportError:  # pragma: no cover
    szmodel = None

# ---------------------------------------------------------------------------
# Low-level helpers (kept from original)
# ---------------------------------------------------------------------------

def uvpoint(dx,dy,u,v,off,amp):
    return off+amp*np.exp(2.00*np.pi*1j*(u*dx+v*dy)) 


def getspec(stype,freq,reffreq=None,alpha=None):
    """Compute simple spectral weighting.
    stype:
      powlaw : power-law with index alpha at reffreq
      tSZ    : thermal SZ (requires szmodel)
    freq : [f_min, f_max] Hz (integration bounds or single frequency twice)
    reffreq, alpha : reference frequency & index for powlaw
    Returns average spectral factor over band.
    """
    if stype=='powlaw':
        if reffreq is None or alpha is None:
            raise ValueError('Reference frequency and spectral index must be provided')
        def foospec(fval):
            return (fval/reffreq)**alpha
    elif stype=='tSZ':
        if szmodel is None:
            raise ImportError("szmodel required for tSZ spectral conversion")
        def foospec(fval):
            return szmodel.comptonToJyPix(fval,np.rad2deg(1.00),np.rad2deg(1.00))
    else:
        raise ValueError(f"Unknown spectral type: {stype}")

    if freq[0]!=freq[1]:    
        outspec = scipy.integrate.quad(foospec,freq[0],freq[1])[0]
        outspec = outspec/(freq[1]-freq[0])
    else:
        outspec = foospec(freq[0])
    return outspec

# ---------------------------------------------------------------------------
# Fourier / NUFFT operations
# ---------------------------------------------------------------------------

class FourierTransformer:
    """Utility for mapping between image and visibility (uv) domains using jax_finufft.

    Provides:
      - image_to_vis : sample a model image (optionally PB * conv factors) at uv points
      - vis_to_image : grid visibilities onto a regular image plane (dirty map style)
    Uses Type 2 NUFFT (uniform -> nonuniform) and Type 1 NUFFT (nonuniform -> uniform).
    """

    def __init__(self, npix: int, pixel_scale_deg: float):
        """Parameters
        npix : image dimension (assumed square)
        pixel_scale_deg : absolute CDELT (deg / pix); sign ignored.
        """
        self.npix = int(npix)
        self.delt_deg = float(abs(pixel_scale_deg))
        self.delt_rad = np.deg2rad(self.delt_deg)

    # -------------------------------------
    # Phase coordinate helpers
    # -------------------------------------
    def phase_coords(self, u: np.ndarray, v: np.ndarray):
        """Return phase coordinates (x, y) for NUFFT given uv (in wavelengths).
        Matches notebook convention:
          x = -2π v Δ
          y =  2π u Δ
        Δ is pixel scale in radians.
        """
        x = -2.0 * np.pi * v * self.delt_rad
        y =  2.0 * np.pi * u * self.delt_rad
        return x, y

    # -------------------------------------
    # Visibility sampling (image -> vis)
    # -------------------------------------
    def image_to_vis(self, image: np.ndarray, u: np.ndarray, v: np.ndarray,
                     pb: np.ndarray | None = None, conv: float | np.ndarray | None = None) -> np.ndarray:
        """Sample an image at provided uv points using Type-2 NUFFT.
        image : 2D array (npix, npix)
        u, v  : arrays of same shape giving uv coordinates (wavelengths)
        pb    : primary beam map (same shape as image) multiplied before transform
        conv  : multiplicative spectral / unit conversion factor (scalar or map)
        Returns complex visibilities.
        """
        if image.shape != (self.npix, self.npix):
            raise ValueError("Image shape does not match configured npix")
        grid = image.astype(np.complex128)
        if pb is not None:
            if pb.shape != image.shape:
                raise ValueError("Primary beam shape mismatch")
            grid = grid * pb
        if conv is not None:
            grid = grid * conv
        x, y = self.phase_coords(u, v)
        # Type-2 NUFFT: uniform grid -> nonuniform samples
        vis = jax_finufft.nufft2(grid, x, y)
        return vis

    # -------------------------------------
    # Imaging (vis -> image)
    # -------------------------------------
    def vis_to_image(self, u: np.ndarray, v: np.ndarray, vis: np.ndarray,
                     weights: np.ndarray | None = None, factor: float = 1.0) -> np.ndarray:
        """Grid visibilities onto a regular image using Type-1 NUFFT.
        u,v    : uv coordinates
        vis    : complex visibility data (same shape)
        weights: optional weights; defaults to ones
        factor : scalar divisor applied at end (e.g. unit conversion / beam factor)
        Returns real-valued dirty map (numpy array shape (npix, npix)).
        """
        if weights is None:
            weights = np.ones_like(vis.real)
        if not (u.shape == v.shape == vis.shape == weights.shape):
            raise ValueError("u, v, vis, weights must have same shape")
        x, y = self.phase_coords(u, v)
        # Weighted inverse NUFFT (Type-1): nonuniform -> uniform grid
        coeffs = weights * vis / weights.sum()
        grid = jax_finufft.nufft1((self.npix, self.npix), coeffs, x, y)
        return np.array(grid.real) / factor

    # -------------------------------------
    # Convenience wrapper replicating notebook pattern
    # -------------------------------------
    def dirty_map(self, u, v, data_vis, weights, factor=1.0):
        return self.vis_to_image(u, v, data_vis, weights=weights, factor=factor)

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
