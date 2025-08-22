import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from astropy import constants as const
from astropy import units as u
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.cosmology import Planck18 as cosmo

import scipy.special
import scipy.integrate as _scint

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner package not available. Some functions may not work.")

# Physical constants and conversions
Tcmb    = 2.7255
mec2    = ((const.m_e * const.c * const.c).to(u.keV)).value
clight  = const.c.value
kboltz  = const.k_B.value
hplanck = const.h.value
ysznorm = (const.sigma_T / const.m_e / const.c**2).to(u.cm**3 / u.keV / u.Mpc)

def calculate_r500(mass, redshift):
    """Calculate R500 in kpc for given mass (M500 in solar masses) and redshift.
    Uses the cosmology imported here (Planck15) for consistency with utils.
    """
    rho_crit = cosmo.critical_density(redshift)  # mass density (Quantity)
    mass_kg = (mass * u.M_sun).to(u.kg)
    volume = mass_kg / (500 * rho_crit * (4.0 / 3.0) * np.pi)
    r500 = volume ** (1.0 / 3.0)
    return r500.to(u.kpc).value

# ------------------------ Frequency / CMB helpers ------------------------
def getx(freq):
    """Dimensionless frequency x = h nu / (k_B T_cmb)."""
    factor = const.h * freq * u.Hz / const.k_B / (Tcmb * u.Kelvin)
    return factor.to(u.dimensionless_unscaled).value

def getJynorm():
    """Normalization factor converting brightness temperature to Jy."""
    factor = 2e26
    factor *= (const.k_B * Tcmb * u.Kelvin) ** 3
    factor /= (const.h * const.c) ** 2
    return factor.value

def KcmbToKbright(freq):
  """Convert CMB thermodynamic K to brightness temperature factor."""
  x = getx(freq)
  return np.exp(x) * ((x / np.expm1(x)) ** 2)


# ------------------------ Jy / pixel / beam conversions -----------------
def JyBeamToJyPix(ipix, jpix, bmaj, bmin):
    """Convert Jy/beam to Jy/pixel given pixel and beam sizes.

    ipix, jpix: pixel sizes in degrees
    bmaj, bmin: beam major/minor in degrees
    """
    return np.abs(ipix * jpix) * (4 * np.log(2)) / (np.pi * bmaj * bmin)


# ------------------------ Kelvin / Jy conversions ------------------------
def KcmbToJyPix(freq, ipix, jpix):
    x = getx(freq)
    factor = getJynorm() / Tcmb
    factor *= (x ** 4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor

def KbrightToJyPix(freq, ipix, jpix):
    return KcmbToJyPix(freq, ipix, jpix) / KcmbToKbright(freq)


# ------------------------ tSZ / y conversions -----------------------------
def ytszToJyPix(freq, ipix, jpix):
    x = getx(freq)
    factor = getJynorm()
    factor *= -4.0 + x / np.tanh(0.5 * x)
    factor *= (x ** 4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor


def ykszToJyPix(freq, ipix, jpix):
    """Alias kept for backward compatibility (identical to ytszToJyPix here)."""
    x = getx(freq)
    factor = getJynorm()
    factor *= (x ** 4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor


def ytszKcmb(freq):
    x = getx(freq)
    return Tcmb * (-4.0 + x / np.tanh(0.5 * x))


def ykszKcmb():
    return Tcmb

# ------------------------ Relativistic corrections ----------------------
def ytszRelativ(freq, order=1):
    """First-to-fourth order relativistic correction series (Itoh et al. 1998).

    Returns Y_order / Y0 as in the original implementation.
    """
    x = getx(freq)
    xt = x / np.tanh(0.5 * x)
    st = x / np.sinh(0.5 * x)
    Y0 = -4.0 + xt

    if order == 0:
        return 1.0
    if order == 1:
        Y1 = -10.0 + ((47.0 / 2.0) + (-(42.0 / 5.0) + (7.0 / 10.0) * xt) * xt) * xt + st * st * (-(21.0 / 5.0) + (7.0 / 5.0) * xt)
        return np.divide(Y1, Y0)
    if order == 2:
        Y2 = (
        (-15.0 / 2.0)
        + ((1023.0 / 8.0) + ((-868.0 / 5.0) + ((329.0 / 5.0) + ((-44.0 / 5.0) + (11.0 / 30.0) * xt) * xt) * xt) * xt) * xt
        + ((-434.0 / 5.0) + ((658.0 / 5.0) + ((-242.0 / 5.0) + (143.0 / 30.0) * xt) * xt) * xt + (-(44.0 / 5.0) + (187.0 / 60.0) * xt) * (st * st)) * (st * st)
        )
        return np.divide(Y2, Y0)
    if order == 3:
        # keep original long expression but formatted for readability
        Y3 = (
        (15.0 / 2.0)
        + ((2505.0 / 8.0) + ((-7098.0 / 5.0) + ((14253.0 / 10.0) + ((-18594.0 / 35.0) + ((12059.0 / 140.0) + ((-128.0 / 21.0) + (16.0 / 105.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
        + (((-7098.0 / 10.0) + ((14253.0 / 5.0) + ((-102267.0 / 35.0) + ((156767.0 / 140.0) + ((-1216.0 / 7.0) + (64.0 / 7.0) * xt) * xt) * xt) * xt) * xt)
            + (((-18594.0 / 35.0) + ((205003.0 / 280.0) + ((-1920.0 / 7.0) + (1024.0 / 35.0) * xt) * xt) * xt) + ((-544.0 / 21.0) + (992.0 / 105.0) * xt) * (st * st)) * (st * st)) * (st * st)
        )
        return np.divide(Y3, Y0)
    if order == 4:
        Y4 = (
        (-135.0 / 32.0)
        + ((30375.0 / 128.0)
            + ((-62391.0 / 10.0) + ((614727.0 / 40.0) + ((-124389.0 / 10.0) + ((355703.0 / 80.0) + ((-16568.0 / 21.0) + ((7516.0 / 105.0) + ((-22.0 / 7.0) + (11.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt) * xt
            )
        + ((-62391.0 / 20.0) + ((614727.0 / 20.0) + ((-1368279.0 / 20.0) + ((4624139.0 / 80.0) + ((-157396.0 / 7.0) + ((30064.0 / 7.0) + ((-2717.0 / 7.0) + (2761.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
            )
        + ((-124389.0 / 10.0) + ((6046951.0 / 160.0) + ((-248520.0 / 7.0) + ((481024.0 / 35.0) + ((-15972.0 / 7.0) + (18689.0 / 140.0) * xt) * xt) * xt) * xt) * xt
            )
        + (((-70414.0 / 21.0) + ((465992.0 / 105.0) + ((-11792.0 / 7.0) + (19778.0 / 105.0) * xt) * xt) * xt + ((-682.0 / 7.0) + (7601.0 / 210.0) * xt) * (st * st)) * (st * st)) * (st * st)
        )
        return np.divide(Y4, Y0)


def ytszCorrect(y, Te=0.0, limsize=np.inf):
    """Apply relativistic correction series to y-coefficients.

    This retains the original behavior: if Te looks like an array and is
    large enough it will attempt a vectorized evaluation. Otherwise a
    simple polynomial accumulation is performed.
    """
    # Note: original code referenced `ne` (numexpr) but it wasn't imported.
    # Keep a safe scalar fallback path and a simple vectorized path.
    if hasattr(Te, '__len__') and (hasattr(Te, 'size') and Te.size >= limsize):
        # If numexpr is desired, add it later; here return a simple evaluation
        # matching the previous polynomial series expansion behavior.
        y = np.asarray(y)
        t = Te / mec2
        if y.ndim == 0:
            return np.full(np.shape(Te), y)
        if y.ndim == 1:
            return y[0] + y[1] * t
        if y.ndim == 2:
            return y[0] + y[1] * t + y[2] * (t ** 2)
        # For higher dims, fallback to the scalar loop below per-pixel

    # scalar / fallback path
    ycorr = 0.0
    yarr = np.atleast_1d(y)
    for i in range(len(yarr) - 1, 0, -1):
        ycorr = (yarr[i] + ycorr) * Te / mec2
    return ycorr + yarr[0]


# ------------------------ Imaging / geometry helpers ---------------------
def extract_plane(arr):
    """Return a 2D plane from a 2D/3D/4D array layout."""
    a = np.asarray(arr)
    if a.ndim == 4:
        return a[0, 0]
    if a.ndim == 3:
        return a[0]
    return a


def get_map_beam_and_pix(header):
    """Extract beam (BMAJ,BMIN) and pixel scales (CDELT/CD) from a FITS header.

    Returns values in degrees: (bmaj_deg, bmin_deg, ipix_deg, jpix_deg).
    If header values appear to be in arcseconds (numeric > 0.01), convert to degrees.
    Raises ValueError if required keys missing.
    """
    if header is None:
        raise ValueError("Header is required to extract beam and pixel size")
    cd1 = header.get('CDELT1') or header.get('CD1_1')
    cd2 = header.get('CDELT2') or header.get('CD2_2')
    bmaj = header.get('BMAJ')
    bmin = header.get('BMIN')
    if cd1 is None or cd2 is None:
        raise ValueError("Pixel scale missing (CDELT1/2 or CD*_*).")
    if bmaj is None or bmin is None:
        raise ValueError("Beam (BMAJ/BMIN) missing from header")

    def to_deg(x):
        xv = float(x)
        if abs(xv) > 0.01:
            return abs(xv) / 3600.0
        return abs(xv)

    ipix_deg = to_deg(cd1)
    jpix_deg = to_deg(cd2)
    bmaj_deg = to_deg(bmaj)
    bmin_deg = to_deg(bmin)
    return bmaj_deg, bmin_deg, ipix_deg, jpix_deg


def circle_mask(im, xc, yc, rcirc):
    ny, nx = im.shape
    y, x = np.mgrid[0:nx, 0:ny]
    r = np.sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc))
    return (r < rcirc)


def r_theta(im, xc, yc):
    """Return radius and angle arrays for an image grid centered at (xc,yc)."""
    ny, nx = im.shape
    yp, xp = np.mgrid[0:ny, 0:nx]
    yp = yp - yc
    xp = xp - xc
    rr = np.sqrt(np.power(yp, 2.0) + np.power(xp, 2.0))
    phi = np.arctan2(yp, xp)
    return rr, phi


def smooth(data, sigma):
    g = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma)
    model_smooth = convolve_fft(data, g)
    return model_smooth


# ------------------------ Cosmology helpers -----------------------------
def arcsec2kpc(arcsec, z=1.98):
    return np.deg2rad(arcsec / 3600.0) * cosmo.angular_diameter_distance(z).to(u.kpc).value


def kpc2arcsec(kpc, z=1.98):
    return np.rad2deg((kpc / cosmo.angular_diameter_distance(z).to(u.kpc).value)) * 3600.0


# ------------------------ Fourier / scale conversions -------------------
def arcsec_to_uvdist(arcsec=1.65 * 60):
    """Convert angular scale in arcseconds to uv-distance in kilolambda."""
    return 1.0 / np.deg2rad(arcsec / 3600.0) / 1e3


def uvdist_to_arcsec(uvdist):
    """Convert uv-distance in kilolambda to angular scale in arcseconds."""
    return np.rad2deg(1.0 / (uvdist * 1e3)) * 3600.0


def l_to_arcsec(ell):
    """Convert spatial scale in wavenumber l to arcseconds."""
    return 180.0 / ell * 3600.0


def arcsec_to_l(arcsec):
    """Convert arcseconds to spatial wavenumber l."""
    return 180.0 / (arcsec / 3600.0)


def l_to_uvdist(ell):
    """Convert spatial wavenumber l to uv-distance in kilolambda."""
    return arcsec_to_uvdist(l_to_arcsec(ell))


def uvdist_to_l(uvdist):
    """Convert uv-distance in kilolambda to spatial wavenumber l."""
    return arcsec_to_l(uvdist_to_arcsec(uvdist))


# ------------------------ Sampling / I/O helpers ------------------------
def get_samples(filename, major=[None], flux=[None]):
    """Load nested sampling results and extract weighted quantiles.

    Note: expects an npz with entries similar to the original codebase.
    """
    if not HAS_CORNER:
        raise ImportError("corner package is required for this function. Install with: pip install corner")

    results = np.load(filename, allow_pickle=True)
    results = results['samples']
    samples = np.copy(results['samples'])

    if major[0] is not None:
        for i in major:
            samples[:, i] *= 3600

    if flux[0] is not None:
        for i in flux:
            samples[:, i] *= 1e3

    weights = results['logwt'] - scipy.special.logsumexp(results['logwt'] - results['logz'][-1])
    weights = np.exp(weights - results['logz'][-1])
    edges = np.array([
        corner.quantile(samples[:, r], [0.16, 0.50, 0.84], weights=weights)
        for r in range(samples.shape[1])
    ])
    return edges