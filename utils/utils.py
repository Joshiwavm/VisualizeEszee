import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from astropy.cosmology import Planck15 as cosmo

import scipy.special

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner package not available. Some functions may not work.")
import astropy.constants as const
import astropy.units as u

global Tcmb; Tcmb = 2.7255
global mec2; mec2 = ((const.m_e*const.c*const.c).to(u.keV)).value
global clight; clight = const.c.value
global kboltz; kboltz = const.k_B.value
global hplanck; hplanck = const.h.value
global ysznorm; ysznorm = (const.sigma_T/const.m_e/const.c**2).to(u.cm**3/u.keV/u.Mpc)

def calculate_r500(mass, redshift):
    """Calculate R500 in kpc for given mass (M500 in solar masses) and redshift.
    Uses the cosmology imported here (Planck15) for consistency with utils.
    """
    rho_crit = cosmo.critical_density(redshift)  # mass density (Quantity)
    mass_kg = (mass * u.M_sun).to(u.kg)
    volume = mass_kg / (500 * rho_crit * (4.0/3.0) * np.pi)
    r500 = (volume) ** (1/3)
    return r500.to(u.kpc).value

# Adimensional frequency
# ----------------------------------------------------------------------
def getx(freq):
  factor = const.h*freq*u.Hz/const.k_B/(Tcmb*u.Kelvin)
  return factor.to(u.dimensionless_unscaled).value

# CMB surface brightness
# ----------------------------------------------------------------------
def getJynorm():
  factor  = 2e26
  factor *= (const.k_B*Tcmb*u.Kelvin)**3 # (kboltz*Tcmb)**3.0
  factor /= (const.h*const.c)**2         # (hplanck*clight)**2.0
  return factor.value

# Jy/beam to Jy/pix
# ----------------------------------------------------------------------
def JyBeamToJyPix(ipix,jpix,bmaj,bmin):
  return np.abs(ipix*jpix)*(4*np.log(2))/(np.pi*bmaj*bmin)

# Jy/deg2 to Jy/pix
# ----------------------------------------------------------------------
def JyDegsToJyPix(ipix,jpix,bmaj,bmin):
  return np.abs(ipix*jpix)

# Kelvin CMB to Kelvin brightness
# ----------------------------------------------------------------------
def KcmbToKbright(freq): 
  x = getx(freq)
  return np.exp(x)*((x/np.expm1(x))**2)

# Kelvin brightness to Jy/pixel
# ----------------------------------------------------------------------
def KbrightToJyPix(freq,ipix,jpix):
  return KcmbToJyPix(freq,ipix,jpix)/KcmbToKbright(freq)

# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
def KcmbToJyPix(freq,ipix,jpix):
  x = getx(freq)
  factor  = getJynorm()/Tcmb
  factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
  factor *= np.abs(ipix*jpix)*(np.pi/1.8e2)*(np.pi/1.8e2)
  return factor

# Compton y to Jy/pixel
# ----------------------------------------------------------------------
def ytszToJyPix(freq,ipix,jpix):
  x = getx(freq)
  factor  = getJynorm()
  factor *= -4.0+x/np.tanh(0.5*x)
  factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
  factor *= np.abs(ipix*jpix)*(np.pi/1.8e2)*(np.pi/1.8e2)
  return factor

# Compton y to Kelvin CMB
# ----------------------------------------------------------------------
def ytszKcmb(freq):
  x = getx(freq)
  return Tcmb*(-4.0+x/np.tanh(0.5*x))

# Compton y to Jy/pixel
# ----------------------------------------------------------------------
def ykszToJyPix(freq,ipix,jpix):
  x = getx(freq)
  factor  = getJynorm()
  factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
  factor *= np.abs(ipix*jpix)*(np.pi/1.8e2)*(np.pi/1.8e2)
  return -factor

# Compton y to Kelvin CMB
# ----------------------------------------------------------------------
def ykszKcmb(): return Tcmb

# First to fourth order relativistic terms (Itoh et al. 1998)
# ----------------------------------------------------------------------
def ytszRelativ(freq,order=1):
    x = getx(freq)
    xt = x/np.tanh(0.5*x)
    st = x/np.sinh(0.5*x)
    Y0 = -4.0+xt

    if (order==0):
        return 1.0
    if (order==1):
        Y1 = -10.+((47./2.)+(-(42./5.)+(7./10.)*xt)*xt)*xt+st*st*(-(21./5.)+(7./5.)*xt)
        return np.divide(Y1,Y0)
    if (order==2):
        Y2 = (-15./2.)+((1023./8.)+((-868./5.)+((329./5.)+((-44./5.)+(11./30.)*xt)*xt)*xt)*xt)*xt+ \
                ((-434./5.)+((658./5.)+((-242./5.)+(143./30.)*xt)*xt)*xt+(-(44./5.)+(187./60.)*xt)*(st*st))*st*st
        return np.divide(Y2,Y0)
    if (order==3):
        Y3 = (15./2.)+((2505./8.)+((-7098./5.)+((14253./10.)+((-18594./35.)+((12059./140.)+((-128./21.)+(16./105.)*xt)*xt)*xt)*xt)*xt)*xt)*xt+ \
                (((-7098./10.)+((14253./5.)+((-102267./35.)+((156767./140.)+((-1216./7.)+(64./7.)*xt)*xt)*xt)*xt)*xt) +
                (((-18594./35.)+((205003./280.)+((-1920./7.)+(1024./35.)*xt)*xt)*xt) +((-544./21.)+(992./105.)*xt)*st*st)*st*st)*st*st
        return np.divide(Y3,Y0)
    if (order==4):
        Y4 = (-135./32.)+((30375./128.)+((-62391./10.)+((614727./40.)+((-124389./10.)+((355703./80.)+((-16568./21.)+((7516./105.)+((-22./7.)+(11./210.)*xt)*xt)*xt)*xt)*xt)*xt)*xt)*xt)*xt + \
                ((-62391./20.)+((614727./20.)+((-1368279./20.)+((4624139./80.)+((-157396./7.)+((30064./7.)+((-2717./7.)+(2761./210.)*xt)*xt)*xt)*xt)*xt)*xt)*xt + \
                ((-124389./10.)+((6046951./160.)+((-248520./7.)+((481024./35.)+((-15972./7.)+(18689./140.)*xt)*xt)*xt)*xt)*xt +\
                ((-70414./21.)+((465992./105.)+((-11792./7.)+(19778./105.)*xt)*xt)*xt+((-682./7.)+(7601./210.)*xt)*st*st)*st*st)*st*st)*st*st
        return np.divide(Y4,Y0)

# Relativistic corrections
def ytszCorrect(y,Te=0.0,limsize=np.inf):
  if hasattr(Te,'__len__') and (Te.size>=limsize):
    local_dict = {'t': ne.evaluate('Te/mec2')}
    for i in range(len(y)):
      local_dict.update({'y{0}'.format(i): y[i]})
    if (not np.shape(y)): return y
    elif y.ndim==1: return np.full(np.shape(Te),y[0])
    elif y.ndim==2: return ne.evaluate('y0+y1*t',local_dict=local_dict)
    elif y.ndim==3: return ne.evaluate('y0+y1*t+y2*(t**2)',local_dict=local_dict)
    elif y.ndim==4: return ne.evaluate('y0+y1*t+y2*(t**2)+y3*(t**3)',local_dict=local_dict)
    elif y.ndim==5: return ne.evaluate('y0+y1*t+y2*(t**2)+y3*(t**3)+y4*(t**4)',local_dict=local_dict)
  else:
    ycorr = 0.0
    if (not np.shape(y)): y = [y]
    for i in range(len(y)-1,0,-1): ycorr = (y[i]+ycorr)*Te/mec2
    return ycorr+y[0]


def arcsec_to_uvdist(arcsec=1.65 * 60):
    """Converts angular scale in arcseconds to uv-distance in kilolambda."""
    return 1 / np.deg2rad(arcsec / 3600) / 1e3

def uvdist_to_arcsec(uvdist):
    """Converts uv-distance in kilolambda to angular scale in arcseconds."""
    return np.rad2deg(1 / (uvdist * 1e3)) * 3600

def l_to_arcsec(ell):
    """Converts spatial scale in wavenumber l to arcseconds."""
    return 180 / ell * 3600

def arcsec_to_l(arcsec):
    """Converts arcseconds to spatial wavenumber l."""
    return 180 / (arcsec / 3600)

def l_to_uvdist(ell):
    """Converts spatial scale in wavenumber l to uv-distance in kilolambda."""
    return arcsec_to_uvdist(l_to_arcsec(ell))

def uvdist_to_l(uvdist):
    """Converts uv-distance in kilolambda to spatial wavenumber l."""
    return arcsec_to_l(uvdist_to_arcsec(uvdist))

def circle_mask(im, xc, yc, rcirc):
        ny, nx = im.shape
        y,x = np.mgrid[0:nx,0:ny]
        r = np.sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc))
        return ( (r < rcirc))

def r_theta(im, xc, yc):
    # returns the radius rr and the angle phi for point (xc,yc)
    ny, nx = im.shape
    yp, xp = np.mgrid[0:ny,0:nx]
    yp = yp - yc
    xp = xp - xc
    rr = np.sqrt(np.power(yp,2.) + np.power(xp,2.))
    phi = np.arctan2(yp, xp)
    return(rr, phi)

def smooth(data, sigma):
    g = Gaussian2DKernel(x_stddev=sigma,
                         y_stddev=sigma)
    
    model_smooth = convolve_fft(data, g)
    return model_smooth

def arcsec2kpc(arcsec, z = 1.98):
    return (np.deg2rad(arcsec/3600)* cosmo.angular_diameter_distance(z).to(u.kpc).value)

def kpc2arcsec(kpc, z = 1.98):
    return np.rad2deg((kpc/ cosmo.angular_diameter_distance(z).to(u.kpc).value))*3600







def get_samples(filename, major=[None], flux=[None]):
    """
    Load nested sampling results and extract quantiles.
    """
    if not HAS_CORNER:
        raise ImportError("corner package is required for this function. Install with: pip install corner")
        
    results = np.load(filename, allow_pickle=True)
    results = results['samples']    
    samples = np.copy(results['samples'])

    if major[0] is not None:
        for i in major:
            samples[:,i] *= 3600

    if flux[0] is not None:
        for i in flux:
            samples[:,i] *= 1e3

    weights = results['logwt'] - scipy.special.logsumexp(results['logwt'] - results['logz'][-1])
    weights = np.exp(weights - results['logz'][-1])
    edges = np.array([corner.quantile(samples[:,r], [0.16, 0.50, 0.84], weights=weights) 
                     for r in range(samples.shape[1])])    
    return edges

# Add legacy-style names for compatibility with Veszee FourierTransform expectations
comptonToJyPix = ytszToJyPix  # identical spectral factor for pure tSZ y to Jy/pix

# Relativistic series (alias of ytszRelativ for naming parity)
comptonRelativ = ytszRelativ

# Wrapper matching old comptonCorrect signature (uses ytszCorrect)
def comptonCorrect(y, Te=0.0, limsize=np.inf):
    return ytszCorrect(y, Te=Te, limsize=limsize)

# Flat (band-averaged) Compton coefficient integration
import scipy.integrate as _scint

def yszCorrect(freq, cdelt, order):
    return comptonToJyPix(freq, cdelt[0], cdelt[1]) * comptonRelativ(freq, order)

def computeFlatCompton(freq_range, cdelt, order):
    """Integrate yszCorrect over a frequency interval [nu1, nu2].
    freq_range: (nu1, nu2) in Hz (list/tuple/array)
    cdelt: (cdelt1, cdelt2) pixel size degrees
    order: relativistic expansion order (0-4)
    """
    nu1, nu2 = freq_range
    if nu1 == nu2:
        return yszCorrect(nu1, cdelt, order)
    val, _ = _scint.quad(lambda nu: yszCorrect(nu, cdelt, order), nu1, nu2, limit=200)
    # Average over bandwidth
    return val / (nu2 - nu1)
