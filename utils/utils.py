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


global Tcmb; Tcmb = 2.7255
global C; C = 299792458
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

def getx(freq):
    factor = const.h*freq*u.Hz/const.k_B/(Tcmb*u.Kelvin)
    return factor.to(u.dimensionless_unscaled).value

def getJynorm():
    factor  = 2e26
    factor *= (const.k_B*Tcmb*u.Kelvin)**3 # (kboltz*Tcmb)**3.0
    factor /= (const.h*const.c)**2         # (hplanck*clight)**2.0
    return factor.value

def KcmbToJyPix(freq,ipix,jpix):
    x = getx(freq)
    factor  = getJynorm()/Tcmb
    factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
    factor *= np.abs(ipix*jpix)*(np.pi/1.8e2)*(np.pi/1.8e2)
    return factor

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

def get_A10params(proftype, RA, Dec, z, Mass):
    if  proftype[-3:]=='_cc': # cool core
        alpha = 2.3000E+00; beta = 3.3400E+00; gamma =  0.2100E+00
        pnorm = 3.7000E+00; c500 = 2.8000E+00; ap    = -1.0000E-01
        pnorm*= 1.026
    elif proftype[-3:]=='_md': # morphologically disturbed
        alpha = 1.7000E+00; beta = 5.7400E+00; gamma =  0.0500E+00
        pnorm = 3.9100E+00; c500 = 1.5000E+00; ap    = -1.0000E-01
        pnorm*= 1.043
    elif proftype[-3:]=='_up': # universal profile
        alpha = 2.2700E+00; beta = 3.4800E+00; gamma =  0.1500E+00
        pnorm = 3.4700E+00; c500 = 2.5900E+00; ap    = -1.0000E-01
        pnorm*= 1.034
        
    popt = {'RA':          RA, 
            'Dec':         Dec, 
            'log10':       Mass, 
            'c500':        c500, 
            'e':           0.0, 
            'Angle':       0.0, 
            'Offset':      0.0, 
            'Temperature': 0.0, 
            'Alpha':       alpha, 
            'Beta':        beta, 
            'Gamma':       gamma, 
            'P0':          pnorm, 
            'Alpha_p':     ap, 
            'z':           z, 
            'bias':        0.0
            }
           
    
    return popt
