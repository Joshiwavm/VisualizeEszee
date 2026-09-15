import numpy as np
from astropy import units as u
from scipy.integrate import quad

from ..utils.utils import cosmo

def elos(e): return 1.00-(1.00-e)/np.sqrt(0.50*(1.00+(1.00-e)**2))

# Morphological distortions (Sanders+2025, arXiv:2502.02239)
# ----------------------------------------------------------------------
# Mirrors the implementation in eszee/model.py. Slosh H perturbs the
# radius before the profile lookup; the multipole magnitudes M1-M4
# modulate the surface brightness afterwards. Only the model types in
# morphmodels carry these parameters.
morphlab = np.array(['H', 'Slosh angle (deg)',
                     'M1', 'M1 angle (deg)', 'M2', 'M2 angle (deg)',
                     'M3', 'M3 angle (deg)', 'M4', 'M4 angle (deg)'])
morphmodels = {'gnfwPressure', 'gnfwEmulator'}


def _morphval(params, key):
    """Named lookup tolerating an explicit YAML null."""
    value = params.get(key, 0.00)
    return 0.00 if value is None else value


def morphpars(params):
    """(H, theta_H, [M1..M4], [theta_1..theta_4]) from a named dict.

    eszee reads these from the tail of a positional vector; here they
    are named keys, but the returned tuple is deliberately the same
    shape so the call sites read alike in both packages.
    """
    mags = np.array([_morphval(params, f'M{m}') for m in range(1, 5)])
    angs = np.array([_morphval(params, f'M{m}_angle')
                     for m in range(1, 5)])
    return (_morphval(params, 'H'), _morphval(params, 'slosh_angle'),
            mags, angs)


def sloshGrid(grid, gridx, gridy, mag, ang):
    """Radial perturbation in the elliptical frame (eq. 18/20).

    r_eff >= r*(1-H) >= 0 for H < 1, so no clipping is needed.
    """
    if mag == 0.00: return grid
    ang = np.deg2rad(ang)
    return grid+mag*(gridx*np.cos(ang)-gridy*np.sin(ang))


def sloshNorm(mag):
    """Brightness rescaling keeping the sloshed flux fixed (eq. 19)."""
    return (1.00-mag**2)**1.5


def multiPole(gridx, gridy, mags, angs):
    """Azimuthal modulation summed over orders (eq. 21).

    Angles follow the mbproj2d rotation convention
    sin(m*(theta-theta_m)), not the paper's sin(m*theta+theta_0).
    Left unnormalised to match eszee: against a symmetric profile each
    sine averages to zero over azimuth, but once slosh is active the
    flux does shift by a few percent and the amplitude absorbs it.
    """
    if not np.any(mags): return 1.00
    azim = np.arctan2(gridy, gridx)
    factor = np.ones_like(azim)
    for m, (mag, ang) in enumerate(zip(mags, angs), start=1):
        if mag != 0.00:
            factor = factor+mag*np.sin(m*(azim-np.deg2rad(ang)))
    return factor

# 3D A10 model profile
# ----------------------------------------------------------------------
def a10RadialProfile(x,alpha,beta,gamma,ap,c500,mass):
    return (x**(-gamma))*((1.00+(x**alpha))**((gamma-beta)/alpha))*(mass**((ap+0.10)/(1.00+(2.00*x/c500)**3.00)))

def a10ProfileIntegrand(x,xi,alpha,beta,gamma,ap,c500,mass): 
    return (x**(-gamma))*((1.00+(x**alpha))**((gamma-beta)/alpha))*(x/((x*x-xi*xi)**0.50))*(mass**((ap+0.10)/(1.00+(2.00*x/c500)**3.00)))

# A10 model integrale
# ----------------------------------------------------------------------
def _a10ProfileIntegral(x,alpha,beta,gamma,ap,c500,mass,limdist=np.inf,epsrel=1.00E-06,radial=False): 
    if not radial: return 2.00*quad(a10ProfileIntegrand,x,limdist,args=(x,alpha,beta,gamma,ap,c500,mass),epsrel=epsrel)[0]
    else: return a10RadialProfile(x,alpha,beta,gamma,ap,c500,mass)
a10ProfileIntegral = np.vectorize(_a10ProfileIntegral)

# Integrated elliptical A10 model profile
# ----------------------------------------------------------------------
def a10Profile(grid,offset,amp,major,e,alpha,beta,gamma,ap,c500,mass, limdist=np.inf,epsrel=1.00E-06,freeLS=None,radial=False): 
    integral = np.zeros_like(grid,dtype=np.float64)
    if not radial: integral[grid<=limdist] = a10ProfileIntegral(grid[grid<=limdist],alpha,beta,gamma,ap,c500,mass,limdist,epsrel,radial)
    else: integral[grid<=limdist] = _a10ProfileIntegral(grid[grid<=limdist],alpha,beta,gamma,ap,c500,mass,limdist,epsrel,radial)

    ellipse = (1.00-elos(e)) if freeLS is None else freeLS
    return offset+amp*major*ellipse*integral

# 3D gNFW model profile
# ----------------------------------------------------------------------
def gnfwRadialProfile(x,alpha,beta,gamma): 
    return (x**(-gamma))*((1.00+(x**alpha))**((gamma-beta)/alpha))

def gnfwProfileIntegrand(x,xi,alpha,beta,gamma): 
    return (x**(-gamma))*((1.00+(x**alpha))**((gamma-beta)/alpha))*x/((x*x-xi*xi)**0.50)

# gNFW model integral
# ----------------------------------------------------------------------
def _gnfwProfileIntegral(x,alpha,beta,gamma,limdist=np.inf,epsrel=1.00E-06,radial=False): 
    if not radial: return 2.00*quad(gnfwProfileIntegrand,x,limdist,args=(x,alpha,beta,gamma),epsrel=epsrel)[0]
    else: return gnfwRadialProfile(x,alpha,beta,gamma)
gnfwProfileIntegral = np.vectorize(_gnfwProfileIntegral)

# Integrated elliptical gNFW model profile
# ----------------------------------------------------------------------
def gnfwProfile(grid,offset,amp,major,e,alpha,beta,gamma,limdist=np.inf,epsrel=1.00E-06,freeLS=None,radial=False): 
    integral = np.zeros_like(grid,dtype=np.float64)
    if not radial: integral[grid<=limdist] = gnfwProfileIntegral(grid[grid<=limdist],alpha,beta,gamma,limdist,epsrel,radial)
    else: integral[grid<=limdist] = _gnfwProfileIntegral(grid[grid<=limdist],alpha,beta,gamma,limdist,epsrel,radial)
    ellipse = (1.00-elos(e)) if freeLS is None else freeLS
    return offset+amp*major*ellipse*integral

# 3D beta model profile
# ----------------------------------------------------------------------
def betaRadialProfile(x,alpha,beta,gamma): 
    return ((1.00+(x**2.00))**(-1.50*beta))

def betaProfileIntegrand(x,xi,alpha,beta,gamma): 
    return ((1.00+(x**2.00))**(-1.50*beta))*x/((x*x-xi*xi)**0.50)

# Beta model integral
# ----------------------------------------------------------------------
def _betaProfileIntegral(x,alpha,beta,gamma,limdist=np.inf,epsrel=1.00E-06,radial=False): 
    if not radial: return 2.00*quad(betaProfileIntegrand,x,limdist,args=(x,beta),epsrel=epsrel)[0]
    else: return betaRadialProfile(x,alpha,beta,gamma)
betaProfileIntegral = np.vectorize(_betaProfileIntegral)

# Integrated elliptical beta model profile 
# ----------------------------------------------------------------------
def betaProfile(grid,offset,amp,major,e,beta,limdist=np.inf,epsrel=1.00E-06,freeLS=None, radial=False):
    integral = np.zeros_like(grid,dtype=np.float64)
    if not radial: integral[grid<=limdist] = betaProfileIntegral(grid[grid<=limdist],beta,limdist,epsrel,radial)
    else: integral[grid<=limdist] = _betaProfileIntegral(grid[grid<=limdist],beta,limdist,epsrel,radial)
    ellipse = (1.00-elos(e)) if freeLS is None else freeLS
    return offset+amp*major*ellipse*integral