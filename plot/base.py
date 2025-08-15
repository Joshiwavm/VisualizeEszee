from .fourier_sensitivity import PlotFourierSensitivity
from .radial_distributions import PlotRadialDistributions
from .pressure_profiles import PlotPressureProfiles
from .maps import PlotMaps

class PlotGatherer(PlotFourierSensitivity, PlotRadialDistributions, PlotPressureProfiles, PlotMaps):
    """Composite plotting mixin aggregating all plot types."""
    pass

__all__ = ['PlotGatherer']
