from .fourier_sensitivity import PlotFourierSensitivity
from .radial_distributions import PlotRadialDistributions
from .pressure_profiles import PlotPressureProfiles
from .maps import PlotMaps

__all__ = ['PlotGatherer']

class PlotGatherer(PlotFourierSensitivity, PlotRadialDistributions, PlotPressureProfiles, PlotMaps):
    """Composite plotting mixin aggregating all plot types."""
    pass
