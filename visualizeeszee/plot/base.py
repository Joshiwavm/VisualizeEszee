from .fourier_sensitivity import PlotFourierSensitivity
from .radial_distributions import PlotRadialDistributions
from .pressure_profiles import PlotPressureProfiles
from .maps import PlotMaps
from .corner_plot import PlotCorner
from .ps_spectra import PlotPointSourceSpectra
from .parameter_table import PlotParameterTable

__all__ = ['PlotGatherer']

class PlotGatherer(PlotFourierSensitivity, PlotRadialDistributions, PlotPressureProfiles, PlotMaps, PlotCorner, PlotPointSourceSpectra, PlotParameterTable):
    """Composite plotting mixin aggregating all plot types."""
    pass
