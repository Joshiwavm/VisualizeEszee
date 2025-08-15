from .data_handler import DataHandler
from .model_handler import ModelHandler
from ..plot import PlotGatherer

class PlotManager(DataHandler, ModelHandler, PlotGatherer):
    """
    Main manager class for loading, processing, and plotting ALMA uv-data and models.
    Inherits from all specialized classes to provide a unified interface.
    """
    def __init__(self, target=None):
        """
        Initialize with optional target name. 
        """
        self.target = target
        
        # Initialize all parent classes
        DataHandler.__init__(self)
        ModelHandler.__init__(self)