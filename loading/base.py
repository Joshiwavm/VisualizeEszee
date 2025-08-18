from .data_handler import DataHandler
from .model_handler import ModelHandler
from ..plot import PlotGatherer
from ..fourier import FourierManager
import warnings

class PlotManager(FourierManager, DataHandler, ModelHandler, PlotGatherer):
    """
    Main manager class for loading, processing, and plotting ALMA uv-data and models.
    Inherits from all specialized classes to provide a unified interface.
    """
    def __init__(self, target=None):
        """Initialize with optional target name."""
        self.target = target
        DataHandler.__init__(self)
        ModelHandler.__init__(self)
        
        self._matched_models = {}
        
        # Fourier-related placeholders populated as needed
        self.model_uvgrids = {}
        self.sampled_model_vis = {}
        self.residual_vis = {}

    # ------------------------------------------------------------------
    # Overridden/extended matching: always build Fourier products
    # (Helper methods _get_uv_struct, _convert_model_map_to_jybeam,
    #  _build_model_uv, _sample_model_uv now live in FourierManager.)
    # ------------------------------------------------------------------
    def _match_single(self, model_name: str, data_name: str, notes=None):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        if data_name not in self.uvdata:
            raise ValueError(f"Data set '{data_name}' not found. Available: {list(self.uvdata.keys())}")
        meta = self.uvdata[data_name].get('metadata', {})
        if meta.get('obstype','').lower() != 'interferometer':
            return None
        model_info = self.models[model_name]
        if data_name not in model_info:
            model_info[data_name] = {
                'band': meta.get('band'),
                'array': meta.get('array'),
                'fields': meta.get('fields'),
                'spws': meta.get('spws'),
                'binvis': meta.get('binvis')
            }
            self.add_model_maps(model_name, dataset_name=data_name)
        assoc = {
            'data': data_name,
            'status': 'fourier_pending',
            'notes': notes
        }
        self._matched_models.setdefault(model_name, {})[data_name] = assoc
        # Build Fourier products now
        fields = meta.get('fields', [])
        spws_nested = meta.get('spws', [])
        for f, field in enumerate(fields):
            field_key = f'field{field}'
            for spw in spws_nested[f]:
                spw_key = f'spw{spw}'
                # Validate uv coverage exists
                try:
                    self._get_uv_struct(data_name, field_key, spw_key)
                except Exception as e:
                    raise ValueError(f"No uv data loaded for {data_name}/{field_key}/{spw_key}: {e}")
                # Build & sample
                self._sample_model_uv(model_name, data_name, field_key, spw_key, recompute=True)
        assoc['status'] = 'fourier_ready'

    def match_model(self, model_name: str | None = None, data_name: str | None = None, **kwargs):
        """Match models to interferometer data sets and always build Fourier products."""
        notes = kwargs.get('notes')
        model_list = list(self.models.keys()) if model_name is None else [model_name]
        if not model_list:
            raise ValueError("No models available to match.")
        def is_interf(d):
            return self.uvdata[d].get('metadata', {}).get('obstype','').lower() == 'interferometer'
        if data_name is None:
            data_list = [k for k in self.uvdata.keys() if is_interf(k)]
        else:
            if data_name not in self.uvdata:
                raise ValueError(f"Data set '{data_name}' not found.")
            data_list = [data_name] if is_interf(data_name) else []
        if not data_list:
            return
        for m in model_list:
            for d in data_list:
                self._match_single(m, d, notes=notes)