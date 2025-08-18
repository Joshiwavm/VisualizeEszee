"""Fourier (visibility) operations package.

Provides the `FourierManager` mixin with static utility methods for:
  * map -> uv grid (FFT)
  * uv grid sampling at arbitrary (u,v)
  * (future) imaging & multi-field combining
  * (future) visibility algebra (subtract / scale / phase shift)

Current stage: skeleton for steps 1-2 only (map_to_uvgrid, sample_uv).
"""
from .manager import FourierManager
__all__ = ["FourierManager"]
