# SPDX-License-Identifier: Apache-2.0
"""Building blocks for Kandinsky6's audio VAE + vocoder.

The registered top-level module class (``Kandinsky6AudioVAE``, the
``EntryClass`` the model registry discovers) lives one directory up, in
``runtime/models/vaes/kandinsky6_audio.py`` -- NOT in this subpackage.
sglang's registry only AST-scans ``.py`` files directly under
``runtime/models/vaes/``; it never recurses into subpackages, so this
package can only hold building blocks imported by that top-level file.
"""

from .bigvgan import BigVGANV2
from .mel_converter import MelConverter
from .mmaudio_vae import DiagonalGaussianDistribution, MMAudioVAE

__all__ = ["BigVGANV2", "DiagonalGaussianDistribution", "MMAudioVAE", "MelConverter"]
