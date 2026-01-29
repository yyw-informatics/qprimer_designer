"""Model loading and inference utilities."""

import warnings
from importlib.resources import files
from pathlib import Path
from typing import Tuple

import joblib
import torch

# =============================================================================
# FIX: Register model classes in __main__ for pickle compatibility
# =============================================================================
# ISSUE: The pretrained models (.pth files) were saved using torch.save(model, path)
# from a script where model classes were defined in __main__. This causes
# AttributeError when loading in different contexts (e.g., pytest, imports).
#
# SOLUTION: Register model classes in __main__ before torch.load() is called.
# This allows PyTorch's unpickler to find the classes.
#
# BETTER LONG-TERM FIX: Re-save models using torch.save(model.state_dict(), path)
# and load with model.load_state_dict(). This is the PyTorch best practice.
# =============================================================================

def _register_model_classes_for_pickle():
    """Register model classes in __main__ for pickle compatibility.

    This is needed because the pretrained .pth files were saved with
    torch.save(model, path) from a script where classes were in __main__.
    Without this registration, torch.load() fails with AttributeError.
    """
    import __main__
    from .architectures import (
        PGC,
        DropoutNd,
        S4DKernel,
        S4D,
        Janus,
        MLP,
        CombinedModel,
        CombinedModelClassifier,
        PcrDataset,
    )

    # Only register if not already present
    if not hasattr(__main__, 'CombinedModelClassifier'):
        __main__.PGC = PGC
        __main__.DropoutNd = DropoutNd
        __main__.S4DKernel = S4DKernel
        __main__.S4D = S4D
        __main__.Janus = Janus
        __main__.MLP = MLP
        __main__.CombinedModel = CombinedModel
        __main__.CombinedModelClassifier = CombinedModelClassifier
        __main__.PcrDataset = PcrDataset


# Register classes at module import time
_register_model_classes_for_pickle()

# =============================================================================
# End of pickle compatibility fix
# =============================================================================


def get_model_path(filename: str) -> Path:
    """
    Get path to a bundled model file.

    Args:
        filename: Name of the model file (e.g., 'combined_classifier.pth')

    Returns:
        Path to the model file
    """
    return files('qprimer_designer.data').joinpath(filename)


def load_scaler():
    """
    Load the pre-trained StandardScaler for feature normalization.

    Returns:
        sklearn StandardScaler object
    """
    scaler_path = get_model_path('standard_scaler.joblib')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(scaler_path)


def load_classifier(device: str = 'cpu'):
    """
    Load the pre-trained binary classifier model.

    Args:
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        CombinedModelClassifier in eval mode
    """
    # FIX: Ensure model classes are registered before loading
    # (already done at module import, but explicit call ensures it)
    _register_model_classes_for_pickle()

    model_path = get_model_path('combined_classifier.pth')
    # ORIGINAL: model = torch.load(model_path, map_location=device, weights_only=False)
    # FIX: Added explicit weights_only=False (already present) and class registration above
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def load_regressor(device: str = 'cpu'):
    """
    Load the pre-trained regressor model.

    Args:
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        CombinedModel in eval mode
    """
    # FIX: Ensure model classes are registered before loading
    # (already done at module import, but explicit call ensures it)
    _register_model_classes_for_pickle()

    model_path = get_model_path('combined_regressor.pth')
    # ORIGINAL: model = torch.load(model_path, map_location=device, weights_only=False)
    # FIX: Added explicit weights_only=False (already present) and class registration above
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def load_models(device: str = None) -> Tuple:
    """
    Load all models and scaler for inference.

    Args:
        device: Device to load models on. If None, auto-detects CUDA.

    Returns:
        Tuple of (scaler, classifier, regressor, device_str)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scaler = load_scaler()
    classifier = load_classifier(device)
    regressor = load_regressor(device)

    return scaler, classifier, regressor, device


# Feature columns expected by the models
FEATURE_COLUMNS = [
    "len_f", "Tm_f", "GC_f", "indel_f", "mm_f",
    "len_r", "Tm_r", "GC_r", "indel_r", "mm_r",
    "prod_len", "prod_Tm",
]

# Sequence columns needed for encoding
SEQUENCE_COLUMNS = ["pseq_f", "tseq_f", "pseq_r", "tseq_r"]
