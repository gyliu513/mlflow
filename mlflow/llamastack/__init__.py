"""
MLflow integration for LlamaStack API autologging.

This module provides autologging capabilities for LlamaStack's chat completions API,
automatically capturing traces, token usage, and model information.
"""

from mlflow.llamastack.autolog import autolog
from mlflow.llamastack.constant import FLAVOR_NAME

__all__ = ["autolog", "FLAVOR_NAME"]
