"""mint -- local Message INTelligence.

A fully local pipeline that classifies messages, extracts tasks and events, and
detects and masks sensitive information. No message text ever leaves the
process: there are no network calls anywhere in this package.
"""

from .pipeline import PipelineResult, ProcessedMessage, run, write_outputs
from .sensitive import mask, scan

__version__ = "1.0.0"

__all__ = [
    "run",
    "write_outputs",
    "PipelineResult",
    "ProcessedMessage",
    "scan",
    "mask",
    "__version__",
]
