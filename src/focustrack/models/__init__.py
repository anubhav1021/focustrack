"""Layer 4 - the intelligence layer.

Two models, each answering a different question:

``focus_state``  what is the user doing right now?
``focus_drop``   is a focused user about to lose focus?

The reminder engine in :mod:`focustrack.engine` consumes both.
"""

from focustrack.models.baseline import RuleBasedClassifier, build_rule_baseline
from focustrack.models.focus_drop import (
    DROP_LABEL,
    FocusDropResult,
    build_drop_labels,
    focused_windows,
    predict_drop_probability,
    train_focus_drop,
)
from focustrack.models.focus_state import (
    FocusStateResult,
    ModelScores,
    build_models,
    classification_text,
    train_focus_state,
)
from focustrack.models.registry import (
    ModelBundle,
    load_bundle,
    models_available,
    save_bundle,
)

__all__ = [
    "DROP_LABEL",
    "FocusDropResult",
    "FocusStateResult",
    "ModelBundle",
    "ModelScores",
    "RuleBasedClassifier",
    "build_drop_labels",
    "build_models",
    "build_rule_baseline",
    "classification_text",
    "focused_windows",
    "load_bundle",
    "models_available",
    "predict_drop_probability",
    "save_bundle",
    "train_focus_drop",
    "train_focus_state",
]
