from .data import (
    FocalPrior,
    FundamentalObservation,
    IntrinsicsEstimate,
    OptimizerBounds,
    PairMatches,
    PrincipalPointPrior,
    ProfileSample,
)
from .colmap_bridge import (
    pinhole_camera_line,
    replace_pinhole_camera_in_txt,
    write_single_pinhole_cameras_txt,
)

__all__ = [
    "FOnlyProfileConfig",
    "FOnlyProfileOptimizer",
    "FocalPrior",
    "FundamentalObservation",
    "IntrinsicsEstimate",
    "KFPPSFocalProfileConfig",
    "KFPPSFocalProfileOptimizer",
    "MogePointLMConfig",
    "MogePointLMOptimizer",
    "OptimizerBounds",
    "PairMatches",
    "PrincipalPointPrior",
    "ProfileSample",
    "RawSampsonConfig",
    "RawSampsonJointOptimizer",
    "pinhole_camera_line",
    "replace_pinhole_camera_in_txt",
    "write_single_pinhole_cameras_txt",
]

_LAZY_IMPORTS = {
    "FOnlyProfileConfig": (".f_only_profile", "FOnlyProfileConfig"),
    "FOnlyProfileOptimizer": (".f_only_profile", "FOnlyProfileOptimizer"),
    "KFPPSFocalProfileConfig": (".f_only_profile", "KFPPSFocalProfileConfig"),
    "KFPPSFocalProfileOptimizer": (".f_only_profile", "KFPPSFocalProfileOptimizer"),
    "MogePointLMConfig": (".moge_point_lm", "MogePointLMConfig"),
    "MogePointLMOptimizer": (".moge_point_lm", "MogePointLMOptimizer"),
    "RawSampsonConfig": (".raw_sampson", "RawSampsonConfig"),
    "RawSampsonJointOptimizer": (".raw_sampson", "RawSampsonJointOptimizer"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
