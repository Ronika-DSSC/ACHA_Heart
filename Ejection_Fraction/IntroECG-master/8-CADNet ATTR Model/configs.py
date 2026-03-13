from dataclasses import dataclass
from typing import Dict

ECG_TABULAR_FLOAT = [
    "age_at_ecg",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
]
ECHO_TABULAR_FLOAT = [
    "lvef_value",
    "ivs_measurement",
    "lvpw_measurement",
    "lv_d_measurement",
]
DX_TABULAR_BOOL = [
    "carpal_tunnel_syndrome",
    "lumbar_spine_stenosis",
    "degenerative_joint_disease",
]
TABULAR_BOOL = ["sex"]

TABULAR_FLOAT = ECG_TABULAR_FLOAT + ECHO_TABULAR_FLOAT
TABULAR_PROCESSED = (
    TABULAR_BOOL + [col + "_standard_scale" for col in TABULAR_FLOAT] + DX_TABULAR_BOOL
)


@dataclass
class CadNetModelConfig:
    model_name: str
    model_version: str
    model_asset_dir: str
    features_table: str
    batch_size: int
    model_params: Dict[str, int]
