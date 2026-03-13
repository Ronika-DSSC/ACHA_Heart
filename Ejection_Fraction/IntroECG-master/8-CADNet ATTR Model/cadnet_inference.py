from model import generate_pred
from utils import preprocess_cadnet

from configs import CadNetModelConfig

import pandas as pd


def run_cadnet_inference(config, ecg_echo_pairs):
    """
    Runs inference using the CadNet model on a batch of ECG and echo data pairs.

    This function:
    - Loads preprocessing pipelines and waveform normalization parameters
    - Preprocesses tabular and waveform data
    - Loads model weights and generates predictions
    - Returns a DataFrame with ECG and echo keys and associated predictions

    Parameters:
        config (CadNetModelConfig): Configuration object containing model metadata and parameters.
        ecg_echo_pairs (pd.DataFrame): Input DataFrame with ECG and echo data, including waveform arrays and tabular features.

    Returns:
        pd.DataFrame: DataFrame with columns ['ecg_key', 'echo_key', 'prediction'].
    """

    ecg_tabular_transformer_path = (
        f"{config.model_asset_dir}/ecg_preprocessing_pipeline.joblib"
    )
    echo_tabular_transformer_path = (
        f"{config.model_asset_dir}/echo_preprocessing_pipeline.joblib"
    )
    waveform_params_path = (
        f"{config.model_asset_dir}/ecg_waveform_preprocessing_params.json"
    )
    checkpoint = f"{config.model_asset_dir}/weights.pt"

    tabular, waveforms = preprocess_cadnet(
        ecg_echo_pairs,
        ecg_tabular_transformer_path,
        echo_tabular_transformer_path,
        waveform_params_path,
    )

    preds = generate_pred(
        config.model_params,
        checkpoint,
        waveforms,
        tabular,
        batch_size=config.batch_size,
    )
    results = ecg_echo_pairs[["ecg_key", "echo_key"]].copy()
    results["prediction"] = preds

    return results


def main():
    """
    Entry point for running CadNet inference.

    This function:
    - Initializes model configuration
    - Loads input data from a Parquet file
    - Runs inference using the CadNet model
    - Saves prediction results to a Parquet file
    """

    cadnet_config = CadNetModelConfig(
        model_name="cadnet",
        model_version="cadnet_non_vent_paced:2",
        model_asset_dir="ml/cadnet/cadnet_non_vent_paced/versions/2/",
        features_table="cadnet_model_features",
        batch_size=256,
        model_params={"len_tabular_feature_vector": 14, "filter_size": 32},
    )

    ecg_echo_pairs = pd.read_parquet(r"ecg_echo_pairs.parquet")
    results = run_cadnet_inference(cadnet_config, ecg_echo_pairs)
    results.to_parquet("results.parquet")


if __name__ == "__main__":
    main()
