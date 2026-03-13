import json
import os

import numpy as np
import scipy
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def baseline_wander_removal(ecg_array, sampling_frequency=250):
    """
    Applies a two-step median filter to remove baseline wander from a single ECG waveform.

    This function expects an input ECG array of shape (12, 2500, 1), where each of the 12 leads
    contains 2500 time samples. It normalizes the baseline of each lead independently and returns
    the processed waveform with the same shape.

    Parameters:
        ecg_array (np.ndarray): ECG input array of shape (12, 2500, 1).
        sampling_frequency (int): Sampling frequency in Hz (default: 250).

    Returns:
        np.ndarray: Baseline-corrected ECG array of shape (12, 2500).
    """

    if ecg_array.shape[0] != 12:
        ecg_array = np.transpose(ecg_array, axes=[1, 0, 2])

    assert ecg_array.shape == (12, 2500, 1), "ecg shape is not (12, 2500, 1)"
    processed_ecg_array = np.zeros(ecg_array.shape)

    for lead in range(ecg_array.shape[0]):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(ecg_array[lead, :], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)

        # Removing baseline wander
        filt_data = ecg_array[lead, :] - baseline
        processed_ecg_array[lead, :] = filt_data

    return processed_ecg_array


def baseline_wander_removal_batch(dataset):
    """
    Applies baseline wander removal to a batch of ECG waveforms using a two-step
    median filter. This function ensures all inputs are in the expected shape
    (12 leads x 2500 samples x 1 channel) before processing.

    Parameters:
        dataset (np.ndarray): A 4D NumPy array of ECG waveforms with shape
                              (N, 2500, 12, 1) or (N, 12, 2500, 1), where N is
                              the number of samples in the batch.

    Returns:
        np.ndarray: A 4D NumPy array of baseline-corrected ECG waveforms with
                    shape (N, 12, 2500, 1).
    """

    output_list = []

    if dataset.shape[1:] == (2500, 12, 1):
        # print(f'dataset shape: {dataset.shape} transposing now...')
        dataset = np.transpose(dataset, axes=[0, 2, 1, 3])

    for ecg in dataset:
        assert ecg.shape == (12, 2500, 1), "ecg is not (12,2500,1)"
        processed_data = baseline_wander_removal(ecg.squeeze(), 250)
        output_list.append(processed_data)

    output_array = np.expand_dims(np.array(output_list), axis=3)
    return output_array


def preprocess_tabular(df, tab_cols, how="fit", save_path=None, silent=True):
    """
    Applies standard scaling and median imputation to tabular features using a scikit-learn pipeline.

    If `how="fit"`, the function fits a pipeline on the specified columns and optionally saves it.
    If `how="transform"`, it loads a saved pipeline and applies it to the data.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing tabular features.
        tab_cols (list): List of column names to preprocess.
        how (str): Either "fit" to train and save the pipeline, or "transform" to apply a saved one.
        save_path (str, optional): Path to save or load the pipeline.
        silent (bool): If False, prints pipeline parameters and output feature names.

    Returns:
        pd.DataFrame: DataFrame with new standardized columns appended (suffix: "_standard_scale").
    """
    if how == "fit":
        pipe = Pipeline(
            [("scale", StandardScaler()), ("impute", SimpleImputer(strategy="median"))]
        )

        tab_scaled = pipe.fit_transform(df[tab_cols])
        new_cols = df[tab_cols].add_suffix("_standard_scale").columns
        df[new_cols] = tab_scaled
        if save_path:
            dump(pipe, save_path)

    if how == "transform":
        pipe = load(save_path)
        tab_scaled = pipe.transform(df[tab_cols])
        new_cols = df[tab_cols].add_suffix("_standard_scale").columns
        df[new_cols] = tab_scaled

    if not silent:
        # Print pipe params
        print("Scaler:")
        print("- Mean:", pipe.named_steps["scale"].mean_)
        print("- Scale:", pipe.named_steps["scale"].scale_)
        print("Imputer:", pipe.named_steps["impute"].statistics_)
        print("Output Feature Names:", pipe.get_feature_names_out())
    return df


def get_train_waveform_truncation_params(params_file):
    """
    Loads per-lead truncation and normalization parameters from a JSON file.

    Parameters:
        params_file (str): Path to the JSON file containing truncation and normalization limits.

    Returns:
        dict: Dictionary with keys: 'lowerbound', 'upperbound', 'mean', and 'std'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """

    if os.path.isfile(params_file):
        with open(params_file, "r") as fid:
            limits = json.load(fid)
    else:
        raise FileNotFoundError("ECG limits file doesn't exsit!")
    return limits


def per_lead_truncation(data, limits, silent=True):
    """
    Applies per-lead truncation to ECG data using specified lower and upper bounds.

    Parameters:
        data (np.ndarray): ECG data of shape (N, 12, 2500, 1).
        limits (dict): Dictionary with 'lowerbound' and 'upperbound' lists for each lead.
        silent (bool): If False, prints min/max values before and after truncation.

    Returns:
        np.ndarray: Truncated ECG data of the same shape.
    """
    assert data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"
    assert len(limits["lowerbound"]) == data.shape[1], (
        "shape of normalization params doesn't match data.shape[1]"
    )

    # Truncate data to 0.1th and 99.9th percentile
    for lead in range(len(limits["lowerbound"])):
        if not silent:
            print(f"Lead {lead} pre-truncation min: {data[:, lead, :, :].min()}")
            print(f"Lead {lead} pre-truncation max: {data[:, lead, :, :].max()}")

        data[:, lead, :, :] = np.where(
            data[:, lead, :, :] > limits["upperbound"][lead],
            limits["upperbound"][lead],
            data[:, lead, :, :],
        )
        data[:, lead, :, :] = np.where(
            data[:, lead, :, :] < limits["lowerbound"][lead],
            limits["lowerbound"][lead],
            data[:, lead, :, :],
        )

    for lead in range(len(limits["lowerbound"])):
        if not silent:
            print(f"Lead {lead} post-truncation min: {data[:, lead, :, :].min()}")
            print(f"Lead {lead} post-truncation max: {data[:, lead, :, :].max()}")

    return data


def per_lead_normalization(data, params, silent=True):
    """
    Applies per-lead normalization to ECG data using mean and standard deviation.

    Parameters:
        data (np.ndarray): ECG data of shape (N, 12, 2500, 1).
        params (dict): Dictionary with 'mean' and 'std' lists for each lead.
        silent (bool): If False, prints min/max values before and after normalization.

    Returns:
        np.ndarray: Normalized ECG data of the same shape.
    """

    assert data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"
    assert len(params["mean"]) == data.shape[1], (
        "shape of normalization params doesn't match data.shape[1]"
    )

    for lead in range(len(params["mean"])):
        if not silent:
            print(f"Lead {lead} pre-normalizing min:", data[:, lead, :, :].min())
            print(f"Lead {lead} pre-normalizing max:", data[:, lead, :, :].max())

        data[:, lead, :, :] = (data[:, lead, :, :] - params["mean"][lead]) / params[
            "std"
        ][lead]

        if not silent:
            print(f"Lead {lead} post-normalizing min: {data[:, lead, :, :].min()}")
            print(f"Lead {lead} post-normalizing max: {data[:, lead, :, :].max()}")

    return data


def per_lead_truncation_normalization_batch(data, params, silent=True):
    """
    Applies per-lead truncation and normalization to ECG data, then transposes it
    to the shape expected by the model.

    Parameters:
        data (np.ndarray): Raw ECG data of shape (N, 12, 2500, 1).
        params (dict): Dictionary with 'lowerbound', 'upperbound', 'mean', and 'std' per lead.
        silent (bool): If False, prints debug information during processing.

    Returns:
        np.ndarray: Processed ECG data of shape (N, 1, 2500, 12), ready for model input.
    """

    data = per_lead_truncation(data, params, silent)
    data = per_lead_normalization(data, params, silent)

    # Transpose data for model
    data = np.transpose(data, axes=[0, 3, 2, 1])
    if not silent:
        print(f"Normalized data shape for model: {data.shape}")
    assert data.shape[1:] == (1, 2500, 12), "dataset is not X,1,2500,12"
    return data


def preprocess_ecg_waveforms(
    ecg_data,
    ecg_param_path,
    baseline_wander_removal=False,
    per_lead_truncation_normalization=True,
    silent=True,
):
    """
    Preprocesses ECG waveform data by optionally applying baseline wander removal
    and per-lead truncation and normalization.

    Parameters:
        ecg_data (pd.DataFrame): DataFrame containing a 'waveform_array' column with ECG arrays.
        ecg_param_path (str): Path to JSON file with per-lead truncation and normalization parameters.
        baseline_wander_removal (bool): Whether to apply baseline wander removal (default: False).
        per_lead_truncation_normalization (bool): Whether to apply per-lead truncation and normalization (default: True).
        silent (bool): If False, prints progress and debug information.

    Returns:
        np.ndarray: Preprocessed ECG waveform data of shape (N, 1, 2500, 12), ready for model input.
    """
    # stacked waveform array
    ecg_waveform_data = np.stack(ecg_data["waveform_array"])

    if baseline_wander_removal:
        if not silent:
            print("Running baseline wander removal.")
        ecg_waveform_data = baseline_wander_removal_batch(ecg_waveform_data)
        if not silent:
            print(
                f"Completed baseline wander removal. Output shape: {ecg_waveform_data.shape}"
            )

    if per_lead_truncation_normalization:
        if not silent:
            print("Running per-lead truncation and mean normalization.")
        params = get_train_waveform_truncation_params(params_file=ecg_param_path)
        ecg_waveform_data = per_lead_truncation_normalization_batch(
            ecg_waveform_data, params
        )
        if not silent:
            print(
                f"Completed per-lead truncation and mean normalization. Output shape: {ecg_waveform_data.shape}"
            )

    return ecg_waveform_data


def preprocess_cadnet(
    data, ecg_joblib_path, echo_joblib_path, ecg_waveform_param_path, float_precision=32
):
    """
    Preprocesses both tabular and waveform data for input into the CadNet model.

    This includes:
    - Encoding categorical variables (e.g., sex)
    - Filling missing values in tabular features
    - Applying saved preprocessing pipelines to ECG and echo features
    - Running waveform preprocessing (truncation, normalization)

    Parameters:
        data (pd.DataFrame): Input DataFrame containing tabular and waveform data.
        ecg_joblib_path (str): Path to saved preprocessing pipeline for ECG tabular features.
        echo_joblib_path (str): Path to saved preprocessing pipeline for echo tabular features.
        ecg_waveform_param_path (str): Path to JSON file with waveform normalization parameters.
        float_precision (int): Precision for output arrays (default: 32-bit floats).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (tabular_data, waveform_data), both as NumPy arrays.
    """

    from configs import (
        DX_TABULAR_BOOL,
        ECG_TABULAR_FLOAT,
        ECHO_TABULAR_FLOAT,
        TABULAR_PROCESSED,
    )

    # encode tabular as numeric
    data["sex"] = (
        data["sex"]
        .map(lambda x: 1 if x == "male" else 0 if x == "female" else None)
        .astype(np.int64)
    )
    data["atrial_rate"] = data["atrial_rate"].fillna(0)
    data["pr_interval"] = data["pr_interval"].fillna(0)
    data[DX_TABULAR_BOOL] = data[DX_TABULAR_BOOL].fillna(0)

    # ECG
    preprocess_tabular(
        data, ECG_TABULAR_FLOAT, how="transform", save_path=ecg_joblib_path
    )

    # Echo
    preprocess_tabular(
        data, ECHO_TABULAR_FLOAT, how="transform", save_path=echo_joblib_path
    )

    tabular_data = data[TABULAR_PROCESSED]
    assert tabular_data.isna().sum().sum() == 0, "Error! pred_tabular_data has NaNs"

    waveform_data = preprocess_ecg_waveforms(
        data,
        ecg_waveform_param_path,
        baseline_wander_removal=False,
        per_lead_truncation_normalization=True,
    )

    tabular_data = tabular_data.to_numpy().astype(f"float{float_precision}")
    waveform_data = waveform_data.astype(f"float{float_precision}")
    return tabular_data, waveform_data


def array_splitter(array: np.array, records: int, generator=True):
    """
    This takes a dataframe as input and splits
    it into a list of dataframes of size `records`

    This replaces the deprecated use of np.array_split
    on pandas DataFrames.
    """
    if generator:
        n_splits = int(np.ceil(len(array) / records))
        return (array[records * i : records * (i + 1)] for i in range(n_splits))
    else:
        arrays = []
        n_splits = int(np.ceil(len(array) / records))
        for i in range(n_splits):
            start = records * i
            stop = records * (i + 1)
            arrays.append(array[start:stop])
        return arrays
