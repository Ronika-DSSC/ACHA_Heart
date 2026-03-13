# Cadnet Inference Pipeline

This repository provides a minimal implementation of the inference pipeline for the Cadnet (ATTRACTnet) model, designed to identify patients at risk for transthyretin amyloid cardiomyopathy (ATTR-CM) using ECG and echocardiographic data.

This code is intended for peer review purposes only and is not suitable for clinical use.

## Repository Structure
```
8-CADNet ATTR Model/
├── run_cadnet_inference.py         # Main script to run inference
├── utils.py                        # Preprocessing and utility functions
├── model.py                        # Model definition: ResNet1d with tabular 
├── configs.py                      # Model configuration and feature column definitions  
├── pyproject.toml                  # Python dependencies
└── README.md                       # Project documentation
```

## Environment 
For python environment managemnent we use [uv](https://docs.astral.sh/uv/):
- dependencies are in the `pyproject.toml` file
- run `uv sync` to install the virtual environment

## Model Assets

To run the CADNet inference pipeline, specific model assets are required and must be located in the designated directory structure. These model assets are proprietary and are not distributed with this repository. 

The following files are expected to be present in the `ml/cadnet/cadnet_non_vent_paced/versions/2/` directory:

*   `ecg_preprocessing_pipeline.joblib`: Preprocessing pipeline for ECG tabular features.
*   `echo_preprocessing_pipeline.joblib`: Preprocessing pipeline for echo tabular features.
*   `ecg_waveform_preprocessing_params.json`: Truncation and normalization parameters for ECG waveforms.
*   `weights.pt`: Trained PyTorch model weights.

# Feature Specification: `cadnet_model_features`

The model expects a structured input table with the following features, derived from ECG and echocardiographic studies:

## Identifiers
- `ecg_key`: Unique ECG identifier
- `echo_key`: Unique echo identifier
- `patient_key`: Patient identifier

## Demographics
- `sex`: Biological sex (`male` or `female`)
- `birth_date`: Date of birth
- `age_at_ecg`: Age at time of ECG

## Timestamps
- `acquisition_datetime`: ECG acquisition time
- `study_datetime`: Echo study time
- `partition_datetime`: Max of ECG or echo datetime

## ECG Features
- `ventricular_pacing_flag`
- `poor_data_quality_flag`
- `ventricular_rate`
- `atrial_rate`
- `pr_interval`
- `qrs_duration`
- `qt_corrected`

## Echo Features
- `lvef_value`: Left ventricular ejection fraction
- `ivs_measurement`: Interventricular septum thickness
- `lvpw_measurement`: Left ventricular posterior wall thickness
- `lv_d_measurement`: LV diastolic dimension
- `max_lv_wall_thickness`: Max of IVS or LVPW

## Waveform Path
- `processed_ecg_blob_path`: Path to preprocessed ECG waveform blob
    > baseline wander removal has already been applied to processed ECG arrays
- `waveform_array`: NumPy array containing 12 lead ECG of shape (12, 2500, 1)

## Comorbidity Flags
- `carpal_tunnel_syndrome`: Binary flag (0  or 1)
- `lumbar_spine_stenosis`: Binary flag (0  or 1)
- `degenerative_joint_disease`: Binary flag (0  or 1)
