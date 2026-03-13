

NAMES = {
    'age_at_ecg': 'Age',
    'ventricular_rate': 'Ventricular Rate',
    'atrial_rate': 'Atrial Rate',
    'pr_interval': 'PR Interval',
    'qrs_duration': 'QRS Duration',
    'qt_corrected': 'QTc',
    'qt_interval': 'QT interval',

    'sex': 'Male',
    'Sex': 'Male',
    'lvef_lte_45_flag': 'LVEF $\leq$ 45',
    'lvwt_gte_13_flag': 'LVWT $\geq$ 1.3',
    'aortic_stenosis_moderate_or_greater_flag': 'Moderate or Greater Aortic Stenosis',
    'aortic_regurgitation_moderate_or_greater_flag': 'Moderate or Greater Aortic Regurgitation',
    'mitral_regurgitation_moderate_or_greater_flag': 'Moderate or Greater Mitral Regurgitation',
    'tricuspid_regurgitation_moderate_or_greater_flag': 'Moderate or Greater Tricuspid Regurgitation',
    'pulmonary_regurgitation_moderate_or_greater_flag': 'Moderate or Greater Pulmonary Regurgitation',
    'rv_systolic_dysfunction_moderate_or_greater_flag': 'Moderate or Greater RV Systolic Dysfunction',
    'pericardial_effusion_moderate_large_flag': 'Moderate or Greater Pericardial Effusion',
    'pasp_gte_45_flag': 'PASP $\geq$ 45',
    'tr_max_gte_32_flag': 'TR Max V $\geq$ 3.2',
    'shd_moderate_or_greater_flag': 'Moderate or Greater Structural Heart Disease',

    'lvef_lte_35_flag': 'LVEF $\leq$ 35',
    'lvwt_gte_16_flag': 'LVWT $\geq$ 1.6',
    'aortic_stenosis_severe_flag': 'Severe Aortic Stenosis',
    'aortic_regurgitation_severe_flag': 'Severe Aortic Regurgitation',
    'mitral_regurgitation_severe_flag': 'Severe Mitral Regurgitation',
    'tricuspid_regurgitation_severe_flag': 'Severe Tricuspid Regurgitation',
    'pulmonary_regurgitation_severe_flag': 'Severe Pulmonary Regurgitation',
    'rv_systolic_dysfunction_severe_flag': 'Severe RV Systolic Dysfunction',
    'pericardial_effusion_large_flag': 'Large Pericardial Effusion',
    'pasp_gte_60_flag': 'PASP $\geq$ 60',
    'tr_max_gte_36_flag': 'TR Max V $\geq$ 36',

    'inpatient': 'Inpatient',
    'emergency': 'Emergency',
    'outpatient': 'Outpatient',
    'procedural': 'Procedural',
    'hispanic': 'Hispanic',
    'white': 'White',
    'black': 'Black',
    'unknown': 'Unknown',
    'other': 'Other',
    'asian': 'Asian',
    'race_ethnicity': 'Race/Ethnicity',
    'location_setting': 'ECG Setting',
    'acquisition_year': 'Year',
    'most_recent_ecg': 'Most Recent ECG',

    'aortic_stenosis_value': 'Aortic Stenosis Value',
    'aortic_regurgitation_value': 'Aortic Regurgitation Value',
    'mitral_regurgitation_value': 'Mitral Regurgitation Value',
    'tricuspid_regurgitation_value': 'Tricuspid Regurgitation Value',
    'pulmonary_regurgitation_value': 'Pulmonary Regurgitation Value',
    'rv_systolic_function_value': 'RV Systolic Function Value',
    'pericardial_effusion_value': 'Pericardial Effusion Value',

    'ivs_measurement': 'Intraventricular Septum Thickness',
    'lvpw_measurement': 'Left Ventricular Posterior Wall Thickness',
    'lvwt_measurement': 'Left Ventricular Wall Thickness',
    'pasp_value': 'PASP',
    'tr_max_velocity_value': 'TR Max V',
    'lvef_value': 'LVEF',
}

def get_name(field):
    if field in NAMES:
        return NAMES[field]
    else:
        return field

SHD_MODERATE_OR_GREATER_COLS = [
    'lvef_lte_45_flag',
    'lvwt_gte_13_flag',
    'aortic_stenosis_moderate_or_greater_flag',
    'aortic_regurgitation_moderate_or_greater_flag',
    'mitral_regurgitation_moderate_or_greater_flag',
    'tricuspid_regurgitation_moderate_or_greater_flag',
    'pulmonary_regurgitation_moderate_or_greater_flag',
    'rv_systolic_dysfunction_moderate_or_greater_flag',
    'pericardial_effusion_moderate_large_flag',
    'pasp_gte_45_flag',
    'tr_max_gte_32_flag',
    'shd_moderate_or_greater_flag'
]

RAW_ECHO_CATEGORICAL = [
    'aortic_stenosis_value',
    'aortic_regurgitation_value',
    'mitral_regurgitation_value',
    'tricuspid_regurgitation_value',
    'pulmonary_regurgitation_value',
    'rv_systolic_function_value',
    'pericardial_effusion_value',
]

RAW_ECHO_NUMERICAL = [
    'lvef_value',
    'ivs_measurement',
    'lvpw_measurement',
    'lvwt_measurement',
    'pasp_value',
    'tr_max_velocity_value',
]

TABULAR_NUMERICAL = ['age_at_ecg', 'ventricular_rate', 'atrial_rate', 
                'pr_interval', 'qrs_duration',  'qt_corrected', 'qt_interval']

FORMAT_AS_FLOAT = [
    'lvef_value',
    'ivs_measurement',
    'lvpw_measurement',
    'lvwt_measurement',
    'tr_max_velocity_value',
    'qt_interval'

]

NUMERIC_COLS = TABULAR_NUMERICAL + RAW_ECHO_NUMERICAL
HIST_COLS = NUMERIC_COLS.copy()
HIST_COLS.remove('lvwt_measurement')


BOOL_COLS = (
    ['sex'] +
    SHD_MODERATE_OR_GREATER_COLS
)

METADATA_CAT_COLS = ['race_ethnicity', 'location_setting']

CAT_COLS = METADATA_CAT_COLS + RAW_ECHO_CATEGORICAL

CUSTOM_ORDER = [
    'NaN', 'none', 'normal', 'presumed none',
    'trace',
    'mild', 'mildly_reduced', 'small',
    'moderate', 'moderately_reduced',
    'severe', 'severely_reduced', 'large'
]