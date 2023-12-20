import streamlit as st

PDF_FILE = "automatic_reports/pdf_results/result.pdf"

ACTIVITY_THREASHOLD = 20

def TYPE():
    TYPE = {}
    TYPE["ECG_FILTERED"]                 = "ecg_filtered"
    TYPE["ECG"]                          = "ecg"
    TYPE["BREATH_THORACIC_FILTERED"]     = "breath_1_filtered"
    TYPE["BREATH_ABDOMINAL_FILTERED"]    = "breath_2_filtered"
    TYPE["BREATH_THORACIC"]              = "breath_1"
    TYPE["BREATH_ABDOMINAL"]             = "breath_2"
    TYPE["ACCELERATION_X"]      = "accx"
    TYPE["ACCELERATION_Y"]      = "accy"
    TYPE["ACCELERATION_Z"]      = "accz"
    TYPE["RAW_SIGNALS"]         = (TYPE['ECG'] +","+
                                    TYPE["BREATH_THORACIC"] +"," +
                                    TYPE["BREATH_ABDOMINAL"] + ","+
                                    TYPE["ACCELERATION_X"] + "," + 
                                    TYPE["ACCELERATION_Y"] + "," + 
                                    TYPE["ACCELERATION_Z"]) 
    TYPE["FILTERED_SIGNALS"]   = (TYPE["ECG_FILTERED"] + "," + 
                                TYPE["BREATH_THORACIC_FILTERED"] + "," +
                                TYPE["BREATH_ABDOMINAL_FILTERED"])
    
    TYPE["SIGNALS"]            = (TYPE["RAW_SIGNALS"] + "," + 
                                TYPE["FILTERED_SIGNALS"])

    TYPE["CHRONOLIFE_INDICATORS"] = 'heartbeat,HRV,respiratory_rate,averaged_activity,steps_number' 
    TYPE["GARMIN_INDICATORS"]     = 'dailies,epochs,sleeps,allDayRespiration,stressDetails,pulseox'
    TYPE["SLEEP"]                 = 'sleeps'
    
    return TYPE

def COLORS():
    COLORS = {}
    COLORS["text"]          = '#3E738D'
    COLORS["garmin"]        = "#6CCFF6"
    COLORS["chronolife"]    = "#F7921E"
    COLORS["sma"]           = "#2ca02c"
    COLORS["sma_th"]        = "#7f7f7f"
    COLORS["acc_x"]         = "#E7B10A"
    COLORS["acc_y"]         = "#898121"
    COLORS["acc_z"]         = "#4C4B16"
    COLORS["ecg"]           = "#A2407E"
    COLORS["breath"]        = "#3C7290"
    COLORS["breath_chart"]  = "#1f77b4" 
    COLORS["breath_cst"]    = "#1E90FF"
    COLORS["breath_garmin"] = "#164B60"
    COLORS["breath_2"]      = "#51C3CA"
    COLORS["temp"]          = "#FF9300"
    COLORS["temp_2"]        = "#ff6a00"
    COLORS["cardio_chart"]  = "#d62728"
    COLORS["bpm_cst"]       = "#730800"
    COLORS["bpm_garmin"]    = "#730800"

    
    COLORS['stress']        = "#FF9300"
    COLORS['stress_rest']   = "#4594F3"
    COLORS['stress_low']    = "#FFAF54"
    COLORS['stress_medium'] = "#F97516"
    COLORS['stress_high']   = "#DD5809"
    
    COLORS['sleep']         = "#A2407E"
    COLORS['sleep_deep']    = "#044A9A"
    COLORS['sleep_light']   = "#1878CF"
    COLORS['sleep_rem']     = "#9D0FB1"
    COLORS['sleep_awake']   = "#EB79D2"

    COLORS['spo2']          = "#6BA439"
    COLORS['spo2_green']    = "#17A444"
    COLORS['spo2_low']      = "#F8CB4B"
    COLORS['spo2_medium']   = "#F77517"
    COLORS['spo2_high']     = "#CE4A14"
    
    COLORS['pulseox']       = "#6BA439"
    COLORS['bodybattery']   = "#51C3CA"
    
    COLORS['bpm']           = COLORS["ecg"]
    COLORS['sdnn']          = COLORS["ecg"]
    COLORS['hrv']           = COLORS["ecg"]
    COLORS['tachycardia']   = COLORS["ecg"]
    COLORS['bradycardia']   = COLORS["ecg"]
    COLORS['qt']            = COLORS["ecg"]
    COLORS['brpm']          = COLORS["breath"]
    COLORS['rpm_1']         = COLORS["breath"]
    COLORS['brv_1']         = COLORS["breath"]
    COLORS['rpm_2']         = COLORS["breath_2"]
    COLORS['brv_2']         = COLORS["breath_2"]
    COLORS['bior_1']        = COLORS["breath"]
    COLORS['bior_2']        = COLORS["breath_2"]
    COLORS['tachypnea_1']   = COLORS["breath"]
    COLORS['bradypnea_1']   = COLORS["breath"]
    COLORS['tachypnea_2']   = COLORS["breath_2"]
    COLORS['bradypnea_2']   = COLORS["breath_2"]
    # COLORS['bior'] = '%'
    
    return COLORS

def SHORTCUT():
    
    translate = st.session_state.translate
    
    SHORTCUT = {}
    SHORTCUT['bpm']         = translate["health_indicators_heart_bpm_title"]
    SHORTCUT['hrv']         = translate["health_indicators_heart_hrv_title"]
    SHORTCUT['tachycardia'] = translate["tachycardia"]
    SHORTCUT['bradycardia'] = translate["bradycardia"] 
    SHORTCUT['qt']          = translate["qt_length"] 
    SHORTCUT['brpm']        = translate["health_indicators_breath_brpm_title"]
    SHORTCUT['brv']         = translate["health_indicators_breath_brv_title"]
    SHORTCUT['bior']        = translate["inout_length_ratio"] 
    SHORTCUT['tachypnea']   = translate["tachypnea"] 
    SHORTCUT['bradypnea']   = translate["bradypnea"] 
    SHORTCUT['temp']        = translate["temperature"]
    SHORTCUT['n_steps']     = translate["steps_number"]
    SHORTCUT['stress']      = translate["stress_score"]
    SHORTCUT['sleep']       = translate["sleep_quality"]
    SHORTCUT['spo2']        = translate["spo2"]
    SHORTCUT['pulseox']     = translate["pulseox"]
    SHORTCUT['bodybattery'] = translate["bodybattery"]
    
    return SHORTCUT

def UNIT():
    UNIT = {}
    UNIT['bpm'] = 'bpm'
    UNIT['sdnn'] = 'ms'
    UNIT['hrv'] = 'ms'
    UNIT['tachycardia'] = 'bpm'
    UNIT['bradycardia'] = 'bpm'
    UNIT['qt'] = 'ms'
    UNIT['brpm'] = 'brpm'
    UNIT['brv'] = 's'
    UNIT['rpm_1'] = 'brpm'
    UNIT['brv_1'] = 's'
    UNIT['rpm_2'] = 'brpm'
    UNIT['brv_2'] = 's'
    UNIT['tachypnea_1'] = 'brpm'
    UNIT['bradypnea_1'] = 'brpm'
    UNIT['tachypnea_2'] = 'brpm'
    UNIT['bradypnea_2'] = 'brpm'
    UNIT['bior_1'] = ''
    UNIT['bior_2'] = ''
    UNIT['temp'] = '°C'
    UNIT['n_steps'] = ''
    UNIT['stress'] = ''
    UNIT['sleep'] = '%'
    UNIT['spo2'] = ''
    UNIT['pulseox'] = ''
    UNIT['bodybattery'] = '%'
    UNIT['inspi_expi'] = '%'
    
    return UNIT

def STANDARD():
    STANDARD = {}
    STANDARD['bpm'] = '[60 - 100] Resting'
    STANDARD['sdnn'] = '> 100 Resting'
    STANDARD['hrv'] = '> 100 Resting'
    STANDARD['tachycardia'] = ''
    STANDARD['bradycardia'] = ''
    STANDARD['qt'] = ''
    STANDARD['rpm_1'] = '[6 - 20] Resting'
    STANDARD['brv_1'] = ''
    STANDARD['rpm_2'] = '[6 - 20] Resting'
    STANDARD['brv_2'] = ''
    STANDARD['tachypnea_1'] = ''
    STANDARD['bradypnea_1'] = ''
    STANDARD['tachypnea_2'] = ''
    STANDARD['bradypnea_2'] = ''
    STANDARD['bior_1'] = ''
    STANDARD['bior_2'] = ''
    STANDARD['temp'] = ''
    STANDARD['n_steps'] = ''
    STANDARD['stress'] = ''
    STANDARD['sleep'] = ''
    STANDARD['spo2'] = ''
    STANDARD['bodybattery'] = ''
    
    return STANDARD

def VALUE_TYPE():
    VALUE_TYPE = {}
    VALUE_TYPE['bpm'] = 'Median'
    VALUE_TYPE['sdnn'] = 'Median'
    VALUE_TYPE['hrv'] = 'Median'
    VALUE_TYPE['tachycardia'] = 'Median'
    VALUE_TYPE['bradycardia'] = 'Median'
    VALUE_TYPE['qt'] = 'Median'
    VALUE_TYPE['rpm_1'] = 'Median'
    VALUE_TYPE['brv_1'] = 'Median'
    VALUE_TYPE['rpm_2'] = 'Median'
    VALUE_TYPE['brv_2'] = 'Median'
    VALUE_TYPE['bior_1'] = 'Median'
    VALUE_TYPE['bior_2'] = 'Median'
    VALUE_TYPE['tachypnea_1'] = 'Median'
    VALUE_TYPE['bradypnea_1'] = 'Median'
    VALUE_TYPE['tachypnea_2'] = 'Median'
    VALUE_TYPE['bradypnea_2'] = 'Median'
    VALUE_TYPE['temp'] = 'Median'
    VALUE_TYPE['n_steps'] = ''
    VALUE_TYPE['stress'] = ''
    VALUE_TYPE['sleep'] = ''
    VALUE_TYPE['spo2'] = ''
    VALUE_TYPE['pulseox'] = ''
    VALUE_TYPE['bodybattery'] = ''
    
    return VALUE_TYPE

def RANGE():
    RANGE = {}
    RANGE['bpm'] = [30, 140]
    RANGE['sdnn'] = [0, 1000]
    RANGE['hrv'] = [0, 1000]
    RANGE['tachycardia'] = []
    RANGE['bradycardia'] = []
    RANGE['qt'] = [350, 750]
    RANGE['brpm'] = [5, 35]
    RANGE['brv'] = [0, 4]
    RANGE['rpm_1'] = [5, 35]
    RANGE['brv_1'] = [0, 4]
    RANGE['rpm_2'] = [5, 35]
    RANGE['brv_2'] = [0, 4]
    RANGE['tachypnea_1'] = []
    RANGE['bradypnea_1'] = []
    RANGE['tachypnea_2'] = []
    RANGE['bradypnea_2'] = []
    RANGE['bior_1'] = [0, 2]
    RANGE['bior_2'] = [0, 2]
    RANGE['temp'] = [20, 40]
    RANGE['n_steps'] = [0, 200]
    RANGE['stress'] = []
    RANGE['sleep'] = []
    RANGE['spo2'] = []
    RANGE['pulseox'] = []
    RANGE['bodybattery'] = [-5, 110]
    RANGE['bior']=[10,150]
    
    return RANGE

def DEFINITION():
    DEFINITION = {}
    DEFINITION['bpm'] = "Number of Heart Beat Per Minute"
    DEFINITION['sdnn'] = "Standard deviation of times between successive QRS complexes (RR intervals)"
    DEFINITION['hrv'] = "Standard deviation of times between successive QRS complexes (RR intervals)"
    DEFINITION['tachycardia'] = "Number of Heart Beat higher than 100 bpm at rest"
    DEFINITION['bradycardia'] = "Number of Heart Beat lower than 50 bpm at rest"
    DEFINITION['qt'] = "Time between Q and T waves in millisecond normalized by Framingham formula"
    DEFINITION['rpm_1'] = "Number of Respiratory Cycle Per Minute"
    DEFINITION['brv_1'] = "Standard deviation of durations between successive respiratory cycles"
    DEFINITION['rpm_2'] = "Number of Respiratory Cycle Per Minute"
    DEFINITION['brv_2'] = "Standard deviation of durations between successive respiratory cycles"
    DEFINITION['tachypnea_1'] = "Number of Respiratory Cycle higher than 20 cpm at rest"
    DEFINITION['bradypnea_1'] = "Number of Respiratory Cycle lower than 6 cpm at rest"
    DEFINITION['tachypnea_2'] = "Number of Respiratory Cycle higher than 20 cpm at rest"
    DEFINITION['bradypnea_2'] = "Number of Respiratory Cycle lower than 6 cpm at rest"
    DEFINITION['bior_1'] = "Ratio of inhalation time to exhalation time"
    DEFINITION['bior_2'] = "Ratio of inhalation time to exhalation time" 
    DEFINITION['temp'] = ''
    DEFINITION['n_steps'] = ''
    DEFINITION['stress'] = ''
    DEFINITION['sleep'] = ''
    DEFINITION['spo2'] = ''
    DEFINITION['pulseox'] = ''
    DEFINITION['bodybattery'] = ''
    
    return DEFINITION