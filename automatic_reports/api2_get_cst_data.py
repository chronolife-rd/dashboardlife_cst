
# Imports
import json
import copy
import math
import requests
import pandas as pd
import numpy as np
import statistics
from datetime import datetime, timedelta

import streamlit as st

from automatic_reports.useful_functions import find_time_intervals, sum_time_intervals, timedelta_formatter, unwrap, unwrap_ratio_inspi_expi
from automatic_reports.config import CST_SIGNAL_TYPES
from automatic_reports.config import RED_ALERT, GREEN_ALERT, ALERT_SIZE, ACTIVITY_THREASHOLD, TEMPERATURE_THREASHOLD
from automatic_reports.config import TACHYPNEA_TH, BRADYPNEA_TH, TACHYCARDIA_TH, BRADYCARDIA_TH,\
    QT_MAX_TH, QT_MIN_TH
    
# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
# Request user's data from servers

def get_cst_data(user_id, date, api, url):

    # We have to request data from date-1 and date+1, because we don't know 
    # the offset (if there is no data in requested date (utc))
    date_after = change_date(date, sign = +1)
    date_before = change_date(date, sign = -1)
    
    print("-- Requesting Chronolife data for 3 days --")
    datas = get_datas(user_id, date, CST_SIGNAL_TYPES, api, url)
    datas_before = get_datas(user_id, date_before, CST_SIGNAL_TYPES, api, url)
    datas_after = get_datas(user_id, date_after, CST_SIGNAL_TYPES, api, url)
    
    datas_3_days = datas + datas_before + datas_after
   
   #  Manage display of chronolife indicators when the user use the t-shirt 
    if len(datas)==0:
        st.session_state.chronolife_data_available = False
    else :
        st.session_state.chronolife_data_available = True

    # Get offset zone value 
    offset = get_cst_offset(datas_3_days) 
    # Save the results in a dictionary
    results_dict = initialize_dictionary_with_template()
    results_dict['user_id']  = user_id
    results_dict['offset']  = offset
    add_cardio(date, datas_3_days, results_dict['cardio'], offset)
    add_breath(date, datas_3_days, results_dict['breath'], offset)                  
    add_activity(date, datas_3_days, results_dict['activity'], offset)
    add_temperature(date, datas_3_days, results_dict['temperature'], offset)
    add_durations(date, results_dict) 
    
    # Add activity level to each indicator
    results_dict_2 = add_activity_to_indicators(copy.deepcopy(results_dict))

    # Add temperature to each indicator
    results_dict_3 = add_temperature_to_indicators(copy.deepcopy(results_dict_2))

    # Add anomalie (alerts)
    add_anomalies(results_dict_3, date)
      
    return results_dict_3

# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------
def change_date(date:str, sign = -1) -> str:
    date_today = datetime.strptime(date, "%Y-%m-%d")
    new_date = date_today + sign*timedelta(days=1)
    new_date =  datetime.strftime(new_date, "%Y-%m-%d")
    return new_date

def get_datas(user_id, date, signal_types, api, url):
    params = {
        'user'    : user_id,  
        'types'   : signal_types,
        'date'    : date,
        }
    
    reply = request_data_from_servers(params, api, url)
    datas = error_management(date, reply)

    return datas

def request_data_from_servers(params, api_key, url):
    # Perform the POST request authenticated with YOUR API key (NOT the one of 
    # the sub-user!).
    reply = requests.get(url, headers={"X-API-Key": api_key}, params=params)
    
    return reply

def error_management(date, reply) :
    datas = []
    
    # Error management 
    if reply.status_code == 200:
      # Convert the reply content into a json object.
      json_list_of_records = json.loads(reply.text) 
      for record in json_list_of_records:
          datas.append(record)
    elif reply.status_code == 400:
        print('Part of the request could not be parsed or is incorrect.')
    elif reply.status_code == 401:
        print('Invalid authentication')
    elif reply.status_code == 403:
        print('Not authorized.')
    elif reply.status_code == 404:
        print('Invalid url')
    elif reply.status_code == 500:
        print('Invalid user ID')
    
    if len(datas) == 0:
        print('No Chronolife data found for , day:', date)

    return datas

def initialize_dictionary_with_template() -> dict :  
    activity_dict = {
        'steps' : "",
        'averaged_activity' : "",
        'distance' : "",
        }
    anomalies_dict = initialize_alerts_with_template()
    breath_dict = {
        'rate' : "", 
        'rate_var' : "",
        'inspi_expi' : "",
        }
    cardio_dict = {
        'rate' : "", 
        'rate_var' : "",
         }
    duration_dict = {
        'intervals' : "", 
        'collected' : "",
        'day' : "", 
        'night' : "",
        'rest' : "",
        'active' : "",
        }
    temperature_dict = {
        'values' : "",
        'mean' : "", 
        'min' : "", 
        'max' : "", 
    }

    dict_template = {
                    'user_id' : "",
                    'offset' : 7200,
                    'activity' : copy.deepcopy(activity_dict),
                    'anomalies' : copy.deepcopy(anomalies_dict),
                    'breath' : copy.deepcopy(breath_dict),
                    'cardio' : copy.deepcopy(cardio_dict),
                    'temperature' : copy.deepcopy(temperature_dict),
                    'duration': copy.deepcopy(duration_dict),
                    }
    return copy.deepcopy(dict_template)

def initialize_alerts_with_template() -> dict :
    pdf_info = {
        "path" : GREEN_ALERT,
        "x" : "",
        "y" : "",
        "w" : ALERT_SIZE,
        "h" : ALERT_SIZE,
        "exists" : False,
        "night" : "",
        "morning" : "",
        "evening" : "",
        "mean" : "",
        "percentage" : "",
        "duration" : "",
        "values" : ""
        }

    dict_template = {
        "tachypnea"   : copy.deepcopy(pdf_info), 
        "bradypnea"   : copy.deepcopy(pdf_info),
        "tachycardia" : copy.deepcopy(pdf_info), 
        "bradycardia" : copy.deepcopy(pdf_info),
        "qt"          : copy.deepcopy(pdf_info),
    }
    return copy.deepcopy(dict_template)

def add_cardio(date, datas, cardio_dict, offset):
    rate = get_cst_result_info(date, datas, offset, result_type='heartbeat')
    rate_var = get_cst_result_info(date, datas, offset, result_type='HRV')
    qt = get_cst_result_info_segment(date, datas, offset, result_type='qt_c_framingham_per_seg', type = 'qt')

    cardio_dict['rate'] = rate
    cardio_dict['rate_var'] = rate_var
    cardio_dict['qt'] = qt

def add_breath(date, datas, breath_dict, offset):
    rate = get_cst_result_info(date, datas, offset, result_type='breath_2_brpm')
    rate_var = get_cst_result_info(date, datas, offset, result_type='breath_2_brv')
    inspi_expi = get_cst_result_info_segment(date, datas, offset, result_type='breath_2_inspi_over_expi', type = 'ratio')

    breath_dict['rate'] = rate
    breath_dict['rate_var'] = rate_var
    breath_dict['inspi_expi'] = inspi_expi

def add_temperature(date, datas, temperature_dict, offset):
    right = get_cst_result_info(date, datas, offset, result_type='temp_1')
    left = get_cst_result_info(date, datas, offset, result_type='temp_2')

    right = right.drop_duplicates(subset=['times'])
    left = left.drop_duplicates(subset=['times'])
    values = merge_on_times(right, left)
    values = get_max_values(values)

    mean_value = ""
    min_value = ""
    max_values = ""

    if len(values) > 0:
        mean_value = round(np.mean(values['values']))/100
        min_value = round(np.quantile(values['values'], 0.1))/100
        max_values = round(np.quantile(values['values'], 0.9))/100

    temperature_dict['values'] = values
    temperature_dict['mean'] = mean_value
    temperature_dict['min'] = min_value
    temperature_dict['max'] = max_values

def add_activity(date, datas, activity_dict, offset) : 
    averaged_activity = get_cst_result_info(date, datas, offset, result_type='averaged_activity')
    steps_number = get_cst_result_info(date, datas, offset, result_type='steps_number')
    activity_level = get_cst_result_info(date, datas, offset,result_type= 'activity_level')
    activity_level["values"] = activity_level["values"].astype(str).str[0]

    # Compose the distance dataframe  
    times = steps_number["times"]
    values = steps_number["values"]*0.76
    distance = pd.DataFrame({'times' : times, 'values' : values})
    
    activity_dict['steps'] = data_per_15_min(steps_number)
    activity_dict['averaged_activity' ] = averaged_activity
    activity_dict['distance'] = distance
    activity_dict['activity_level'] = activity_level

def add_activity_to_indicators(results_dict) -> dict:
    averaged_activity_df = results_dict["activity"]["averaged_activity"]
    averaged_activity_df.rename(columns={"values": "activity_values"}, inplace=True)
    
    # ECG indicators
    sig_indicators = results_dict["cardio"]
    sig_indicators["rate"] = merge_on_times(sig_indicators["rate"], averaged_activity_df)
    sig_indicators["rate_var"] = merge_on_times(sig_indicators["rate_var"], averaged_activity_df)
    sig_indicators["qt"] = merge_on_times(sig_indicators["qt"], averaged_activity_df)
    
    # Breath 
    sig_indicators = results_dict["breath"]
    sig_indicators["rate"] = merge_on_times(sig_indicators["rate"], averaged_activity_df)
    sig_indicators["rate_var"] = merge_on_times(sig_indicators["rate_var"], averaged_activity_df)
    sig_indicators["inspi_expi"] = merge_on_times(sig_indicators["inspi_expi"], averaged_activity_df)

    return copy.deepcopy(results_dict)

def add_temperature_to_indicators(results_dict_2) -> dict:

    temperature_df = results_dict_2["temperature"]["values"]
    temperature_df.rename(columns={"values": "temperature_values"}, inplace=True)
    
    # ECG indicators
    sig_indicators = results_dict_2["cardio"]
    rate = merge_on_times(sig_indicators["rate"], temperature_df)
    sig_indicators["rate"] = temperature_filter(rate)

    rate_var = merge_on_times(sig_indicators["rate_var"], temperature_df)
    sig_indicators["rate_var"] = temperature_filter(rate_var)

    qt = merge_on_times(sig_indicators["qt"], temperature_df)
    sig_indicators["qt"] = temperature_filter(qt)
    
    # Breath 
    sig_indicators = results_dict_2["breath"]
    rate= merge_on_times(sig_indicators["rate"], temperature_df)
    sig_indicators["rate"] = temperature_filter(rate)
    rate_var = merge_on_times(sig_indicators["rate_var"], temperature_df)
    sig_indicators["rate_var"] = temperature_filter(rate_var)
    inspi_expi = merge_on_times(sig_indicators["inspi_expi"], temperature_df)
    sig_indicators["inspi_expi"] = temperature_filter(inspi_expi)

    return copy.deepcopy(results_dict_2)

def add_durations(date, results_dict):
    # Times constants
    YEAR = int(date[:4])
    M = int(date[5:7])
    D = int(date[8:10])
    NIGHT_LIMIT = datetime(YEAR, M, D, 6, 0, 0)

    ref_df = results_dict['activity']['averaged_activity']

    if len(ref_df)>0:
        day_times    = ref_df.loc[ref_df["times"] > NIGHT_LIMIT,
                                "times"].reset_index(drop=True)
        night_times  = ref_df.loc[ref_df["times"] < NIGHT_LIMIT, 
                                        "times"].reset_index(drop=True)
        
        rest_times   = ref_df.loc[ref_df["values"] <= ACTIVITY_THREASHOLD,
                                "times"].reset_index(drop=True)   
        
        time_intervals = find_time_intervals(ref_df['times'])
        day_time_intervals = find_time_intervals(day_times)
        night_time_intervals = find_time_intervals(night_times)
        rest_time_intervals = find_time_intervals(rest_times)
        
        collected_in_s = sum_time_intervals(time_intervals)
        day_in_s = sum_time_intervals(day_time_intervals)
        night_in_s = sum_time_intervals(night_time_intervals)
        rest_in_s = sum_time_intervals(rest_time_intervals)
        active_in_s = collected_in_s - rest_in_s
        
        duration_dict = results_dict["duration"]
        duration_dict["intervals"] = time_intervals
        duration_dict["collected"] = timedelta_formatter(collected_in_s)
        duration_dict["day"] = timedelta_formatter(day_in_s)
        duration_dict["night"] = timedelta_formatter(night_in_s)
        duration_dict["rest"] = timedelta_formatter(rest_in_s)
        duration_dict["active"] = timedelta_formatter(active_in_s)
    
def add_anomalies(results_dict, date):
    alerts_dict = results_dict['anomalies']
    # --- Set alerts image positions ---
    # Tachypnea
    alerts_dict["tachypnea"]["x"]  = 4.56
    alerts_dict["tachypnea"]["y"]  = 7.01 + ALERT_SIZE
    # Bradypnea
    alerts_dict["bradypnea"]["x"]  = 4.56
    alerts_dict["bradypnea"]["y"]  = 7.23 + ALERT_SIZE
    # Tachycardia
    alerts_dict["tachycardia"]["x"]  = 1.95
    alerts_dict["tachycardia"]["y"]  = 7.01 + ALERT_SIZE
    # Bradycardia
    alerts_dict["bradycardia"]["x"]  = 1.95
    alerts_dict["bradycardia"]["y"]  = 7.23 + ALERT_SIZE
    # QT
    alerts_dict["qt"]["x"]  = 1.95
    alerts_dict["qt"]["y"]  = 7.45 + ALERT_SIZE

    # ------------------- detect anomaly ----------------------
    # Breath Tachy/Brady
    df_aux = results_dict['breath']['rate']
    values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, "values"].dropna().reset_index(drop=True)

    values_tachy = [i for i in values if i > TACHYPNEA_TH]
    if len(values_tachy) > 0:
        alerts_dict["tachypnea"]["path"]  = RED_ALERT
        alerts_dict["tachypnea"]["exists"] = True
        alerts_dict["tachypnea"]["values"] = values_tachy
        alerts_dict["tachypnea"]["mean"] = round(np.mean(values_tachy))
        alerts_dict["tachypnea"]["duration"] = len(values_tachy)
        alerts_dict["tachypnea"]["percentage"] = round(len(values_tachy)/len(values)*100, 1)

    values_brady = [i for i in values if i < BRADYPNEA_TH] 
    if len(values_brady) > 0:
        alerts_dict["bradypnea"]["path"]  = RED_ALERT
        alerts_dict["bradypnea"]["exists"] = True
        alerts_dict["bradypnea"]["values"] = values_brady
        alerts_dict["bradypnea"]["mean"] = round(np.mean(values_brady))
        alerts_dict["bradypnea"]["duration"] = len(values_brady)
        alerts_dict["bradypnea"]["percentage"] = round(len(values_brady)/len(values)*100, 1)

    # Cardio Tachy/Brady
    df_aux = results_dict['cardio']['rate']
    values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, "values"].dropna().reset_index(drop=True)
    
    values_tachy = [i for i in values if i > TACHYCARDIA_TH]
    if len(values_tachy) > 0:
        alerts_dict["tachycardia"]["path"] = RED_ALERT
        alerts_dict["tachycardia"]["exists"] = True
        alerts_dict["tachycardia"]["values"] = values_tachy
        alerts_dict["tachycardia"]["mean"] = round(np.mean(values_tachy))
        alerts_dict["tachycardia"]["duration"] = len(values_tachy)
        alerts_dict["tachycardia"]["percentage"] = round(len(values_tachy)/len(values)*100, 1)

    values_brady = [i for i in values if i < BRADYCARDIA_TH] 
    if len(values_brady) > 0:
        alerts_dict["bradycardia"]["path"]  = RED_ALERT
        alerts_dict["bradycardia"]["exists"] = True
        alerts_dict["bradycardia"]["values"] = values_brady
        alerts_dict["bradycardia"]["mean"] = round(np.mean(values_brady))
        alerts_dict["bradycardia"]["duration"] = len(values_brady)
        alerts_dict["bradycardia"]["percentage"] = round(len(values_brady)/len(values)*100, 1)
    
    # Cardio QTc length TO CHANGE TO CHANGE when indicator is updateted !!!
    YEAR = int(date[:4])
    M = int(date[5:7])
    D = int(date[8:10])
    night_limit = datetime(YEAR, M, D, 6, 0, 0)
    morning_limit = datetime(YEAR, M, D, 12, 0, 0)
    
    df_aux = results_dict['cardio']['qt']
    df_night = df_aux.loc[df_aux["times"] <= night_limit].dropna().reset_index(drop=True)
    df_morning = df_aux.loc[df_aux["times"] > night_limit].dropna().reset_index(drop=True)
    df_morning = df_morning.loc[df_morning["times"] <= morning_limit].dropna().reset_index(drop=True)
    df_evening = df_aux.loc[df_aux["times"] > morning_limit].dropna().reset_index(drop=True)

    values_night = df_night.loc[df_night["activity_values"] <= ACTIVITY_THREASHOLD, "values"].dropna().reset_index(drop=True)
    values_morning = df_morning.loc[df_morning["activity_values"] <= ACTIVITY_THREASHOLD, "values"].dropna().reset_index(drop=True)
    values_evening = df_evening.loc[df_evening["activity_values"] <= ACTIVITY_THREASHOLD, "values"].dropna().reset_index(drop=True)
    
    qt_alert = False
    if len(values) > 0:
        alerts_dict["qt"]["values"] = values

    if len(values_night) > 0:
        qt_night  = round(statistics.median(values_night))
        alerts_dict["qt"]["night"] = qt_night
        if qt_night > QT_MAX_TH:
            qt_alert = True

    if len(values_morning) > 0:
        qt_morning  = round(statistics.median(values_morning))
        alerts_dict["qt"]["morning"] = qt_morning
        if qt_morning > QT_MAX_TH:
            qt_alert = True

    if len(values_evening) > 0:
        qt_evening  = round(statistics.median(values_evening))
        alerts_dict["qt"]["evening"] = qt_evening
        if qt_evening > QT_MAX_TH:
            qt_alert = True

    if qt_alert:
        alerts_dict["qt"]["path"]  = RED_ALERT
        alerts_dict["qt"]["exists"] = True

def merge_on_times(df_1, df_2):
    df_result = pd.merge(df_1, df_2, how="outer", on="times")
    df_result.sort_values(by="times", inplace = True)
    df_result = df_result.reset_index(drop=True)
                   
    return copy.deepcopy(df_result)

def get_cst_result_info(date, datas, offset, result_type):
    times = []
    values = []
    output = pd.DataFrame({
    'times' : times,
    'values' : values,
    })

    for data in datas:
        if result_type == data['type']:
            timestamp = compute_local_timestamp(data['mtimestamp'], offset)
            times.append(timestamp)
            values.append(data['values'])

    output['times'] = times
    output['values'] = values
    output.sort_values(by = 'times', inplace = True) 

    # Get output where times = date
    YEAR = int(date[:4])
    M = int(date[5:7])
    D = int(date[8:10])
    start_date = datetime(YEAR, M, D, 0, 0, 0)
    end_date = datetime(YEAR, M, D, 23, 59, 59)
    mask = (output['times'] > start_date) & (output['times'] <= end_date)
    output = output.loc[mask]
    output = output.reset_index(drop=True)

    if len(output['times']) > 0:
        # Round times to minutes
        output['times'] = output["times"].dt.round("min")
        # Format type from pandas._libs.tslibs.timestamps.Timestamp to datetime
        output['times'] = output["times"]

    return output

def get_cst_result_info_segment(date, datas, offset, result_type, type):
    times = []
    values = []
    output = pd.DataFrame({
    'times' : times,
    'values' : values,
    })

    for data in datas:
        if result_type == data['type']:
            timestamp = compute_local_timestamp(data['mtimestamp'], offset)
            times.append(timestamp)
            if type == "ratio" : 
                segment_values = unwrap_ratio_inspi_expi(data['values'])
            else :
                segment_values = unwrap(data['values'])

            if np.size(segment_values) > 1:         # Nan values have size = 1       
                mean_value = round(np.mean(segment_values))
            else: mean_value = math.nan
            values.append(mean_value)

    output['times'] = times
    output['values'] = values
    output.sort_values(by = 'times', inplace = True) 

    # Get output where times = date
    YEAR = int(date[:4])
    M = int(date[5:7])
    D = int(date[8:10])
    start_date = datetime(YEAR, M, D, 0, 0, 0)
    end_date = datetime(YEAR, M, D, 23, 59, 59)
    mask = (output['times'] > start_date) & (output['times'] <= end_date)
    output = output.loc[mask]
    output = output.reset_index(drop=True)

    if len(output['times']) > 0:
        # Round times to minutes
        output['times'] = output["times"].dt.round("min")
        # Format type from pandas._libs.tslibs.timestamps.Timestamp to datetime
        output['times'] = output["times"]

    return output

def get_cst_offset(datas):
    offset = 7200
    result_type='temp_1'

    for data in datas:
        if result_type == data['type']:
            offset = data['timezone_offset']
            break
    print(offset)
    print(type(offset))
    st.session_state.chronolife_offset = offset

    return offset

def compute_local_timestamp(mtimestamp:str, offset:int):
    new_timestamp_dt = 0
    sign = get_offset_sign(offset)

    # String to datetime object
    timestamp_dt = datetime.strptime(mtimestamp, '%Y-%m-%dT%H:%M:%S') 

    # Add offset 
    new_timestamp_dt = timestamp_dt + sign*timedelta(seconds=abs(offset))

    return new_timestamp_dt

def get_offset_sign(offset:int):
    sign = 1
    if offset>=0:
        sign = 1
    elif offset<0:
        sign = -1
    return sign

def data_per_15_min(input_df):

    output_df = pd.DataFrame({"times" : [], "values" : []})

    if len(input_df) > 0 : 
        first_minute = input_df["times"][0]
        first_minute = pd.Timestamp(first_minute).round(freq='H') # round to minute

        index_last_minut = input_df.tail(1).index[0]
        last_minute = input_df["times"][index_last_minut]
        last_minute = pd.Timestamp(last_minute).round(freq='T') # round to minute
    
        while first_minute < last_minute:
            values_df = input_df.loc[input_df['times'] >= first_minute, ]
            values = values_df.loc[values_df['times'] < first_minute + timedelta(minutes = 15), 'values']

            aux_df = pd.DataFrame({"times" : first_minute, "values" : np.mean(values)}, index=[0])

            output_df = pd.concat([output_df, aux_df ])
            first_minute += + timedelta(minutes = 15)
        output_df = output_df.reset_index(drop=True)

    return output_df

def get_max_values(input_df):
    input_df["values"] = input_df[["values_x", "values_y"]].max(axis=1)

    output_df = input_df[["times", "values"]]

    output_df = output_df.reset_index(drop=True)

    return output_df

def temperature_filter(input_df) :
    
    output_df = input_df.loc[input_df["temperature_values"] > TEMPERATURE_THREASHOLD].dropna().reset_index(drop=True)

    return output_df 

# %% ------------- Test the main function-------------------------------------
# from config import API_KEY_PREPROD, API_KEY_PROD, URL_CST_PREPROD, URL_CST_PROD
# prod = False

# # -- Ludo
# # user_id = "4vk5VJ"
# # date = "2023-05-17"
# # -- Fernando
# # user_id = "5Nwwut"
# # date = "2023-05-17"
# # -- Michel
# # user_id = "5Nwwut" 
# # date = "2023-05-04" 
# # -- Adriana
# user_id = "6o2Fzp"
# date = "2023-06-13"

# if prod == True :
#     api = API_KEY_PROD
#     url = URL_CST_PROD
# else :
#     api = API_KEY_PREPROD
#     url = URL_CST_PREPROD

# results_dict = get_cst_data(user_id, date, api, url)
