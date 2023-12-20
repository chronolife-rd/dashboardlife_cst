# Imports
import streamlit as st
import json
import copy
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import template.constant as constant

# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
# Request user's data from servers
def get_garmin_indicators():
    
    result_dict = []
    
    api_key     = st.session_state.api_key
    end_user    = st.session_state.end_user
    date        = st.session_state.date
    url         = st.session_state.url_garmin
    
    # Build the query parameters object
    params = {
           'user'    : end_user,   
           'types'   : constant.TYPE()["GARMIN_INDICATORS"],
           'date'    : date,
         }
    
    # Request data from servers
    reply = request_data_from_servers(params, api_key, url)

    # Convert the reply content into a json object
    datas = error_management(reply)

    # result_dict = save_datas_in_dict(date, end_user, datas)
    
    try:
        # Organize the data in a dictionary 
        result_dict = save_datas_in_dict(date, end_user, datas)
    except:
        result_dict = initialize_dictionary_with_template()
    
    st.session_state.garmin_indicators = result_dict
   
# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------
def request_data_from_servers(params, api_key, url):
    # Perform the POST request authenticated with YOUR API key (NOT the one of
    # the sub-user!).
    reply = requests.get(url, headers={"X-API-Key": api_key}, params=params)
    
    return reply

def error_management(reply):
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
        print('No data found')
    
    return datas

def save_datas_in_dict(date, user_id, datas) -> dict :
    # Save the results in a dictionary
    results_dict = initialize_dictionary_with_template()
    # User id
    results_dict['user_id']  = user_id
    # Activity 
    add_activity(datas, results_dict['activity'])
    # Calories 
    add_calories(datas, results_dict['calories'])
    # Intensity minutes
    add_intensity_minutes(datas, results_dict['intensity_min'])
    # Stress
    add_stress(datas, results_dict['stress'])
    # Cardio 
    add_cardio(datas, results_dict['cardio'])
    # # Breath
    add_breath(datas, results_dict['breath'])
    # # Spo2 
    add_spo2(datas, results_dict['spo2'])
    # # Body battery
    add_body_battery(datas, results_dict["body_battery"])
    # # Sleep
    add_sleep(date, user_id, datas, results_dict['sleep'])

    return copy.deepcopy(results_dict)

def initialize_dictionary_with_template() -> dict :  
    activity_dict = {
        "distance" : None,
        "goal" : None,
        "intensity" : None,
        "steps" : None
    }
    body_battery_dict = {
        "all_values" : None,
        "highest" : None,
        "lowest" : None,
    }
    breath_dict = {
        "rate" : None
    }
    calories_dict = {
        "active" : None, 
        "resting" : None, 
        "total" : None, 
    }
    cardio_dict = {
        "rate" : None
    }
    duration_dict = {
    }
    intensity_min_dict = {
        "moderate" : None,
        "total" : None, 
        "vigurous" : None,
    }
    sleep_dict = {
        "awake" : None, 
        "deep" : None, 
        "light" : None, 
        "quality" : None,
        "recorded_time" : None,
        "rem" : None, 
        "score" : None,
        "timestamp_end" : None,
    }
    spo2_dict = {
        "all_values" : None,
        "averege" : None,
        "lowest" : None,
    }
    stress_dict = {
        "high" : None, 
        "low" : None, 
        "medium" : None,
        "recorded_time" : None,
        "rest" : None, 
        "score" : None,
    }
    
    dict_template = {
                    'activity'     : copy.deepcopy(activity_dict),                 
                    'body_battery' : copy.deepcopy(body_battery_dict),
                    'breath'       : copy.deepcopy(breath_dict),
                    'calories'     : copy.deepcopy(calories_dict),
                    'cardio'       : copy.deepcopy(cardio_dict),
                    'duration'     : copy.deepcopy(duration_dict),  
                    'intensity_min': copy.deepcopy(intensity_min_dict),
                    'sleep'        : copy.deepcopy(sleep_dict),
                    'spo2'         : copy.deepcopy(spo2_dict),
                    'stress'       : copy.deepcopy(stress_dict),
                    'user_id'      : None,
                    }

    return copy.deepcopy(dict_template)

def add_calories(datas, calories_dict):
    calories_dict["resting"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'bmrKilocalories')
    calories_dict["active"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'activeKilocalories')
    calories_dict["total"] = calories_dict["resting"]+calories_dict["active"]

def add_intensity_minutes(datas, intensity_min_dict):
    intensity_min_dict["moderate"] = get_garmin_result_info(datas, result_type =\
                    'dailies', output_type = 'moderateIntensityDurationInSeconds')
    intensity_min_dict["vigurous"] = get_garmin_result_info(datas, result_type =\
                    'dailies', output_type = 'vigorousIntensityDurationInSeconds')
    intensity_min_dict["total"] = intensity_min_dict["moderate"] +\
                                  intensity_min_dict["vigurous"] 
def add_stress(datas, stress_dict):
    stress_dict["score"]  = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'averageStressLevel')
    stress_dict["rest"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'restStressDurationInSeconds')
    stress_dict["low"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'lowStressDurationInSeconds')
    stress_dict["medium"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'mediumStressDurationInSeconds')
    stress_dict["high"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'highStressDurationInSeconds')
    stress_dict["recorded_time"] = stress_dict["rest"] + stress_dict["low"] +\
                                   stress_dict["medium"] + stress_dict["high"]
    
def add_cardio(datas, cardio_dict):
    start_time = get_garmin_result_info(datas=datas, result_type =\
                        'dailies', output_type = 'mtimestamp')
    cardio_data = get_garmin_result_info(datas=datas, result_type =\
                        'dailies', output_type = 'timeOffsetHeartRateSamples')
 
    cardio_dict['rate'] = convert_dict_to_df(cardio_data, start_time)
    
    # TO CHANGE !!!
    #/ cardio_dict['rate_var'] = get_garmin_result_info(datas=datas, result_type =\
    #                     'hrv', output_type = 'hrvValues')
    
def add_breath(datas, breath_dict):
    result_type = 'allDayRespiration'
    output_type = 'timeOffsetEpochToBreaths'
    df_output = pd.DataFrame()
    for data in datas:
        if data['type'] == result_type:
            start_time = data['value']['mtimestamp']
            values_dict = data['value'][output_type]
            df_temp = convert_dict_to_df(values_dict, start_time)
            df_output = pd.concat([df_output, df_temp])
    df_output.sort_values(by='times', inplace=True)
    df_output.reset_index(drop = True)         
    breath_dict['rate'] = df_output

def add_spo2(datas, spo2_dict):
    start_time = get_garmin_result_info(datas=datas, result_type =\
                        'pulseox', output_type = 'mtimestamp')
    values = get_garmin_result_info(datas=datas, result_type =\
                        'pulseox', output_type = 'timeOffsetSpo2Values')
    values_df = convert_dict_to_df(values, start_time)

    spo2_dict["all_values"] = values_df
    spo2_dict["averege"] = round(np.mean(values_df['values']))
    spo2_dict["lowest"] = min(values_df['values'])

def add_body_battery(datas, body_battery_dict):
    start_time = get_garmin_result_info(datas=datas, result_type =\
                        'stressDetails', output_type = 'mtimestamp')
    values = get_garmin_result_info(datas=datas, result_type =\
                    'stressDetails', output_type = 'timeOffsetBodyBatteryValues')
    values_df = convert_dict_to_df(values, start_time)
    body_battery_dict["all_values"] = values_df
    if len(values):
        body_battery_dict["highest"] = max(values_df['values'])
        body_battery_dict["lowest"] = min(values_df['values'])

def add_sleep(date, user_id, datas, sleep_dict):
    
    sleep_dict["score"] = []
    
    value_dict = get_sleep_data(date, user_id, datas)
    timestamp_end = value_dict['startTimeInSeconds'] + value_dict['durationInSeconds']
    sleep_dict["timestamp_end"] = datetime.fromtimestamp(timestamp_end)
    sleep_dict["score"]   = value_dict['overallSleepScore']['value']   
    sleep_dict["quality"] = value_dict['overallSleepScore']['qualifierKey']            
    sleep_dict["deep"]    = value_dict['deepSleepDurationInSeconds']
    sleep_dict["light"]   = value_dict['lightSleepDurationInSeconds']
    sleep_dict["rem"]     = value_dict['remSleepInSeconds']
    sleep_dict["awake"]   = value_dict[ 'awakeDurationInSeconds']
    sleep_dict["recorded_time"] = sleep_dict["deep"] + sleep_dict["light"] +\
                                  sleep_dict["rem"] + sleep_dict["awake"]

def add_activity(datas, activity_dict):
    # TO CHANGE !! 
    activity_dict["goal"] = get_garmin_result_info(datas, result_type =\
                        'dailies', output_type = 'stepsGoal')   
    activity_dict["steps"] = get_garmin_activity_indicator(datas,
                        result_type = 'epochs', output_type = 'steps')
    activity_dict["distance"] = get_garmin_activity_indicator(datas,
                        result_type = 'epochs', output_type = 'distanceInMeters')
    activity_dict['intensity'] = get_garmin_activity_indicator(datas,
                        result_type = 'epochs', output_type = 'intensity')

def get_garmin_activity_indicator(datas, result_type, output_type):
    i = 0
    df_output = pd.DataFrame(columns = ['values', 'times'])
    for data in datas:
        if data['type'] == result_type:
            start_time = data['value']['mtimestamp']
            start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
            value = data['value'][output_type]
            df_temp = pd.DataFrame({'values': value, 'times': start_time}, index = [i])
            df_output = pd.concat([df_output, df_temp])
            i +=1
    df_output.sort_values(by='times', inplace=True)
    df_output.reset_index(drop = True)  

    return  df_output
    
def get_sleep_data(date, user_id, datas) -> dict:
    result_type = 'sleeps'
    value_dict = []
    there_is_sleeps_batch = False

    # Hypothesis: sleep batches are in chronological order
    for data in datas:
        # If this is a 'sleeps' batch 
        if data['type'] == result_type:
            there_is_sleeps_batch = True
            # Get the timestamp_end of the sleep batch
            timestamp_end = data['mtimestamp_end'][:10]
            # If timestamp_end = date
            if timestamp_end == date:
                # Get value dictionary that contains all sleep info
                value_dict = data['value']
                # Break loop
                break

            # if timestamp_end != date, request data for the day before
            else:
                date_before = get_date_before_str(date)
                value_dict = get_sleep_of_day_before(date, date_before, user_id)

    # If there is no sleep batch in datas, get the datas of day before
    if there_is_sleeps_batch == False:
        date_before = get_date_before_str(date)
        value_dict = get_sleep_of_day_before(date, date_before, user_id)

    return copy.deepcopy(value_dict)

def get_date_before_str(date:str) -> str:
    date_today = datetime.strptime(date, "%Y-%m-%d")
    date_before = date_today - timedelta(days=1)
    date_before =  datetime.strftime(date_before, "%Y-%m-%d")
    return date_before

def get_sleep_of_day_before(date, date_before, user_id):
    
    api_key         = st.session_state.api_key
    url_garmin      = st.session_state.url_garmin
    url             = url_garmin
    
    value_dict = []

    # Build the query parameters object
    params = {
        'user'    : user_id,   
        'types'   : constant.TYPE()["SLEEP"],
        'date'    : date_before,
        }
    
    # Request data from servers
    reply = request_data_from_servers(params, api_key, url)
    # Convert the reply content into a json object
    datas_day_before = error_management(reply)
    # Get the sleep of the day before
    for data in datas_day_before:
        if data['type'] == constant.TYPE()["SLEEP"]:
            timestamp_end = data['mtimestamp_end'][:10]

            # If timestamp_end = date, get value ductionary
            if timestamp_end == date:
                value_dict = data['value']

    return copy.deepcopy(value_dict)

def get_garmin_result_info(datas, result_type, output_type):
    output = []
    for data in datas:
        if data['type'] == result_type:
            output = data['value'][output_type]
    return output

def convert_dict_to_df(signal_data, start_time) -> pd.DataFrame:
    start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    output = pd.DataFrame.from_dict(signal_data, orient='index', columns=['values'])

    # Get the offsets
    offsets = output.index
    # Add times column to the daaframe
    output['times'] = np.nan
    # Reset the index
    output = output.reset_index(drop = True)

    for i in range(len(offsets)):
        offset = timedelta(seconds = int(offsets[i]))
        output.loc[i, 'times'] = start_time + offset
    
    return copy.deepcopy(output)
    


