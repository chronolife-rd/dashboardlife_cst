# Imports
import json
import copy
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from automatic_reports.useful_functions import find_time_intervals, sum_time_intervals, timedelta_formatter
# from automatic_reports.config import GARMIN_SIGNAL_TYPES

import streamlit as st 
# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
# Request user's data from servers
# def get_garmin_data(user_id, date, api, url):

#     # Build the query parameters object
#     params = {
#            'user'    : user_id,   
#            'types'   : GARMIN_SIGNAL_TYPES,
#            'date'    : date,
#          }
    
#     # Request data from servers
#     print("-- Requesting Garmin data for the day", date, "--")
#     reply = request_data_from_servers(params, api, url)
#     # Convert the reply content into a json object
#     datas = error_management(date, reply)

#     #  Manage display of Garmin indicators when the user use the t-shirt 

#     if len(datas)==0:
#         st.session_state.garmin_data_available = False
#         result_dict = {}
#     else :
#         st.session_state.garmin_data_available = True
#          # Organize the data in a dictionary 
#         result_dict = save_datas_in_dict(date, user_id, datas, api, url)
   


   

#     return result_dict
   
# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------
# def request_data_from_servers(params, api, url):

#     # Perform the POST request authenticated with YOUR API key (NOT the one of
#     # the sub-user!).
#     reply = requests.get(url, headers={"X-API-Key": api}, params=params)
    
#     return reply

# def error_management(date, reply):
#     datas = []
#     # Error management 
#     if reply.status_code == 200:
#       # Convert the reply content into a json object.
#       json_list_of_records = json.loads(reply.text) 
#       for record in json_list_of_records:
#           datas.append(record)
#     elif reply.status_code == 400:
#         print('Part of the request could not be parsed or is incorrect.')
#     elif reply.status_code == 401:
#         print('Invalid authentication')
#     elif reply.status_code == 403:
#         print('Not authorized.')
#     elif reply.status_code == 404:
#         print('Invalid url')
#     elif reply.status_code == 500:
#         print('Invalid user ID')
    
#     if len(datas) == 0:
#         print('No Garmin data found for day:', date)
    
#     return datas

# # def save_datas_in_dict(date, user_id, datas, api, url) -> dict :
# #     # Save the results in a dictionary
# #     results_dict = initialize_dictionary_with_template()
# #     # User id
# #     results_dict['user_id']  = user_id

# #     if len(datas) > 0:
# #         # Activity 
# #         add_activity(datas, results_dict['activity'])
# #         # Body battery
# #         add_body_battery(datas, results_dict["body_battery"])
# #         # Breath
# #         add_breath(datas, results_dict['breath'])
# #         # Calories 
# #         add_calories(datas, results_dict['calories'])
# #         # Cardio 
# #         add_cardio(datas, results_dict['cardio'])
# #         # Compute durations
# #         add_durations(date, results_dict)
# #         # Intensity minutes
# #         add_intensity_minutes(datas, results_dict['intensity_min'])
# #         # Sleep
# #         add_sleep(date, user_id, datas, results_dict['sleep'], api, url)

# #         # Spo2 
# #         add_spo2(datas, results_dict['spo2'])
# #         # Stress
# #         add_stress(datas, results_dict['stress'])

# #     return copy.deepcopy(results_dict)

# # def add_activity(datas, activity_dict):
# #     if datas_exist(datas, result_type = 'dailies'):
# #         activity_dict["goal"] = get_garmin_result_info(datas, result_type =\
# #                             'dailies', output_type = 'stepsGoal')   
# #         activity_dict["steps"] = get_garmin_result_info(datas, result_type =\
# #                             'dailies', output_type = 'steps')   
# #         activity_dict["distance"] = get_garmin_result_info(datas, result_type =\
# #                             'dailies', output_type = 'distanceInMeters')  

# #         activity_dict['intensity'] = get_garmin_activity_indicator(datas,
# #                             result_type = 'epochs')

# # def add_body_battery(datas, body_battery_dict):
# #     if datas_exist(datas, result_type = 'stressDetails'):
# #         start_time = get_garmin_result_info(datas=datas, result_type =\
# #                             'stressDetails', output_type = 'mtimestamp')
# #         values = get_garmin_result_info(datas=datas, result_type =\
# #                         'stressDetails', output_type = 'timeOffsetBodyBatteryValues')
# #         if type(values) != int :
# #             values_df = convert_dict_to_df(values, start_time)
# #             body_battery_dict["all_values"] = values_df
# #             if len(values):
# #                 body_battery_dict["highest"] = max(values_df['values'])
# #                 body_battery_dict["lowest"] = min(values_df['values'])

# # def add_breath(datas, breath_dict):
# #     if datas_exist(datas, result_type = 'allDayRespiration'):
# #         result_type = 'allDayRespiration'
# #         output_type = 'timeOffsetEpochToBreaths'
# #         df_output = pd.DataFrame()
# #         for data in datas:
# #             if data['type'] == result_type:
# #                 start_time = data['value']['mtimestamp']
# #                 values_dict = data['value'][output_type]
# #                 df_temp = convert_dict_to_df(values_dict, start_time)
# #                 df_output = pd.concat([df_output, df_temp])
# #         df_output = df_output[(df_output["values"] > 0)]
# #         df_output.sort_values(by='times', inplace=True)
# #         df_output = df_output.reset_index(drop = True)         
# #         breath_dict['rate'] = df_output
        
# # def add_calories(datas, calories_dict):
# #     if datas_exist(datas, result_type = 'dailies'):
# #         calories_dict["resting"] = get_garmin_result_info(datas, result_type =\
# #                             'dailies', output_type = 'bmrKilocalories')
# #         calories_dict["active"] = get_garmin_result_info(datas, result_type =\
# #                             'dailies', output_type = 'activeKilocalories')
# #         calories_dict["total"] = calories_dict["resting"]+calories_dict["active"]

# # def add_cardio(datas, cardio_dict):
# #     if datas_exist(datas, result_type = 'dailies') :
# #         start_time = get_garmin_result_info(datas=datas, result_type =\
# #                             'dailies', output_type = 'mtimestamp')
# #         cardio_data = get_garmin_result_info(datas=datas, result_type =\
# #                             'dailies', output_type = 'timeOffsetHeartRateSamples')
        
# #         if (len(cardio_data) > 0) :
# #             rate = convert_dict_to_df(cardio_data, start_time)
# #             cardio_dict['rate'] = data_per_min(rate)
# #         else :
# #             st.session_state.garmin_data_available = False

    
# # def add_durations(date, results_dict):
# #     # Times constants
# #     YEAR = int(date[:4])
# #     M = int(date[5:7])
# #     D = int(date[8:10])
# #     NIGHT_LIMIT = datetime(YEAR, M, D, 6, 0, 0)

# #     ref_df = results_dict['cardio']['rate']

# #     if len(ref_df)>0:
# #         day_times    = ref_df.loc[ref_df["times"] > NIGHT_LIMIT,
# #                                 "times"].reset_index(drop=True)
# #         night_times  = ref_df.loc[ref_df["times"] < NIGHT_LIMIT, 
# #                                         "times"].reset_index(drop=True)
        
# #         df_activity = results_dict['activity']['intensity'] 
# #         medium_duration_df = df_activity.loc[df_activity['intensity'] == 'ACTIVE', 'duration']
# #         high_duration_df = df_activity.loc[df_activity['intensity'] == 'HIGHLY_ACTIVE', 'duration']
        
# #         time_intervals = find_time_intervals(ref_df['times'])
# #         day_time_intervals = find_time_intervals(day_times)
# #         night_time_intervals = find_time_intervals(night_times)

# #         collected_in_s = sum_time_intervals(time_intervals)
# #         day_in_s = sum_time_intervals(day_time_intervals)
# #         night_in_s = sum_time_intervals(night_time_intervals)
# #         active_in_s = sum(medium_duration_df) + sum(high_duration_df)
# #         rest_in_s = collected_in_s - active_in_s 
        
# #         duration_dict = results_dict["duration"]
# #         duration_dict["intervals"] = time_intervals
# #         duration_dict["collected"] = timedelta_formatter(collected_in_s)
# #         duration_dict["day"] = timedelta_formatter(day_in_s)
# #         duration_dict["night"] = timedelta_formatter(night_in_s)
        
# #         duration_dict["rest"] = timedelta_formatter(rest_in_s)
# #         duration_dict["active"] = timedelta_formatter(active_in_s)

# # def add_intensity_minutes(datas, intensity_min_dict):
# #     if datas_exist(datas, result_type = 'dailies'):
# #         intensity_min_dict["moderate"] = get_garmin_result_info(datas, result_type =\
# #                         'dailies', output_type = 'moderateIntensityDurationInSeconds')
# #         intensity_min_dict["vigurous"] = get_garmin_result_info(datas, result_type =\
# #                         'dailies', output_type = 'vigorousIntensityDurationInSeconds')
# #         intensity_min_dict["total"] = intensity_min_dict["moderate"] +\
# #                                     intensity_min_dict["vigurous"] 

# # def add_sleep(date, user_id, datas, sleep_dict, api, url):
# #     value_dict = get_sleep_data(date, user_id, datas, api, url)
# #     timestamp_end = value_dict['startTimeInSeconds'] + value_dict['durationInSeconds']
# #     sleep_dict["timestamp_end"] = datetime.fromtimestamp(timestamp_end)
# #     sleep_dict["sleep_map"] = value_dict['sleepLevelsMap']
# #     sleep_dict["score"] = value_dict['overallSleepScore']['value']   
# #     sleep_dict["quality"] = value_dict['overallSleepScore']['qualifierKey']            
# #     sleep_dict["deep"] = value_dict['deepSleepDurationInSeconds']
# #     sleep_dict["light"] = value_dict['lightSleepDurationInSeconds']
# #     sleep_dict["rem"] = value_dict['remSleepInSeconds']
# #     sleep_dict["awake"] = value_dict[ 'awakeDurationInSeconds']
# #     sleep_dict["recorded_time"] = sleep_dict["deep"] + sleep_dict["light"] +\
# #                                   sleep_dict["rem"] + sleep_dict["awake"]
    
# #     if(sleep_dict["recorded_time"] > 0):
# #         sleep_dict["percentage_deep"]   = int(round(sleep_dict["deep"]/sleep_dict["recorded_time"]*100))
# #         sleep_dict["percentage_light"]  = int(round(sleep_dict["light"]/sleep_dict["recorded_time"]*100))
# #         sleep_dict["percentage_rem"]    = int(round(sleep_dict["rem"]/sleep_dict["recorded_time"] *100))
# #         sleep_dict["percentage_awake"]  = int(round(sleep_dict["awake"]/sleep_dict["recorded_time"] *100))

# # def add_spo2(datas, spo2_dict):
# #     if datas_exist(datas, result_type = 'pulseox'):
# #         start_time = get_garmin_result_info(datas, result_type =\
# #                             'pulseox', output_type = 'mtimestamp')
        
# #         values = get_garmin_result_info(datas, result_type =\
# #                             'pulseox', output_type = 'timeOffsetSpo2Values')
# #         values_df = convert_dict_to_df(values, start_time)

# #         spo2_dict["all_values"] = values_df

# #         if (len(spo2_dict["all_values"]) > 0) :
# #             spo2_dict["averege"] = round(np.mean(values_df["values"]))
# #             spo2_dict["lowest"] = min(values_df['values'])

# # def add_stress(datas, stress_dict):
#     start_time = get_garmin_result_info(datas, result_type =\
#                         'stressDetails', output_type = 'mtimestamp')
#     values = get_garmin_result_info(datas, result_type =\
#                         'stressDetails', output_type = 'timeOffsetStressLevelValues')
#     stress_dict["all_values"] = convert_dict_to_df(values, start_time)
#     stress_dict["score"] = get_garmin_result_info(datas, result_type =\
#                         'dailies', output_type = 'averageStressLevel')
#     stress_dict["rest"] = get_garmin_result_info(datas, result_type =\
#                         'dailies', output_type = 'restStressDurationInSeconds')
#     stress_dict["low"] = get_garmin_result_info(datas, result_type =\
#                         'dailies', output_type = 'lowStressDurationInSeconds')
#     stress_dict["medium"] = get_garmin_result_info(datas, result_type =\
#                         'dailies', output_type = 'mediumStressDurationInSeconds')
#     stress_dict["high"] = get_garmin_result_info(datas, result_type =\
#                         'dailies', output_type = 'highStressDurationInSeconds')
#     stress_dict["recorded_time"] = stress_dict["rest"] + stress_dict["low"] +\
#                                    stress_dict["medium"] + stress_dict["high"]

# def datas_exist(datas, result_type):
#     output = False
    
#     for data in datas:
        
#         if data['type'] == result_type:
#             output = True
#             break
#     return output

# def get_garmin_result_info(datas, result_type, output_type):
#     output = 0
#     for data in datas:
#         if data['type'] == result_type:
#             value_dict = data['value']
#             if output_type in value_dict:
#                 output = value_dict[output_type] 
#     return output

# def get_garmin_activity_indicator(datas, result_type):
#     i = 0
#     df_output = pd.DataFrame(columns = ['times', 'intensity', 'duration', 'steps', 'distance'])
#     for data in datas:
#         if data['type'] == result_type:
#             garmin_data = data['value']
#             start_time = garmin_data['mtimestamp']
#             start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')

#             list_of_dict = garmin_data['value']
#             for dict_i in list_of_dict:
#                 intensity = dict_i['intensity']
#                 steps = dict_i['steps']
#                 duration_in_s = dict_i['activeTimeInSeconds']
#                 distance_in_m =  dict_i['distanceInMeters']
#                 df_temp = pd.DataFrame({
#                     'times': start_time,
#                     'intensity': intensity, 
#                     'duration': duration_in_s,
#                     'steps' : steps,
#                     'distance' : distance_in_m
#                     }, index = [i])
#                 df_output = pd.concat([df_output, df_temp])
#                 i +=1
#     df_output.sort_values(by='times', inplace=True)
#     df_output = df_output.reset_index(drop = True)  

#     return  df_output
    
# def get_sleep_data(date, user_id, datas, api, url) -> dict:
#     result_type = 'sleeps'
#     value_dict = init_sleep_dict_tamplate()
#     there_is_sleeps_batch = False

#     # Hypothesis: sleep batches are in chronological order
#     for data in datas:
#         # If this is a 'sleeps' batch 
#         if data['type'] == result_type:
#             there_is_sleeps_batch = True
#             # Get the timestamp_end of the sleep batch
#             timestamp_end = data['mtimestamp_end'][:10]
#             # If timestamp_end = date
#             if timestamp_end == date:
#                 # Get value dictionary that contains all sleep info
#                 value_dict = data['value']
#                 # Break loop
#                 break

#             # if timestamp_end != date, request data for the day before
#             else:
#                 date_before = change_date(date, sign = -1)
#                 value_dict = get_sleep_of_day_before(date, date_before, user_id, api, url)

#     # If there is no sleep batch in datas, get the datas of day before
#     if there_is_sleeps_batch == False:
#         date_before = change_date(date, sign = -1)
#         value_dict = get_sleep_of_day_before(date, date_before, user_id, api, url)

#     return copy.deepcopy(value_dict)

# def change_date(date:str, sign:int) -> str:
#     date_today = datetime.strptime(date, "%Y-%m-%d")
#     new_date = date_today + sign*timedelta(days=1)
#     new_date =  datetime.strftime(new_date, "%Y-%m-%d")
#     return new_date

# def get_sleep_of_day_before(date, date_before, user_id, api, url):
#     value_dict = init_sleep_dict_tamplate()
#     result_type = 'sleeps'

#     # Build the query parameters object
#     params = {
#         'user'    : user_id,   
#         'types'   : result_type,
#         'date'    : date_before,
#         }
#     # Request data from servers
#     print("-- Requesting Garmin data for sleep --")
#     reply = request_data_from_servers(params, api, url)
#     # Convert the reply content into a json object
#     datas_day_before = error_management(date, reply)
#     # Get the sleep of the day before
#     for data in datas_day_before:
#         if data['type'] == result_type:
#             timestamp_end = data['mtimestamp_end'][:10]

#             # If timestamp_end = date, get value ductionary
#             if timestamp_end == date:
#                 value_dict = data['value']

#     return copy.deepcopy(value_dict)

# def init_sleep_dict_tamplate():
#     output = {
#         'startTimeInSeconds' : 0,
#         'durationInSeconds' : 0,
#         'sleepLevelsMap' : 0,
#         'overallSleepScore' : {
#             'value' : 0,
#             'qualifierKey' : 'no data',
#         },
#         'deepSleepDurationInSeconds' : 0,
#         'lightSleepDurationInSeconds' : 0,
#         'remSleepInSeconds' : 0,
#         'awakeDurationInSeconds' : 0,
#     }
#     return output

# def convert_dict_to_df(signal_data, start_time) -> pd.DataFrame:
#     output = []
#     if st.session_state.garmin_data_available == True:
#         output = pd.DataFrame.from_dict(signal_data, orient='index', columns=['values'])

#         # String to datetime object
#         start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
#         # Get the offsets
#         offsets = output.index
#         # Add times column to the dataframe
#         output['times'] = np.nan
#         # Reset the index
#         output = output.reset_index(drop = True)

#         for i in range(len(offsets)):
#             offset = timedelta(seconds = int(offsets[i]))
#             output.loc[i, 'times'] = start_time + offset
#         output.reset_index(drop=True)
        
#     return copy.deepcopy(output)

# def data_per_min(input_df):
#     output_df = pd.DataFrame({"times" : [], "values" : []})
#     first_minute = input_df["times"][0]
#     first_minute = pd.Timestamp(first_minute).round(freq='T') # round to minute

#     index_last_minut = input_df.tail(1).index[0]
#     last_minute = input_df["times"][index_last_minut]
#     last_minute = pd.Timestamp(last_minute).round(freq='T') # round to minute
 
#     while first_minute < last_minute:
#         values_df = input_df.loc[input_df['times'] >= first_minute, ]
#         values = values_df.loc[values_df['times'] < first_minute + timedelta(minutes = 1), 'values']

#         aux_df = pd.DataFrame({"times" : first_minute, "values" : np.mean(values)}, index=[0])

#         output_df = pd.concat([output_df, aux_df ])
#         first_minute += + timedelta(minutes = 1)
       
#     return output_df.reset_index(drop=True)

# def initialize_dictionary_with_template() -> dict :  
#     activity_dict = {
#         "distance" : "",
#         "goal" : "",
#         "intensity" : "",
#         "steps" : ""
#     }
#     body_battery_dict = {
#         "all_values" : "",
#         "highest" : "",
#         "lowest" : "",
#     }
#     breath_dict = {
#         "rate" : ""
#     }
#     calories_dict = {
#         "active" : "", 
#         "resting" : "", 
#         "total" : "", 
#     }
#     cardio_dict = {
#         "rate" : ""
#     }
#     duration_dict = {
#         'intervals' : "", 
#         'collected' : "",
#         'day' : "", 
#         'night' : "",
#         'rest' : "",
#         'active' : ""
#     }
#     intensity_min_dict = {
#         "moderate" : "",
#         "total" : "", 
#         "vigurous" : "",
#     }
#     sleep_dict = {
#         "sleep_map" : "",
#         "awake" : "",
#         "deep" : "", 
#         "light" : "", 
#         "quality" : "",
#         "recorded_time" : "",
#         "rem" : "",
#         "score" : "",
#         "timestamp_end" : "",
#         'percentage_deep' : 0,
#         'percentage_light' : 0,
#         'percentage_rem' : 0,
#         'percentage_awake' : 0,
#     }
#     spo2_dict = {
#         "all_values" : "",
#         "averege" : "",
#         "lowest" : "",
#     }
#     stress_dict = {
#         "all_values" : "",
#         "high" : "", 
#         "low" : "", 
#         "medium" : "",
#         "recorded_time" : "",
#         "rest" : "", 
#         "score" : "",
#     }
    
#     dict_template = {
#                     'activity'     : copy.deepcopy(activity_dict),                 
#                     'body_battery' : copy.deepcopy(body_battery_dict),
#                     'breath'       : copy.deepcopy(breath_dict),
#                     'calories'     : copy.deepcopy(calories_dict),
#                     'cardio'       : copy.deepcopy(cardio_dict),
#                     'duration'     : copy.deepcopy(duration_dict),  
#                     'intensity_min': copy.deepcopy(intensity_min_dict),
#                     'sleep'        : copy.deepcopy(sleep_dict),
#                     'spo2'         : copy.deepcopy(spo2_dict),
#                     'stress'       : copy.deepcopy(stress_dict),
#                     'user_id'      : "",
#                     }
#     return copy.deepcopy(dict_template)

# %% ---------------------------- Test function ------------------------------ 
# ----------------------------------------------------------------------------
# from config import API_KEY_PREPROD, API_KEY_PROD, URL_GARMIN_PREPROD, URL_GARMIN_PROD
# prod = False
# # Michel
# # user_id = "5Nwwut" 
# # date = "2023-05-04" 
# # Adriana
# user_id = "6o2Fzp"
# date = "2023-06-01"

# # -- Ludo
# # user_id = "4vk5VJ"
# # date = "2023-05-25"

# if prod == True :
#     api = API_KEY_PROD
#     url = URL_GARMIN_PROD
# else :
#     api = API_KEY_PREPROD
#     url = URL_GARMIN_PREPROD
# results_dict =  get_garmin_data(user_id, date, api, url)

