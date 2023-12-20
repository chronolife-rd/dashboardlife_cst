import streamlit as st
import template.constant as constant
import template.test as test
import requests
import json
from datetime import datetime, timedelta
from pylife.api_functions import map_data
from pylife.api_functions import map_data_filt
from pylife.api_functions import get_sig_info

   
def get_raw_data():

    offset = st.session_state.chronolife_offset
    print("OFFSET :")
    print(offset)
    sign = 1
    if offset>=0:
        sign = 1
    elif offset<0:
        sign = -1

    offset_info = {
        'value' : abs(offset),
        'sign' : sign
    }
    
    url         = st.session_state.url_data
    api_key     = st.session_state.api_key
    user        = st.session_state.end_user
    types       = constant.TYPE()["SIGNALS"]
    date        = st.session_state.date
    # start_time  =  f"{st.session_state.start_time_dl}:00" # str
    # end_time    =  f"{st.session_state.end_time_dl}:00"   # str
    start_time  = st.session_state.start_time + ":00" # str
    end_time    = st.session_state.end_time + ":00"   # str

    time_gte    = start_time 
    time_lt     = end_time

    if isinstance(offset_info["value"], str) == False:
        sign = offset_info["sign"] 
        format_datetime = "%H:%M:%S"
        # Str to datetime
        start_time = datetime.strptime(start_time, format_datetime) # datetime
        end_time   = datetime.strptime(end_time, format_datetime)   # datetime
        
        time_gte  = start_time - sign*timedelta(seconds = offset_info["value"])
        time_lt   = end_time - sign*timedelta(seconds = offset_info["value"])

        # Datetime to str 
        time_gte = datetime.strftime(time_gte, format_datetime)  # str
        time_lt  = datetime.strftime(time_lt, format_datetime)   # str
    
    params = {
           'user':      user, # sub-user username
           'types':     types, 
           'date':      date,
           'time_gte':  time_gte, # UTC
           'time_lt':   time_lt,  # UTC
         }
    
    # Perform the POST request authenticated with YOUR API key (NOT the one of the sub-user!).
    reply = get_reply(params, url, api_key)
    message, status_code = test.api_status(reply)

    # Convert the reply content into a json object
    raw_data = convert_reply_to_datas(reply, status_code, offset_info)

    # Declare as global variable
    st.session_state.smart_textile_raw_data = raw_data

def get_raw_data_to_dowload():
    offset_info = {
        'value' : 7200,
        'sign' : 1
    }
    
    url         = st.session_state.url_data
    api_key     = st.session_state.api_key
    user        = st.session_state.end_user
    types       = constant.TYPE()["SIGNALS"]
    date        = st.session_state.date
    start_time  = "00:00:00" # str
    end_time    = "23:59:59"   # str

    params = {
           'user':      user, # sub-user username
           'types':     types, 
           'date':      date,
           'time_gte':  start_time, # UTC
           'time_lt':   end_time,  # UTC
         }
    
    # Perform the POST request authenticated with YOUR API key (NOT the one of the sub-user!).
    reply = get_reply(params, url, api_key)
    message, status_code = test.api_status(reply)

    # Convert the reply content into a json object
    raw_data = convert_reply_to_datas(reply, status_code, offset_info)

    # Declare as global variable
    st.session_state.smart_textile_raw_data_all_day = raw_data

    error = False
    if len(st.session_state.smart_textile_raw_data_all_day) == 0:
        error = True 
    if error:
        return error




def get_reply(params, url, api_key):
    reply = requests.get(url, headers={"X-API-Key": api_key}, params=params)
    return reply

def convert_reply_to_datas(reply, status_code, offset_info):
    datas  = []
    result = []

    if status_code == 200:  
        json_list_of_records = json.loads(reply.text) 
        for record in json_list_of_records:
            datas.append(record)
        
        if len(datas) == 0:
            status_code = 600
            print("Lenght of datas is 0")
    
    # If len(datas) != 0
    if status_code == 200:
        result = {}

        # --- Map raw data 
        map_raw_data(datas, result, offset_info)
        
        # --- Map filtered data 
        map_filtered_data(datas, result, offset_info)

    return result

def map_raw_data(datas, result, offset_info):
    types_raw    = constant.TYPE()["RAW_SIGNALS"].split(',')
    datas_mapped = map_data(datas, types_raw)
    
    for key_type in types_raw:
        result[key_type] = get_sig_info(datas_mapped, key_type, verbose=0)
        result[key_type]["times"] = add_offset(result[key_type]["times"], offset_info)
    
def map_filtered_data(datas, result, offset_info):
    types_filtered  = constant.TYPE()["FILTERED_SIGNALS"].split(',')
    datas_filtered_mapped = map_data_filt(datas, types_filtered)
    for key_type in types_filtered:
        result[key_type] = get_sig_info(datas_filtered_mapped, key_type, verbose=0)
        result[key_type]["times"] = add_offset(result[key_type]["times"], offset_info)

def add_offset(times, offset_info):
    if isinstance(offset_info["value"], str) == False:
        value = offset_info["value"] 
        sign = offset_info["sign"] 

        new_times = []
        for i, times_seg in enumerate((times)):
            new_times_seg = []
            for i, time_value in enumerate(times_seg):
                time_value = time_value.astype(datetime)   # datetime
                time_value = time_value + sign*timedelta(seconds = value)
                new_times_seg.append(time_value)

            new_times.append(new_times_seg)
            
    return new_times
     









