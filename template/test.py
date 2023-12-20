import requests 
import streamlit as st

def authentication():
    
    api_key     = st.session_state.api_key
    username    = st.session_state.username
    url_root    = st.session_state.url_root
    status_code = None
    message     = None
    
    url = url_root + "/user/{userId}".format(userId=username)
    
    reply = requests.get(url, headers={"X-API-Key": api_key})
    message, status_code = api_status(reply)

    return message, status_code

def authentication2():
    
    api_key     = st.session_state.api_key
    url         = st.session_state.url_data
    status_code = None
    message     = None
    
    params = {
           'user':      "5Nwwut", # sub-user username 6o2Fzp 5P4svk
           'types':     'temp_1', 
           'date':      '2010-05-02',
         }
    
    reply = requests.get(url, headers={"X-API-Key": api_key}, params=params)
    message, status_code = api_status(reply)

    return message, status_code

def api_status(reply, user_text='Username'):
    
    translate = st.session_state.translate
    
    status_code = reply.status_code
        
    if status_code == 200:
        message = translate["api_auth_200"] #'Connected'
    elif status_code == 400:
        message = translate["api_auth_400"] #'Part of the request could not be parsed or is incorrect'
    elif status_code == 401:
        message = translate["api_auth_401"] #'Incorrect API key'
    elif status_code == 403:
        message = translate["api_auth_403"] #'Not authorized'
    elif status_code == 404:
        message = translate["api_auth_404"] #'Incorrect url'
    elif status_code == 500:
        message = translate["api_auth_500"] 
    elif status_code == 0:
        message = translate["api_auth_0"] #"You are disconnect"
        
    return message, status_code

def string(string, name, layout):
    
    translate = st.session_state.translate
    
    error = False
    message = False
    if len(string) > 0:
        string = string.replace(" ", "")
    else:
        message = translate["error_string_empty"] + " " + name 
        layout.error(message)
        error = True
        return string, message, error
        
    return string, message, error
    
def end_user(end_user, layout):
    
    translate = st.session_state.translate
    
    error = False
    message = False
    if end_user == "":
        message = translate["error_enduser_empty"] 
        error = True
        return message, error
    
    if len(end_user) != 6:
        message = translate["error_enduser_length"] 
        error = True
        return message, error

    return message, error
    
