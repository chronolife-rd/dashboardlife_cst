import streamlit as st
import template.language.en as en
import template.language.fr as fr 
from template.util import img_to_bytes

def init():
    
    if 'chronolife_data_available' not in st.session_state:
        st.session_state.chronolife_data_available = False

    # if 'garmin_data_available' not in st.session_state:
    #     st.session_state.garmin_data_available = False

    
    if 'prod' not in st.session_state:
        st.session_state.prod = True
        
    if 'url_root' not in st.session_state:
        st.session_state.url_root = ""
        
    if 'url_data' not in st.session_state:
        st.session_state.url_data = ""
        
    if 'url_user' not in st.session_state:
        st.session_state.url_user = ""
        
    # if 'url_garmin' not in st.session_state:
    #     st.session_state.url_garmin = ""
        
    if 'username' not in st.session_state:
        st.session_state.username = ''
        
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
        
    if 'myendusers' not in st.session_state:
        st.session_state.myendusers = ''

    if 'date' not in st.session_state:
            st.session_state.date = ''
            
    if 'start_time' not in st.session_state:
            st.session_state.start_time = ''
            
    if 'end_time' not in st.session_state:
            st.session_state.end_time = ''
            
    if 'end_user' not in st.session_state:
            st.session_state.end_user = ''
            
    if 'timezone' not in st.session_state:
            st.session_state.timezone = ''
            
    if 'enduser_sessions' not in st.session_state:
            st.session_state.enduser_sessions = ''
            
    if 'form_indicators_layout' not in st.session_state:
            st.session_state.form_indicators_layout = ''
            
    if 'form_indicators_submit' not in st.session_state:
            st.session_state.form_indicators_submit = False
            
    if 'form_raw_layout' not in st.session_state:
            st.session_state.form_raw_layout = False
            
    if 'form_raw_submit' not in st.session_state:
            st.session_state.form_raw_submit = False
    
    if 'form_raw_dowload_submit_dl' not in st.session_state:
            st.session_state.form_raw_dowload_submit_dl = False
    
    if 'form_raw_dowload_submit' not in st.session_state:
            st.session_state.form_raw_dowload_submit = False
            
    if 'is_logged' not in st.session_state:
            st.session_state.is_logged = False
    
    if 'logout_submit' not in st.session_state:
            st.session_state.logout_submit = False
            
    if 'is_data' not in st.session_state:
            st.session_state.is_data = False

    if 'al' not in st.session_state:
            st.session_state.al = None
    
    if 'end_users_list' not in st.session_state:
            st.session_state.end_users_list = None
            
    if 'all_end_users_sessions' not in st.session_state:
            st.session_state.all_end_users_sessions = None
            
    if 'end_user_sessions' not in st.session_state:
            st.session_state.end_user_sessions = None
            
    if 'end_user_sessions_run' not in st.session_state:
            st.session_state.end_user_sessions_run = False
            
    if 'tachycardia_alert_icon' not in st.session_state:
            st.session_state.tachycardia_alert_icon = None
            
    if 'bradycardia_alert_icon' not in st.session_state:
            st.session_state.bradycardia_alert_icon = None
            
    if 'qt_alert_icon' not in st.session_state:
            st.session_state.qt_alert_icon = None
            
    if 'tachypnea_alert_icon' not in st.session_state:
            st.session_state.tachypnea_alert_icon = None
            
    if 'bradypnea_alert_icon' not in st.session_state:
            st.session_state.bradypnea_alert_icon = None
            
    if 'smart_textile_raw_data' not in st.session_state:
            st.session_state.smart_textile_raw_data = None
            
    if 'background_wave' not in st.session_state:
        st.session_state.background_wave = img_to_bytes('template/images/background_wave.png')
            
    if 'logo_clife_white' not in st.session_state:
        st.session_state.logo_clife_white = img_to_bytes('template/images/logo_clife_white.png')
    
    if 'night' not in st.session_state:
        st.session_state.night = img_to_bytes('template/images/night.png')

    if 'day' not in st.session_state:
        st.session_state.day = img_to_bytes('template/images/day.png')

    if 'rest' not in st.session_state:
        st.session_state.rest = img_to_bytes('template/images/rest.png')

    if 'activity' not in st.session_state:
        st.session_state.activity = img_to_bytes('template/images/activity.png')

    if 'tshirt_right' not in st.session_state:
        st.session_state.tshirt_right = img_to_bytes('template/images/T-shirt.png')
    
    # if 'garrmin' not in st.session_state:
    #     st.session_state.garrmin = img_to_bytes('template/images/watch.png')
    
    if 'alert' not in st.session_state:
        st.session_state.alert = img_to_bytes('template/images/alert.png')
    
    if 'alert_no' not in st.session_state:
        st.session_state.alert_no = img_to_bytes('template/images/alert_no.png')
    
    if 'heart_icon' not in st.session_state:
        st.session_state.heart_icon = img_to_bytes('template/images/heart.png')
    
    if 'breath_icon' not in st.session_state:
        st.session_state.breath_icon = img_to_bytes('template/images/breath.png')
    
    if 'steps_icon' not in st.session_state:
        st.session_state.steps_icon = img_to_bytes('template/images/steps.png')

    # if 'stress_icon' not in st.session_state:
    #     st.session_state.stress_icon = img_to_bytes('template/images/stress.png')
    
    # if 'stress_rest' not in st.session_state:
    #     st.session_state.stress_rest = img_to_bytes('template/images/stress_rest.png')
    
    # if 'stress_low' not in st.session_state:
    #     st.session_state.stress_low = img_to_bytes('template/images/stress_low.png')
    
    # if 'stress_medium' not in st.session_state:
    #     st.session_state.stress_medium = img_to_bytes('template/images/stress_medium.png')
        
    # if 'stress_high' not in st.session_state:
    #     st.session_state.stress_high = img_to_bytes('template/images/stress_high.png')
        
    # if 'pulseox_icon' not in st.session_state:
    #     st.session_state.pulseox_icon = img_to_bytes('template/images/pulseox.png')
        
    # if 'spo2_green' not in st.session_state:
    #     st.session_state.spo2_green = img_to_bytes('template/images/spo2_green.png')

    # if 'spo2_yellow' not in st.session_state:
    #     st.session_state.spo2_yellow = img_to_bytes('template/images/spo2_yellow.png')

    # if 'spo2_orange' not in st.session_state:
    #     st.session_state.spo2_orange = img_to_bytes('template/images/spo2_orange.png')
    
    # if 'spo2_red' not in st.session_state:
    #     st.session_state.spo2_red = img_to_bytes('template/images/spo2_red.png')
        
    # if 'sleep_icon' not in st.session_state:
    #     st.session_state.sleep_icon = img_to_bytes('template/images/sleep.png')
    
    # if 'sleep_deep' not in st.session_state:
    #     st.session_state.sleep_deep = img_to_bytes('template/images/sleep_deep.png')
        
    # if 'sleep_light' not in st.session_state:
    #     st.session_state.sleep_light = img_to_bytes('template/images/sleep_light.png')    
        
    # if 'sleep_rem' not in st.session_state:
    #     st.session_state.sleep_rem = img_to_bytes('template/images/sleep_rem.png')
    
    # if 'sleep_awake' not in st.session_state:
    #     st.session_state.sleep_awake = img_to_bytes('template/images/sleep_awake.png')
        
    # if 'calories_icon' not in st.session_state:
    #     st.session_state.calories_icon = img_to_bytes('template/images/calories.png')
    
    # if 'intensity_icon' not in st.session_state:
    #     st.session_state.intensity_icon = img_to_bytes('template/images/intensity_minutes.png')
    
    # if 'bodybattery_icon' not in st.session_state:
    #     st.session_state.bodybattery_icon = img_to_bytes('template/images/body_battery.png')
        
    if 'temperature_icon' not in st.session_state:
        st.session_state.temperature_icon = img_to_bytes('template/images/temperature.png')
        
    if 'to_top' not in st.session_state:
        st.session_state.to_top = img_to_bytes('template/images/to_top.png')
        
    if 'user_icon' not in st.session_state:
        st.session_state.user_icon = img_to_bytes('template/images/user.png')
        
    # if 'stress_donut' not in st.session_state:
    #     st.session_state.stress_donut = None
        
    # if 'spo2_donut' not in st.session_state:
    #     st.session_state.spo2_donut = None
        
    # if 'sleep_donut' not in st.session_state:
    #     st.session_state.sleep_donut = None
        
    # if 'steps_donut' not in st.session_state:
    #     st.session_state.steps_donut = None
        
    # if 'garmin_indicators' not in st.session_state:
    #     st.session_state.garmin_indicators = []
    
    if 'chronolife_indicators' not in st.session_state:
            st.session_state.chronolife_indicators = []
            
    if 'common_data' not in st.session_state:
            st.session_state.common_data = []
    
    if 'common_indicators' not in st.session_state:
            st.session_state.common_indicators = []

    if 'common_indicators_pdf' not in st.session_state:
            st.session_state.common_indicators_pdf = []

    # if 'garmin_indicators_pdf' not in st.session_state:
    #         st.session_state.garmin_indicators_pdf = []

    if 'chronolife_indicators_pdf' not in st.session_state:
            st.session_state.chronolife_indicators_pdf = []
    
    if 'chronolife_offset' not in st.session_state:
            st.session_state.chronolife_offset = ''
                        
    if 'translate' not in st.session_state:
        st.session_state.translate = ''
        
    if 'language' not in st.session_state:
        st.session_state.language = 'FR'
    
def init_simul():
    st.session_state.tachycardia_alert_icon = st.session_state.alert_no
    st.session_state.bradycardia_alert_icon = st.session_state.alert_no
    st.session_state.qt_alert_icon = st.session_state.alert_no
    st.session_state.tachypnea_alert_icon = st.session_state.alert_no
    st.session_state.bradypnea_alert_icon = st.session_state.alert_no
    
        
def restart():
    for key in st.session_state.keys():
        del st.session_state[key]
    init()
    st.experimental_rerun()
    
def set_translation():
    
    translate = st.session_state.translate
    language = st.session_state.language
    
    if language == "FR":
        translate = fr.translate()
    elif language == "EN":
        translate = en.translate()
    else:
        st.sidebar.error("Language has not been set properly")
    
    st.session_state.translate = translate
    
def set_url():
    
    # prod = st.session_state.prod
    url_root = "https://prod.chronolife.net/api/2"
    
    # if prod:
    #     url_root = "https://prod.chronolife.net/api/2"
    # else:
    #     url_root = "https://preprod.chronolife.net/api/2"
        
    st.session_state.url_root       = url_root
    st.session_state.url_data       = url_root + "/data"
    st.session_state.url_user       = url_root + "/user"
    # st.session_state.url_garmin     = url_root + "/garmin/data" 