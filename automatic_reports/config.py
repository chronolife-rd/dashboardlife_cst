from enum import Enum
from datetime import timedelta
# Request data from servers inputs 
API_KEY_PROD = 'CLjfUipLb32dfMC8ZCCwUA' 
# API_KEY_PREPROD = '3-1krbPQoufnmNVN6semRA'

URL_CST_PROD = "https://prod.chronolife.net/api/2/data"
# URL_CST_PREPROD = "https://preprod.chronolife.net/api/2/data"
# URL_GARMIN_PROD = "https://prod.chronolife.net/api/2/garmin/data" 
# URL_GARMIN_PREPROD = "https://preprod.chronolife.net/api/2/garmin/data" 

CST_SIGNAL_TYPES = 'heartbeat,HRV,qt_c_framingham_per_seg,breath_2_brpm,breath_2_brv,breath_2_inspi_over_expi,averaged_activity,steps_number,temp_1,temp_2,activity_level' 
# GARMIN_SIGNAL_TYPES = 'dailies,epochs,sleeps,allDayRespiration,stressDetails,pulseox'

# Constants used in functions 
TEMPERATURE_THREASHOLD = 2500 # min temperature of weared t-shirt
ACTIVITY_THREASHOLD = 18 # constant used in computing alerts
BRADYPNEA_TH = 6
TACHYPNEA_TH = 20

BRADYCARDIA_TH = 60
TACHYCARDIA_TH = 100

QT_MIN_TH = 350
QT_MAX_TH = 500

DELTA_TIME = timedelta(minutes = 3) # constant used in computing times intervals 

# Path to the pdf results folder 
PATH_PDF = "automatic_reports/pdf_results"
# Path save reports images  
PATH_SAVE_IMG = "automatic_reports/report_images"
# Path to alerts images for PDF
RED_ALERT = PATH_SAVE_IMG + "/alerts/red.png"
GREEN_ALERT = PATH_SAVE_IMG + "/alerts/green.png"

# Constants for PDF
ICON_SIZE = 0.17
ALERT_SIZE = 0.16
HEIGHT_CIRCLE = 2
WIDTH_CIRCLE = 2

# Garmin Indicators variables for PDF
# class GarminIndicator(Enum):
#     DURATION = "duration"
#     STRESS = "stress"
#     SLEEP = "sleep"
#     CALORIES = "calories"
#     INTENSITY_MIN = "intensity_min"
#     BODY_BATTERY = "body_battery"
    
class CstIndicator(Enum):
    HEADER = "header"
    DURATION = "duration"

class CommonIndicator(Enum):
    CARDIO = "cardio"
    BREATH = "breath"
    ACTIVITY = "activity"

class Alert(Enum):
    BRADYPNEA = "bradypnea"
    TACHYPNEA = "tachypnea"
    BRADYCARDIA = "bradycardia"
    TACHYCARDIA = "tachycardia"
    QT = "qt"

# Images parameters for PDF
class ImageForPdf(Enum):
    STEPS = {
        "path" : PATH_SAVE_IMG + "/steps.png",
        "x" : 6.23,
        "y" : 6.63 + HEIGHT_CIRCLE,
        "w" : WIDTH_CIRCLE,
        "h" : HEIGHT_CIRCLE,
    }

    # STRESS = {
    #     "path" : PATH_SAVE_IMG + "/stress.png",
    #     "x" : 1,
    #     "y" : 8.4 + HEIGHT_CIRCLE, 
    #     "w" : WIDTH_CIRCLE,
    #     "h" : HEIGHT_CIRCLE,
    # }

    # SPO2 = {
    #     "path" : PATH_SAVE_IMG + "/spo2.png",
    #     "x" : 3.63,
    #     "y" : 8.4 + HEIGHT_CIRCLE, 
    #     "w" : WIDTH_CIRCLE,
    #     "h" : HEIGHT_CIRCLE,
    # }

    # SLEEP = { 
    #     "path" : PATH_SAVE_IMG + "/sleep.png",
    #     "x" : 6.26,
    #     "y" : 8.4 + HEIGHT_CIRCLE, 
    #     "w" : WIDTH_CIRCLE,
    #     "h" : HEIGHT_CIRCLE,
    # }
    
    DURATION = { 
        "path" : PATH_SAVE_IMG + "/duration.png",
        "x" : 0.19,
        "y" : 4.74 + 1.46, 
        "w" : 7.78, 
        "h" : 1.46,
    }
    # ------------ icons ---------------
    # NIGHT_ICON_LEFT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/night.png",
    #     "x" : 2.18,
    #     "y" : 2.4 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # DAY_ICON_LEFT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/day.png",
    #     "x" : 2.18,
    #     "y" : 2.74 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # REST_ICON_LEFT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/rest.png",
    #     "x" : 2.18,
    #     "y" : 3.06 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # ACTIVE_ICON_LEFT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/active.png",
    #     "x" : 2.18,
    #     "y" : 3.39 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # NIGHT_ICON_RIGHT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/night.png",
    #     "x" : 6.15,
    #     "y" : 2.4 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # DAY_ICON_RIGHT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/day.png",
    #     "x" : 6.15,
    #     "y" : 2.74 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # REST_ICON_RIGHT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/rest.png",
    #     "x" : 6.15,
    #     "y" : 3.06 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }
    # ACTIVE_ICON_RIGHT = { 
    #     "path" : PATH_SAVE_IMG + "/icons/active.png",
    #     "x" : 6.15,
    #     "y" : 3.39 + ICON_SIZE, 
    #     "w" : ICON_SIZE, 
    #     "h" : ICON_SIZE,
    # }

