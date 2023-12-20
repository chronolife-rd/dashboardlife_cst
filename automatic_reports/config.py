from enum import Enum
from datetime import timedelta
# Request data from servers inputs 
API_KEY_PROD = 'CLjfUipLb32dfMC8ZCCwUA' 

URL_CST_PROD = "https://prod.chronolife.net/api/2/data"
CST_SIGNAL_TYPES = 'heartbeat,HRV,qt_c_framingham_per_seg,breath_2_brpm,breath_2_brv,breath_2_inspi_over_expi,averaged_activity,steps_number,temp_1,temp_2,activity_level' 


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
HEIGHT_INDICATOR_PLOT = 1.63
WIDTH_INDICATOR_PLOT = 7.61

    
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
    DURATION = { 
        "path" : PATH_SAVE_IMG + "/duration.png",
        "x" : 0.19,
        "y" : 4.74 + 1.46, 
        "w" : 7.78, 
        "h" : 1.46,
    }

class IndicatorsImageForPdf(Enum):
    HR = { 
        "path" : PATH_SAVE_IMG + "/heart_bpm.png",
        "x" : 0.33,
        "y" : 2.33 + HEIGHT_INDICATOR_PLOT, 
        "w" : WIDTH_INDICATOR_PLOT, 
        "h" : HEIGHT_INDICATOR_PLOT,
    }

    BR = { 
        "path" : PATH_SAVE_IMG + "/breath_brpm.png",
        "x" : 0.27,
        "y" : 4.77 + HEIGHT_INDICATOR_PLOT, 
        "w" : WIDTH_INDICATOR_PLOT, 
        "h" : HEIGHT_INDICATOR_PLOT,
    }

    HRV = { 
        "path" : PATH_SAVE_IMG + "/heart_hrv.png",
        "x" : 0.27,
        "y" : 7.2 + HEIGHT_INDICATOR_PLOT, 
        "w" : WIDTH_INDICATOR_PLOT, 
        "h" : HEIGHT_INDICATOR_PLOT,
    }

    BRV = { 
        "path" : PATH_SAVE_IMG + "/breath_brv.png",
        "x" : 0.27,
        "y" : 9.64 + HEIGHT_INDICATOR_PLOT, 
        "w" : WIDTH_INDICATOR_PLOT, 
        "h" : HEIGHT_INDICATOR_PLOT,
    }
    