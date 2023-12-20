# -*- coding: utf-8 -*-
"""
Last updated on 07/06/2023
@author: aterman

CST = Chronolife Smart Textile
"""
# Imports 
import time 
from api2_get_cst_data import get_cst_data
# from api2_get_garmin_data import get_garmin_data
from useful_functions import combine_data

from compute_common_for_html import get_common_indicators
from compute_common_for_pdf import get_common_indicators_pdf
from plot_images import plot_images

API_KEY_PROD = 'CLjfUipLb32dfMC8ZCCwUA' 
# API_KEY_PREPROD = '3-1krbPQoufnmNVN6semRA'

URL_CST_PROD = "https://prod.chronolife.net/api/2/data"
# URL_CST_PREPROD = "https://preprod.chronolife.net/api/2/data"
# URL_GARMIN_PROD = "https://prod.chronolife.net/api/2/garmin/data" 
# URL_GARMIN_PREPROD = "https://preprod.chronolife.net/api/2/garmin/data" 

# ----------------------------- Chronolife -------------------------------
# Start timer   
# -- Ludo
# user_id = "4vk5VJ"
# date = "2023-05-25"
# -- Fernando
# user_id = "5Nwwut"
# date = "2023-05-17"
# -- Michel
# user_id = "5Nwwut" 
# date = "2023-05-04" 
# # -- Adriana
user_id = "6o2Fzp"
date = "2023-06-01"

begin = time.time()

cst_data = get_cst_data(
                        user_id = user_id, 
                        date = date,
                        api = API_KEY_PROD,
                        url = URL_CST_PROD
                        )

# End timer
end = time.time()
print('Time taken to get CST data:',round((end-begin)/60,2),'min')

# ----------------------------- Garmin -----------------------------------
# Start timer    
begin = time.time()

# garmin_data = get_garmin_data(
#     user_id = user_id, 
#     date = date,
#     api = API_KEY_PREPROD,
#     url = URL_GARMIN_PREPROD
#     )

# End timer
end = time.time()
print('Time taken to get Garmin data:', round((end-begin)/60,2),'min')

# %% ------------------------------ Common --------------------------------------
# Compute common indicators: cardio, respiration and steps
common_data = combine_data(cst_data)

common_indicators = get_common_indicators(common_data) 
common_indicators_pdf = get_common_indicators_pdf(common_data) 

# Plot and save graphs
# Time intervals
cst_time_intervals = cst_data['duration']['intervals']
# Time intervals
# garmin_time_intervals = garmin_data['duration']['intervals']

plot_images(cst_time_intervals,date)
