# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:23:57 2023

@author: aterman
"""
import copy
import numpy as np
import pandas as pd 
from collections import deque
from automatic_reports.config import ACTIVITY_THREASHOLD
from datetime import datetime

TEXT_FONT = "Helvetica"
BLUE_COLOR = "#3E738D"

# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
def get_common_indicators_pdf(common_data) :

    # Initialize the dictionary where the new data will be saved
    common_indicators_pdf = initialize_dictionary_with_template()

    # ========================== Cardio dict ===============================
    # Add rate high
    dict_aux = common_indicators_pdf["cardio"]["rate_high"]
    dict_aux["x"] = 1.95
    dict_aux["y"] = 7.81
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["cardio"]["rate_resting"]
    dict_aux["x"] = 1.95
    dict_aux["y"] = 8.02
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["cardio"]["rate_var_resting"]
    dict_aux["x"] = 1.95
    dict_aux["y"] = 8.26
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    # Add rate high
    if len(common_data["cardio"]["rate"]) >0:
        value = round(max(common_data["cardio"]["rate"]["values"].dropna()))
        dict_aux = common_indicators_pdf["cardio"]["rate_high"]
        dict_aux["text"] = str(value) + " bpm"

    # Add rate resting 
        df_values = common_data['cardio']['rate']["values"].dropna()
        mean_values = sliding_window(df_values, minutes = 30)
        value = round(min(mean_values))
        dict_aux = common_indicators_pdf["cardio"]["rate_resting"]
        dict_aux["text"] = str(value) + " bpm"


    # Add rate variability
    if len(common_data["cardio"]["rate_var"]) >0:        
        df_aux = common_data['cardio']['rate_var']
        values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, 
                            "values"].dropna().reset_index(drop=True)
        value = round(np.mean(values))
        dict_aux = common_indicators_pdf["cardio"]["rate_var_resting"]
        dict_aux["text"] = str(value) + " ms"

    # ========================== Breath dict =================================
    dict_aux = common_indicators_pdf["breath"]["rate_high"]
    dict_aux["x"] = 4.56
    dict_aux["y"] = 7.55
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["breath"]["rate_resting"]
    dict_aux["x"] = 4.56
    dict_aux["y"] = 7.81
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["breath"]["rate_var_resting"]
    dict_aux["x"] = 4.56
    dict_aux["y"] = 8.06
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["breath"]["inspi_expi"]
    dict_aux["x"] = 4.56
    dict_aux["y"] = 8.28
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 8
    dict_aux["color"] = BLUE_COLOR

    # Add rate high
    if len(common_data["breath"]["rate"]) >0:
        value = round(max(common_data["breath"]["rate"]["values"].dropna()))
        dict_aux = common_indicators_pdf["breath"]["rate_high"]
        dict_aux["text"] = str(value) + " brpm"

    # Add rate resting
        values_df = common_data['breath']['rate']['values'].dropna()
        mean_values = sliding_window(values_df, minutes = 30)
        value = round(min(mean_values))
        dict_aux = common_indicators_pdf["breath"]["rate_resting"]
        dict_aux["text"] = str(value) + " brpm"

    # Add rate variability 
    if len(common_data["breath"]["rate_var"]) >0:
        df_aux = common_data['breath']['rate_var']
        values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, 
                            "values"].dropna().reset_index(drop=True)
        value = round(np.mean(values))

        dict_aux = common_indicators_pdf["breath"]["rate_var_resting"]
        dict_aux["text"] = str(value) + " s"

    # Add inhale/exhale ratio 
    if len(common_data["breath"]["inspi_expi"]) >0:
        df_aux = common_data['breath']['inspi_expi']
        values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, 
                            "values"].dropna().reset_index(drop=True)
        value = round(np.mean(values))

        dict_aux = common_indicators_pdf["breath"]["inspi_expi"]
        dict_aux["text"] = str(value) + " %"

    # ========================== Activity dict ===============================
    dict_aux = common_indicators_pdf["activity"]["steps"]
    dict_aux["x"] = 5.73
    dict_aux["y"] = 7.43
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 12
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["activity"]["goal"]
    dict_aux["x"] = 5.73
    dict_aux["y"] = 7.81
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 12
    dict_aux["color"] = BLUE_COLOR

    dict_aux = common_indicators_pdf["activity"]["distance"]
    dict_aux["x"] = 5.73
    dict_aux["y"] = 8.23
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 12
    dict_aux["color"] = BLUE_COLOR

    if len(common_data["activity"]["steps"]) >0:
        # Steps
        value = round(sum(common_data["activity"]["steps"]["values"].dropna()))
        dict_aux = common_indicators_pdf["activity"]["steps"]
        dict_aux["text"] = str(value) 

        # Goal
        value = common_data["activity"]["goal"]
        dict_aux = common_indicators_pdf["activity"]["goal"]
        dict_aux["text"] = str(value)

        # Distance
        value = round(sum(common_data["activity"]["distance"]["values"].dropna()))
        dict_aux = common_indicators_pdf["activity"]["distance"]
        dict_aux["text"] = str(value) + " m"

    return copy.deepcopy(common_indicators_pdf)

# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------
def initialize_dictionary_with_template() -> dict :
    pdf_info = {
        "text" : "", 
        "x" : "",
        "y" : "",
        "font" : "",
        "size" : "",
        "color" : "",
        }

    cardio_dict = {
        "rate_high"        : copy.deepcopy(pdf_info),
        "rate_resting"     : copy.deepcopy(pdf_info),
        "rate_var_resting" : copy.deepcopy(pdf_info),
    } 
    breath_dict = {
        "rate_high"        : copy.deepcopy(pdf_info),
        "rate_resting"     : copy.deepcopy(pdf_info),
        "rate_var_resting" : copy.deepcopy(pdf_info),
        "inspi_expi"  : copy.deepcopy(pdf_info),
    } 
    activity_dict = {
        "steps"            : copy.deepcopy(pdf_info),
        "distance"         : copy.deepcopy(pdf_info),
        "goal"             : copy.deepcopy(pdf_info),
    }
    
    dict_template = {
                    'cardio'   : copy.deepcopy(cardio_dict),
                    'breath'   : copy.deepcopy(breath_dict),
                    'activity' : copy.deepcopy(activity_dict)
                    }
    
    return copy.deepcopy(dict_template)
    
# Sliding window to compute the average rate on 30 min sliding window
# TO CHANGE !! it has to be adapted to time intervals 
def sliding_window(sequence, minutes):
    """Calcule une moyenne sur des fenêtres glissantes.
    k est la taille de la fenêtre glissante
 
    >>> fenetre_glissante([40, 30, 50, 46, 39, 44], 3)
    [40.0, 42.0, 45.0, 43.0]
    """
    # on initialise avec les k premiers élements
    d = deque(sequence[:minutes])  
    avg = []

    nan_count = d.count('Nan')
    if(nan_count < 10):
        s = sum(d)
        avg.append(s / minutes)  # la moyenne sur la fenêtre
    
    # Calcul de la moyenne sur le fenetre glissante 
    for element in sequence[minutes:]:
        d.append(element)
        s += element - d.popleft()  # on enlève la 1re valeur, on ajoute la nouvelle
        
        nan_count = d.count('Nan')
        if(nan_count < 10):
            s = sum(d)
            avg.append(s / minutes)  # la moyenne sur la fenêtre
 
    return avg