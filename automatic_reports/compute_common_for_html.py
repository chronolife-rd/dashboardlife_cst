# -*- coding: utf-8 -*-
"""
Last updated on 01/06/2023

@author: aterman
"""
import copy
import numpy as np
from collections import deque
from automatic_reports.config import ACTIVITY_THREASHOLD

# ------------------------ The main function ---------------------------------
# ----------------------------------------------------------------------------
def get_common_indicators(common_data) :

    # Initialize the dictionary where the new data will be saved
    common_indicators = initialize_dictionary_with_template()

    # ========================== Cardio dict ===============================
    # ----------------------------- rate -------------------------------------
    if len(common_data["cardio"]["rate"]) >0:
    # Add rate high and max
        values_df = common_data["cardio"]["rate"]["values"].dropna()
        value = round(max(values_df))
        common_indicators["cardio"]["rate_high"] = value
        common_indicators["cardio"]["rate_max"] = value

        # Add rate min
        value = round(min(values_df))
        common_indicators["cardio"]["rate_min"] = value

        # Add rate mean
        value = round(np.mean(values_df))
        common_indicators["cardio"]["rate_mean"] = value
        
        # Add rate resting 
        mean_values = sliding_window(values_df, minutes = 30)
        value = round(min(mean_values))
        common_indicators["cardio"]["rate_resting"] = value

    # --------------------------- rate var -----------------------------------
    if len(common_data["cardio"]["rate_var"]) >0:
    # Add rate variability max and high
        values_df = common_data["cardio"]["rate_var"]["values"].dropna()
        value = round(max(values_df))
        common_indicators["cardio"]["rate_var_high"] = value
        common_indicators["cardio"]["rate_var_max"] = value

        # Add rate variability min
        value = round(min(values_df))
        common_indicators["cardio"]["rate_var_min"] = value

        # Add rate variability mean
        value = round(np.mean(values_df))
        common_indicators["cardio"]["rate_var_mean"] = value

        # Add rate variability resting
        df_aux = common_data['cardio']['rate_var']
        values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, 
                            "values"].dropna().reset_index(drop=True)
        value = round(np.mean(values))
        common_indicators["cardio"]["rate_var_resting"] = value

    # ========================== Breath dict =================================
    # ----------------------------- rate -------------------------------------
    if len(common_data["breath"]["rate"]) >0:
        # Add rate high and max
        values_df = common_data["breath"]["rate"]["values"].dropna()
        value = round(max(values_df))
        common_indicators["breath"]["rate_high"] = value
        common_indicators["breath"]["rate_max"] = value

        # Add rate min
        value = round(min(values_df))
        common_indicators["breath"]["rate_min"] = value

        # Add rate mean
        value = round(np.mean(values_df))
        common_indicators["breath"]["rate_mean"] = value

        # Add rate resting
        mean_values = sliding_window(values_df, minutes = 30)
        value = round(min(mean_values))
        common_indicators["breath"]["rate_resting"] = value

    # --------------------------- rate var -----------------------------------
    if len(common_data["breath"]["rate_var"]) >0:
        # Add rate variability high and max
        values_df = common_data["breath"]["rate_var"]["values"].dropna()
        common_indicators["breath"]["rate_var_max"] = round(max(values_df))
        common_indicators["breath"]["rate_var_high"] = round(max(values_df))

        # Add rate variability min
        common_indicators["breath"]["rate_var_min"] = round(min(values_df))

        # Add rate variability mean
        common_indicators["breath"]["rate_var_mean"] = round(np.mean(values_df))

        # Add rate variability resting
        df_aux = common_data['breath']['rate_var']
        values = df_aux.loc[df_aux["activity_values"] <= ACTIVITY_THREASHOLD, 
                            "values"].dropna().reset_index(drop=True)
        value = round(np.mean(values))
        common_indicators["breath"]["rate_var_resting"] = value

    # ------------------------- inspi expi -----------------------------------
    if len(common_data["breath"]["inspi_expi"]) >0:
        values_df = common_data["breath"]["inspi_expi"]["values"].dropna()

        # Add rate variability high and max
        common_indicators["breath"]["inspi_expi_max"] =  round(max(values_df))
        
        # Add rate variability min
        common_indicators["breath"]["inspi_expi_min"] = round(min(values_df))

        # Add rate variability mean
        common_indicators["breath"]["inspi_expi_mean"] = round(np.mean(values_df))

    # ========================== Activity dict ===============================
    # dictionary with data used to plot steps graph 
    steps_dict_for_plot = {
        "total_steps" : "",
        "goal" : ""
    }

    if len(common_data["activity"]["steps"]) >0:
        # Steps
        value = round(sum(common_data["activity"]["steps"]["values"].dropna()))
        common_indicators["activity"]["steps"] = value
        steps_dict_for_plot["total_steps"] = value

        # Goal        
        common_indicators["activity"]["goal"] = common_data["activity"]["goal"]

        # Distance
        value = round(sum(common_data["activity"]["distance"]["values"].dropna()))
        common_indicators["activity"]["distance"] = value

    return common_indicators

# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------
def initialize_dictionary_with_template() -> dict :
    cardio_dict = {
        "rate_high"        : "",
        "rate_resting"     : "",
        "rate_min"         : "",
        "rate_max"         : "",
        "rate_mean"        : "",

        "rate_var_resting" : "",
        "rate_var_high"    : "",
        "rate_var_min"     : "",
        "rate_var_max"     : "",
        "rate_var_mean"    : "",
    } 
    breath_dict = {
        "rate_high"        : "",
        "rate_max"         : "",
        "rate_min"         : "",
        "rate_mean"        : "",
        "rate_resting"     : "",

        "rate_var_high"    : "",
        "rate_var_max"     : "",
        "rate_var_min"     : "",
        "rate_var_mean"    : "",
        "rate_var_resting" : "",
        
        "inspi_expi_max"  : "",
        "inspi_expi_min"  : "",
        "inspi_expi_mean" : "",
    } 
    activity_dict = {
        "steps"            : "",
        "distance"         : "",
        "goal"             : "",
    }
    dict_template = {
                    'cardio'   : copy.deepcopy(cardio_dict),
                    'breath'   : copy.deepcopy(breath_dict),
                    'activity' : copy.deepcopy(activity_dict)
                    }
    return copy.deepcopy(dict_template)

# Sliding window to compute the average rate on 30 min sliding window
# TO CHANGE !! adapt to time intervals (when the session is interrupted)
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