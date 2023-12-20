import matplotlib.pyplot as plt
import numpy as np

from automatic_reports.config import PATH_SAVE_IMG as path_save
from datetime import datetime, timedelta
from data.data import get_steps
from automatic_reports.config import ACTIVITY_THREASHOLD
import pandas as pd
import os

# Constants (colors)
BLUE ='#3E738D'
GREY = '#6F6F6F'
LEGEND_POSITION = 'upper right'
FIG_SIZE = (14,3)

# ------------------------ The main function ---------------------------------#
# -----------------------------cst only----------------------------------#
def plot_images_old(cst_time_intervals, date, steps_score):
    plt.close("all")
    plot_steps(path_save, steps_score)
    plot_duration(cst_time_intervals, date)

# -----------------------------old ^----------------------------------#
# -----------------------------new v----------------------------------#
    
def plot_images_new(cst_data, date):
    # Get steps score
    steps = get_steps()
    steps_score = steps["score"]

    # Get intervals
    cst_time_intervals = cst_data['duration']['intervals']

    plt.close("all")
    plot_steps(path_save, steps_score)

    if cst_time_intervals != None:
        plot_duration(cst_time_intervals, date)
    
    heart_bpm(cst_data, path_save)
    breath_brpm(cst_data, path_save)
    heart_hrv(cst_data, path_save)
    breath_brv(cst_data, path_save)


    

# ----------------------- Internal functions ---------------------------------
# ----------------------------------------------------------------------------

# --- Steps ---
def plot_steps(path_save, steps_score):
    image_path = path_save +"/steps.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    plt.figure(figsize=(5,5))
    plt.axis('off')

    if isinstance(steps_score, str) == False:
        if (steps_score < 100):
            size_of_groups=[steps_score, 100-steps_score]
            plt.pie(size_of_groups, 
                    colors=["#4393B4", "#E6E6E6"], 
                    startangle=90,
                    counterclock=False)
        if (steps_score >= 100) :
            size_of_groups=[100]
            plt.pie(size_of_groups, 
                    colors=["#13A943"], 
                    startangle=90,
                    counterclock=False)
            
        my_circle=plt.Circle( (0,0), 0.9, color="white")
        p=plt.gcf()
        p.gca().add_artist(my_circle)  

        plt.text(0, 0, (str(steps_score) + '%'), fontsize=30, color=BLUE,
                  horizontalalignment = "center")
        plt.text(0, -0.25, 'of Goal', fontsize=20, color=GREY,  
                 horizontalalignment = "center") 

    else:
        plt.text(0, -1.5, "no data", fontsize=40, color=BLUE,
                 horizontalalignment='center')
    plt.savefig(image_path, transparent=True)
    
    
# --- Duration ---
def plot_duration(cst_times, date):
    image_path = path_save +"/duration.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    # Set the figure parameters
    CST_COLOR = "#F7921E"
    CHART_COLOR = "#3E738D"
    NIGHT_COLOR = "#EEF3F5"
    DAY_COLOR = "white"

    # Times constants
    YEAR = int(date[:4])
    M = int(date[5:7])
    D = int(date[8:10])

    DATE = datetime(YEAR, M, D, 0, 0, 0)
    X_LIMIT_MIN = DATE - timedelta(minutes = 15) 
    X_LIMIT_MAX = DATE + timedelta(days = 1, minutes = 15)
    TIMES_RANGE = [DATE + timedelta(hours=1*x) for x in range(0, 25)]
    TIMES_SPECIAL = [DATE + timedelta(hours=3*x) for x in range(0, 9)]
    TIMES_LABELS = ["12 am", "3 am", "6 am", "9 am", "12 pm", "3 pm", "6 pm", "9 pm", "12 am" ]
    
    with plt.rc_context({'axes.edgecolor':'white', 'xtick.color': CHART_COLOR, 'ytick.color': "white"}):
        
        # Set size of the figure 
        plt.figure(figsize=(16,3))
        
        # Set x and y limits
        plt.xlim([X_LIMIT_MIN, X_LIMIT_MAX])
        plt.ylim([-1, 2])
                
        # Gange x ticks 
        plt.xticks(TIMES_SPECIAL, TIMES_LABELS, color = CHART_COLOR, size = 16) 
    
        # Color the backround of "night" and "day" period
        plt.axvspan(xmin = TIMES_SPECIAL[0], xmax = TIMES_SPECIAL[2], ymin = 0.15, ymax = 2, facecolor = NIGHT_COLOR)
        plt.axvspan(xmin = TIMES_SPECIAL[2], xmax = TIMES_SPECIAL[8], ymin = 0.15, ymax = 2, facecolor = DAY_COLOR)

        # Plot the design of the chart (the x axis)
        plt.hlines(-0.5, xmin = TIMES_SPECIAL[0], xmax = TIMES_SPECIAL[8], color = CHART_COLOR, linewidth=3)
        for i in TIMES_RANGE:
            plt.vlines(i, ymin = -0.5, ymax = 2, color = CHART_COLOR, linewidth=1)
            plt.scatter(x = i, y = -0.5, s = 50, color = CHART_COLOR )
        for i in TIMES_SPECIAL:
            plt.vlines(i, ymin = -0.5, ymax = 2, color = CHART_COLOR, linewidth=1.5)
            plt.scatter(x = i, y = -0.5, s = 200, color = CHART_COLOR )
    
        # Plot CST
        if cst_times != None:
            for i in range(len(cst_times)):
                interval = np.array(cst_times[i][:])
                if (len(interval) > 10):
                    y_value = np.zeros(len(interval)) + 0.25
                    plt.plot(interval, y_value, color = CST_COLOR, linewidth=20, solid_capstyle='round')
    
        plt.tight_layout()
        
        # Save as image
        plt.savefig(image_path, transparent=True)

# --- Charts indicators ---
def heart_bpm(cst_data, path_save):
    
    image_path = path_save +"/heart_bpm.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    _, ax = plt.subplots(2, figsize=FIG_SIZE, sharex=True)

    # Add cst 
    df_ref = cst_data['cardio']['rate']
    if len(df_ref) > 0:
        ax[0].plot(df_ref['times'], df_ref['values'],color='C0', label = 'CST HR' )
        ax[0].legend(loc=LEGEND_POSITION)

    # Add activity 
    df_ref = cst_data['activity']['averaged_activity']
    if len(df_ref) > 0:
        ax[1].plot(df_ref['times'], df_ref['activity_values'],color='C2', label = 'ACTIVITY' )
        # Add activity limit 
        ax[1].axhline(y=ACTIVITY_THREASHOLD, color='black', linestyle='--', label='ACTIVITY THREASHOLD')
        ax[1].legend(loc=LEGEND_POSITION)

    plt.tight_layout()
    
    # Save as image
    plt.savefig(image_path, transparent=True)

def breath_brpm(cst_data, path_save):
    
    image_path = path_save +"/breath_brpm.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    _, ax = plt.subplots(2, figsize=FIG_SIZE, sharex=True)

    # Add cst 
    df_ref = cst_data['breath']['rate']
    if len(df_ref) > 0:
        ax[0].plot(df_ref['times'], df_ref['values'],color='C0', label = 'CST BR' )
        ax[0].legend(loc=LEGEND_POSITION)

    # Add activity 
    df_ref = cst_data['activity']['averaged_activity']
    if len(df_ref) > 0:
        ax[1].plot(df_ref['times'], df_ref['activity_values'],color='C2', label = 'ACTIVITY' )
        # Add activity limit 
        ax[1].axhline(y=ACTIVITY_THREASHOLD, color='black', linestyle='--', label='ACTIVITY THREASHOLD')
        ax[1].legend(loc=LEGEND_POSITION)

    plt.tight_layout()
    
    # Save as image
    plt.savefig(image_path, transparent=True)

def heart_hrv(cst_data, path_save):
    
    image_path = path_save +"/heart_hrv.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    _, ax = plt.subplots(2, figsize=FIG_SIZE, sharex=True)

    # Add cst 
    df_ref = cst_data['cardio']['rate_var']
    if len(df_ref) > 0:
        ax[0].plot(df_ref['times'], df_ref['values'],color='C0', label = 'CST HRV' )
        ax[0].legend(loc=LEGEND_POSITION)
    
    # Add activity 
    df_ref = cst_data['activity']['averaged_activity']
    if len(df_ref) > 0:
        ax[1].plot(df_ref['times'], df_ref['activity_values'],color='C2', label = 'ACTIVITY' )
        # Add activity limit 
        ax[1].axhline(y=ACTIVITY_THREASHOLD, color='black', linestyle='--', label='ACTIVITY THREASHOLD')
        ax[1].legend(loc=LEGEND_POSITION)

    plt.tight_layout()
    
    # Save as image
    plt.savefig(image_path, transparent=True)

def breath_brv(cst_data, path_save):
    
    image_path = path_save +"/breath_brv.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    _, ax = plt.subplots(2, figsize=FIG_SIZE, sharex=True)

    # Add cst 
    df_ref = cst_data['breath']['rate_var']
    if len(df_ref) > 0:
        ax[0].plot(df_ref['times'], df_ref['values'],color='C0', label = 'CST BRV' )
        ax[0].legend(loc=LEGEND_POSITION)
    
    # Add activity 
    df_ref = cst_data['activity']['averaged_activity']
    if len(df_ref) > 0:
        ax[1].plot(df_ref['times'], df_ref['activity_values'],color='C2', label = 'ACTIVITY' )
        # Add activity limit 
        ax[1].axhline(y=ACTIVITY_THREASHOLD, color='black', linestyle='--', label='ACTIVITY THREASHOLD')
        ax[1].legend(loc=LEGEND_POSITION)

    plt.tight_layout()
    
    # Save as image
    plt.savefig(image_path, transparent=True)