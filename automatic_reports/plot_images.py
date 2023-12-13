import matplotlib.pyplot as plt
import numpy as np

from automatic_reports.config import PATH_SAVE_IMG as path_save
from datetime import datetime, timedelta
import os

# Constants (colors)
BLUE ='#3E738D'
GREY = '#6F6F6F'

# # ------------------------ The main function ---------------------------------
# # ----------------------------------------------------------------------------
# def plot_images(garmin_data, cst_time_intervals, garmin_time_intervals, date, steps_score):
#     plt.close("all")
#     plot_steps(path_save, steps_score)
#     plot_sleep(garmin_data["sleep"], path_save)
#     plot_stress(garmin_data["stress"], path_save)
#     plot_spo2(garmin_data["spo2"], path_save)
    
#     plot_duration(cst_time_intervals, garmin_time_intervals, date)

# ------------------------ The main function ---------------------------------#
# -----------------------------cst only----------------------------------#
def plot_images(cst_time_intervals, date, steps_score):
    plt.close("all")
    plot_steps(path_save, steps_score)
    
    plot_duration(cst_time_intervals, date)
    

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
    
# --- Spo2 ---
def plot_spo2(spo2_dict, path_save):
    
    GREEN = "#17A444"
    YELLOW = "#F8CB4B"
    ORANGE = "#F77517"
    RED = "#CE4A14"

    image_path = path_save +"/spo2.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    size_of_groups=[4,23,23,23,23,4]

    plt.figure(figsize=(5,5))
    if isinstance(spo2_dict["averege"], str) == False:
        averege_score = spo2_dict["averege"]
        lowest_score = spo2_dict["lowest"]

        plt.pie(size_of_groups, 
                colors=["white", GREEN, YELLOW, ORANGE, RED, "white"], 
                startangle=270)
        
        my_circle=plt.Circle( (0,0), 0.9, color="white")
        p=plt.gcf()
        p.gca().add_artist(my_circle)

        plt.text(0, 0.30, 'Averege Spo2', fontsize=18, color=GREY,  horizontalalignment = "center")
        plt.text(0, 0, (str(averege_score) + '%'), fontsize=30, color = BLUE, horizontalalignment = "center")
        plt.text(0, -0.40, 'Lowest', fontsize=18, color=GREY,  horizontalalignment = "center")
        plt.text(0, -0.65, (str(lowest_score) + '%'), fontsize=20, color=BLUE,  horizontalalignment = "center")

        # setting the axes projection as polar
        plt.axes(projection = 'polar')
        radius = 0.75

        score_reshape = ((100 - 0)/(100-60)) * (averege_score - 60)
        deg = (270-(4/100*360)) - score_reshape/100*(92/100*360)
        rad = np.deg2rad(deg)

        if averege_score < 70:
            color_spo2_score=RED
        elif 70 <= averege_score < 80:
            color_spo2_score=ORANGE
        elif 80 <= averege_score < 90:
            color_spo2_score=YELLOW
        elif 90 <= averege_score <= 100:
            color_spo2_score=GREEN
            
        plt.polar(rad, radius, '.', markersize=60, color=color_spo2_score)
        plt.polar(rad, 1, '.', color = "white")

    plt.axis('off')
    plt.savefig(image_path, transparent=True)

# --- Sleep ---
def plot_sleep(sleep_dict, path_save):
    image_path = path_save +"/sleep.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    if isinstance(sleep_dict["recorded_time"], str) == False:
        plt.figure(figsize=(5,5))
        if sleep_dict["recorded_time"] > 0:
            score   = sleep_dict["score"]
            quality = sleep_dict["quality"]
            deep    = sleep_dict["deep"]/sleep_dict["recorded_time"]
            light   = sleep_dict["light"]/sleep_dict["recorded_time"]
            rem     = sleep_dict["rem"]/sleep_dict["recorded_time"]
            awake   = sleep_dict["awake"]/sleep_dict["recorded_time"]

            
            size_of_groups=[deep, light, rem, awake]
            
            plt.pie(size_of_groups, 
                    colors = ['#044A9A', '#1878CF', '#9D0FB1', '#EB79D2'], 
                    startangle = 90,
                    counterclock = False,
                    wedgeprops = {"linewidth": 1, "edgecolor": "white"},
                )
            
            my_circle=plt.Circle( (0,0), 0.9, color="white")
            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.text(0, 0, (str(score) + '/100'), fontsize=30, color= BLUE,  horizontalalignment = "center")
            plt.text(0, -.35, 'Quality:', fontsize=20, color=GREY,  horizontalalignment = "center")
            plt.text(0, -.60, quality, fontsize=20, color=GREY,  horizontalalignment = "center")
        plt.savefig(image_path, transparent=True)
    
# --- Stress ---
def plot_stress(stress_dict, path_save):
    image_path = path_save +"/stress.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    recorded_time = stress_dict["recorded_time"]
    plt.figure(figsize=(5,5))
    plt.axis('off')

    if stress_dict["score"] == "":
        plt.text(0, -1.5, 'No data', fontsize=40, color=BLUE,
                 horizontalalignment='center',verticalalignment='center')
        
    elif stress_dict["score"] == "" and isinstance(recorded_time, str) == False:
        stress_score = stress_dict["score"]
        rest = stress_dict["rest"]/recorded_time
        low = stress_dict["low"]/recorded_time
        medium = stress_dict["medium"]/recorded_time
        high = stress_dict["high"]/recorded_time

        size_of_groups=[rest, low, medium, high]
        plt.pie(size_of_groups, 
                colors=['#4594F3', '#FFAF54', '#F97516', '#DD5809'], 
                startangle=90,
                counterclock=False,
                wedgeprops = {"linewidth": 1, "edgecolor": "white"})
        my_circle=plt.Circle( (0,0), 0.9, color="white")
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.text(0, 0, (str(stress_score)), fontsize=30, color=BLUE, horizontalalignment = "center")
        plt.text(0, -0.25, 'Overall', fontsize=20, color=GREY, horizontalalignment = "center")
    plt.savefig(image_path, transparent=True)
    
# --- Duration ---
def plot_duration(cst_times, date):
    image_path = path_save +"/duration.png"

    # Delete image if exists
    if os.path.exists(image_path):
        os.remove(image_path)

    # Set the figure parameters
    GARMIN_COLOR = "#6CCFF6"
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
        for i in range(len(cst_times)):
            interval = np.array(cst_times[i][:])
            if (len(interval) > 10):
                y_value = np.zeros(len(interval)) + 0.25
                plt.plot(interval, y_value, color = CST_COLOR, linewidth=20, solid_capstyle='round')
    
        # Plot Garmin
        # for i in range(len(garmin_times)):
        #     interval = np.array(garmin_times[i][:])
        #     if (len(interval) > 10):
        #         y_value = np.ones(len(interval)) + 0.25
        #         plt.plot(interval, y_value, color = GARMIN_COLOR, linewidth=20, solid_capstyle='round')
    
        plt.tight_layout()
        
        # Save as image
        plt.savefig(image_path, transparent=True)
        