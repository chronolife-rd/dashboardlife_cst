# # -*- coding: utf-8 -*-
# """
# Created on Fri Mar 17 13:23:57 2023

# @author: aterman
# """
# import copy

# TEXT_FONT = "Helvetica"
# BLUE_COLOR = "#3E738D"

# def garmin_data_for_pdf(garmin_data:dict) -> dict:
#     garmin_data_pdf = initialize_dictionary_with_template()
    
#     # ========================== Duration dict ================================
#     # Add collected
#     dict_aux = garmin_data_pdf["duration"]["collected"]
#     dict_aux["text"] = "Collected data: " + str(garmin_data["duration"]["collected"])
#     dict_aux["x"] = 6.13
#     dict_aux["y"] = 2.24
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR
    
#     # Add night
#     dict_aux = garmin_data_pdf["duration"]["night"]
#     dict_aux["text"] = "Night: " + str(garmin_data["duration"]["night"])
#     dict_aux["x"] = 6.43
#     dict_aux["y"] = 2.55
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # Add day
#     dict_aux = garmin_data_pdf["duration"]["day"]
#     dict_aux["text"] = "Day: " + str(garmin_data["duration"]["day"])
#     dict_aux["x"] = 6.43
#     dict_aux["y"] = 2.87
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR
    
#     # Add rest
#     dict_aux = garmin_data_pdf["duration"]["rest"]
#     dict_aux["text"] = "Rest: " + str(garmin_data["duration"]["rest"])
#     dict_aux["x"] = 6.43
#     dict_aux["y"] = 3.19
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR
    
#     # Add activity
#     dict_aux = garmin_data_pdf["duration"]["activity"]
#     dict_aux["text"] = "Active: " + str(garmin_data["duration"]["active"])
#     dict_aux["x"] = 6.43
#     dict_aux["y"] = 3.52
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR
    
#     # ========================== stress dict =================================
#     recorded_time = garmin_data["stress"]["recorded_time"]
#     if isinstance(recorded_time, str) == False and recorded_time > 0:
#         # Add rest info
#         td_str = td_to_hhmm_str(garmin_data["stress"]["rest"])
#         percentage = garmin_data["stress"]["rest"]/recorded_time*100
#         dict_aux = garmin_data_pdf["stress"]["rest"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 0.6
#         dict_aux["y"] = 9.13
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add low info
#         td_str = td_to_hhmm_str(garmin_data["stress"]["low"])
#         percentage = garmin_data["stress"]["low"]/recorded_time*100
#         dict_aux = garmin_data_pdf["stress"]["low"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 0.6
#         dict_aux["y"] = 9.45
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add medium info
#         td_str = td_to_hhmm_str(garmin_data["stress"]["medium"])
#         percentage = garmin_data["stress"]["medium"]/recorded_time*100
#         dict_aux = garmin_data_pdf["stress"]["medium"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 0.6
#         dict_aux["y"] = 9.77
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add high info
#         td_str = td_to_hhmm_str(garmin_data["stress"]["high"])
#         percentage = garmin_data["stress"]["high"]/recorded_time*100
#         dict_aux = garmin_data_pdf["stress"]["high"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 0.6
#         dict_aux["y"] = 10.09
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#     # ========================== sleep dict ==================================
#     recorded_time = garmin_data["sleep"]["recorded_time"]
#     if isinstance(recorded_time, str) == False and recorded_time > 0:
#         # Add deep info
#         td_str = td_to_hhmm_str(garmin_data["sleep"]["deep"])
#         percentage = garmin_data["sleep"]["deep"]/recorded_time*100
#         dict_aux = garmin_data_pdf["sleep"]["deep"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 5.85
#         dict_aux["y"] = 9.11
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add light info
#         td_str = td_to_hhmm_str(garmin_data["sleep"]["light"])
#         percentage = garmin_data["sleep"]["light"]/recorded_time*100
#         dict_aux = garmin_data_pdf["sleep"]["light"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 5.85
#         dict_aux["y"] = 9.43
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add rem info
#         td_str = td_to_hhmm_str(garmin_data["sleep"]["rem"])
#         percentage = garmin_data["sleep"]["rem"]/recorded_time*100
#         dict_aux = garmin_data_pdf["sleep"]["rem"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 5.85
#         dict_aux["y"] = 9.75
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#         # Add awake info
#         td_str = td_to_hhmm_str(garmin_data["sleep"]["awake"])
#         percentage = garmin_data["sleep"]["awake"]/recorded_time*100
#         dict_aux = garmin_data_pdf["sleep"]["awake"]
#         dict_aux["text"] = td_str + " (" + str(round(percentage)) + "%)"
#         dict_aux["x"] = 5.85
#         dict_aux["y"] = 10.07
#         dict_aux["font"] = TEXT_FONT
#         dict_aux["size"] = 8
#         dict_aux["color"] = BLUE_COLOR

#     # ========================== Calories dict ==================================
#     # Add active calories info
#     dict_aux = garmin_data_pdf["calories"]["active"]
#     dict_aux["text"] = str(garmin_data["calories"]["active"]) + " kcals "
#     dict_aux["x"] = 1.94
#     dict_aux["y"] = 10.92
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # Add resting calories info
#     dict_aux = garmin_data_pdf["calories"]["resting"]
#     dict_aux["text"] = str(garmin_data["calories"]["resting"]) + " kcals"
#     dict_aux["x"] = 1.94
#     dict_aux["y"] = 11.33
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # Add total calories info
#     dict_aux = garmin_data_pdf["calories"]["total"]
#     dict_aux["text"] = str(garmin_data["calories"]["total"]) + " kcals"
#     dict_aux["x"] = 0.43
#     dict_aux["y"] = 11.05
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 12
#     dict_aux["color"] = BLUE_COLOR

#     # ====================== Intensity min dict ==============================
#     # Add moderate info
#     dict_aux = garmin_data_pdf["intensity_min"]["moderate"]
#     value = garmin_data["intensity_min"]["moderate"]
#     if isinstance(value, str) == False:
#         dict_aux["text"] = td_to_hhmm_str(value)
#     dict_aux["x"] = 3.11
#     dict_aux["y"] = 11.14
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # Add vigurous info
#     dict_aux = garmin_data_pdf["intensity_min"]["vigurous"]
#     value = garmin_data["intensity_min"]["vigurous"]
#     if isinstance(value, str) == False:
#         dict_aux["text"] = td_to_hhmm_str(value)
#     dict_aux["x"] = 3.83
#     dict_aux["y"] = 11.14
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # Add total info
#     dict_aux = garmin_data_pdf["intensity_min"]["total"]
#     value = garmin_data["intensity_min"]["total"]
#     if isinstance(value, str) == False:
#         dict_aux["text"] = td_to_hhmm_str(value)
#     dict_aux["x"] = 4.55
#     dict_aux["y"] = 11.14
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 10
#     dict_aux["color"] = BLUE_COLOR

#     # ====================== Body battery min dict ===========================
#     # Add highest info
#     dict_aux = garmin_data_pdf["body_battery"]["highest"]
#     dict_aux["text"] = str(garmin_data["body_battery"]["highest"]) + " /100"
#     dict_aux["x"] = 5.71
#     dict_aux["y"] = 11.14
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 12
#     dict_aux["color"] = BLUE_COLOR

#     # Add lowest info
#     dict_aux = garmin_data_pdf["body_battery"]["lowest"]
#     dict_aux["text"] = str(garmin_data["body_battery"]["lowest"]) + " /100"
#     dict_aux["x"] = 6.9
#     dict_aux["y"] = 11.14
#     dict_aux["font"] = TEXT_FONT
#     dict_aux["size"] = 12
#     dict_aux["color"] = BLUE_COLOR


#     return copy.deepcopy(garmin_data_pdf)

# def initialize_dictionary_with_template() -> dict :
#     pdf_info = {
#         "text" : "", 
#         "x" : "",
#         "y" : "",
#         "font" : "",
#         "size" : "",
#         "color" : "",
#         }
 
#     duration_dict = {
#         "collected" : copy.deepcopy(pdf_info),
#         "day"       : copy.deepcopy(pdf_info),
#         "night"     : copy.deepcopy(pdf_info),
#         "rest"      : copy.deepcopy(pdf_info),
#         "activity"  : copy.deepcopy(pdf_info),
#     }  
#     stress_dict = {
#         "rest"   : copy.deepcopy(pdf_info), 
#         "low"    : copy.deepcopy(pdf_info), 
#         "medium" : copy.deepcopy(pdf_info), 
#         "high"   : copy.deepcopy(pdf_info), 
#     }
#     sleep_dict = {
#         "deep"  : copy.deepcopy(pdf_info), 
#         "light" : copy.deepcopy(pdf_info), 
#         "rem"   : copy.deepcopy(pdf_info), 
#         "awake" : copy.deepcopy(pdf_info), 

#     }
#     calories_dict = {
#         "active"  : copy.deepcopy(pdf_info), 
#         "resting" : copy.deepcopy(pdf_info), 
#         "total"   : copy.deepcopy(pdf_info), 
#     }
#     intensity_min_dict = {
#         "moderate" : copy.deepcopy(pdf_info), 
#         "vigurous" : copy.deepcopy(pdf_info), 
#         "total"    : copy.deepcopy(pdf_info),  
#     }

#     body_battery_dict = {
#         "highest" : copy.deepcopy(pdf_info), 
#         "lowest"  : copy.deepcopy(pdf_info), 
#     }
    
#     dict_template = {
#                     'duration'      : copy.deepcopy(duration_dict),
#                     'stress'        : copy.deepcopy(stress_dict),
#                     'sleep'         : copy.deepcopy(sleep_dict),
#                     'calories'      : copy.deepcopy(calories_dict),
#                     'intensity_min' : copy.deepcopy(intensity_min_dict),
#                     'body_battery'  : copy.deepcopy(body_battery_dict),
#                     }
    
#     return copy.deepcopy(dict_template)

# def td_to_hhmm_str(td_seconds):
   
#     sign = ''
#     tdhours, rem = divmod(td_seconds, 3600)
#     tdminutes, rem = divmod(rem, 60)
#     tdstr = '{}{:}h {:02d}m'.format(sign, int(tdhours), int(tdminutes))
#     return tdstr
