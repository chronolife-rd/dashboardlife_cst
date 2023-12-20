# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:23:57 2023

@author: aterman
"""
import copy
import datetime
from datetime import timedelta

TEXT_FONT = "Helvetica"
BLUE_COLOR = "#3E738D"

def cst_data_for_pdf(uder_id:str, date:str, cst_data:dict) -> dict:
    cst_data_pdf = initialize_dictionary_with_template()
    
    # ========================== Header dict ==================================
    # Add user id
    dict_aux = cst_data_pdf["header"]["user_id"]
    dict_aux["text"] = "ID: " + uder_id
    dict_aux["x"] = 0.29
    dict_aux["y"] = 0.36
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 12
    dict_aux["color"] = "white"
    
    # Add date
    dict_aux = cst_data_pdf["header"]["date"]
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    dict_aux["text"] = "Daily report for " + datetime.date.strftime(date, "%B %d, %Y")
    dict_aux["x"] = 0.29
    dict_aux["y"] = 1.05
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 12
    dict_aux["color"] = "white"
    
    # ========================== Duration dict ===============================
    # Add collected
    dict_aux = cst_data_pdf["duration"]["collected"]
    dict_aux["text"] = "Collected data: " + str(cst_data["duration"]["collected"])
    dict_aux["x"] = 2.18
    dict_aux["y"] = 2.24
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 10
    dict_aux["color"] = BLUE_COLOR
    
    # Add night
    dict_aux = cst_data_pdf["duration"]["night"]
    dict_aux["text"] = "Night: " + str(cst_data["duration"]["night"])
    dict_aux["x"] = 2.45
    dict_aux["y"] = 2.55
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 10
    dict_aux["color"] = BLUE_COLOR

    # Add day
    dict_aux = cst_data_pdf["duration"]["day"]
    dict_aux["text"] = "Day: " + str(cst_data["duration"]["day"])
    dict_aux["x"] = 2.45
    dict_aux["y"] = 2.87
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 10
    dict_aux["color"] = BLUE_COLOR
    
    # Add rest
    dict_aux = cst_data_pdf["duration"]["rest"]
    dict_aux["text"] = "Rest: " + str(cst_data["duration"]["rest"])
    dict_aux["x"] = 2.45
    dict_aux["y"] = 3.19
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 10
    dict_aux["color"] = BLUE_COLOR
    
    # Add activity
    dict_aux = cst_data_pdf["duration"]["active"]
    dict_aux["text"] = "Active: " + str(cst_data["duration"]["active"])
    dict_aux["x"] = 2.45
    dict_aux["y"] = 3.52
    dict_aux["font"] = TEXT_FONT
    dict_aux["size"] = 10
    dict_aux["color"] = BLUE_COLOR

    return copy.deepcopy(cst_data_pdf)

def initialize_dictionary_with_template() -> dict :
    pdf_info = {
        "text" : "", 
        "x" : "",
        "y" : "",
        "font" : "",
        "size" : "",
        "color" : "",
        }
    
    header_dict = {
        "user_id" : copy.deepcopy(pdf_info),
        "date"    : copy.deepcopy(pdf_info),
    }  
    duration_dict = {
        "collected" : copy.deepcopy(pdf_info),
        "day"       : copy.deepcopy(pdf_info),
        "night"     : copy.deepcopy(pdf_info),
        "rest"      : copy.deepcopy(pdf_info),
        "active"  : copy.deepcopy(pdf_info),
    }  

    dict_template = {
                    'header'   : copy.deepcopy(header_dict),
                    'duration' : copy.deepcopy(duration_dict),
                    }
    
    return copy.deepcopy(dict_template)