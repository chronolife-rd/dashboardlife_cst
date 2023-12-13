import streamlit as st
import io
import os
import pandas as pd
import template.constant as constant

from data.data import get_health_indicators
from automatic_reports.generate_pdf import generate_pdf
# from automatic_reports.compute_garmin_for_pdf import garmin_data_for_pdf
from automatic_reports.compute_cst_for_pdf import cst_data_for_pdf
from automatic_reports.compute_common_for_pdf import get_common_indicators_pdf
from automatic_reports.plot_images import plot_images
from data.data import *
from pylife.useful import unwrap, unwrap_signals_dashboard
from data.data_process import filter_ecg_scipy

def data_report_pdf():
    
    # Delete pdf if exists
    if os.path.exists(constant.PDF_FILE):
        os.remove(constant.PDF_FILE)

    date                        = st.session_state.date
    # garmin_data                 = st.session_state.garmin_indicators    
    chronolife_data             = st.session_state.chronolife_indicators  
    common_indicators_pdf       = st.session_state.common_indicators_pdf 
    chronolife_indicators_pdf   = st.session_state.chronolife_indicators_pdf 
    # garmin_indicators_pdf       = st.session_state.garmin_indicators_pdf 

    # Get intervals and alerts
    # if "duration" in garmin_data :
    # garmin_time_intervals = garmin_data['duration']['intervals']
    cst_time_intervals = chronolife_data['duration']['intervals']
    alerts_dict = chronolife_data['anomalies']
    steps           = get_steps()
    steps_score     = steps["score"]
    
    # Plot and save graphs
    # plot_images(garmin_data, cst_time_intervals, garmin_time_intervals, date, steps_score)
    plot_images(cst_time_intervals, date, steps_score)
    
    # Construct pdf
    generate_pdf(chronolife_indicators_pdf,common_indicators_pdf ,alerts_dict)
    
    with open(constant.PDF_FILE, "rb") as pdf_file:
        return pdf_file.read()
      

def health_indicators_to_excel():
    # Cache the conversion to prevent computation on every rerun
    output = io.BytesIO()
    writer = pd.ExcelWriter(output,engine='xlsxwriter')
    workbook = writer.book
  
   # init
   # Chronolife or Garmin data
    HR_dict = {}
    Tachycardia_dict = {}
    Bradycardia_dict = {}
    BR_dict = {}
    Tachypnea_dict = {}
    Bradypnea_dict = {}
    
    
    #Only Chronolife data
    HRV_dict = {}
    Inex_dict ={}
    BRV_dict = {}
    Qt_dict = {}
    Temp_dict = {}

    #Only Garmin data
    # Stress_dict ={}
    # Pulseox_dict = {}
    # Bodybatt_dict = {}
    # Sleep_dict = {}

    HR_value = get_bpm_values()
    HR_dict = {
    'Time': HR_value['times'],
    'Value': HR_value['values'] }
     #-------Sheet for HR-------#
    df = pd.DataFrame.from_dict(HR_dict)
    df.to_excel(writer, index=False, sheet_name='HR')

    BR_value = get_brpm()
    BR_raw_value = get_brpm_values()
    BR_dict = {
        'Times' : BR_raw_value["times"],
        'Values' : BR_raw_value["values"],
    }
    #-------Sheet for BR-------#
    df = pd.DataFrame.from_dict(BR_dict)
    df.to_excel(writer, index=False, sheet_name='BR')
    worksheet = writer.sheets['BR']


    tachycardia_value = get_tachycardia()
    Tachycardia_dict = {
    'Mean' : tachycardia_value['mean'],
    'Duration' : tachycardia_value['duration'],
    'Percentage' : tachycardia_value['percentage']
    }
    #-------Sheet for Tachycardie-------#
    df = pd.DataFrame.from_dict([Tachycardia_dict])
    df.to_excel(writer, index=False, sheet_name='Tachycardia')
    worksheet = writer.sheets['Tachycardia']


    tachypnea_value = get_tachypnea()
    Tachypnea_dict = {
    'HR' : tachypnea_value['mean'],
    'Duration' : tachypnea_value['duration'],
    'Percentage' : tachypnea_value['percentage']
    }
    #-------Sheet for Tachypnée-------#
    df = pd.DataFrame.from_dict([Tachypnea_dict])
    df.to_excel(writer, index=False, sheet_name='Tachypnea')
    worksheet = writer.sheets['Tachypnea']
    
    bradycardia_value = get_bradycardia()
    Bradycardia_dict = {
    'HR' : bradycardia_value['mean'],
    'Duration' : bradycardia_value['duration'],
    'Percentage' : bradycardia_value['percentage']
    }
    #-------Sheet for Bradycardie-------#
    df = pd.DataFrame.from_dict([Bradycardia_dict])
    df.to_excel(writer, index=False, sheet_name='Bradycardia')
    worksheet = writer.sheets['Bradycardia']

    bradypnea_value = get_bradypnea()
    Bradypnea_dict = {
    'HR' : bradypnea_value['mean'],
    'Duration' : bradypnea_value['duration'],
    'Percentage' : bradypnea_value['percentage']
    }
        #-------Sheet for Bradypnée-------#
    df = pd.DataFrame.from_dict([Bradypnea_dict])
    df.to_excel(writer, index=False, sheet_name='Bradypnea')
    worksheet = writer.sheets['Bradypnea']
    

    #Only Chronolife data
    if (st.session_state.chronolife_data_available == True):
        
        HRV_value = get_hrv_values()
        HRV_dict = {
        'Time': HRV_value['times'],
        'Value': HRV_value['values'],
        }
            #-------Sheet for HRV-------#
        df = pd.DataFrame.from_dict(HRV_dict)
        df.to_excel(writer, index=False, sheet_name='HRV')
        worksheet = writer.sheets['HRV']
        
        Inex_value = get_inex_values()
        Inex_dict = {
        'Time': Inex_value['times'],
        'Value' : Inex_value['values'],
        }
        #-------Sheet for Ratio Inspi-Expi -------#
        df = pd.DataFrame(Inex_dict)
        df.to_excel(writer, index=False, sheet_name='Inex Ratio')
        worksheet = writer.sheets['Inex Ratio']
        
        BRV_value = get_brv_values()
        BRV_dict = {
        'Time': BRV_value['times'],
        'Value': BRV_value['values'],
        }
        #-------Sheet for BRV -------#
        df = pd.DataFrame.from_dict(BRV_dict)
        df.to_excel(writer, index=False, sheet_name='BRV')
        worksheet = writer.sheets['BRV']

        Qt_value = get_qt()
        Qt_dict = {
        'Value' : Qt_value['values'],
        'Night' : Qt_value['night'],
        'Morning' : Qt_value['morning'],
        'Evening' : Qt_value['evening']
        }
        #-------Sheet for Intervalle de QT-------#
        df = pd.DataFrame.from_dict([Qt_dict])
        df.to_excel(writer, index=False, sheet_name='QT interval')
        worksheet = writer.sheets['QT interval']

        #-------Sheet for Activity level-------#

        activity_level_value = get_activity_level()
        activity_level_dict = {
        'Valeur' : activity_level_value['values'],
        'Time' :activity_level_value['times']
        }
        #-------Sheet for Intervalle de QT-------#
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in activity_level_dict.items()]))
        df.to_excel(writer, index=False, sheet_name='Activity level')
        worksheet = writer.sheets['Activity level']

        Temp_value = get_temperature()
        Temp_dict = {
        'Time' : Temp_value['values']['times'],
        'Value °C' : Temp_value['values']['temperature_values'].div(100),
        'Mean' : Temp_value['mean'],
        'Minimum' : Temp_value['min'],
        'Maximum' : Temp_value['max']
        }
         #-------Sheet for Temperature-------#
        df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in Temp_dict.items()]))
        df.to_excel(writer, index=False, sheet_name='Skin temperature')
        worksheet = writer.sheets['Skin temperature']

    
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)
    writer.close()
    data = output.getvalue()
    return data


# @st.cache_data
def raw_data_to_excel():
        
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    
    df_acc = parse_acceleration_for_excel()
    df_acc.to_excel(writer, index=False, sheet_name='Acceleration')
    
    df_breath = parse_breath_for_excel()
    df_breath.to_excel(writer, index=False, sheet_name='Respiratory')
    
    df_ecg = parse_ecg_for_excel()
    df_ecg.to_excel(writer, index=False, sheet_name='ECG')
    
    worksheet = writer.sheets['Acceleration']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    data = output.getvalue()
    return data

def parse_acceleration_for_excel():
    accx    = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_X"]]
    accy    = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_Y"]]
    accz    = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_Z"]]
    
    times   = unwrap_signals_dashboard(accx['times'])
    x       = unwrap_signals_dashboard(accx['sig'])
    y       = unwrap_signals_dashboard(accy['sig'])
    z       = unwrap_signals_dashboard(accz['sig'])
    
    data_dict = {
      "Date": times,
      "X values": x,
      "Y values": y,
      "Z values": z,
    }
    df = pd.DataFrame(data_dict)
    
    return df

def parse_breath_for_excel():
    breath_1    = st.session_state.smart_textile_raw_data[constant.TYPE()["BREATH_THORACIC"]]
    breath_2    = st.session_state.smart_textile_raw_data[constant.TYPE()["BREATH_ABDOMINAL"]]
    times       = unwrap_signals_dashboard(breath_1['times'])
    breath_tho  = unwrap_signals_dashboard(breath_1['sig'])
    breath_abd  = unwrap_signals_dashboard(breath_2['sig'])
    
    data_dict = {
      "Date": times,
      "Abdominal values": breath_abd,
      "Thoracic values": breath_tho
    }
    df = pd.DataFrame(data_dict)
    
    return df

def parse_ecg_for_excel():
    ecg     = st.session_state.smart_textile_raw_data[constant.TYPE()['ECG']]
    times   = unwrap_signals_dashboard(ecg['times'])
    values  = unwrap_signals_dashboard(ecg['sig'])
    
    data_dict = {
      "Date": times,
      "ECG values": values
    }
    df = pd.DataFrame(data_dict)
    
    return df

def acc_to_csv():
    acc_x     = st.session_state.smart_textile_raw_data_all_day[constant.TYPE()["ACCELERATION_X"]]
    acc_z     = st.session_state.smart_textile_raw_data_all_day[constant.TYPE()["ACCELERATION_X"]]
    acc_y    = st.session_state.smart_textile_raw_data_all_day[constant.TYPE()["ACCELERATION_X"]]

    times   = unwrap_signals_dashboard(acc_x['times'])
    values_x  = unwrap_signals_dashboard(acc_x['sig'])
    values_z  = unwrap_signals_dashboard(acc_z['sig'])
    values_y  = unwrap_signals_dashboard(acc_y['sig'])


    df_acc = pd.DataFrame({'times': unwrap(times),
                'signal_x': unwrap(values_x),
                'signal_z': unwrap(values_z),
                'signal_y': unwrap(values_y)},

                )
    return df_acc.to_csv().encode('utf-8')

def ecg_to_csv(filered = False):
    if filered == False :
      ecg     = st.session_state.smart_textile_raw_data_all_day[constant.TYPE()['ECG_FILTERED']]
      times   = unwrap_signals_dashboard(ecg['times'])
      values  = unwrap_signals_dashboard(ecg['sig'])
      
      ecg_filt = filter_ecg_scipy(values,fs=200)
      df_ecg_filt = pd.DataFrame({'times': unwrap(times),
                    'signal': unwrap(ecg_filt)},
                    )
      
    
    return df_ecg_filt.to_csv().encode('utf-8')
        


def breath_to_csv(BREATH_TYPE, filered = False):
    if filered == False :

      raw_breath    = st.session_state.smart_textile_raw_data_all_day[constant.TYPE()[f"{BREATH_TYPE}"]]
      times       = unwrap_signals_dashboard(raw_breath['times'])
      filter_breath  = unwrap_signals_dashboard(raw_breath['sig'])

      
      data_dict = {
        "time": times,
        f"{BREATH_TYPE} values": filter_breath,
      }
      df_breath_filt = pd.DataFrame(data_dict)
    
    return df_breath_filt.to_csv().encode('utf-8')
