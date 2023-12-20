import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import data.data as data
import data.cst_raw_data as cst_raw_data
import numpy as np
import datetime 
import math
from template.util import img_to_bytes

from data.data import *
from template.raw_signal_filter import *
from template.constant import COLORS, UNIT, RANGE, SHORTCUT, TYPE, ACTIVITY_THREASHOLD
from plotly.subplots import make_subplots
from pylife.useful import unwrap, unwrap_signals_dashboard
from automatic_reports.useful_functions import find_time_intervals


BGCOLOR = 'rgba(255,255,255,1)'


def manage_no_data_color(df):
    """
    
    Retrun
    ---------
    naw_df = {"times","values}
    """
    new_df = df
    
    for val in range (len(df)-1):
        if (df["times"][val+1]-df["times"][val] > datetime.timedelta(days=0,hours=0,minutes=1)):
            new_df.loc[val+0.5]=[ df["times"][val] + datetime.timedelta(days=0,hours=0,minutes=1), None]
    
    new_df = new_df.sort_index().reset_index(drop=True)

    return new_df

def temperature_mean():

    translate = st.session_state.translate
    line_width = 2

    fig = go.Figure()

    values_df = get_temperature()["values"]
    values_df = values_df[["times","temperature_values"]]
    values_df = values_df.rename(columns= {"times":"times","temperature_values":"values"})

    if st.session_state.chronolife_data_available == True:
        df = manage_no_data_color(values_df)

        fig.add_trace(
            go.Scatter(
                x=df["times"], 
                y=df["values"]/100,
                mode='lines+markers',
                line=dict(color=COLORS()['temp'], width=line_width),
                name='tmp')
        )
            
    fig.update_layout(xaxis_title = translate["times"],
                      yaxis_title=UNIT()['temp'],
                      font=dict(size=14,),
                      height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=SHORTCUT()['temp'],
                       yaxis = dict(range=RANGE()['temp']),
                      )
    
    return fig


# ---- Charts with indicators ----
def heart_bpm():
    translate = st.session_state.translate
    line_width = 2
    values_df_cst = get_bpm_values()
    dict_activity = get_averaged_activity()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add CST
    if st.session_state.chronolife_data_available == True:
        df_cst = manage_no_data_color(pd.DataFrame.from_dict(values_df_cst)) # gère le fait qu'on lie pas les données quand on plot
        fig.append_trace(
            go.Scatter(
                x=df_cst['times'], 
                y=df_cst['values'],
                connectgaps=False,
                mode='lines+markers',
                line=dict(color=COLORS()['cardio_chart'], width=line_width),
                name='HR',
                showlegend=True), 
            row=1, col=1
        )
        # Add activity in the second plot
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"],
                showlegend=True), 
            row=2, col=1
        )
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )
    fig.update_layout(
        xaxis_title=translate["times"],
        yaxis_title=UNIT()['bpm'],
        font=dict(size=14),
        height=300, 
        template="plotly_white",
        paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
        title=SHORTCUT()['bpm'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,),
    )
    return fig

def heart_hrv():
    translate = st.session_state.translate
    line_width = 2
    dict_activity = get_averaged_activity()
    values_df = get_hrv_values()
    df_hrv = values_df[['times','values']]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # Add CST
    if st.session_state.chronolife_data_available == True:
        df = manage_no_data_color(df_hrv) 
        fig.append_trace(
            go.Scatter(
                y=df['values'],
                x=df['times'],
                mode='lines+markers', connectgaps=False,
                line=dict(color=COLORS()["cardio_chart"]),
                name = 'HRV',
                showlegend=True),
            row=1, col=1
        )
        # Add activity in the second plot
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"],
                showlegend=True), 
            row=2, col=1
        )
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )
    fig.update_layout(
        xaxis_title=translate["times"],
        yaxis_title=UNIT()['hrv'],
        font=dict(size=14,),
        height=300, 
        template="plotly_white",
        paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
        title=SHORTCUT()['hrv'],
        legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01)
    )
    return fig

def heart_qt():
    translate = st.session_state.translate
    line_width =2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add CST
    if st.session_state.chronolife_data_available == True:
        dict_activity = get_averaged_activity()
        values_df = get_qt()
        values_df = pd.DataFrame.from_dict (values_df)
        values_df = values_df[["times","values"]]
        df = manage_no_data_color(values_df)
        # Ad qt 
        fig.append_trace(
            go.Bar(x=df["times"], y=df["values"],orientation='v', 
                   name = 'QTc Chroolife', marker_color=COLORS()["cardio_chart"], showlegend=True),
            row=1, col=1
        )
        # Add activity in the second plot
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"], 
                showlegend=True), 
            row=2, col=1
        )
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )  
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=UNIT()['qt'],
                      height=400,
                      font=dict(size=14,)
                      )
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=SHORTCUT()['qt'],
                      legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01)
                      )
    return fig

def breath_brpm():
    translate = st.session_state.translate    
    line_width = 2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    values_df_cst = get_brpm_values_for_excel()
    dict_activity = get_averaged_activity()
    
    # Add Chronolife
    if st.session_state.chronolife_data_available == True:
        df_cst = manage_no_data_color(pd.DataFrame.from_dict(values_df_cst))  
        fig.append_trace(
            go.Scatter(
                x=df_cst["times"], 
                y=df_cst["values"],
                mode='lines+markers',
                connectgaps=False,
                line=dict(color=COLORS()['breath_chart'], width=line_width),
                name='BRPM',
                showlegend=True), 
            row=1, col=1
        )
        # Add activity in the second plot
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"], 
                showlegend=True), 
            row=2, col=1
        )       
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )
    fig.update_layout(
        xaxis_title=translate["times"],
        yaxis_title=UNIT()['brpm'],
        font=dict(size=14,),
        height=300, 
        template="plotly_white",
        paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
        title=SHORTCUT()['brpm'],
        legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01,),
    )    
    return fig

def breath_brv():
    translate = st.session_state.translate
    line_width = 2
    values_df = get_brv_values()
    values_df = values_df[["times","values"]]
    dict_activity = get_averaged_activity()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # Add BRV
    if st.session_state.chronolife_data_available == True:
        df = manage_no_data_color(values_df)
        fig.append_trace(
            go.Scatter(
                x=df["times"], 
                y=df["values"],
                mode='lines+markers',
                line=dict(color=COLORS()['breath_chart'], width=line_width),
                name='BRV',
                showlegend=True), 
            row=1, col=1
        )
        # Add activity in the second plot
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"],
                showlegend=True), 
            row=2, col=1
        ) 
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )
    fig.update_layout(
        yaxis_title=UNIT()['brv'],
        font=dict(size=14),
        height=300, 
        template="plotly_white",
        paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
        title=SHORTCUT()['brv'],
        legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01,),
    )
    return fig

def breath_inex():
    translate = st.session_state.translate
    line_width = 2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    if st.session_state.chronolife_data_available == True:
        values_df = get_inex_values()
        values_df = values_df[["times","values"]]
        df = manage_no_data_color(values_df)
        dict_activity = get_averaged_activity()

        # Add in/ex ratio
        fig.append_trace(
            go.Scatter(
                x=df["times"], 
                y=df["values"],
                mode='lines+markers',
                line=dict(color=COLORS()['breath_chart'], width=line_width),
                name='Ratio',
                showlegend=True),
            row=1, col=1
        )
        # Add activity
        fig.append_trace(
            go.Scatter(
                x=dict_activity["times"], 
                y=dict_activity["values"],
                fill='tozeroy',
                line=dict(color=COLORS()['sma'], width=line_width),
                name=translate["activity"],
                showlegend=True),
            row=2, col=1
        )
        # Add a horizontal dashed line in the second subplot
        x_range_activity_th = [dict_activity["times"].min(), dict_activity["times"].max()]
        y_range_activity_th = [ACTIVITY_THREASHOLD, ACTIVITY_THREASHOLD] 
        fig.add_trace(
            go.Scatter(
                x= x_range_activity_th, 
                y= y_range_activity_th, 
                mode='lines', line=dict(color=COLORS()['sma_th'], width=1, dash="dash"),
                name=translate["activity_threshold"]),
            row=2, col=1
        )
    fig.update_layout(
        xaxis_title=translate["times"],
        yaxis_title=UNIT()['inspi_expi'],
        font=dict(size=14,),
        height= 300, 
        template="plotly_white",
        paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
        title= SHORTCUT()['bior'],
        legend=dict(yanchor="top",y=0.99, xanchor="left", x=0.01)
        )
    return fig

def breath_brpm_combine():
    
    translate = st.session_state.translate    
    line_width = 2

    fig = go.Figure()

    dict_values = get_brpm_values()
    values_df = dict_values[["times","values"]]
    df = manage_no_data_color(values_df)
    
    fig.add_trace(go.Scatter(x=df["times"], 
                            y=df["values"],
                        mode='lines+markers',
                        line=dict(color=COLORS()['breath'], width=line_width),
                        name='tmp'))
        
    fig.update_layout(xaxis_title=translate["times"],
                      height=400,
                      font=dict(size=14,))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=SHORTCUT()['brpm'],
                      yaxis = dict(range=RANGE()['brpm']),
                      )
    return fig

def heart_bpm_conbine():
    
    translate = st.session_state.translate
           
    line_width = 2
    fig = go.Figure()

    values_df = get_bpm_values()
    values_df = values_df[["times","values"]]

    df = manage_no_data_color(values_df)
    fig.add_trace(go.Scatter(x=df['times'], 
                            y=df['values'],
                        mode='lines+markers',
                        line=dict(color=COLORS()['ecg'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=UNIT()['bpm'],
                      height=400,
                      font=dict(size=14))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=SHORTCUT()['bpm'],
                       yaxis = dict(range=RANGE()['bpm']),
                      )
    return fig

def duration():
    
    translate = st.session_state.translate

    date = st.session_state.date
    date = datetime.datetime.strptime(date, "%Y-%m-%d")   
    xmin = date
    xmax = date + datetime.timedelta(days = 1)
        
    y_chronolife    = np.repeat("Smart Textile", 2)

    x_chronolife    = get_duration_chronolife()['intervals']

    width = 20
    fig = go.Figure()
    
    # Add Chronolife
    for i in range(len(x_chronolife)):
        interval = x_chronolife[i][:]
        if (len(interval) > 1):
            interval_start = interval.iloc[0]
            interval_end = interval.iloc[-1]
            fig.add_trace(go.Scatter(y = y_chronolife, x=[interval_start, interval_end],
                        mode="lines", line=dict(color=COLORS()["chronolife"],width=width)))
    
    
    fig.update_layout(barmode='stack', height=250, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      title="<span style='font-size:20px;'>" + translate["data_collection_duration"] + "</span>",
                      xaxis = dict(range=[xmin, xmax]))
    
    return fig

def smart_textile_raw_data():
    # time_start_get_raw_data = datetime.datetime.now() # Mesure time dowload data file 
    # print("time_start_get_raw_data",time_start_get_raw_data)
    cst_raw_data.get_raw_data()

    # time_recive_raw_data = datetime.datetime.now()
    # print("time_recive_raw_data",time_recive_raw_data)

    
    error = False
    if len(st.session_state.smart_textile_raw_data) == 0:
        error = True 
    if error:
        return error
    
    # Figure Params
    template = 'plotly_white'
    width_line = 2
    height = 275
    
    fig_ecg     = ecg_signal(template=template, width_line=width_line, height=height)
    fig_breath  = breath_signal(template=template, width_line=width_line, height=height)
    fig_acc     = acceleration_raw_signal(template=template, width_line=width_line, height=height)
    time_finish_process_data = datetime.datetime.now()
    print("time_finish_process_data",time_finish_process_data)
    
    st.plotly_chart(fig_ecg, use_container_width=True)
    st.plotly_chart(fig_acc,use_container_width=True)
    st.plotly_chart(fig_breath, use_container_width=True)
    
    
def ecg_signal(template='plotly_white', width_line=2, height=500):
    fig = go.Figure()
    
    # ECG
    raw_data = st.session_state.smart_textile_raw_data[TYPE()['ECG']]

    ecg_filt = filter_ecg_scipy_unwrap(raw_data['sig'],fs=200) # Localy filtered


    ymin = max([min(unwrap_signals_dashboard(ecg_filt))*1.1, -1000])
    ymax = min([max(unwrap_signals_dashboard(ecg_filt))*1.1, 1000])

    
    for seg in range(len(raw_data["times"])):
        
        #Show legende only one time 
        if seg == 0 :
            fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=ecg_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["ecg"], width=width_line),
                        name='ECG',
                        showlegend=True
                        ))
        else :
             fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=ecg_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["ecg"], width=width_line),
                        name='ECG',
                        showlegend=False
                        ))
             
    output = get_averaged_activity()
    
    fig.update_layout(height=height,
                      title='Electrocardiogram',
                      yaxis = dict(range=[ymin, ymax]), 
                      template=template,
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=True,
                      legend=dict(yanchor="top",
                                  y=0.99, xanchor="left", 
                                  x=0.01,
                                  ),
                      )
    
    return fig
    
def breath_signal(template='plotly_white', width_line=2, height=500):
    fig = go.Figure()
    # Abdominal
    raw_data = st.session_state.smart_textile_raw_data[TYPE()["BREATH_ABDOMINAL"]]
    abdominal_filt = filter_breath_unwrap(raw_data['sig'],fs=20) # Localy filtered
    for seg in range(len(raw_data["times"])):

        if seg == 0 :

            fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=abdominal_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["breath_2"], width=width_line),
                        connectgaps=False,
                        name='Abdominal Breath', 
                        showlegend=True
                        ))
            
        else :
            fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=abdominal_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["breath_2"], width=width_line),
                        connectgaps=False,
                        name='Abdominal Breath', 
                        showlegend=False
                        ))
            
    # Thoracic 
    raw_data = st.session_state.smart_textile_raw_data[TYPE()["BREATH_THORACIC"]]
    thoracic_filt = filter_breath_unwrap(raw_data['sig'],fs=20) # Localy filtered

    for seg in range(len(raw_data["times"])):

        if seg == 0 :
            fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=thoracic_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["breath"], width=width_line),
                        connectgaps=False,
                        name='Thoracic Breath', 
                        showlegend=True
                        )) 
        else :
            fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=thoracic_filt[seg],
                        mode='lines',
                        line=dict(color=COLORS()["breath"], width=width_line),
                        connectgaps=False,
                        name='Thoracic Breath', 
                        showlegend=False
                        )) 

        
    fig.update_layout(height=height, 
                      template=template, 
                      title='Breath', 
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      legend=dict(yanchor="top",
                                  y=0.99, xanchor="left", 
                                  x=0.01,
                                  ),
                      )
    return fig



def acceleration_signal(template='plotly_white', width_line=2, height=500):
    # Acceleration
    ymin = -200
    ymax = 200
    fig = go.Figure()
    output = get_averaged_activity()
    sig = output["values"]
    time = output["times"]
    
    fig.add_trace(go.Scatter(x=time, 
                              y=sig,
                        mode='lines',
                        line=dict(color=COLORS()["acc"], width=width_line),
                        name='Acceleration'))
    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[ymin, ymax]),
                      title='Acceleration',
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      )
    
    return fig 


def acceleration_raw_signal(template='plotly_white', width_line=2, height=500):
    # Acceleration
    ymin = 1500
    ymax = 2500
    fig = go.Figure()
    acc_x     = st.session_state.smart_textile_raw_data[TYPE()["ACCELERATION_X"]]
    acc_z     = st.session_state.smart_textile_raw_data[TYPE()["ACCELERATION_Z"]]
    acc_y    = st.session_state.smart_textile_raw_data[TYPE()["ACCELERATION_Y"]]


    # st.write(acc_x['times'])
    for seg in range(len(acc_x['times'])):

        if seg == 0 :
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_x["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_x"], width=width_line),
                        name='acc_x',
                        showlegend=True))
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_y["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_y"],width=width_line),
                                    name='acc_y',showlegend=True))
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_z["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_z"],width=width_line),
                                    name='acc_z',showlegend=True))
        
        else : 
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_x["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_x"], width=width_line),
                        name='acc_x',
                        showlegend=False))
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_y["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_y"],width=width_line),
                                    name='acc_y',showlegend=False))
            fig.add_trace(go.Scatter(x=acc_x["times"][seg], y=acc_z["sig"][seg],
                        mode='lines',
                        line=dict(color=COLORS()["acc_z"],width=width_line),
                                    name='acc_z',showlegend=False))
    

    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[ymin, ymax]),
                      title='Acceleration',
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=True,
                      legend_title_text='Acc',
                      legend=dict(yanchor="top",
                                  y=0.99, xanchor="left", 
                                  x=0.01,
                                  ),
                      )
    
    return fig 


def chart_activity_level (template='plotly_white', width_line=2, height=500):
    # Activity Level
    translate = st.session_state.translate
    ymin = -200
    ymax = 200
    fig = go.Figure()
    output = get_averaged_activity()
    sig = output["values"]
    time = output["times"]
    
    fig.add_trace(go.Scatter(x=time, 
                              y=sig,
                        mode='lines',
                        line=dict(color=COLORS()["acc"], width=width_line),
                        name='Activity Level'))
    
    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[ymin, ymax]),
                      title= translate["activity_level"],
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      )
    
    return fig 


def temperature(al, template='plotly_white', width_line=2, height=500):
    # Temperature
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unwrap(al.temp_2.times_), 
                              y=1/2*(unwrap(al.temp_2.sig_)+unwrap(al.temp_1.sig_)),
                        mode='lines',
                        line=dict(color=COLORS()["temp_2"], width=width_line),
                        name='Temperature'))
    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[21, 40]), 
                      title='Skin temperature',
                      )
    
    return fig 
    
def steps_donut_old():
    plt.close('all')
    color_text = '#3E738D'

    translate = st.session_state.translate
    steps           = get_steps()
    steps_score     = steps["score"]
            
    wedgeprops = {"linewidth": 1, "edgecolor": "white"}
    
    plt.figure()
    if isinstance(steps_score, str) == False:
        if (steps_score < 100):
            size_of_groups=[steps_score, 100-steps_score]
            plt.pie(size_of_groups, 
                    colors=["#4393B4", "#E6E6E6"], 
                    startangle=90,
                    counterclock=False,
                    wedgeprops=wedgeprops)
        if (steps_score >= 100) :
            size_of_groups=[100]
            plt.pie(size_of_groups, 
                colors=["#13A943"], 
                startangle=90,
                counterclock=False,
                wedgeprops=wedgeprops)  

    my_circle=plt.Circle( (0,0), 0.8, color="white")
    if steps["score"] == "":
        plt.text(0, -1.5, translate["no_data"], fontsize=40, color=COLORS()["text"],
                 horizontalalignment='center',verticalalignment='center')
    else:
        plt.text(0, 0, (str(steps_score) + '%'), fontsize=40, color=color_text,
                 horizontalalignment='center')
        
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.savefig("template/images/steps_donut.png", transparent=True)
    st.session_state.steps_donut = img_to_bytes('template/images/steps_donut.png')


def steps_donut():
    
    translate = st.session_state.translate
    steps           = data.get_steps()
    steps_score     = steps["score"]
    
    color_text = '#3E738D'
    
    if steps_score == "":
        size_of_groups=[0, 100]

    elif (steps_score < 100):
        size_of_groups=[steps_score, 100-steps_score]
        plt.pie(size_of_groups, 
                colors=["#4393B4", "#E6E6E6"], 
                startangle=90,
                counterclock=False)
        
    elif (steps_score >= 100) :
        size_of_groups=[100]
        plt.pie(size_of_groups,  
        colors=["#13A943"], 
        startangle=90,
        counterclock=False)
        
    wedgeprops = {"linewidth": 1, "edgecolor": "white"}
    plt.close('all')
    plt.figure()
    plt.pie(size_of_groups, 
            colors=['green', "#e8e8e8"], 
            startangle=90,
            counterclock=False,
            wedgeprops=wedgeprops)
    

    my_circle=plt.Circle( (0,0), 0.8, color="white")
    
    if steps["score"] == "":
        plt.text(0, -1.5, translate["no_data"], fontsize=40, color=COLORS()["text"],
                 horizontalalignment='center',verticalalignment='center')
    else:
        plt.text(0, 0, (str(steps_score) + '%'), fontsize=40, color=color_text,
                 horizontalalignment='center')
        
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.savefig("template/images/steps_donut.png", transparent=True)
    st.session_state.steps_donut = img_to_bytes('template/images/steps_donut.png')
    
