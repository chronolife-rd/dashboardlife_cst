import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import data.data as data
import data.cst_raw_data as cst_raw_data
import template.constant as constant 
from pylife.useful import unwrap, unwrap_signals_dashboard
from scipy.signal import medfilt
import numpy as np
import datetime 
from template.util import img_to_bytes
from data.data import *
import random
from template.raw_signal_filter import *
import plotly.express as px
import time


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


    df = manage_no_data_color(values_df)

    fig.add_trace(go.Scatter(x=df["times"], 
                            y=df["values"]/100,
                            mode='lines+markers',
                            line=dict(color=constant.COLORS()['temp'], width=line_width),
                            name='tmp'))
        
    fig.update_layout(xaxis_title = translate["times"],
                      yaxis_title=constant.UNIT()['temp'],
                      font=dict(size=14,),
                      height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['temp'],
                       yaxis = dict(range=constant.RANGE()['temp']),
                      )
    
    return fig

def breath_brv():
    
    translate = st.session_state.translate
    
    # !!! TO BE UPDATED WITH REAL DATA !!!
    values_df = get_brv_values()
    values_df = values_df[["times","values"]]
    df = manage_no_data_color(values_df)
    # x = values_df["times"]
    # y = values_df["values"]

    line_width = 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["times"], 
                             y=df["values"],
                        mode='lines+markers',
                        line=dict(color=constant.COLORS()['breath'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['brv'],
                      height=400,
                      font=dict(size=14))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['brv'],
                       yaxis = dict(range=constant.RANGE()['brv']),
                      )
    
    return fig

def breath_brpm():
    
    translate = st.session_state.translate    
    line_width = 2

    fig = go.Figure()

    values_df = get_brpm_values()
    values_df = values_df[["times","values"]]
    df = manage_no_data_color(values_df)

    fig.add_trace(go.Scatter(x=df["times"], 
                            y=df["values"],
                        mode='lines+markers',
                        line=dict(color=constant.COLORS()['breath'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['brpm'],
                      height=400,
                      font=dict(size=14,))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['brpm'],
                       yaxis = dict(range=constant.RANGE()['brpm']),
                      )
    
    return fig

def breath_inex():
    
    translate = st.session_state.translate
    
    values_df = get_inex_values()
    values_df = values_df[["times","values"]]
    df = manage_no_data_color(values_df)
    
    line_width = 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["times"], 
                             y=df["values"],
                        mode='lines+markers',
                        line=dict(color=constant.COLORS()['breath'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['inspi_expi'],
                      height=400,
                      font=dict(size=14,))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['bior'],
                       yaxis = dict(range=constant.RANGE()['bior']),
                      )
    
    return fig



def heart_hrv():
    
    translate = st.session_state.translate
    
    values_df = get_hrv_values()
    df_hrv = values_df[['times','values']]
    df = manage_no_data_color(df_hrv) 
    
    line_width = 2
    fig = go.Figure()
    # Add Chronolife


    fig.add_trace(go.Scatter(y=df['values'],x=df['times'],mode='lines+markers',
                             connectgaps=False,
                             line=dict(color=constant.COLORS()["chronolife"])

                             ))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['hrv'],
                      height=400,
                      font=dict(size=14,))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['hrv'],
                      )
    
    return fig

def heart_qt():
    
    translate = st.session_state.translate
    
    values_df = get_qt()
    values_df = pd.DataFrame.from_dict (values_df)
    
    values_df = values_df[["times","values"]]

    df = manage_no_data_color(values_df)
   
    
    line_width = 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["times"], 
                             y=df["values"],
                        mode='lines+markers',
                        connectgaps=False,
                        line=dict(color=constant.COLORS()['hrv'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['hrv'],
                      height=400,
                      font=dict(size=14,))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['qt'],
                      )
    
    return fig


def heart_bpm():
    
    translate = st.session_state.translate
           
    line_width = 2
    fig = go.Figure()

    values_df = get_bpm_values()
    values_df = values_df[["times","values"]]

    df = manage_no_data_color(values_df)


    fig.add_trace(go.Scatter(x=df['times'], 
                            y=df['values'],
                        mode='lines+markers',
                        line=dict(color=constant.COLORS()['ecg'], width=line_width),
                        name='tmp'))
    
    fig.update_layout(xaxis_title=translate["times"],
                      yaxis_title=constant.UNIT()['bpm'],
                      height=400,
                      font=dict(size=14))
    fig.update_layout(height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      title=constant.SHORTCUT()['bpm'],
                       yaxis = dict(range=constant.RANGE()['bpm']),
                      )
    
    return fig

def duration():
    
    translate = st.session_state.translate

    date = st.session_state.date
    date = datetime.datetime.strptime(date, "%Y-%m-%d")   
    xmin = date
    xmax = date + datetime.timedelta(days = 1)
        
    y_chronolife    = np.repeat("Smart Textile", 2)
    y_empty         = np.repeat(" ", 2)
    y_empty2        = np.repeat("", 2)

    # x_garmin        = get_duration_garmin()['intervals']
    x_chronolife    = get_duration_chronolife()['intervals']
    
    width = 20
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_empty,
                              x=[datetime.datetime(date.year, date.month, date.day, 00, 0, 0),
                                 datetime.datetime(date.year, date.month, date.day, 23, 59, 59)],
                              mode="lines", line=dict(color="white",width=width)))

    # Add Chronolife
    for i in range(len(x_chronolife)):
        interval = x_chronolife[i][:]
        if (len(interval) > 1):
            interval_start = interval.iloc[0]
            interval_end = interval.iloc[-1]
            fig.add_trace(go.Scatter(y = y_chronolife, x=[interval_start, interval_end],
                        mode="lines", line=dict(color=constant.COLORS()["chronolife"],width=width)))
    
    fig.add_trace(go.Scatter(y=y_empty2,
                              x=[datetime.datetime(date.year, date.month, date.day, 00, 0, 0),
                                 datetime.datetime(date.year, date.month, date.day, 23, 0, 0)],
                              mode="lines", line=dict(color="white",width=width)))
    
    fig.update_layout(barmode='stack', height=300, 
                      template="plotly_white",
                      paper_bgcolor=BGCOLOR, plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      title="<span style='font-size:20px;'>" + translate["data_collection_duration"] + "</span>",
                      xaxis = dict(range=[xmin, xmax]), 
                      )
    
    return fig

def smart_textile_raw_data():

    time_start_get_raw_data = datetime.datetime.now()
    print("time_start_get_raw_data",time_start_get_raw_data)

    cst_raw_data.get_raw_data()

    time_recive_raw_data = datetime.datetime.now()
    print("time_recive_raw_data",time_recive_raw_data)
    
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
    #/ fig_acc     = acceleration_signal(template=template, width_line=width_line, height=height)
    fig_acc     = acceleration_raw_signal(template=template, width_line=width_line, height=height)
    time_finish_process_data = datetime.datetime.now()
    print("time_finish_process_data",time_finish_process_data)
    
    st.plotly_chart(fig_ecg, use_container_width=True)
    st.plotly_chart(fig_breath, use_container_width=True)
    st.plotly_chart(fig_acc,use_container_width=True)
    
def ecg_signal(template='plotly_white', width_line=2, height=500):
    fig = go.Figure()
    # offset = data.get_offset
    # if isinstance(offset["value"], str) == False:
    #     value = offset["value"]
    #     sign = offset["sign"]
    
    # ECG
    raw_data = st.session_state.smart_textile_raw_data[constant.TYPE()['ECG']]
    ecg_filt = filter_ecg_scipy_unwrap(raw_data['sig'],fs=200) # Localy filtered

    ymin = max([min(unwrap_signals_dashboard(ecg_filt))*1.1, -1000])
    ymax = min([max(unwrap_signals_dashboard(ecg_filt))*1.1, 1000])
    
    for seg in range(len(raw_data["times"])):
        fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=ecg_filt[seg],
                    mode='lines',
                    line=dict(color=constant.COLORS()["ecg"], width=width_line),
                                name='ECG'))
        
    # output = get_averaged_activity()

    fig.update_layout(height=height,
                      title='Electrocardiogram',
                      yaxis = dict(range=[ymin, ymax]), 
                      template=template,
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      )
    
    return fig
    
def breath_signal(template='plotly_white', width_line=2, height=500):
    fig = go.Figure()
    # Abdominal
    raw_data = st.session_state.smart_textile_raw_data[constant.TYPE()["BREATH_ABDOMINAL"]]
    abdominal_filt = filter_breath_unwrap(raw_data['sig'],fs=20) # Localy filtered
    for seg in range(len(raw_data["times"])):
        fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=abdominal_filt[seg],
                    mode='lines',
                    line=dict(color=constant.COLORS()["breath_2"], width=width_line),
                    name='Abdominal Breath', showlegend=False))
    # Thoracic 
    raw_data = st.session_state.smart_textile_raw_data[constant.TYPE()["BREATH_THORACIC"]]
    thoracic_filt = filter_breath_unwrap(raw_data['sig'],fs=20) # Localy filtered

    for seg in range(len(raw_data["times"])):
        fig.add_trace(go.Scatter(x=raw_data["times"][seg], y=thoracic_filt[seg],
                    mode='lines',
                    line=dict(color=constant.COLORS()["breath"], width=width_line),
                    name='Thoracic Breath', showlegend=False)) 
        
    fig.update_layout(showlegend=False)
        


    fig.update_layout(height=height, 
                      template=template, 
                      title='Breath',
                      showlegend=True, 
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
                        line=dict(color=constant.COLORS()["acc"], width=width_line),
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
    acc_x     = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_X"]]
    acc_z     = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_Z"]]
    acc_y    = st.session_state.smart_textile_raw_data[constant.TYPE()["ACCELERATION_Y"]]

    times   = unwrap_signals_dashboard(acc_x['times'])
    values_x  = unwrap_signals_dashboard(acc_x['sig'])
    values_z  = unwrap_signals_dashboard(acc_z['sig'])
    values_y  = unwrap_signals_dashboard(acc_y['sig'])


    df_acc = pd.DataFrame({'times': unwrap(times),
                'signal_x': unwrap(values_x),
                'signal_z': unwrap(values_z),
                'signal_y': unwrap(values_y)},

                )

    fig.add_trace(go.Scatter(x=df_acc["times"], 
                            y=df_acc["signal_x"],
                        mode='lines',
                        line=dict( width=width_line),
                        name='acc_x'))

    fig.add_trace(go.Scatter(x=df_acc["times"], 
                            y=df_acc["signal_y"],
                        mode='lines',
                        line=dict( width=width_line),
                        name='acc_y'))

    fig.add_trace(go.Scatter(x=df_acc["times"], 
                            y=df_acc["signal_z"],
                        mode='lines',
                        line=dict( width=width_line),
                        name='acc_z'),
                        )

    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[ymin, ymax]),
                      title='Acceleration',
                      paper_bgcolor=BGCOLOR, 
                      plot_bgcolor=BGCOLOR,
                      showlegend=False,
                      legend_title_text='Acc'

                      )

    return fig 


def chart_activity_level (template='plotly_white', width_line=2, height=500):
    # Activity Level
    translate = st.session_state.translate
    ymin = -200
    ymax = 200
    fig = go.Figure()
    output = get_activity_level()
    sig = output["values"]
    time = output["times"]

    fig.add_trace(go.Scatter(x=time, 
                              y=sig,
                        mode='lines',
                        line=dict(color=constant.COLORS()["acc"], width=width_line),
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
                        line=dict(color=constant.COLORS()["temp_2"], width=width_line),
                        name='Temperature'))
    fig.update_layout(height=height, 
                      template=template, 
                      yaxis = dict(range=[21, 40]), 
                      title='Skin temperature',
                      )
    
    return fig 

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
        plt.text(0, -1.5, translate["no_data"], fontsize=40, color=constant.COLORS()["text"],
                 horizontalalignment='center',verticalalignment='center')
    else:
        plt.text(0, 0, (str(steps_score) + '%'), fontsize=40, color=color_text,
                 horizontalalignment='center')
        
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.savefig("template/images/steps_donut.png", transparent=True)
    st.session_state.steps_donut = img_to_bytes('template/images/steps_donut.png')
    