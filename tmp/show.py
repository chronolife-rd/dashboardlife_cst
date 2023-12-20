import streamlit as st
import numpy as np
import plotly.graph_objects as go
from back_functions import get_data
from pylife.useful import unwrap
from back_functions import test_time, test_end_user 
# from back_functions import convert_data_to_excel
from back_functions import process_indicators
# from back_functions import test_end_user
from back_functions import get_end_users
from back_functions import test_string
from back_functions import get_sessions
from back_functions import run_quality_check
# from back_functions import get_data
from back_functions import aggregate_stream_data
from back_functions import download
# import pandas as pd
from constant import (COLORS, SHORTCUT, UNIT, VALUE_TYPE, 
                      RANGE, DEFINITION, IMAGE) #STANDARD
import time
from PIL import Image
from scipy.signal import medfilt
from datetime import datetime as dt
from util import img_to_bytes


# def show_historical_data(form_indicators_layout, form_indicators_submit):
    
#     username    = st.session_state.username
#     date        = st.session_state.date
#     start_time  = st.session_state.start_time
#     end_time    = st.session_state.end_time
#     end_user    = st.session_state.end_user
#     timezone    = st.session_state.timezone
    
#     if form_indicators_submit: 
        
#         with st.spinner("Getting Data..."):
            
            # # ----- DURATION
            # color_garmin = '#6CCFF6'
            # color_clife  = '#F7921E'
            # # Graph test
            # y_garmin    = 'Garmin'
            # y_clife     = 'Smart Textile'
            # fig = go.Figure()
            # fig.add_trace(go.Bar(
            #     y=[" "],
            #     x=[0],
            #     name='Night',
            #     orientation='h',
            #     marker=dict(
            #         color=color_clife, opacity=[1, 0, 1],
            #     )
            # ))
            # fig.add_trace(go.Bar(
            #     y=[y_garmin, y_garmin, y_garmin],
            #     x=[7, 3, 14],
            #     name='Garmin',
            #     orientation='h',
            #     marker=dict(
            #         color=color_garmin, opacity=[1, 0, 1],
            #     )
            # ))
            # fig.add_trace(go.Bar(
            #     y=[y_clife, y_clife, y_clife],
            #     x=[12, 2, 10],
            #     name='Smart Textile',
            #     orientation='h',
            #     marker=dict(
            #         color=color_clife, opacity=[1, 0, 1],
            #     )
            # ))
            # fig.add_trace(go.Bar(
            #     y=[""],
            #     x=[0],
            #     name='Night',
            #     orientation='h',
            #     marker=dict(
            #         color=color_clife, opacity=[1, 0, 1],
            #     )
            # ))
            
            # fig.add_trace(go.Scatter(x=[0, 6], 
            #                          y=["", ""], 
            #                          name="night",
            #                          opacity=0.1,
            #                          mode="lines",
            #                          fill='tozeroy',
            #                          line=dict(width=0.1, color="#E5EEF3"),
            #                          )) # fill down to xaxis
            
            # fig.update_layout(barmode='stack', height=300, 
            #                   template="plotly_white",
            #                   paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)',
            #                   bargap=0.50, title='<span style="font-size:20px;">DURATION OF DATA COLLECTION</span>',
            #                   showlegend=False,
            #                   )
            
            # ----- HEALTH OVERVIEW
            # ----- Cardio
            # alert_icon      = st.session_state.alert
            # no_alert_icon   = st.session_state.alert_no
            
            # tachycardia = True
            # if tachycardia:
            #     tachycardia_alert_icon = alert_icon
            # else:
            #     tachycardia_alert_icon = no_alert_icon
                
            # bradycardia = True
            # if bradycardia:
            #     bradycardia_alert_icon = alert_icon
            # else:
            #     bradycardia_alert_icon = no_alert_icon

            # qt = 435
            # qt_threshold = 550
            # if qt > qt_threshold:
            #     qt_alert_icon = alert_icon
            # else:
            #     qt_alert_icon = no_alert_icon
                
            # hr_high = 135
            # hr_rest = 62
            # hrv_rest = 125
            # cardio_html_string = """
            # <div class='health_section'> 
            #     <div class="container-fluid">
            #         <div class="row">
            #             <div class="col-lg-12">
            #                 <img class=icon src='data:image/png;base64,""" + st.session_state.cardiology + """'/> 
            #                 <p class='indicator_name'>Cardiology</p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>Tachycardia</p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p><img class=miniicon src='data:image/png;base64,""" + tachycardia_alert_icon + """'/></p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>Bradycardia</p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p><img class=miniicon src='data:image/png;base64,""" + bradycardia_alert_icon + """'/> </p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>QTc Length anomaly </p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p><img class=miniicon src='data:image/png;base64,""" + qt_alert_icon + """'/> </p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>HR high</p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p>""" + str(hr_high) + """ bpm</p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>HR resting</p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p>""" + str(hr_rest) + """ bpm</p>
            #             </div>
            #             <div class="col-lg-8">
            #                 <p>HRV resting</p>
            #             </div>
            #             <div class="col-lg-4">
            #                 <p>""" + str(hrv_rest) + """ ms</p>
            #             </div>
            #         </div>
            #     </div>
            # </div>"""
            
            # ----- Breath
            # tachypnea = False
            # if tachypnea:
            #     tachypnea_alert_icon = alert_icon
            # else:
            #     tachypnea_alert_icon = no_alert_icon
                
            # bradypnea = True
            # if bradypnea:
            #     bradypnea_alert_icon = alert_icon
            # else:
            #     bradypnea_alert_icon = no_alert_icon

            # br_high = 25
            # br_rest = 7
            # brv_rest = 2
            # breath_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.respiratory + """'/> 
            #             <p class='indicator_name'>Respiratory</p>
            #         </div>
            #         <div class="col-lg-8">
            #             <p>Tachypnea</p>
            #         </div>
            #         <div class="col-lg-4">
            #             <p><img class=miniicon src='data:image/png;base64,""" + tachypnea_alert_icon + """'/> </p>
            #         </div>
            #         <div class="col-lg-8">
            #             <p>Bradypnea</p>
            #         </div>
            #         <div class="col-lg-4">
            #             <p><img class=miniicon src='data:image/png;base64,""" + bradypnea_alert_icon + """'/> </p>
            #         </div>
            #         <div class="col-lg-8">
            #             <p>BR high</p>
            #         </div>
            #         <div class="col-lg-4">
            #             <p>""" + str(br_high) + """ brpm</p>
            #         </div>
            #         <div class="col-lg-8">
            #             <p>BR resting</p>
            #         </div>
            #         <div class="col-lg-4">
            #             <p>""" + str(br_rest) + """ brpm</p>
            #         </div>
            #         <div class="col-lg-8">
            #             <p>BRV resting</p>
            #         </div>
            #         <div class="col-lg-4">
            #             <p>""" + str(brv_rest) + """ s</p>
            #             <br>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # # ----- Steps
            # steps = 1267
            # goal = 3457
            # distance = 2.8
            
            # steps_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.steps + """'/> 
            #             <p class='indicator_name'>Steps</p>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <div class="row">
            #                         <div class="col-lg-12">
            #                             <p>Number of steps</p>
            #                         </div>
            #                         <div class="col-lg-12">
            #                             <p class='steps_number'>""" + str(steps) + """</p>
            #                         </div>
            #                     </div>
            #                 </div>
            #                 <div class="col-lg-6">
            #                     <div class="row">
            #                         <div class="col-lg-12">
            #                             <p>Goal</p>
            #                         </div>
            #                         <div class="col-lg-12">
            #                             <p>""" + str(goal) + """</p>
            #                         </div>
            #                     </div>
            #                 </div>
            #                 <div class="col-lg-6">
            #                     <div class="row">
            #                         <div class="col-lg-12">
            #                             <p>Distance</p>
            #                         </div>
            #                         <div class="col-lg-12">
            #                             <p>""" + str(distance) + """ km</p>
            #                         </div>
            #                     </div>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <center>
            #                 <img class=donut src='data:image/png;base64,""" + img_to_bytes('assets/donut_steps.png') + """'/> 
            #             </center>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # stress_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.stress + """'/> 
            #             <p class='indicator_name'>Stress</p>
            #         </div>
            #         <div class="col-lg-7">
            #             <div class="row">
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.stress_rest + """'/> 
            #                     <p>Rest</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.stress_low + """'/>
            #                     <p>Low</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.stress_medium + """'/>
            #                     <p>Medium</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.stress_high + """'/>
            #                     <p>High</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-5">
            #             <center>
            #                 <img class=donut src='data:image/png;base64,""" + img_to_bytes('assets/donut_stress.png') + """'/> 
            #             </center>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # pulseox_html_string = """ 
            # # <div class='health_section'> 
            # #     <div class="row">
            # #         <div class="col-lg-12">
            # #             <img class=icon src='data:image/png;base64,""" + st.session_state.pulseox + """'/> 
            # #             <p class='indicator_name'>Pulse Ox</p>
            # #         </div>
            # #         <div class="col-lg-5">
            # #                 <img class=stressicon src='data:image/png;base64,""" + st.session_state.spo2_green + """'/>
            # #                 <p>90 - 100 %</p>
            # #                 <img class=stressicon src='data:image/png;base64,""" + st.session_state.spo2_yellow + """'/>
            # #                 <p>80 - 89 %</p>
            # #                 <img class=stressicon src='data:image/png;base64,""" + st.session_state.spo2_orange + """'/>
            # #                 <p>70 - 79 %</p>
            # #                 <img class=stressicon src='data:image/png;base64,""" + st.session_state.spo2_red + """'/>
            # #                 <p>< 70 %</p>
            # #         </div>
            # #         <div class="col-lg-7">
            # #             <center>
            # #                 <img class=donut src='data:image/png;base64,""" + img_to_bytes('assets/donut_spo2.png') + """'/> 
            # #             </center>
            # #         </div>
            # #     </div>
            # # </div>"""
            
            # sleep_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.sleep + """'/> 
            #             <p class='indicator_name'>Sleep</p>
            #         </div>
            #         <div class="col-lg-7">
            #             <div class="row">
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.sleep_deep + """'/>
            #                     <p>Deep</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.sleep_light + """'/>
            #                     <p>Light</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.sleep_rem + """'/>
            #                     <p>REM</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #                 <div class="col-lg-5">
            #                     <img class=stressicon src='data:image/png;base64,""" + st.session_state.sleep_awake + """'/>
            #                     <p>Awake</p>
            #                 </div>
            #                 <div class="col-lg-7">
            #                     <p>10h 04m (25%)</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-5">
            #             <center>
            #                 <img class=donut src='data:image/png;base64,""" + img_to_bytes('assets/donut_sleep.png') + """'/> 
            #             </center>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # Calories
            # calories = 1234
            # calories_rest = 817
            # calories_active = 324
            # calories_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.calories + """'/> 
            #             <p class='indicator_name'>Calories</p>
            #         </div>
            #         <div class="col-lg-12">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Number of total calories</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p class='total_calories'>""" + str(calories) + """ kcals</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Active</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p>""" + str(calories_rest) + """ kcals</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Resting</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p>""" + str(calories_active) + """ kcals</p>
            #                 </div>
            #             </div>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # Intensity
            # intensity_total = 45
            # intensity_moderate = 20
            # intensity_vigorous = 13
            # intensity_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.intensity_minutes + """'/> 
            #             <p class='indicator_name'>Intensity Minutes</p>
            #         </div>
            #         <div class="col-lg-12">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Total Intensity Minutes</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p class='total_intensity_minutes'>""" + str(intensity_total) + """ min</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Moderate</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p>""" + str(intensity_moderate) + """ min</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Vigorous</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p>""" + str(intensity_vigorous) + """ min</p>
            #                 </div>
            #             </div>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # # Body Battery
            # body_bat_high = 89
            # body_bat_low = 32
            # body_battery_html_string = """
            # <div class='health_section'> 
            #     <div class="row">
            #         <div class="col-lg-12">
            #             <img class=icon src='data:image/png;base64,""" + st.session_state.body_battery + """'/> 
            #             <p class='indicator_name'>Body Battery</p>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>High</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p class='body_battery_scores'>""" + str(body_bat_high) + """%</p>
            #                 </div>
            #             </div>
            #         </div>
            #         <div class="col-lg-6">
            #             <div class="row">
            #                 <div class="col-lg-12">
            #                     <p>Low</p>
            #                 </div>
            #                 <div class="col-lg-12">
            #                     <p class='body_battery_scores'>""" + str(body_bat_low) + """%</p>
            #                     <br>
            #                     <br>
            #                 </div>
            #             </div>
            #         </div>
            #     </div>
            # </div>
            # """
            
            # layout_health = """
            # <div class="container-fluid">
            #     <div class="row">
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + cardio_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + breath_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + pulseox_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + steps_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + stress_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + sleep_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + calories_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + intensity_html_string + """</div>
            #         <div class="col-xl-4 col-lg-6 col-md-12 col-sm-12">""" + body_battery_html_string + """</div>
            #     </div>
            # </div>
            # """
            
            # # ----- Overview
            # st.markdown("""
            #             <div id="overview" class="main_title">
            #                 <p>Overview</p>
            #             </div>
            #             <div class="second_title">
            #                 <p>Data Collection</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            # st.markdown(layout_overview, unsafe_allow_html=True)
            # st.markdown("")
            # st.markdown("""
            #             <div class="second_title">
            #                 <p>Duration Of Data Collection</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            
            # config = {'displayModeBar': False}
            # st.plotly_chart(fig, config=config, use_container_width=True)
            # st.markdown("")
            # st.markdown("""
            #             <div class="second_title">
            #                 <p>Health Indicators Overview</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            # st.markdown(layout_health, unsafe_allow_html=True)
            
            # Raw Data
            # st.markdown("""
            #             <div id="smart_textile_raw_data" class="main_title">
            #                 <p>Smart Textile Raw Data</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            
            # st.markdown("""
            #             <div class="second_title">
            #                 <p>Raw Data</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            # error = show_raw_data(username, end_user, date, start_time, end_time, timezone, form_indicators_layout)
            
            # Horizontal line
            
            # # Tabs
            # st.markdown("""
            #             <div id="health_indicators" class="main_title">
            #                 <p>Health Indicators</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            # st.markdown("""
            #             <div>
            #                 <p>Download the daily health indicators (XLS)</p>
            #             </div>
            #             """, 
            #             unsafe_allow_html=True)
            # download_col,_ = st.columns([2,6])
            # download_layout = st.empty()
            
            # tab_heart, tab_breath, tab_temp, tab_activity, tab_stress, tab_sleep, tab_spo2, tab_body_battery = st.tabs(["Heart", 
            #                                                                                                             "Breath", 
            #                                                                                                             "Temperature",
            #                                                                                                             "Activity",
            #                                                                                                             "Stress",
            #                                                                                                             "Sleep",
            #                                                                                                             "Pulse Ox",
            #                                                                                                             "Body Battery",
            #                                                                                                             ])
            
            # if not error:
            #     # form_indicators_layout.success("Data has been successfully requested")
                
            #     with tab_heart:
            #         show_indicator("bpm", minmax=True)
            #         show_indicator("sdnn", minmax=True)
            #         show_indicator("tachycardia", pc=True, dur=True)
            #         show_indicator("bradycardia", pc=True, dur=True)
            #         show_indicator("qt", minmax=True)
    
            #     with tab_breath:
            #         show_indicator_breath("rpm", minmax=True)
            #         show_indicator_breath("brv", minmax=True)
            #         show_indicator_breath("bior", minmax=True)
            #         show_indicator_breath("tachypnea", pc=True, dur=True)
            #         show_indicator_breath('bradypnea', pc=True, dur=True)
                    
            #     with tab_temp:
            #         show_indicator("temp", minmax=True)
                    
            #     with tab_activity:
            #         show_indicator("n_steps")
                    
            #     with tab_stress:
            #         show_indicator("stress", disp=False)
                    
            #     with tab_sleep:
            #         show_indicator("sleep", disp=False)
                    
            #     with tab_spo2:
            #         show_indicator("spo2", disp=False)
                    
            #     with tab_body_battery:
            #         show_indicator("fatigue", disp=False)
                    
                # # Download function
                # download(download_layout)
    

def show_indicator(indicator, val='value', unit=None, disp=True, minmax=False, dur=False, pc=False):
    col_ind_size = [1, 5, 2, 2, 1, 3]
    img_path = 'assets/'
    value = '-'
    mini = '-'
    maxi = '-'
    
    al = st.session_state.al
    if disp:
        output, indicators = process_indicators(al)
        st.session_state.indicators = indicators
        value = output[indicator][val]
        if minmax:
            mini = output[indicator]['min']
            maxi = output[indicator]['max']
        if dur:
            duration = output[indicator]['duration']
        if pc:
            percent = output[indicator]['percent']
    else:
        value = 'Garmin package'        
        
    c1, c2, c3, c4, c5, c6 = st.columns(col_ind_size)
    image = Image.open(img_path + IMAGE[indicator])
    c1.image(image, width=30)
    c2.write(SHORTCUT[indicator])
    c2.write(DEFINITION[indicator])
    if disp:
        if value != '':
            c3.write(VALUE_TYPE[indicator])
            value += " " + UNIT[indicator]
            c4.write(value)
    else:
        c3.write(value)
    if minmax:
        c3.write('Min')
        c4.write((mini + " " + UNIT[indicator]))
        c3.write('Max')
        c4.write((maxi + " " + UNIT[indicator]))
    
    if minmax:
        show_indicator_graph(indicator, indicators)
    
    if dur:
        c3.write('Duration')
        c4.write((duration + " min"))
    if pc:
        c3.write('Proportion')
        c4.write((percent + " % of rest time"))
    
    st.markdown("---")

def show_indicator_breath(indicator, val='value', unit=None, disp=True, minmax=False, dur=False, pc=False):
    col_ind_size = [1, 5, 3, 3, 3, 2]
    img_path = 'assets/'
    value = '-'
    mini = '-'
    maxi = '-'
    
    indicator_1 = indicator + "_1"
    indicator_2 = indicator + "_2"
    
    al = st.session_state.al
    if disp:
        output, result = process_indicators(al)
        value = output[indicator_1][val]
        value2 = output[indicator_2][val]
        if minmax:
            mini = output[indicator_1]['min']
            maxi = output[indicator_1]['max']
            mini2 = output[indicator_2]['min']
            maxi2 = output[indicator_2]['max']
        if dur:
            duration = output[indicator_1]['duration']
            duration2 = output[indicator_2]['duration']
        if pc:
            percent = output[indicator_1]['percent']
            percent2 = output[indicator_2]['percent']
            
    else:
        value = 'Garmin package'        
    c1, c2, c3, c4, c5, c6 = st.columns(col_ind_size)
    image = Image.open(img_path + IMAGE[indicator_1])
    c1.image(image, width=30)
    c2.write(SHORTCUT[indicator])
    c2.write(DEFINITION[indicator_1])
    c3.markdown("<span style='color:black;'>.</span>", unsafe_allow_html=True)
    c4.write('Thoracic')
    c5.write('Abdominal')
    
    if disp:
        if value != '' or value2 != '':
            c3.write(VALUE_TYPE[indicator_1])
            if value == '':
                c4.markdown("<span style='color:black;'>.</span>", unsafe_allow_html=True)
            else:
                value += " " + UNIT[indicator_1]
                c4.write(value)

            if value2 == '':
                c5.markdown("<span style='color:black;'>.</span>", unsafe_allow_html=True)
            else:
                value2 += " " + UNIT[indicator_2]
                c5.write(value2)
            
    if minmax:
        c3.write('Min')
        c4.write((mini + " " + UNIT[indicator_1]))
        c5.write((mini2 + " " + UNIT[indicator_2]))
        c3.write('Max')
        c4.write((maxi + " " + UNIT[indicator_1]))
        c5.write((maxi2 + " " + UNIT[indicator_2]))
        
        show_indicator_breath_graph(indicator, result)
        
    if dur:
        c3.write('Duration')
        c4.write((duration + " min"))
        c5.write((duration2 + " min"))
    if pc:
        c3.write('Proportion')
        c4.write((percent + " % of rest time"))
        c5.write((percent2 + " % of rest time"))
        
        
    
    st.markdown("---")
    
def show_indicator_graph(indicator, result):
    line_width = 2
    template = 'plotly_dark'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result[result.label==indicator]['times'], 
                             y=result[result.label==indicator]['value'],
                        mode='lines',
                        line=dict(color=COLORS[indicator], width=line_width),
                        name=indicator))
    
    fig.update_layout(xaxis_title="Times",
                      yaxis_title=UNIT[indicator],
                      height=400,
                      font=dict(
                          # family="Courier New, monospace",
                          size=14,
                          ))
    fig.update_layout(yaxis = dict(range=RANGE[indicator]))
    fig.update_layout(template=template)
    fig.update_layout(title=SHORTCUT[indicator])
    st.plotly_chart(fig, True)
    
    
def show_indicator_breath_graph(indicator, result):
    
    indicator_1 = indicator + "_1"
    indicator_2 = indicator + "_2"
    
    line_width = 2
    template = 'plotly_dark'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result[result.label==indicator_1]['times'], 
                             y=result[result.label==indicator_1]['value'],
                        mode='lines',
                        line=dict(color=COLORS[indicator_1], width=line_width),
                        name=SHORTCUT[indicator_1]))
    fig.add_trace(go.Scatter(x=result[result.label==indicator_2]['times'], 
                             y=result[result.label==indicator_2]['value'],
                        mode='lines',
                        line=dict(color=COLORS[indicator_2], width=line_width),
                        name=SHORTCUT[indicator_2]))
    
    fig.update_layout(xaxis_title="Times",
                      yaxis_title=UNIT[indicator_1],
                      height=400,
                      font=dict(
                          size=14,
                          ))
    fig.update_layout(yaxis = dict(range=RANGE[indicator_1]))
    fig.update_layout(template=template)
    fig.update_layout(title=SHORTCUT[indicator])
    st.plotly_chart(fig, True)
        
def show_raw_data(username, end_user, date, start_time, end_time, timezone, data_form):
    error = False
    error = test_time(date, start_time, end_time, data_form)
    if error:
        return error
    error = test_end_user(end_user, data_form)
    st.session_state.end_user = end_user
    if error:
        return error
    
    if timezone == 'France (Winter Time)':
        time_zone       = 'CET'
    elif timezone == 'France (Summer Time)':
        time_zone       = 'CEST'    
    elif timezone == 'GMT':
        time_zone       = 'GMT'
    elif timezone == "Riyadh":
        time_zone       = 'AST1'
        
    start_time      += ':00'
    end_time        += ':00'
    from_datetime   = str(date) + " " + start_time
    to_datetime     = str(date) + " " + end_time
    
    al = get_data(username, end_user, time_zone, from_datetime, to_datetime)
    st.session_state.al = al
    
    if al.is_empty_:
        data_form.warning("No data found")
        error = True
        return error
        
    # ------------------------------------------------------------------------
    width_line = 2
    template = 'plotly_dark'
    width = 1000
    height = 500
    
    # ECG
    ymin = max([min(unwrap(al.ecg.sig_filt_))*1.1, -1000])
    ymax = min([max(unwrap(al.ecg.sig_filt_))*1.1, 1000])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unwrap(al.ecg.times_), y=unwrap(al.ecg.sig_filt_),
                        mode='lines',
                        line=dict(color=COLORS["ecg"], width=width_line),
                                  name='ECG'))
    fig.update_layout(width=width, height=height)
    fig.update_layout(title='Electrocardiogram')
    fig.update_layout(yaxis = dict(range=[ymin, ymax]))
    fig.update_layout(template=template)
    st.plotly_chart(fig)
    
    # Breath
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=unwrap(al.breath_2.times_), y=unwrap(al.breath_2.sig_filt_),
                              mode='lines',
                              line=dict(color=COLORS["breath_2"], width=width_line),
                              name='Abdominal Breath'))
    fig.add_trace(go.Scatter(x=unwrap(al.breath_1.times_), y=unwrap(al.breath_1.sig_filt_),
                        mode='lines',
                        line=dict(color=COLORS["breath"], width=width_line),
                        name='Thoracic Breath'))
    fig.update_layout(width=width, height=height)
    fig.update_layout(template=template)
    fig.update_layout(title='Breath')
    st.plotly_chart(fig)
    
    # Acceleration
    ymin = -200
    ymax = 200
    fig = go.Figure()
    sig = 1/3*(abs(unwrap(al.accx.sig_clean_)) + abs(unwrap(al.accy.sig_clean_)) + abs(unwrap(al.accz.sig_clean_)))
    sig = medfilt(sig, kernel_size=11)
    fig.add_trace(go.Scatter(x=unwrap(al.accx.times_clean_), 
                              y=sig,
                        mode='lines',
                        line=dict(color=COLORS["acc"], width=width_line),
                        name='Acceleration'))
    fig.update_layout(width=width, height=height)
    fig.update_layout(template=template)
    fig.update_layout(yaxis = dict(range=[ymin, ymax]))
    fig.update_layout(title='Acceleration')
    st.plotly_chart(fig)
    
    # # Temperature
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=unwrap(al.temp_2.times_), 
    #                           y=1/2*(unwrap(al.temp_2.sig_)+unwrap(al.temp_1.sig_)),
    #                     mode='lines',
    #                     line=dict(color=COLORS["temp_2"], width=width_line),
    #                     name='Temperature'))
    # fig.update_layout(width=width, height=height)
    # fig.update_layout(template=template)
    # fig.update_layout(yaxis = dict(range=[21, 40]))
    # fig.update_layout(title='Skin temperature')
    # st.plotly_chart(fig)
    
