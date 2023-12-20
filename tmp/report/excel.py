import os
path_root = 'C:/Users/MichelClet/Desktop/Files/pylife'
import sys
sys.path.append(path_root)
from pylife.env import get_env
DEV = get_env()
from pylife.useful import unwrap
from pylife.useful import get_stats_clean_sig
from pylife.useful import mean_quadratic
from pylife.time_functions import datetime_np2str
import numpy as np
import pandas as pd

def create_report_folder(results_folder, entity, end_user, from_time, to_time, activity_type=None):
    reports_folder = 'reports'
    path_reports = results_folder + '/' + reports_folder + '/'
    if reports_folder not in os.listdir(results_folder):
        os.mkdir(path_reports)
    path_entity = path_reports + '/' + entity + '/'
    if entity not in os.listdir(path_reports):
        os.mkdir(path_entity)
    path_user = path_entity + end_user + '/'
    if end_user not in os.listdir(path_entity):
        os.mkdir(path_user)
    folder_data = from_time.replace('-', '').replace(' ', '-').replace(':', '')[:-2]
    folder_data += '_' + to_time.replace('-', '').replace(' ', '-').replace(':', '')[:-2]
    if activity_type is not None:
        folder_data += ' ' + activity_type
    path_save = path_user + folder_data + '/'
    if folder_data not in os.listdir(path_user):
        os.mkdir(path_save)
        
    return path_save


def reshape_results(times, results, window_time, op, per_minute=False):
    
    new_results = []
    new_starts = []
    new_stops = []
    
    times = np.array(unwrap(times))
    results = np.array(unwrap(results))
    
    if per_minute:
        results = 60/results
    
    if len(results) > 0:
        last_ts = times[0]
        last_idx = 0
        
        for i in range(len(times)):
            ts = times[i]
            delta = (ts - last_ts)/np.timedelta64(1, 's')
            
            if i == len(times)-1 and delta > window_time/2:
                delta = window_time
                
            if delta >= window_time:
                
                if op == 'mean':
                    new_result = np.mean(results[last_idx:i])
                if op == 'std':
                    new_result = np.std(results[last_idx:i])
                if op == 'median':
                    new_result = np.median(results[last_idx:i])
                if op == 'iqr':
                    new_result = np.percentile(results[last_idx:i], 75) - np.percentile(results[last_idx:i], 25)
                if op == 'sum':
                    new_result = np.sum(results[last_idx:i])
                if op == 'rmssd':
                    new_result = mean_quadratic(results[last_idx:i])
                if op == 'lnrmssd':
                    new_result = np.log(mean_quadratic(results[last_idx:i]))
                if op == 'pnn50':
                    res = results[last_idx:i]
                    diff = abs(res[1:] - res[:-1])
                    pnn50  = 0
                    diff = np.array(diff)
                    if len(diff) > 0:
                        pnn50 = len(diff[diff > 50])/len(diff)*100
                    new_result = np.sum(pnn50)
                
                # new_result = results[last_idx:i]
                new_results.append(new_result)
                new_starts.append(last_ts)
                new_stops.append(ts)
                last_ts = ts
                last_idx = i
    
    return np.array(new_starts), np.array(new_stops), np.array(new_results)

def usable_percentage(starts, stops, times, indicators, fs):
    times       = unwrap(times)
    indicators  = unwrap(indicators)
    
    percentages = []
    for i in range(len(starts)):
        start       = starts[i]
        stop        = stops[i]
        istart      = np.where(times == start)[0][0]
        istop       = np.where(times == stop)[0][0]
        ind         = indicators[istart:istop]
        info        = get_stats_clean_sig(ind, fs)
        percentages.append(info['percentage'])
    
    return np.array(percentages)

def reshape_as_sma(sma_stops, times, values, window_time, append='nan'):
    new_values = []
    # idex_s = []
    
    if len(values) > 0:
        for sma_t in sma_stops:
            delta = abs((sma_t - times)/np.timedelta64(1, 's'))
            imin = np.argmin(delta)
            delta = abs((sma_t - times[imin])/np.timedelta64(1, 's'))
            if delta < window_time:
                new_values.append(values[imin])
            else:
                if append == 'nan':
                    new_values.append(np.NAN)
                elif append == '0':
                    new_values.append(0)
    else:
        if append == 'nan':
            new_values = np.repeat(np.NAN, len(sma_stops))
        elif append == '0':
            new_values = np.repeat(0, len(sma_stops))
            
    return np.array(new_values)

def np_2_df(label, times, sma_values, values, value_type):
    columns = ['label', 'times', 'sma_values', value_type]
    labels = np.repeat(label, len(values))
    data = np.array([labels, times, sma_values.round(), values.round(2)])
    df = pd.DataFrame(data.T, columns=columns)
    
    return df

def excel_result_split(al, window_time, path_save):
    
    if al.accx and not al.accx.is_empty_:
        sma_starts, sma_stops, sma = reshape_results(al.accx.times_, 
                                                     al.accx.sma_, 
                                                     window_time,
                                                     op='median')
    
    # ------------------------------------------------------------------
    # acc
    steps_starts, steps_stops, steps = reshape_results(al.accx.steps_times_start_, 
                                                       al.accx.steps_, 
                                                       window_time,
                                                       op='sum')
    
    # RR intervals
    
    ecg_starts, ecg_stops, hr = reshape_results(al.ecg.times_rr_, 
                                                al.ecg.rr_, 
                                                window_time,
                                                op='median')
    
    # sdnn (Heart rate variability)
    _, _, sdnn = reshape_results(al.ecg.times_rr_, 
                                al.ecg.rr_, 
                                window_time,
                                op='std')
    
    # rmssd (HRV : Root Mean Square of the Successive Differences)
    _, _, rmssd = reshape_results(al.ecg.times_rr_, 
                                al.ecg.rr_, 
                                window_time,
                                op='rmssd')
    
    # lnrmssd (HRV : Log of Root Mean Square of the Successive Differences)
    _, _, lnrmssd = reshape_results(al.ecg.times_rr_, 
                                al.ecg.rr_, 
                                window_time,
                                op='lnrmssd')
    

    # pnn50
    _, _, pnn50 = reshape_results(al.ecg.times_rr_, 
                                  al.ecg.rr_, 
                                  window_time,
                                  op='pnn50')

    # QT intervals
    _, _, qt = reshape_results(al.ecg.q_start_time_, 
                               al.ecg.qt_length_, 
                               window_time,
                               op='median')

    iqh = np.where(qt >= 550)[0]
    if len(iqh) > 0:
        qt[iqh] = np.nan
    iql = np.where(qt <= 350)[0]
    if len(iql) > 0:
        qt[iql] = np.nan
        
    # BPM 
    _, _, bpm = reshape_results(al.ecg.times_rr_, 
                                al.ecg.rr_, 
                                window_time,
                                op='median',
                                per_minute=True)
    bpm *= 1000

    # BPM variability
    _, _, bpmv = reshape_results(al.ecg.times_rr_, 
                                al.ecg.rr_, 
                                window_time,
                                op='iqr',
                                per_minute=True)
    bpmv *= 1000
      

    # Breath rate THO
    breath_1_starts, breath_1_stops, br_1 = reshape_results(al.breath_1.times_rr_, 
                                                             al.breath_1.rr_, 
                                                             window_time,
                                                             op='median')
    # br_1 /= 1000

    # Breath rate variability THO
    breath_1_starts, breath_1_stops, brv_1 = reshape_results(al.breath_1.times_rr_, 
                                                             al.breath_1.rr_, 
                                                             window_time,
                                                             op='std')
    # brv_1 /= 1000

    # Respiration per minute THO
    _, _, rpm_1 = reshape_results(al.breath_1.times_rr_, 
                                  al.breath_1.rr_, 
                                  window_time,
                                  op='median',
                                  per_minute=True)
    # rpm_1 *= 1000

    # Respiration per minute variability THO
    _, _, rpmv_1 = reshape_results(al.breath_1.times_rr_, 
                                  al.breath_1.rr_, 
                                  window_time,
                                  op='iqr',
                                  per_minute=True)
    # rpmv_1 *= 1000


    # Breath rate ABD
    breath_2_starts, breath_2_stops, br_2 = reshape_results(al.breath_2.times_rr_, 
                                                             al.breath_2.rr_, 
                                                             window_time,
                                                             op='median')

    # br_2 /= 1000

    # Breath rate variability ABD
    breath_2_starts, breath_2_stops, brv_2 = reshape_results(al.breath_2.times_rr_, 
                                                             al.breath_2.rr_, 
                                                             window_time,
                                                             op='std')
    # brv_2 /= 1000

    # Respiration per minute ABD
    _, _, rpm_2 = reshape_results(al.breath_2.times_rr_, 
                                  al.breath_2.rr_, 
                                  window_time,
                                  op='median',
                                  per_minute=True)
    # rpm_2 *= 1000

    # Respiration per minute variability ABD
    _, _, rpmv_2 = reshape_results(al.breath_2.times_rr_, 
                                  al.breath_2.rr_, 
                                  window_time,
                                  op='iqr',
                                  per_minute=True)
    
    _, _, bior_1 = reshape_results(al.breath_1.rsp_features_['rsp_cycles_times'], 
                                  al.breath_1.rsp_features_['inhale_exhale_interval_ratio'], 
                                  window_time,
                                  op='median')
    
    _, _, bior_2 = reshape_results(al.breath_2.rsp_features_['rsp_cycles_times'], 
                                  al.breath_2.rsp_features_['inhale_exhale_interval_ratio'], 
                                  window_time,
                                  op='median')
    
    # rpmv_2 *= 1000


    #  temp
    temp_1_starts, temp_1_stops, temp_1 = reshape_results(al.temp_1.times_clean_, 
                                           al.temp_1.sig_clean_, 
                                           window_time,
                                           op='median')

    temp_2_starts, temp_2_stops, temp_2 = reshape_results(al.temp_2.times_clean_, 
                                           al.temp_2.sig_clean_, 
                                           window_time,
                                           op='median')

    steps_sma       = reshape_as_sma(sma_stops, steps_stops, steps, window_time, append='0')

    hr_sma          = reshape_as_sma(sma_stops, ecg_stops, hr, window_time, append='nan')
    sdnn_sma        = reshape_as_sma(sma_stops, ecg_stops, sdnn, window_time, append='nan')
    rmssd_sma       = reshape_as_sma(sma_stops, ecg_stops, rmssd, window_time, append='nan')
    lnrmssd_sma     = reshape_as_sma(sma_stops, ecg_stops, lnrmssd, window_time, append='nan')
    pnn50_sma       = reshape_as_sma(sma_stops, ecg_stops, pnn50, window_time, append='nan')
    
    bpm_sma         = reshape_as_sma(sma_stops, ecg_stops, bpm, window_time, append='nan')
    bpmv_sma        = reshape_as_sma(sma_stops, ecg_stops, bpmv, window_time, append='nan')
    
    qt_sma          = reshape_as_sma(sma_stops, ecg_stops, qt, window_time, append='nan')

    br_1_sma        = reshape_as_sma(sma_stops, breath_1_stops, br_1, window_time, append='nan')
    brv_1_sma       = reshape_as_sma(sma_stops, breath_1_stops, brv_1, window_time, append='nan')

    rpm_1_sma       = reshape_as_sma(sma_stops, breath_1_stops, rpm_1, window_time, append='nan')
    rpmv_1_sma      = reshape_as_sma(sma_stops, breath_1_stops, rpmv_1, window_time, append='nan')

    br_2_sma        = reshape_as_sma(sma_stops, breath_2_stops, br_2, window_time, append='nan')
    brv_2_sma       = reshape_as_sma(sma_stops, breath_2_stops, brv_2, window_time, append='nan')

    rpm_2_sma       = reshape_as_sma(sma_stops, breath_2_stops, rpm_2, window_time, append='nan')
    rpmv_2_sma      = reshape_as_sma(sma_stops, breath_2_stops, rpmv_2, window_time, append='nan')
    
    bior_1_sma      = reshape_as_sma(sma_stops, breath_1_stops, bior_1, window_time, append='nan')
    bior_2_sma      = reshape_as_sma(sma_stops, breath_2_stops, bior_2, window_time, append='nan')

    temp_1_sma      = reshape_as_sma(sma_stops, temp_1_stops, temp_1, window_time, append='nan')
    temp_2_sma      = reshape_as_sma(sma_stops, temp_1_stops, temp_1, window_time, append='nan')

    temp_sma        = (temp_1_sma + temp_2_sma)/2

    # MAKE DATAFRAMES

    sma_df      = np_2_df('sma',       sma_stops, sma, sma,            'value')
    steps_df    = np_2_df('steps',     sma_stops, sma, steps_sma,      'value')

    hr_df       = np_2_df('hr',        sma_stops, sma, hr_sma,        'value')
    sdnn_df     = np_2_df('sdnn',      sma_stops, sma, sdnn_sma,      'value')
    rmssd_df    = np_2_df('rmssd',     sma_stops, sma, rmssd_sma,      'value')
    lnrmssd_df  = np_2_df('lnrmssd',   sma_stops, sma, lnrmssd_sma,    'value')
    pnn50_df    = np_2_df('pnn50',     sma_stops, sma, pnn50_sma,      'value')
    
    bpm_df      = np_2_df('bpm',       sma_stops, sma, bpm_sma,        'value')
    bpmv_df     = np_2_df('bpmv',      sma_stops, sma, bpmv_sma,       'value')
    
    qt_df       = np_2_df('qt',         sma_stops, sma, qt_sma,         'value')

    br_1_df     = np_2_df('br_1',      sma_stops, sma, br_1_sma,      'value')
    brv_1_df    = np_2_df('brv_1',     sma_stops, sma, brv_1_sma,      'value')

    rpm_1_df    = np_2_df('rpm_1',     sma_stops, sma, rpm_1_sma,      'value')
    rpmv_1_df   = np_2_df('rpmv_1',    sma_stops, sma, rpmv_1_sma,      'value')

    br_2_df     = np_2_df('br_2',      sma_stops, sma, br_2_sma,      'value')
    brv_2_df    = np_2_df('brv_2',     sma_stops, sma, brv_2_sma,      'value')

    rpm_2_df    = np_2_df('rpm_2',     sma_stops, sma, rpm_2_sma,      'value')
    rpmv_2_df   = np_2_df('rpmv_2',    sma_stops, sma, rpmv_2_sma,      'value')
    
    bior_1_df   = np_2_df('bior_1',    sma_stops, sma, bior_1_sma,      'value')
    bior_2_df   = np_2_df('bior_2',    sma_stops, sma, bior_2_sma,      'value')
    
    temp_df     = np_2_df('temp',    sma_stops, sma, temp_sma,     'value')

    df_values = pd.DataFrame()
    df_values = df_values.append(sma_df)
    df_values = df_values.append(steps_df)
    df_values = df_values.append(hr_df)
    df_values = df_values.append(sdnn_df)
    df_values = df_values.append(rmssd_df)
    df_values = df_values.append(lnrmssd_df)
    df_values = df_values.append(pnn50_df)
    
    df_values = df_values.append(bpm_df)
    df_values = df_values.append(bpmv_df)
    
    df_values = df_values.append(qt_df)

    df_values = df_values.append(br_1_df)
    df_values = df_values.append(brv_1_df)
    df_values = df_values.append(rpm_1_df)
    df_values = df_values.append(rpmv_1_df)

    df_values = df_values.append(br_2_df)
    df_values = df_values.append(brv_2_df)
    df_values = df_values.append(rpm_2_df)
    df_values = df_values.append(rpmv_2_df)
    df_values = df_values.append(bior_1_df)
    df_values = df_values.append(bior_2_df)

    df_values = df_values.append(temp_df)

    # # EXPORT AS XLSX
    # with pd.ExcelWriter(path_save + al.end_user_ + '_Report_ts.xlsx') as writer:
    #     df_values.to_excel(writer, sheet_name='result')
        
    output = {}
    output['result']    = df_values

    return output

def excel_usable_split(al, window_time, path_save):
    
    sma_starts, sma_stops, sma = reshape_results(al.accx.times_, 
                                                     al.accx.sma_, 
                                                     window_time,
                                                     op='median')
    # RR intervals
    ecg_starts, ecg_stops, _ = reshape_results(al.ecg.times_rr_, 
                                                al.ecg.rr_, 
                                                window_time,
                                                op='median')
    # Breath rate THO
    breath_1_starts, breath_1_stops, _ = reshape_results(al.breath_1.times_rr_, 
                                                             al.breath_1.rr_, 
                                                             window_time,
                                                             op='median')
    # Breath rate ABD
    breath_2_starts, breath_2_stops, _ = reshape_results(al.breath_2.times_rr_, 
                                                             al.breath_2.rr_, 
                                                             window_time,
                                                             op='median')
    #  temp
    temp_1_starts, temp_1_stops, _ = reshape_results(al.temp_1.times_clean_, 
                                           al.temp_1.sig_clean_, 
                                           window_time,
                                           op='median')
    
    temp_2_starts, temp_2_stops, _ = reshape_results(al.temp_2.times_clean_, 
                                           al.temp_2.sig_clean_, 
                                           window_time,
                                           op='median')
    # ------------------------------------------------------------------
    #  USABLES
    sma_usable = usable_percentage(sma_starts, 
                                    sma_stops, 
                                    al.accx.times_, 
                                    al.accx.indicators_clean_, 
                                    al.accx.fs_)
    
    ecg_usable = usable_percentage(ecg_starts, 
                                    ecg_stops, 
                                    al.ecg.times_, 
                                    al.ecg.indicators_clean_3_, 
                                    al.ecg.fs_)
    
    breath_1_usable = usable_percentage(breath_1_starts, 
                                        breath_1_stops, 
                                        al.breath_1.times_, 
                                        al.breath_1.indicators_clean_bis_, 
                                        al.breath_1.fs_)
    
    breath_2_usable = usable_percentage(breath_2_starts, 
                                        breath_2_stops, 
                                        al.breath_2.times_, 
                                        al.breath_2.indicators_clean_bis_, 
                                        al.breath_2.fs_)
    
    
    temp_1_usable = usable_percentage(temp_1_starts, 
                                      temp_1_stops, 
                                      al.temp_1.times_, 
                                      al.temp_1.indicators_clean_, 
                                      al.temp_1.fs_)
    
    temp_2_usable = usable_percentage(temp_2_starts, 
                                      temp_2_stops, 
                                      al.temp_2.times_, 
                                      al.temp_2.indicators_clean_, 
                                      al.temp_2.fs_)
    
    # RESHAPE RESULTS AND USABLE AS SMA TO MERGE TIMESTAMPS
    
    rpm_1_usable_sma    = reshape_as_sma(sma_stops, breath_1_stops, breath_1_usable, window_time, append='0')
    rpm_2_usable_sma    = reshape_as_sma(sma_stops, breath_2_stops, breath_2_usable, window_time, append='0')
    bpm_usable_sma      = reshape_as_sma(sma_stops, ecg_stops, ecg_usable, window_time, append='0')
    temp_1_usable_sma   = reshape_as_sma(sma_stops, temp_1_stops, temp_1_usable, window_time, append='0')
    temp_2_usable_sma   = reshape_as_sma(sma_stops, temp_2_stops, temp_2_usable, window_time, append='0')
    
    # MAKE DATAFRAMES
    
    acc_usable_df       = np_2_df('acc',        sma_stops, sma, sma_usable,          'percentage')
    breath_1_usable_df  = np_2_df('breath_1',   sma_stops, sma, rpm_1_usable_sma,    'percentage')
    breath_2_usable_df  = np_2_df('breath_2',   sma_stops, sma, rpm_2_usable_sma,    'percentage')
    ecg_usable_df       = np_2_df('ecg',        sma_stops, sma, bpm_usable_sma,      'percentage')
    temp_1_usable_df    = np_2_df('temp_1',     sma_stops, sma, temp_1_usable_sma,   'percentage')
    temp_2_usable_df    = np_2_df('temp_2',     sma_stops, sma, temp_2_usable_sma,   'percentage')
    
    df_usable = pd.DataFrame()
    df_usable = df_usable.append(acc_usable_df)
    df_usable = df_usable.append(breath_1_usable_df)
    df_usable = df_usable.append(breath_2_usable_df)
    df_usable = df_usable.append(ecg_usable_df)
    df_usable = df_usable.append(temp_1_usable_df)
    df_usable = df_usable.append(temp_2_usable_df)
    
    # EXPORT AS XLSX
    with pd.ExcelWriter(path_save + al.end_user_ + '_Report_ts.xlsx') as writer:
        df_usable.to_excel(writer, sheet_name='usable')
        
    output = {}
    output['usable']    = df_usable
    
    return output

def excel_report(al, path_save=None, verbose=1, from_time=None, to_time=None,
                   flag_clean=False, flag_analyze=False, activity_types=None):
            
        filename = None
        df_activity = pd.DataFrame()
        
        sig_start_stop          = al.get_sig_start_stop(from_time=from_time, to_time=to_time, verbose=verbose)
        sig_duration            = al.get_sig_duration(from_time=from_time, to_time=to_time, time_format='m', verbose=verbose)
        disconnection           = al.get_disconnections(from_time=from_time, to_time=to_time, time_format='s', verbose=verbose)
        disconnection_detail    = al.get_disconnections_details(from_time=from_time, to_time=to_time)
        if flag_clean:
            sig_clean_stat      = al.get_sig_clean_stats(from_time=from_time, to_time=to_time, time_format='m', verbose=verbose)
        if flag_analyze:
            result = al.get_main_results(from_time=from_time, to_time=to_time, verbose=verbose)

        if from_time is None and to_time is None:
            from_time   = al.from_time_
            to_time     = al.to_time_            
            
        # Signal info: summary
        columns_summary = ['from', 'to', 'start', 'stop', 
                           'duration', 'duration_min', 'duration_max', 
                           'duration_median', 'duration_iqr',
                           'duration_mean', 'duration_std',
                           ]
        index_summary   = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                           'temp_1', 'temp_2', 'imp_1', 'imp_2', 'imp_3', 'imp_4']
        df_summary      = pd.DataFrame(columns=columns_summary, index=index_summary)
        
        df_summary['from']  = from_time
        df_summary['to']    = to_time
        
        # Disconnections
        columns_disconnection = ['from', 'to', 'start', 'stop', 
                                 'number', 'percentage', 
                                 'duration', 'duration_min', 'duration_max', 
                                 'duration_median', 'duration_iqr',
                                 'duration_mean', 'duration_std',
                                 ]
        index_disconnection = index_summary
        df_disconnection    = pd.DataFrame(columns=columns_disconnection, index=index_disconnection)
        
        df_disconnection['from']    = from_time
        df_disconnection['to']      = to_time
        
        # Disconnections details
        columns_disconnection_detail   = ['from', 'to', 'start', 'stop', 'duration']
        
        # Sig clean stats (usable signal)
        columns_usable = ['from', 'to', 'start', 'stop', 
                          'percentage', 'duration', 
                          'duration_min', 'duration_max', 
                          'duration_median', 'duration_iqr',
                          'duration_mean', 'duration_std', 'n_segments',
                          ]
        index_usable    = ['acc', 'breath_1', 'breath_2', 'ecg', 'temp_1', 'temp_2', 
                        'imp', 'ecg_breath', 'ecg_temp', 'ecg_breath_temp']
        df_usable       = pd.DataFrame(columns=columns_usable, index=index_usable)

        df_usable['from']   = from_time
        df_usable['to']     = to_time
            
        # results
        columns_result  = ['from', 'to', 'start', 'stop', 'signal', 'result']
        index_result    = ['n_steps', 'mean_activity', 'mean_activity_level', 
                           'rpm_tho', 'rpm_var_tho', 'amp_tho', 
                           'rpm_abd', 'rpm_var_abd', 'amp_abd',
                           'bpm', 'bpm_var', 'rr', 'hrv', 'pnn50', 'amp_ecg', 
                           'temp_right_mean', 'temp_right_std', 
                           'temp_right_var_mean', 'temp_right_var_std',
                           'temp_left_mean', 'temp_left_std', 
                           'temp_left_var_mean', 'temp_left_var_std']
        df_result       = pd.DataFrame(columns=columns_result, index=index_result)

        df_result['from']   = from_time
        df_result['to']     = to_time
        
        # Define keys 
        key = 'accx'
        keys_start_stop             = sig_start_stop[key].keys()
        keys_sig_duration           = sig_duration[key].keys()
        keys_disconnection          = disconnection[key].keys()
        key = 'acc'
        if flag_clean:
            keys_clean      = sig_clean_stat[key].keys()
        
        # Summary and Disconnections
        rows = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
                'temp_1', 'temp_2']
        for row in rows:
            if sig_start_stop[row]['start'] is not None:
                for col in keys_start_stop:
                    df_summary.loc[row, col] = datetime_np2str(np.array(sig_start_stop[row][col], 
                                                                        dtype='datetime64[s]'))
                    df_disconnection.loc[row, col] = datetime_np2str(np.array(sig_start_stop[row][col], 
                                                                              dtype='datetime64[s]'))
                    
                for col in keys_sig_duration:
                    df_summary.loc[row, col] = str(sig_duration[row][col])
                for col in keys_disconnection:
                    df_disconnection.loc[row, col] = str(disconnection[row][col])
        
        # Disconnections details
        df_disconnection_detail = pd.DataFrame(columns=columns_disconnection_detail)
        for row in disconnection_detail.keys():
            n_val = len(disconnection_detail[row]['start'])
            if n_val > 0:
                data = np.array([np.repeat(from_time, n_val), 
                                 np.repeat(to_time, n_val),
                                 disconnection_detail[row]['start'], 
                                 disconnection_detail[row]['stop'],
                                 disconnection_detail[row]['duration']])
                disco = pd.DataFrame(data.T, columns=columns_disconnection_detail)
                disco.index = np.repeat(row, n_val)
            else:
                disco = pd.DataFrame(columns=columns_disconnection_detail, index=[row])
                disco['from'] = from_time
                disco['to'] = to_time
            df_disconnection_detail = pd.DataFrame.append(df_disconnection_detail, disco)
                
        if len(df_disconnection_detail) > 0:
            df_disconnection_detail.start = df_disconnection_detail.start.astype('datetime64[s]')
            df_disconnection_detail.stop  = df_disconnection_detail.stop.astype('datetime64[s]')
                
        
        # Usable and artifacts
        if flag_clean:
            ref_clean = {'acc': 'accx', 'breath_1': 'breath_1', 'breath_2': 'breath_2',
                         'ecg': 'ecg', 'temp_1': 'temp_1', 'temp_2': 'temp_2', 
                         'imp': 'imp_1', 
                         'ecg_breath': 'ecg', 'ecg_temp': 'ecg', 
                         'ecg_breath_temp': 'ecg'}
            rows = ref_clean.keys()
            for row in rows:
                if sig_start_stop[ref_clean[row]]['start'] is not None:
                    for col in keys_start_stop:
                        df_usable.loc[row, col] = datetime_np2str(np.array(sig_start_stop[ref_clean[row]][col], 
                                                                           dtype='datetime64[s]'))
                
                    for col in keys_clean:
                        df_usable.loc[row, col] = str(sig_clean_stat[row][col])

        # Results
        ref_results     = {'n_steps': 'acc', 'mean_activity': 'acc', 'mean_activity_level': 'acc', 
                           'rpm_tho': 'breath_1', 'rpm_var_tho': 'breath_1', 
                           'rpm_abd': 'breath_2', 'rpm_var_abd': 'breath_2', 
                           'amp_tho': 'breath_1', 'amp_abd': 'breath_2', 
                           'amp_ecg': 'ecg', 'bpm': 'ecg', 'bpm_var': 'ecg', 
                           'rr': 'ecg', 'hrv': 'ecg', 'pnn50': 'ecg', 
                           'temp_right_mean': 'temp_1', 'temp_right_std': 'temp_1', 
                           'temp_right_var_mean': 'temp_1', 'temp_right_var_std': 'temp_1', 
                           'temp_left_mean': 'temp_2', 'temp_left_std': 'temp_2', 
                           'temp_left_var_mean': 'temp_2', 'temp_left_var_std': 'temp_2'}
        ref_results2    = {'n_steps': 'n_steps', 'mean_activity': 'mean_activity', 
                           'mean_activity_level': 'mean_activity_level', 
                           'rpm_tho': 'rpm', 'rpm_var_tho': 'rpm_var',
                           'rpm_abd': 'rpm', 'rpm_var_abd': 'rpm_var', 
                           'amp_tho': 'peaks_amps_mv', 'amp_abd': 'peaks_amps_mv', 
                           'amp_ecg': 'peaks_amps_mv', 'bpm': 'bpm', 
                           'bpm_var': 'bpm_var', 'rr': 'rr', 'hrv': 'hrv', 
                           'pnn50': 'pnn50', 
                           'temp_right_mean': 'mean', 'temp_right_std': 'std',
                           'temp_right_var_mean': 'var_mean', 'temp_right_var_std': 'var_std',
                           'temp_left_mean': 'mean', 'temp_left_std': 'std',
                           'temp_left_var_mean': 'var_mean', 'temp_left_var_std': 'var_std'}
        rows = ref_results.keys()
        if flag_analyze:
            for row in rows:
                if sig_start_stop[ref_clean[ref_results[row]]]['start'] is not None:
                    for col in keys_start_stop:
                        df_result.loc[row, col] = datetime_np2str(np.array(sig_start_stop[ref_clean[ref_results[row]]][col],
                                                                           dtype='datetime64[s]'))
                        df_result.loc[row, 'signal'] = ref_results[row]
                        df_result.loc[row, 'result'] = datetime_np2str(result[ref_results[row]][ref_results2[row]])
            
        # Activities
        if activity_types is not None:
            df_activity = pd.Series(activity_types, name='Activity')
            
        if path_save is not None:
            if os.path.exists(path_save):
                filename = path_save + al.end_user_ + '_' + al.xls_report_name_
                with pd.ExcelWriter(filename) as writer:
                    df_summary.to_excel(writer, sheet_name='summary')
                    df_disconnection.to_excel(writer, sheet_name='disconnection')
                    df_disconnection_detail.to_excel(writer, sheet_name='disconnection_detail')
                    if flag_clean:
                        df_usable.to_excel(writer, sheet_name='usable')
                        # df_artifact.to_excel(writer, sheet_name='artifact')
                    if flag_analyze:
                        df_result.to_excel(writer, sheet_name='result')
                    if activity_types is not None:
                        df_activity.to_excel(writer, sheet_name='activity')
                print('---------------------------------------------------------')
                print('SIGNAL REPORT')
                print(al.end_user_ + '_' + al.xls_report_name_, 'saved in', path_save)
            else:
                raise NameError('path_save folder does not exist')
        
        output = {}
        output['summary']               = df_summary
        output['disconnection']         = df_disconnection
        output['disconnection_detail']  = df_disconnection_detail
        output['usable']                = df_usable
        output['result']                = df_result
        output['activity']              = df_activity
        output['filename']              = filename
        
        return output