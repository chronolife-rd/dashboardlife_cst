from pylife.env import get_env
DEV = get_env()
# --- Add imports for DEV env
if DEV:
    pass
# import numpy as np
#import datetime
from pylife.siglife import Acceleration_x
from pylife.siglife import Acceleration_y
from pylife.siglife import Acceleration_z
from pylife.siglife import Accelerations
from pylife.siglife import Breath_1
from pylife.siglife import Breath_2
from pylife.siglife import Breaths
from pylife.siglife import ECG
from pylife.siglife import Temperature_1
from pylife.siglife import Temperature_2
from pylife.siglife import Temperatures
from pylife.datalife import Datalife
#import math
import numpy as np
import datetime

def process_data_interval(dict_params):
    """ Process data for a given end_user in a time interval 
   input dict_params = {
       "ecg":       dict of sig info,
       "breath_1":  dict of sig info,
       "breath_2":  dict of sig info,
       "temp_1":    dict of sig info,
       "temp_2":    dict of sig info,
       "accx":      dict of sig info,
       "accy":      dict of sig info,
       "accz":      dict of sig info
       }   
       and dict of sig info = {times, sig, fs, fw_version}
   
    returns dict_results
   
    """
    
    keys = ['accx', 'accy', 'accz', 'breath_1', 'breath_2', 'ecg', 
            'temp_1', 'temp_2']
    for key in keys:
        if key not in dict_params.keys():
            dict_params[key]  = {
                                    'times':        [], 
                                    'sig':          [], 
                                    'fs':           [],
                                    'fw_version':   []
                                }
        
    # Define parameters for init
    dict_params['init'] = {'flag_acc': True, 'flag_breath': True, 
                           'flag_ecg': True, 'flag_temp': True}
    
    # Init class for processing
    dl = Datalife(dict_params['init'])
    dl.init()
    dl.check(dict_params['init'])
    dl.assign(dict_params['init'])
    
    # Define signal classes
    dl.accx     = Acceleration_x(dict_params['accx'])
    dl.accy     = Acceleration_y(dict_params['accy'])
    dl.accz     = Acceleration_z(dict_params['accz'])
    dl.accs     = Accelerations(dl.accx, dl.accy, dl.accz)
    dl.breath_1 = Breath_1(dict_params['breath_1'])
    
    dl.breath_2 = Breath_2(dict_params['breath_2'])
    dl.breaths  = Breaths(dl.breath_1, dl.breath_2)
    dl.ecg      = ECG(dict_params['ecg'])
    dl.temp_1   = Temperature_1(dict_params['temp_1'])
    dl.temp_2   = Temperature_2(dict_params['temp_2'])
    dl.temps    = Temperatures(dl.temp_1, dl.temp_2)
    
    # Process
    dl.filt()
    dl.clean()
    dl.analyze()
    
    # Store results
    dict_result = {}
    
    if not dl.accs.is_empty_:
        print('accx')
        if isinstance(dl.accs.mean_activity_level_, float):
            averaged_activity = round(dl.accs.mean_activity_)
        else:
            averaged_activity = 0
            
        dict_result["steps_number"]             = dl.accs.n_steps_
        dict_result["averaged_activity"]        = averaged_activity
        dict_result["activity_level"]           = [[round(respj,4) for respj in respij] for respij in dl.accs.activity_level_] #[item.tolist() for item in (dl.accs.activity_level_)]
        dict_result["activity_level_times"]     = [times[0].astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc) for times in dl.accx.times_]
        
    if not dl.breath_1.is_empty_:
        print('breath_1')
        brpm = None
        brv_s = None
        if len(dl.breath_1.rpm_) > 0:
            brpm      = dl.breath_1.rpm_[0]     # RPM: respirations per minute "how many cycles in a minute?"
            brpm      = int(round(brpm))
           
        if len(dl.breath_1.rpm_var_) > 0:       
            brv_s = dl.breath_1.rpm_var_s_[0]   # "Variation of lengh breath cycles (in seconds)"
            brv_s = int(round(brv_s))
         
        def get_first_time_value(times):
            return [time[0].astype(datetime.datetime).replace(tzinfo=datetime.timezone.utc) if len(time)>0 else [] for time in times ]

        #/ times, br_filtered, indic   = dl.breath_1.select_on_sig('filt')
        
        dict_result["breath_1_filtered"]                = [[round(respj,4) for respj in respij] for respij in dl.breath_1.sig_clean_] #[item.tolist() for item in (dl.breath_1.sig_clean_)]           # This is Sig clean !!!!
        dict_result["breath_1_filtered_times"]          = get_first_time_value(dl.breath_1.times_clean_)                 # breath_1_filtered_times gets the first timestamp of each signal interval, because after it's parsed on servers ? 
        
        dict_result["respiratory_rate_quality_index"]   = 0 if brpm is None else 1                                       # Should be renamed to breath_1_brpm_quality_index ? 
        dict_result["breath_quality_index"]             = (dl.breath_1.indicators_seconds_).tolist()                     # Should be renamed to breath_1_quality_index ?
        dict_result["breath_quality_index_frequency"]   = dl.breath_1.indicators_frequency_                              # Should be renamed to breath_1_quality_index_frequency?

        dict_result["respiratory_rate"]                 = brpm                                                           # This should be deleted at the next release !!!!
        dict_result["breath_1_brpm"]                    = brpm                          ####
        dict_result["breath_1_brv"]                     = brv_s                         #### 
        
        print('breath_1 new indic')
        insp_expi = None
        if len(dl.breath_1.inspi_over_expi_) > 0:
            insp_expi = dl.breath_1.inspi_over_expi_     # RPM: respirations per minute "how many cycles in a minute?"
            insp_expi      = [[int(round(inspex*100)) for inspex in insp_expij] for insp_expij in insp_expi]

        dict_result["breath_1_peaks"]                   = [[int(itemj) for itemj in itemij] for itemij in dl.breath_1.peaks_]             ####
        dict_result["breath_1_valleys"]                 = [[int(itemj) for itemj in itemij] for itemij in dl.breath_1.valleys_]                       ####
        dict_result["breath_1_inspi_over_expi"]         = insp_expi  ####
    
    if not dl.breath_2.is_empty_:
        print('breath_2')
        brpm = None
        brv_s = None
        if len(dl.breath_2.rpm_) > 0:
            brpm      = dl.breath_2.rpm_[0]          # RPM: respirations per minute "how many cycles in a minute?"
            brpm      = int(round(brpm))
           
        if len(dl.breath_2.rpm_var_) > 0:      
            brv_s = dl.breath_2.rpm_var_s_[0]        # "Variation of lengh breath cycles (in seconds)"
            brv_s = int(round(brv_s))

        dict_result["breath_2_filtered"]                = [item.tolist() for item in (dl.breath_2.sig_clean_)]           # This is Sig clean !!!!
        dict_result["breath_2_filtered_times"]          = get_first_time_value(dl.breath_2.times_clean_)                 # breath_2_filtered_times gets the first timestamp of each signal interval, because after it's parsed on servers ? 

        dict_result["respiratory_rate_2_quality_index"] = 0 if brpm is None else 1                                       # Should be renamed to breath_2_brpm_quality_index ? 

        dict_result["breath_2_brpm"]                    = brpm                          ####
        dict_result["breath_2_brv"]                     = brv_s                         #### 
        
        insp_expi = None
        if len(dl.breath_2.inspi_over_expi_) > 0:
            insp_expi = dl.breath_2.inspi_over_expi_     # RPM: respirations per minute "how many cycles in a minute?"
            insp_expi      = [[int(round(inspex*100)) for inspex in insp_expij] for insp_expij in insp_expi]
        print('breath_2 new indic')
        dict_result["breath_2_peaks"]                   = [[int(itemj) for itemj in itemij] for itemij in dl.breath_2.peaks_]        ####
        dict_result["breath_2_valleys"]                 = [[int(itemj) for itemj in itemij] for itemij in dl.breath_2.valleys_ ]     ####
        dict_result["breath_2_inspi_over_expi"]         = insp_expi  #### 

        

    if not dl.ecg.is_empty_:
        print('ecg')
        hr = None
        if len(dl.ecg.bpm_) > 0:
            hr      = dl.ecg.bpm_[0]
            hr      = int(round(hr))
            
        hrv = None
        if len(dl.ecg.bpm_var_ms_) > 0:
            hrv      = dl.ecg.bpm_var_ms_[0]
            hrv      = int(round(hrv))
            
        rr = None
        if len(dl.ecg.bpm_ms_) > 0:
            rr      = dl.ecg.bpm_ms_[0]
            rr      = int(round(rr))
            
        sdnn = None
        if len(dl.ecg.sdnn_) > 0:
            sdnn    = dl.ecg.sdnn_[0]
            sdnn    = int(round(sdnn))
            
        rmssd = None
        if len(dl.ecg.rmssd_) > 0:
            rmssd   = dl.ecg.rmssd_[0]
            rmssd   = int(round(rmssd))
            
        lnrmssd = None
        if len(dl.ecg.lnrmssd_) > 0:
            lnrmssd = dl.ecg.lnrmssd_[0]
            lnrmssd = int(round(lnrmssd))
            
        pnn50 = None
        if len(dl.ecg.pnn50_) > 0:
            pnn50   = dl.ecg.pnn50_[0]
            pnn50   = int(round(pnn50*100)) # it's a percentage so it need be multiplied by 100 to keep making sense
            
        r_peak = None
        if len(dl.ecg.peaks_) > 0:
            r_peak =  dl.ecg.peaks_  
               
        print('ecg new indic')
        q_start = None
        if len(dl.ecg.q_start_time_) > 0:
            q_start   =  dl.ecg.q_start_index_ 
            
        t_stop = None
        if len(dl.ecg.t_stop_time_) > 0:
            t_stop   = dl.ecg.t_stop_index_ 
            
        qt_length_median_corrected = None
        if len(dl.ecg.qt_length_median_corrected_) > 0:
            qt_length_median_corrected   = dl.ecg.qt_length_median_corrected_[0]
            qt_length_median_corrected   = int(round(qt_length_median_corrected))
            
        qt_c_framingham_per_seg = None
        if len(dl.ecg.qt_c_framingham_per_seg_) > 0:
            #print(dl.ecg.qt_c_framingham_per_seg_)
            qt_c_framingham_per_seg   = dl.ecg.qt_c_framingham_per_seg_[0]#.tolist() 
            #print('type(dl.ecg.qt_c_framingham_per_seg_', type(dl.ecg.qt_c_framingham_per_seg_))
            if  (type(qt_c_framingham_per_seg)==list) :
                qt_c_framingham_per_seg = [int(itemj) if itemj>0 else itemj for itemj in qt_c_framingham_per_seg]  # list with one value for each clean segment
                #qt_c_framingham_per_seg   = qt_c_framingham_per_seg.tolist() 
            else :
                qt_c_framingham_per_seg = [int(qt_c_framingham_per_seg) if qt_c_framingham_per_seg>0 else qt_c_framingham_per_seg]
        
        print('ecg dict fill')
        dict_result["heartbeat"]                    = hr
        dict_result["HRV"]                          = hrv
        dict_result["ecg_filtered"]                 = [item.tolist() for item in (dl.ecg.sig_clean_)]                    # This is Sig clean !!!!
        dict_result["ecg_filtered_times"]           = get_first_time_value(dl.ecg.times_clean_)                          # ecg_filtered_times gets the first timestamp of each signal interval, because after it's parsed on servers ?
        dict_result["heartbeat_quality_index"]      = 0 if hrv is None else 1 
        dict_result["ecg_quality_index"]            = (dl.ecg.indicators_seconds_).tolist()
        dict_result["HRV_quality_index"]            = 0 if hrv is None else 1
                                                 
        dict_result["ecg_quality_index_frequency"]  = dl.ecg.indicators_frequency_
        dict_result["rr_interval"]                  = rr
        dict_result["sdnn"]                         = sdnn
        dict_result["rmssd"]                        = rmssd
        dict_result["lnrmssd"]                      = lnrmssd
        dict_result["pnn50"]                        = pnn50
        
        dict_result["q_start"]                      = [[int(itemj) for itemj in itemij] for itemij in q_start ] # lists of indexes for each clean segment, the index number restart from 0 at each segment  ####
        dict_result["r_peak"]                       = [[int(itemj) for itemj in itemij] for itemij in r_peak ]  ####
        dict_result["t_stop"]                       = [[int(itemj) for itemj in itemij] for itemij in t_stop ]  ####
        dict_result["qt_length_median_corrected"]   = qt_length_median_corrected ####
        dict_result["qt_c_framingham_per_seg"]      = qt_c_framingham_per_seg
        
    if not dl.temp_1.is_empty_:
        dict_result["averaged_temp_1"] = int(round(dl.temp_1.mean_*1e4))/1e2
        
    if not dl.temp_2.is_empty_:
        dict_result["averaged_temp_2"] = int(round(dl.temp_2.mean_*1e4))/1e2
        
    dict_result["is_worn"] = int(1) if dl.is_worn_ else int(0)
    
    
    # print every key in the dictionary to provide logs for debug 
    for key in dict_result:
        res = dict_result[key]
        if (type(res)==list)|(isinstance(res, np.ndarray)):
            if (type(res[0])==list)|(isinstance(res[0], np.ndarray)) :
                maxview = min(3, len(res[0]))
                print(key, res[0][:maxview])
            else :
                maxview = min(3, len(res))
                print(key, res[:maxview])
        else :
            print(key, res)
            
            
                    
        
    
    return dict_result

#%%

# for key in dict_result:
#     res = dict_result[key]
#     if (isinstance(res, np.int64)):
#         print(key)
#         if (isinstance(res[0],  np.int64)) :
#             print(key)
# a = np.int64(4)
# isinstance(a, np.int64)
