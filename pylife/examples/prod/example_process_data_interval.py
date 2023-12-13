# Path pylife
path_pylife = 'C:/Users/aterman/OneDrive - Passage innovation/Chronolife/pylife_folder'
#/ path_pylife ='C:/Users/blandrieu/OneDrive - Passage innovation/Documents/GitHub/'

# Path file for API ids
<<<<<<< HEAD
path_ids    = 'C:/Users/aterman/OneDrive - Passage innovation/Chronolife/pylife_folder/api/prod'
#/ path_ids = 'C:/Users/blandrieu/OneDrive - Passage innovation/Documents/GitCode/api/'
=======
#path_ids    = 'C:/Users/aterman/OneDrive - Passage innovation/Chronolife/pylife_folder/api/prod'
path_ids = 'C:/Users/blandrieu/OneDrive - Passage innovation/Documents/GitCode/api/preprod/'
>>>>>>> 545ac0ace607eb7b831ee8eb5f19d1b8784eee1d


import sys
sys.path.append(path_pylife)
from pylife.env import get_env
DEV = get_env()
from pylife.datalife import Apilife
from pylife.process_functions import process_data_interval
from time import time
import matplotlib.pyplot as plt

# %% Load data

# Request info
# from_time       = "2023-01-24 14:00:00"
# to_time         = "2023-01-24 14:50:00"
# time_zone       = 'CET'
# end_user        = "6Bvsyh" #6Bvsyh3h4nFM

from_time       = "2023-05-04 14:20:00" # "2022-10-21 09:00:00"
to_time         = "2023-05-04 15:50:00" # "2022-10-21 09:05:00"
time_zone       = 'UTC'
end_user        = "5Nwwut" # 7k6Hs3 3JLqh6

dict_params         = {}
dict_params['api']  = {'path_ids': path_ids, 'end_user': end_user, 'api_version': 2,
                       'from_time': from_time, 'to_time': to_time, 
                       'time_zone': time_zone, 'device_model': 't-shirt',
                       'flag_acc': True, 'flag_breath': True, 
                       'flag_ecg': True, 'flag_temp': True}
al = Apilife(dict_params['api'])




print('User : ', end_user)

begin = time() 
al.get()
end = time()
print('Time taken for get:',round((end-begin)/60,2), 'min')
   
begin = time() 
al.parse()
end = time()
print('Time taken for parse:',round((end-begin)/60,2), 'min')
   
begin = time() 
al.filt()
end = time()
print('Time taken for filt:',round((end-begin)/60,2), 'min')
   
begin = time() 
al.clean()
end = time()
print('Time taken for clean:',round((end-begin)/60,2), 'min')
   
begin = time() 
al.analyze()
end = time()
print('Time taken for analyze:',round((end-begin)/60,2), 'min')
     
plt.close('all')
# Figure 1
plt.figure()
#al.ecg.show()
#al.ecg.show(on_sig = 'raw')
al.ecg.show(on_sig = 'clean')
#al.accx.show()
plt.show()

# Figure 2
plt.figure()
al.breath_1.show()
al.breath_1.show(on_sig = 'clean')
plt.show()

# Figure 3
plt.figure()
al.breath_2.show()
al.breath_2.show(on_sig = 'filt')
plt.show()

# Figure 4
plt.figure()
al.temp_1.show()
al.temp_2.show()


# %% Define parameters
dict_params              = {}
dict_params['accx']      = {'times':        al.accx.times_, 
                            'sig':          al.accx.sig_, 
                            'fs':           al.accx.fs_, 
                            'fw_version':   al.accx.fw_version_}

dict_params['accy']      = {'times':        al.accy.times_, 
                            'sig':          al.accy.sig_, 
                            'fs':           al.accy.fs_, 
                            'fw_version':   al.accy.fw_version_}

dict_params['accz']      = {'times':        al.accz.times_, 
                            'sig':          al.accz.sig_, 
                            'fs':           al.accz.fs_, 
                            'fw_version':   al.accz.fw_version_}

dict_params['breath_1']  = {'times':        al.breath_1.times_, 
                            'sig':          al.breath_1.sig_, 
                            'fs':           al.breath_1.fs_,
                            'fw_version':   [al.breath_1.fw_version_]}

dict_params['breath_2']  = {'times':        al.breath_2.times_, 
                            'sig':          al.breath_2.sig_, 
                            'fs':           al.breath_2.fs_, 
                            'fw_version':   al.breath_2.fw_version_}

dict_params['ecg']       = {'times':        al.ecg.times_, 
                            'sig':          al.ecg.sig_, 
                            'fs':           al.ecg.fs_, 
                            'fw_version':   al.ecg.fw_version_}

dict_params['temp_1']    = {'times':        al.temp_1.times_, 
                            'sig':          al.temp_1.sig_, 
                            'fs':           al.temp_1.fs_, 
                            'fw_version':   al.temp_1.fw_version_}

dict_params['temp_2']    = {'times':        al.temp_2.times_, 
                            'sig':          al.temp_2.sig_, 
                            'fs':           al.temp_2.fs_,
                            'fw_version':   al.temp_2.fw_version_}

# % Process data
dict_result = process_data_interval(dict_params)

# %%
print("======================================================================")
print(from_time, '==>' ,to_time)
print()
print("======================================================================")
print("=========================== Activity =================================")
print('steps_number             ', dict_result["steps_number"],'   class',  type(dict_result["steps_number"]))
print('averaged_activity        ', dict_result["averaged_activity"],'   class',  type(dict_result["averaged_activity"]))

print()
print("======================================================================")
print("====================== Breath indicators =============================")
print('respiratory_rate_quality_index quality index thoracic', dict_result["respiratory_rate_quality_index"],'   class',  type( dict_result["respiratory_rate_quality_index"]))
print('respiratory_rate thoracic moyenne en brpm            ', dict_result["breath_1_brpm"], 'CPM','   class',  type(dict_result["breath_1_brpm"]))
print('respiratory_rate_var thoracic en s                   ', dict_result["breath_1_brv"],'   class',  type( dict_result["breath_1_brv"]))
print('respiratory_rate abdominal moyenne en brpm           ', dict_result["breath_2_brpm"], 'CPM','   class',  type(dict_result["breath_2_brpm"]))
print('respiratory_rate_var abdominal en s                  ', dict_result["breath_2_brv"],'   class',  type(dict_result["breath_2_brv"]))

print()
print("======================================================================")
print('breath_1_peaks                                     ', dict_result["breath_1_peaks"][0][:3],'   class',  type(dict_result["breath_1_peaks"][0][0]))
print('breath_2_peaks                                     ', dict_result["breath_2_peaks"][0][:3],'   class',  type(dict_result["breath_2_peaks"][0][0]))
print('breath_1_valleys                                   ', dict_result["breath_1_valleys"][0][:3],'   class',  type(dict_result["breath_1_valleys"][0][0]))
print('breath_2_valleys                                   ', dict_result["breath_2_valleys"][0][:3],'   class',  type(dict_result["breath_2_valleys"][0][0]))
print('breath_1_inspi_over_expi                           ', dict_result["breath_1_inspi_over_expi"][0][:3],'%   class',  type(dict_result["breath_1_inspi_over_expi"][0][0]))
print('breath_2_inspi_over_expi                           ', dict_result["breath_2_inspi_over_expi"][0][:3],'%   class',  type(dict_result["breath_2_inspi_over_expi"][0][0]))

print()
print("======================================================================")
print("====================== Cardio indicators =============================")
print('heartbeat quality index                      ', dict_result["heartbeat_quality_index"],'   class',  type(dict_result["heartbeat_quality_index"]))
print('hrv quality index                            ', dict_result["HRV_quality_index"],'   class',  type(dict_result["HRV_quality_index"]))
print('heartbeat                                    ', dict_result["heartbeat"], 'BPM','   class',  type(dict_result["heartbeat"]))
print('HRV                                          ', dict_result["HRV"],'   class',  type(dict_result["HRV"]))
print('RR interval                                  ', dict_result["rr_interval"], 'ms','   class',  type(dict_result["rr_interval"]))
print('sdnn                                         ', dict_result["sdnn"],'   class',  type(dict_result["sdnn"]))
print('rmssd                                        ', dict_result["rmssd"],'   class',  type(dict_result["rmssd"]))
print('lnrmssd                                      ', dict_result["lnrmssd"],'   class',  type(dict_result["lnrmssd"]))
print('pnn50                                        ', dict_result["pnn50"], '%','   class',  type(dict_result["pnn50"])) # a percentage
print('qt_length_median_corrected                   ', dict_result["qt_length_median_corrected"], 'ms','   class',  type(dict_result["qt_length_median_corrected"])) # corrected is made with the supression of obvious outliers
print('qt_c_framingham_per_seg                      ', dict_result["qt_c_framingham_per_seg"][:3], 'ms corrected to BPM','   class',  type(dict_result["qt_c_framingham_per_seg"][0]))
print()
print("======================================================================")
print('5 first q_start_time of first clean segment ', dict_result["q_start"][0][:3],'   class',  type(dict_result["q_start"][0][0]))
print()
print('5 first r_peak_time of first clean segment  ', dict_result["r_peak"][0][:3],'   class',  type(dict_result["r_peak"][0][0]))
print()
print('5 first t_stop_time of first clean segment  ', dict_result["t_stop"][0][:3],'   class',  type(dict_result["t_stop"][0][0]))


print()
print("======================================================================")
print("========================= Temperature ================================")
print('averaged_temp_1          ', dict_result["averaged_temp_1"], '°C', '   class',  type(dict_result["averaged_temp_1"]))
print('averaged_temp_2          ', dict_result["averaged_temp_2"], '°C', '   class',  type(dict_result["averaged_temp_1"]))


# %% Plot

# Figure 5 -> ECG
segment_i = 3
plt.figure()
plt.plot(dict_result["ecg_filtered"][segment_i])
    
for p in dict_result["r_peak"][segment_i]:
    plt.axvline(p, c='r', alpha = 0.2)
plt.title('ECG filt and peaks computed on clean sig')
plt.show()

# Figure 6 -> QT interval
plt.figure()
plt.plot(dict_result["ecg_filtered"][segment_i])
for p in dict_result["q_start"][segment_i]:
    plt.axvline(p, c='g', alpha = 0.2, linestyle = '--')
for p in dict_result["r_peak"][segment_i]:
    plt.axvline(p, c='r', alpha = 0.2)
for p in dict_result["t_stop"][segment_i]:
    plt.axvline(p, c='b', alpha = 0.2, linestyle = '--')
plt.title('QT segments with Q in green, R in red and the end of the T wave in bleue')
plt.show()



# %% Figure 7 -> Breath 2
segment_i = 0
plt.figure()
plt.plot(dict_result["breath_2_filtered"][segment_i])

for p in dict_result["breath_2_peaks"][segment_i]:
    plt.axvline(p, c='r', alpha = 0.2)

for v in dict_result["breath_2_valleys"][segment_i]:
    plt.axvline(v, c='g', alpha = 0.2)

plt.title('Respi filt and peaks and valleys ')
plt.show()

# Figure 8 -> Breath 2 inpi over expi
plt.figure()
plt.plot(dict_result["breath_2_inspi_over_expi"][segment_i])
    
plt.title('Inspiration over expiration intervals of Breath 2')
plt.show()
    
