import numpy as np
import scipy.signal


#### filter used for both
def low_high_pass_f(input_sig, f_sig, filter_order, w_c, type_f):
    """
    low_high_pass_f using butterworth filter
    Input: input_sig, f_sig, filter_order, w_c, type_f
    input_sig : signal to filter
    f_sig : signal sampling frequency
    filter_order : filter order
    w_c : cut off frequency
    type_f : "low" if low pass, "high" if high pass
    Output: output_sig
    """
    coeff_b, coeff_a = scipy.signal.butter(filter_order, w_c/(f_sig/2.),
                                          type_f)
    
    #### sans padding
    output_sig = scipy.signal.filtfilt(coeff_b, coeff_a, input_sig)
    return output_sig #[padwidth:len(input_sig)-padwidth]

    #### with a padding to limit effects on the sides
    #padwidth = min(50, len(input_sig))
    #input_sig = np.pad(input_sig, pad_width = padwidth, mode='reflect')
    #return output_sig[padwidth:len(input_sig)-padwidth]

#### Breath

def filter_breath_scipy(sig, fs):
    """ Filter breath signal

    Parameters
    ------------
    sig: breath signal
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    
    if len(sig) > fs:
        sig_filt = low_high_pass_f(sig, fs, 6, 1, 'low')
        sig_filt = low_high_pass_f(sig_filt, fs, 4, 0.12, 'high')
    else:
        sig_filt = sig - np.mean(sig) #[0]*len(sig) # sig # moyenne de signal (segment)

    return sig_filt

def filter_breath_unwrap(sig, fs):
    """ Filter breath signal

    Parameters
    ------------
    sig: breath signal

    Returns
    ------------
    Filtered signal

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_breath_scipy(seg, fs)
        sig_filt_s.append(sig_filt)

    return sig_filt_s


# sig_filt = filter_breath_unwrap(sig, fs)

#### ECG
def filter_ecg_scipy(sig, fs, low_order=10, high_order=4):
    """ Filter ECG signal

    Parameters
    ------------
    sig: ECG signal (list)
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal

    """
    
    sig_filt = low_high_pass_f(sig, fs, low_order, 40, 'low')
    sig_filt = low_high_pass_f(sig_filt, fs, high_order, 1, 'high')

    return sig_filt


def filter_ecg_scipy_unwrap(sig, fs):
    """ Filter ECG signal

    Parameters
    ------------
    sig: ECG signal (list of list)
    fs: signal's sampling frequency

    Returns
    ------------
    Filtered signal (list of list)

    """
    sig_filt_s = []
    for seg in sig:
        sig_filt = filter_ecg_scipy(seg, fs)
        sig_filt_s.append(sig_filt)

    return sig_filt_s
