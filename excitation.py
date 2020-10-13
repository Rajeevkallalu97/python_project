'''
This script consists of two parts, the function excitation that creates the 
excitation potential and records the resulting current and the funtion analysis
which analyses the data by taking the upper envelope of the second harmonic. 
The function of analysis is a quick and dirty version of the matlab script 
And_AC_Volt and uses a butterworth bandpass to acheive similar if not identical 
results to the fourier transform.

All of the required packages can be installed from PIP with the exception of
intersection which can be found here https://github.com/sukhbinder/intersection/


'''


import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.core.numeric import binary_repr
import sounddevice as sd
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from intersection import intersection
import peakutils
from datetime import datetime
import ntpath

date = datetime.now().strftime("%d-%m-%Y_%I-%M_%p")

date_time = str(datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))
path = os.getcwd()
newpath = str(path) + '/' + 'Electrobe_output_'+ str(datetime.now().strftime("%d_%m_%Y"))
if not os.path.exists(newpath):
    os.makedirs(newpath)
    


################## Settings - Change as you like #######################
#These values are called in temp_call function, while calling the function 
#if any null value is passed default value will be used
harmonic = 2
amplitude = 0.06 # This is as a fraction of the maximum amplitude 1 = 2.96 V 
stable = 2.0 #stable duration in seconds
sample_rate = 44100 #Doesn't necessarily work for other sample rates
duration = 8.0 # recording duration in seconds
frequency = 115.0 # Frequency
v1 = 0.0 #Stable "Voltage" actually a fraction of max output positive values only 
v2 = 0.0 #Recording Start "Voltage" actually a fraction of max output 0.1 = ~0.045V
v3 = 0.7 #Recording stop "Voltage" actually a fraction of max output 1.0 = ~1.265
filename_string = ""
filename = ""
filename_data = ""


def temp_call_citizen(stable_fn, sample_rate_fn, v1_fn, v2_fn, v3_fn, frequency_fn, duration_fn):
    global  stable, sample_rate, duration, frequency, v1, v2, v3, filename_string, filename, filename_data
    stable = stable_fn
    sample_rate = sample_rate_fn
    frequency = frequency_fn
    duration = duration_fn
    v1 = v1_fn
    v2 = v2_fn
    v3 = v3_fn
    filename_string = str('50ppm_'+ '_Amp_'+ str(amplitude) + '_stable_' +str(stable) + 'recording_'+ str(duration)+ '_freq_' + '_v1_'+ str(v1) + '_v2_' + str(v2) +'_v3_' +str(v3))#file identifier in quotes
    filename = str(newpath+'/' +filename_string + '_'+ date + '.wav') 
    filename_data = str(newpath+'/' +filename_string + '_'+ date + '.data') 
    wine = '300ppm_s3_run1'
    excitation(stable,0.06, sample_rate, v1, v2, v3, frequency, duration, filename, filename_string, filename_data)
    result = analysis(filename_data,wine) #uncomment to perform analysis as well as recording the potential
    return result

def temp_call_researcher(stable_func, sample_rate_func, v1_func, v2_func, v3_func, frequency_func, duration_func):
    global  stable, sample_rate, duration, frequency, v1, v2, v3, filename_string, filename, filename_data
    stable = stable_func
    sample_rate = sample_rate_func
    frequency = frequency_func
    duration = duration_func
    v1 = v1_func
    v2 = v2_func
    v3 = v3_func
   
    filename_string = str('50ppm_'+ '_Amp_'+ str(amplitude) + '_stable_' +str(stable) + 'recording_'+ str(duration)+ '_freq_' + '_v1_'+ str(v1) + '_v2_' + str(v2) +'_v3_' +str(v3))#file identifier in quotes
    filename = str(newpath+'/' +filename_string + '_'+ date + '.wav') 
    filename_data = str(newpath+'/' +filename_string + '_'+ date + '.data') 
    wine = '300ppm_s3_run1'
    excitation(stable,0.06, sample_rate, v1, v2, v3, frequency, duration, filename, filename_string, filename_data)
    result = analysis(filename_data,wine) #uncomment to perform analysis as well as recording the potential
    return result

def temp_call_researcher_amp(stable_fun,amplitude_fun, sample_rate_fun, duration_fun, frequency_fun,v1_fun, v2_fun, v3_fun):
    global amplitude, stable, sample_rate, duration, frequency, v1, v2, v3, filename_string, filename, filename_data
    amplitude = amplitude_fun
    stable = stable_fun
    sample_rate = sample_rate_fun
    frequency = frequency_fun
    duration = duration_fun
    v1 = v1_fun
    v2 = v2_fun
    v3 = v3_fun
   
    filename_string = str('50ppm_'+ '_Amp_'+ str(amplitude) + '_stable_' +str(stable) + 'recording_'+ str(duration)+ '_freq_' + '_v1_'+ str(v1) + '_v2_' + str(v2) +'_v3_' +str(v3))#file identifier in quotes
    filename = str(newpath+'/' +filename_string + '_'+ date + '.wav') 
    filename_data = str(newpath+'/' +filename_string + '_'+ date + '.data') 
    wine = '300ppm_s3_run1'

    excitation(stable, amplitude,sample_rate, v1, v2, v3, frequency, duration, filename, filename_string, filename_data)
    result = analysis(filename_data,wine) #uncomment to perform analysis as well as recording the potential
    return result




########################################################################
def excitation(stable, amplitude1,sample_rate, v1, v2, v3, frequency, duration, filename,filename_string, filename_data):

    global amplitude
    stable = float(stable)
    amplitude = float(amplitude1)
    sample_rate = int(sample_rate)
    v1 = float(v1)
    v2 = float (v2)
    v3 = float (v3)
    startTime = datetime.now()
    print('Generating waveforms...')
    filename = str(filename_string + '_'+ date + '.wav') 
    sramp = np.linspace(v1,v1,int(stable*sample_rate)) #ramp for stable period
    ramp = np.linspace(v2,v3,int(duration*sample_rate)) #ramp for excitation
    #stable duration
    xls = np.linspace(0, stable * 2 * np.pi, int(stable * sample_rate)) #Left channel wave form
    xrs = np.linspace(0, stable * 2 * np.pi, int(stable * sample_rate)) #Right Channel waveform
    
   
    s_left_channel = np.sin(frequency * xls)*amplitude 
    s_right_channel = np.sin(frequency * xrs + np.pi)*amplitude
    
    s_left_channel  -= sramp
    s_right_channel += sramp
    
    stable_waveform_stereo = np.vstack((s_left_channel, s_right_channel)).T #combine left and right channels
    #record duration 
    
    xl = np.linspace(0, duration * 2 * np.pi, int(duration * sample_rate))
    xr = np.linspace(0, duration * 2 * np.pi, int(duration * sample_rate))
    
    left_channel = amplitude*np.sin(frequency * xl)
    right_channel = amplitude*np.sin(frequency * xr + np.pi)
    
    left_channel -= ramp

    right_channel += ramp
    
    waveform_stereo = np.vstack((left_channel, right_channel)).T #combine left and right channels
    

    
    total_waveform = (np.append(stable_waveform_stereo, waveform_stereo, axis=0))
    print('Excitation potential generated in '+ str(datetime.now() - startTime))
    print('Now Recording current')
    rec_data = sd.playrec(total_waveform, sample_rate, channels=1)
    time.sleep(stable+duration)
    sd.stop()
    #for i in rec_data:
        #sd.play(i)

    print('Writing data to file')
    import scipy.io.wavfile as wf
    write_data = np.int16(total_waveform * 32767)
    wf.write(filename, sample_rate, write_data)

    #Save file
    #The header excitation_data is used to record the settings used
    
    excitation_data = str(str(stable) + ',' +  str(sample_rate) + ',' + str(v1) +','+ str(v2) +','+ str(v3) +','+ str(frequency) +','+ str(duration) +','+ str(filename))
    np.savetxt(filename_data, rec_data, delimiter=',' , header=excitation_data)
        
    #Microphone input graph
    fig, axs = plt.subplots(2)
    axs[0].plot(total_waveform)
    axs[0].set_title('Output Waveform')
    axs[1].plot(rec_data)
    axs[1].set_title('Microphone input')
    plt.plot(rec_data)
    plt.show()


    print('Writing data complete')
    print('Process completed in '+ str(datetime.now() - startTime))
    



def analysis(filename,wine):
    
    path = os.getcwd()
    newpath = str(path) + '/' + 'Electrobe_output_'+ str(datetime.now().strftime("%d_%m_%Y"))
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    file = ntpath.basename(filename)
    colnames = ['x','y']
    data = pd.read_csv(filename, sep=',',names=colnames, skiprows=1) # x is time, y is potential
    y = data['x']
    
    
    freq_pert = frequency               # Frequency perturbation 
    har_num = harmonic                  # harmonic to use (normally 2)
    
    blank_samples = 4000 # set first samples to zero
    y[0:blank_samples] = 0 
             
    
    sample_rate = 44100.0         # sample rate
    dt = 1/sample_rate            # Distance between the data points
    a = np.size(y);                   # Sample size
    n = np.arange(0,a)                    # Creating a column vector with 
    t = n*dt
    Fs = sample_rate; # sampling rate

    n2 = a # length of the signal
    k = n
    T = n2/Fs
    frq = k/T # two sides frequency range in Hz
    frq = frq[range(n2//2)] # one side frequency range
    Y = np.fft.fft(y)/n2 # fft computing and normalization
    Y = Y[range(n2//2)]
    fig,myplot = plt.subplots(2, 1)
    
    myplot[0].plot(t,y)
    myplot[0].set_xlabel('Time')
    myplot[0].set_ylabel('Current')
    myplot[1].plot(frq,abs(Y),'r') # plotting the spectrum
    myplot[1].set_xlabel('Freq (Hz)')
    myplot[1].set_ylabel('|Y(freq)|')
    plt.show()
    
    # To implement the bandpass filter, the signal is filtered in a range of 
    # plus/minus 2 Hz of the second harmonic frequency
    
    low = freq_pert*har_num -2 
    high = freq_pert*har_num +2
    
    # The butterworth bandpass filter requires scipy 1.21 or newer
    
    sos = signal.butter(10, [low, high], 'bp', fs=Fs, output='sos')
    filtered = signal.sosfilt(sos, y)
    #myplot[2].plot(t, filtered)
    #myplot[2].set_title('After ' + str(low) + ' to ' + str(high) + ' Hz bandpass filter')
    #myplot[2].set_ylim((min(filtered)*1.1),(max(filtered)*1.1))
    #myplot[2].set_xlabel('Time [seconds]')
    #plt.tight_layout()
    #plt.show()
    
    '''
    This function takes the upper envelope of the signal, the code is taken from 
    https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal
    '''
    def hl_envelopes_idx(s,dmin=1,dmax=1):
        """
        s : 1d-array, data signal from which to extract high and low envelopes
        dmin, dmax : int, size of chunks, use this if size of data is too big
        """
    
        # locals min      
        lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
        # locals max
        lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
        """
        # the following might help in some case by cutting the signal in "half"
        s_mid = np.mean(s) (0 if s centered or more generally mean of signal)
        # pre-sort of locals min based on sign 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sort of local max based on sign 
        lmax = lmax[s[lmax]>s_mid]
        """
    
        # global max of dmax-chunks of locals max 
        lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
        # global min of dmin-chunks of locals min 
        lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
        return lmin,lmax
    
    s = filtered #signal after bandpass aka the second harmonic
    high_idx, low_idx = hl_envelopes_idx(s)  #Finds the upper envelope 

    
    # plots the upper envelope
    #myplot[3].plot(t[low_idx], s[low_idx], 'g', label='high')
    #myplot[3].set_title('Upper Envelope')
    #myplot[3].set_ylim(0,(max(s[low_idx])*1.05))
    #myplot[3].set_xlabel('Time [seconds]')
    #plt.tight_layout()
    fig.savefig(newpath+'/'+date_time+'_plot_1.png')
    #plt.show()
    
    ################################################################ 
    # This section attempts to automatically detect the peak and   #
    # take the peak height and area                                #
    ################################################################
    x = t[low_idx]
    y = s[low_idx]
    
    baseline_values = peakutils.baseline(y, deg=1)
    
    xi,yi = intersection(x,y,x,baseline_values) #This section finds the base of 
    #of the peak by finding the intersection then min and max of the intersection points
    xi1 = min(xi)
    xi2 = max(xi)
    yi1 = min(yi)
    yi2 = max(yi)
    
    #This section deals with multiple minima and maxima
    from bisect import bisect_left
    
    def take_closest(myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.
    
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
           return after
        else:
           return before
    xi1 = take_closest(x,xi1)
    xi2 = take_closest(x,xi2)
    yi1 = take_closest(y,yi1)
    yi2 = take_closest(y,yi2)
    
    index_x_min = int(np.where(x==xi1)[0])
    index_x_max = int(np.where(x==xi2)[0])
    
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x, baseline_values)
    ax.plot(x, y-baseline_values)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, (max(y)*1.1))
    fig.savefig(newpath+'/'+date_time+ '_'+ wine +'_plot_2.png')
    plt.show()
    
    x1 = x[index_x_min:index_x_max]
    y1 = y[index_x_min:index_x_max]
    

    Ay = np.linspace(yi1,yi2,len(x1))
    
    
    
    area_under_curve = np.trapz(y1,x1); # Using trapezoid method to find the total area under curve (AC response)
    area_under_baseline = np.trapz(Ay, x1); # Using trapezoid method to find the total area under baseline
    area_between_curves = area_under_curve - area_under_baseline;
    
    peak_height = max(y-baseline_values) # Using trapezoid method to find the area between the curve (AC response)
    absolute_peak = max(y)
    print('Peak area is ' + str(area_between_curves))
    print('Peak height is ' + str(peak_height)) 
    print('Abs Peak height is ' + str(max(y)))
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x, baseline_values)
    ax.plot(xi,yi,"*k")
    ax.fill_between(x1,y1,Ay)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ((max(y)*1.1)))
    ax.set_ylabel('Potential V')
    ax.set_xlabel('Time s')
    ax.text(0.5, (0.9*peak_height), 'Peak Height ' + str(round(peak_height,4)),  style='normal',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.5, (0.75*peak_height), 'Peak Area ' + str(round(area_between_curves,4))  ,  style='normal',
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
    ax.text(0.5, (0.5*peak_height), 'Abs Peak Height ' + str(round(max(y),4))  ,  style='normal',
            bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
    fig.savefig(newpath+'/'+date_time+'_'+wine+'_plot_3.png')
    plt.show()
    df = pd.DataFrame({"time" : x, "potential" : y})
    df.to_csv(newpath+'/' +file + '_' + date_time+ '.csv', index=False)
    
    return peak_height, area_between_curves, absolute_peak
    
#analysis("/Users/sairajeevkallalu/Desktop/assignments/latrobe/project 3A/screens/Electrobe_output_11_10_2020/50ppm__Amp_0.06_stable_2.0recording_5.0_freq__v1_0.1_v2_0.2_v3_0.7_11-10-2020_01-10_am.data","300ppm_s3_run1")