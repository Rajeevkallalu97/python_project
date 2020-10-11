import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import struct
import pandas as pd


		
def withFile(filename,current,current_modified,harmonic):
	current = current
	current_modified = current_modified
	harmonic = harmonic
	inp = []

	with open(filename, 'r') as f:
		for i in f.readlines():
			i = i.split(' ')
			inp.append([int(i[0]), int(i[1])])
	# colnames = ['x','y']
	# data = pd.read_csv(filename, sep=',',names=colnames, skiprows=1) # x is time, y is potential
	# data['x'] = data['x'].astype(int)
	# data['y'] = data['y'].astype(int)
	# inp.append([data['x'],data['y']])
	

	inp = np.array(inp)

	freq_pert = 60	 # Frequency perturbation 
	sec_har_bw = 10 # Band-width window 
	lpf_bw = 10  # env lpf bandwidth
	har_num = 2  #  harmonic to use (normally 2)

	blank_samples = 4000 #  set first samples to zero

	max_time = 1.5 # Maximum time
	max_width = 0.2           

	sample_rate = 8000.0


	max_index_l = round((max_time - max_width) * sample_rate)
	max_index_u = round((max_time + max_width) * sample_rate)

	nod = len(inp)+1

	dt = 1/sample_rate
	df = sample_rate/nod

	n = np.arange(1, nod)

	f = n*df
	t = n*dt

	
	v = inp[:,0]
	i = inp[:,1]

	v[:blank_samples] = 0.0
	i[:blank_samples] = 0.0

	imag = np.abs(np.fft.fft(i))/nod*2
	p = np.fft.fft(i)

	sample_freq_pert = round(freq_pert*2/sample_rate*nod)
	sample_lpf_bw = round(sec_har_bw/sample_rate*nod)

	p_filtered = p.copy()
	p_filtered[:int(sample_freq_pert-sample_lpf_bw/2)] = 0.0
	p_filtered[int(sample_freq_pert+sample_lpf_bw/2):int(nod-sample_freq_pert-sample_lpf_bw/2)] = 0.0
	p_filtered[int(nod-sample_freq_pert+sample_lpf_bw/2):nod] = 0.0

	p_wave = np.real(np.fft.ifft(p_filtered))
	n = 2046

	
	d1 = 1
	d2 = 10001

	fc = freq_pert*har_num
	bw = sec_har_bw

	fs2 = sample_rate/2

	ff = [0, (fc-bw)/fs2*0.99, (fc-bw)/fs2, (fc+bw)/fs2, (fc+bw)/fs2*1.01, 1]
	m = [0, 0, 1, 1, 0, 0]

	b = signal.firwin2(n, ff, m)
	c = b/max(b)
	h = signal.freqz(c, 1, 100001)[1]
	gain = max(abs(h))

	c = c/gain
	with open('newcvalues.txt', 'w') as fin:
		for ele in c:
			fin.write(str(np.real(ele))+'\n')

	ifilt = signal.lfilter(c,1,i)
	Imagfilt = ((abs(np.fft.fft(ifilt)/nod*2)))

	i2sin = np.conj(np.sin(2*np.pi*fc*t))
	i2cosin = np.conj(np.cos(2*np.pi*fc*t))
	ixsin = ifilt * i2sin
	ixcosin = ifilt * i2cosin

	fc2 = lpf_bw
	ff = np.round([0, fc2/fs2, fc2/fs2*1.01, 1], 4)
	m = [0, 0, 1, 0]
	b = signal.firwin2(n,ff,m)
	c = b/max(b)
	w, h = signal.freqz(c,1,100001)

	ixsin_filt = signal.filtfilt(c,1,ixsin)
	ixcosin_filt = signal.filtfilt(c,1,ixcosin)

	ienv = 2*np.sqrt(ixsin_filt * ixsin_filt + ixcosin_filt * ixcosin_filt)
	int_ienv = np.cumsum(ienv)


	filter_length = 200
	ienv_filtered = ienv[:nod]
	ienv_filtered = signal.filtfilt(np.ones(filter_length)/filter_length,1,ienv_filtered)

	doff = 1

	if current == 1:
		current_time(t,nod,ienv,ienv_filtered)
	
	if current_modified == 1:
		current_time_modif(t,nod, doff, ienv_filtered)

	if harmonic == 1:
		harmonic_plot(ienv,nod,t,f,ifilt,Imagfilt,int_ienv,i,imag)
	
	parameters = [freq_pert,sec_har_bw,lpf_bw, har_num, blank_samples, max_time,max_width,sample_rate]

	#save_data('binary_data', [f[1:round(nod/2)],imag[1:round(nod/2)]], parameters)
	#file_parameters, file_x_coords, file_y_coords = load_data('binary_data')
	#print(file_parameters)

def current_time(t,nod, ienv, ienv_filtered):
	plt.plot(t[:nod],ienv[:nod],'-r','linewidth',2)
	plt.plot(t[:nod],ienv_filtered,'b','linewidth',1)
	plt.xlabel('Time (s)')
	plt.ylabel('Current (S.U)')
	plt.title('Results')
	plt.show()

def current_time_modif(t,nod, doff, ienv_filtered):
	plt.plot(t[:nod-doff],ienv_filtered,'-r','linewidth',2)
	plt.xlabel('Time (s)')
	plt.ylabel('Current (S.U)')
	plt.title('Results')
	plt.show()

def harmonic_plot(ienv,nod,t,f,ifilt,Imagfilt,int_ienv,i,imag):
	figure, axes = plt.subplots(6, figsize=(14,7))
	#axes = figure.subplots(6, 1)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.34)
	axes[0].plot(t[:nod],i[:nod])
	axes[1].plot(f[1:round(nod/2)],imag[1:round(nod/2)])
	axes[2].plot(t[:nod],ifilt[:nod])
	axes[3].plot(f[1:round(nod/2)],Imagfilt[1:round(nod/2)])
	axes[4].plot(t[:nod],ienv[:nod])
	axes[5].plot(t[:nod],int_ienv[:nod])
	plt.subplot_tool()
	plt.show()

#withFile("/Users/sairajeevkallalu/Desktop/assignments/latrobe/project 3A/screens/data/Raw_data.data",1,1,1)