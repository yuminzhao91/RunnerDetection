import numpy as np
import obspy as opy
import scipy.signal
import fnmatch
import os
import datetime
import matplotlib.pyplot as plt

from obspy.core import UTCDateTime
from collections import defaultdict
from obspy.core.stream import Stream
from obspy.signal.invsim import corn_freq_2_paz, paz_to_freq_resp
from obspy.signal.filter import envelope
from scipy.signal import find_peaks
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import scipy.signal
import scipy.fft as ft
# Read data


# ----- default parameters ------ #
# remove response

 
tm_extend = 1*60*60 # hrs
paz = {'gain':1.0,
       'poles':[-22.211059 + 22.217768j,
                -22.211059 - 22.217768j],
       'sensitivity': 76.7,
       'zeros': [0j,0j]}

sensitivity = 76.7            
paz_str = corn_freq_2_paz(0.01, damp=0.707)  
paz_str['sensitivity'] = sensitivity
tmz_correction = 8*3600

default_par = {  
    'rm_timeextend'     :   tm_extend,
    'pas'               :   paz,
    'pas_str'           :   paz_str,
    'time_extend'       :   tm_extend,
    'tmz_correction'    :   tmz_correction  
}


def getdatalist(path, station, flag, starttime, endtime, cmp):
    """
    Get data list
    """
    if not path.endswith('/'):
        path = path + '/'
        
    
    data_list = []
    srckey = '%s-%s-*-%s.npz' %(station, flag, cmp)
    for file in sorted(os.listdir(path)):
        if fnmatch.fnmatch(file, srckey):   
            data_list    += [path+file]

    index = []
    for ii in range(0, len(data_list)):
        time_bg  = data_list[ii].split('-')[2]
        time_bg  = time_bg.replace(':','.')
        [year,month,days,hours,minute,second]= time_bg.split('.')
       
        date  = UTCDateTime('%s,%s,%s,%s,%s,%s'%(year,month,days, hours, minute, second))       
        if  starttime<=date and date<endtime:
            index += [ii]
                
    data_list = [data_list[ii] for ii in index]
    return data_list




class DataProcessing:
    
    def __init__(self, project_par=None, processing_par=None, default_par=default_par):
        self.par  = {**project_par, **processing_par, **default_par}
        self.data = Stream()
        self.data_list = []
        self.station_list = []
        self.station = ''
        self.component = ''
        self.current_index = 0
        self.title = ''

        # Judge compliance
        for keys in ['data_input_path', 'data_output_path', 'fig_output_path'] :
            if not self.par[keys].endswith('/'):
                self.par[keys] = self.par[keys] + '/'
                
        for keys in ['data_output_path', 'fig_output_path'] :
            if not os.path.exists(self.par[keys]):
                print('creat %s' %self.par[keys])
                os.makedirs(self.par[keys])


    def getdatalist(self):
        """
        Get data list for read file list
        output: data_list
                station_list
        """
        data_list = []
        station_list = []
        
        ifall = True
        if 'proj_time_bgn' in self.par.keys() and 'proj_time_end' in self.par.keys():
            if self.par['proj_time_bgn'] < self.par['proj_time_end']:
                ifall = False

        stations = self.par['stations']
        for station in stations.keys():
            station_values = stations[station]
            for ii in range(0, len(station_values)):
                for cmp in self.par['components']:
                    srckey = '*%s*%s.sac' %(station_values[ii], cmp)
                    for file in os.listdir(self.par['data_input_path']):
                        if fnmatch.fnmatch(file, srckey):
                            data_list    += [file]
                            station_list += [station]

        if ifall == False:
            index = []
            for ii in range(0, len(data_list)):
                year  = data_list[ii].split('.')[4]
                month = data_list[ii].split('.')[5]
                days  = data_list[ii].split('.')[6]
                hours = data_list[ii].split('.')[7]        
                date  = UTCDateTime('%s,%s,%s,%s,0,0'%(year,month,days, hours))
                
                if  self.par['proj_time_bgn']<=date and date<=self.par['proj_time_end']:
                    index += [ii]

            data_list    = [ data_list[ii]    for ii in index]
            station_list = [ station_list[ii] for ii in index]
                
            
        # add path
        for ii in range(0, len(data_list)):
            data_list[ii] = self.par['data_input_path'] + data_list[ii]
        
        self.data_list = data_list
        self.station_list = station_list
        
        if self.par['verbosity']:
            print('>>> Create input data_list and station_list!')
            for ii in range(0,len(data_list)):
                print(data_list[ii])
        return self.data_list, self.station_list
    
    
    def readdata(self, data_path):
        """
        Read SAC data
        """
        self.data = Stream() 
        self.data = opy.read(data_path, debug_headers=True)
        self.current_index = self.data_list.index(data_path)
        
        
        if self.par['verbosity']:
            print(" >>> Read data from %s" %data_path)
            print(self.data)
            
    def preprocessing(self):
        """
        Preprocessing: sampling the file 
        """
        
        # set station
        self.data[0].stats.network = self.station_list[self.current_index]
        
        # set component
        self.component = self.data[0].stats.channel
        
        # correct time zone
        # local time zone   
        if self.par['local_timezone'] == True:
            self.data[0].stats.starttime = self.data[0].stats.starttime+self.par['tmz_correction']
            if self.par['verbosity']:
                print(" Pre-processing:  correct time-zone ")
                print(self.data)
  
        # set unified title
        self.title = '%s-%s-%s'%(self.station_list[self.current_index], 
                                 self.data[0].stats.starttime.strftime("%Y.%m.%d.%H:%M:%S"),
                                 self.data[0].stats.endtime.strftime("%Y.%m.%d.%H:%M:%S"))
        self.station = self.station_list[self.current_index]
        
        # downsampling from sampling_rate_raw to sampling_rate_down 
        self.data[0].stats.delta = round(self.data[0].stats.delta, 3)
        sampling_rate_raw  = self.data[0].stats.delta
        sampling_rate_down = self.par['downsampling_rate']
        factor = int(sampling_rate_down/sampling_rate_raw)

        if self.data[0].stats.delta != sampling_rate_down:
            # SmartSolo Anti-alias filter
            # 206.5 Hz @ 2ms (82.6% of Nyquist)
            self.data[0].filter("lowpass", corners=30, freq=206.5)  
            self.data[0].decimate(factor, no_filter=True, strict_length=False)
            
            if self.par['verbosity']:
                print(" Pre-processing:  downsampling ")
                print(self.data)
                
    def gettitle(self, flag, starttime, endtime, cmp):
        """
        Generate unified title
        """
        title = '%s-%s-%s-%s-%s'%(self.station_list[self.current_index], 
                                  flag, starttime.strftime("%Y.%m.%d.%H:%M:%S"),
                                  endtime.strftime("%Y.%m.%d.%H:%M:%S"), cmp)
        return title
    
                        
    def remove_response(self):
        """
        Remove receiver response
        
        """
        if self.par['remove_response'] == True:           
            self.data[0].simulate(paz_remove=self.par['paz'], paz_simulate=self.par['paz_str']) 
            print('Remove response ...')
                      
    def high_pass(self):
        """
        High pass filter
        """
        self.data[0].filter("highpass", freq=self.par['freq_highpass'])
        if self.par['verbosity']:
            print(" High pass filter @%f Hz " %self.par['freq_highpass'])
            print(self.data)
            
    def plot_waveform(self, plot_color='blue'):
        """
        Plot wave form
        """
        fig_title = '%s-waveform.png' %self.title
        fig_path = self.par['fig_output_path'] + fig_title
        self.data[0].plot(color=plot_color, outfile = fig_path)
        
    def psd(self):
        """
        Output PSD
        """
        dt = self.data[0].stats.delta
        fs = 1/dt
        
        starttime = self.data[0].stats.starttime
        endtime   = self.data[0].stats.endtime
        win_len   = self.par['PSD_winlen']
        
        nwindow   = int((endtime-starttime)/win_len)
        
        for iid in range(0, nwindow): #nframe
            win_bgn = starttime+iid*win_len
            win_end = win_bgn+win_len
            st_slice = self.data.slice(win_bgn, win_end)
            pxx, fre = plt.psd(st_slice[0].data, Fs=fs)

            psd_title = self.gettitle(flag='PSD', starttime=win_bgn, endtime=win_end, cmp=self.component)
            path = self.par['data_output_path'] + psd_title
            self.writedata(data=pxx, ax1=fre, time=win_bgn.strftime("%Y.%m.%d.%H:%M:%S"), file_path=path, txt_hrd=psd_title)

        if self.par['verbosity']:
            print(" Generate PSD ")
            
    
    def footstep_events(self):
        """
        Find footsteps
        """
        if self.par['winlen_spc'] > self.par['winlen_events']:
            sys.exit("Footstep_events: The winlen_events should be large than winlen_events !")
    
        if self.par['winlen_events']%self.par['winlen_spc'] !=0:
            sys.exit("Footstep_events: The winlen_events should be an integer multiple of winlen_spc !")
            
        if self.par['tw'] > self.par['winlen_spc']:
            sys.exit("Footstep_events: The winlen_spc should be large than tw !")
    
        if self.par['winlen_spc']%self.par['tw'] !=0:
            sys.exit("Footstep_events: The winlen_spc should be an integer multiple of tw !")
 
 
        dt = self.data[0].stats.delta
        fs = 1/(dt*1.0)
        
    
        starttime     = self.data[0].stats.starttime
        endtime       = self.data[0].stats.endtime
        winlen_events = self.par['winlen_events']
        winlen_spc    = self.par['winlen_spc']
        sampleRate  = self.par['sampleRate']
        
        tw            = self.par['tw']
        ntw          =self.par['ntw']
        f_min      = self.par['f_min']
        f_max      = self.par['f_max']
   
        hf_l          =self.par['hf_l']
        hf_h          =self.par['hf_h']
        a_mn        =self.par['a_mn']
        
        pf_min      =self.par['pf_min']
        pf_max      =self.par['pf_max']
        a_p         =self.par['a_p']
        eps          =self.par['eps']
       
        f_l          =self.par['f_l']
        f_h         =self.par['f_h']
        
        
        NyquistFrq  = sampleRate/2.0 # the Nyquist frequency
        lw=tw*sampleRate
        
        winnum_events = int((endtime-starttime+1)/winlen_events)
        
        runner_num = []
        time_line  = []
      
        ##########################################################
        
        for ii in range(0, winnum_events):
            win_bgn  = starttime + ii*winlen_events 
            win_end  = starttime + (ii+1)*winlen_events
            win_plot = starttime + ii*winlen_events +0.5*winlen_events
            winnum_spc = int(winlen_events/winlen_spc)
            win_sp = int(winlen_spc/tw)
            runner_cnt = 0
            C=[]

            for jj in range(0, winnum_spc):
                win_spc_bgn = starttime + ii*winlen_events + jj*winlen_spc
                win_spc_end = win_spc_bgn + winlen_spc

                st_slice = self.data.slice(win_spc_bgn, win_spc_end)
                st_slice.filter("bandpass", freqmin=f_min, freqmax=f_max,corners=2,zerophase=True)
                
                
                lst=len(st_slice[0].data)
                dlst=int(lst%lw)
                sdata=st_slice[0].data[:lst-dlst]
                sdata=np.transpose(sdata.reshape((lst-int(dlst))//lw,lw))
                
                fsdata=ft.fft(sdata,axis=0)
                frqBins = int(np.ceil(fsdata.shape[0]/2))
                freqs = np.linspace(0, NyquistFrq,num=frqBins)
                fsdata=np.absolute(fsdata)/np.amax(np.absolute(fsdata),axis=0)

                ihfl=np.where(np.round(freqs)==hf_l)
                ihfh=np.where(np.round(freqs)==hf_h)
                hsp_m=np.mean(np.absolute(fsdata[ihfl[0][0]:ihfh[0][0],:]),axis=0)
              
                st_env=envelope(st_slice[0].data)
                st_slice[0].data=st_env
                st_slice.filter("lowpass", corners=4, freq=10)  
                st_elf=st_slice[0].data[:lst-int(dlst)]
                st_elf=np.transpose(st_elf.reshape((lst-int(dlst))//lw,lw))
                
                fst_elf=ft.fft(st_elf,axis=0)
                fst_elf=np.absolute(fst_elf)/np.amax(np.absolute(fst_elf),axis=0)
                N=fst_elf.shape[1]
                label=np.zeros((N,))
                
                i=0
                while i<N:
                    
                    if (hsp_m[i]<a_mn):
                        
                        spectrum=fst_elf[:,i]
                        ia=np.where(np.round(freqs)==f_l)
                        ib=np.where(np.round(freqs)==f_h)

                        frqs_a=freqs[ia[0][0]:ib[0][0]]

                        sp_tmp=np.absolute(spectrum[ia[0][0]:ib[0][0]])
                        peak_id= find_peaks(sp_tmp)
        
                        if len(peak_id[0])>0:
                            peak_freq = frqs_a[peak_id[0]]
                            peak_pp=sp_tmp[peak_id[0]]
                            peak_pp=list(peak_pp)
                            max_index1 = peak_pp.index(max(peak_pp))
                            cpeak_pp=peak_pp.copy()
                            cpeak_pp[max_index1]=min(peak_pp)
                            max_index2 = cpeak_pp.index(max(cpeak_pp))
                            f21=peak_freq[max_index2]/peak_freq[max_index1]
                        else:
                            max_index1=0
                            f21=0
                            peak_freq=[0]
                            peak_pp=[0]

                        if (pf_min<=peak_freq[max_index1]<pf_max)&(2-eps<f21<2+eps)&(max(peak_pp)>a_p):
                            label[i]=1
                            runner_cnt=runner_cnt+label[i]
                            C.append(60*peak_freq[max_index1])
                            i+=ntw
                        else:
                
                            i+=1
                    else:
                        i+=1
                   

            runner_num += [runner_cnt]
            time_line  += [win_plot.datetime]

        
            footstep_title = self.gettitle(flag='Footstep', 
                                          starttime=win_bgn, endtime=win_end, 
                                          cmp=self.component)
            path = self.par['data_output_path'] + footstep_title
            self.writedata(data=runner_cnt, ax1=np.array([]), 
                           time=win_plot.strftime("%Y-%m-%d %H:%M:%S"), file_path=path, txt_hrd=footstep_title)
            
            C_title = self.gettitle(flag='Cadence', starttime=win_bgn, endtime=win_end, cmp=self.component)
            path = self.par['data_output_path'] + C_title
            self.writedata(data=C, ax1=np.array([]), time=win_bgn.strftime("%Y.%m.%d.%H:%M:%S"), file_path=path, txt_hrd=C_title)


        if self.par['verbosity']:
            print(" Find footstep events!")
  

    def writedata(self, data=np.array([]), ax1=np.array([]), time='', file_path='./', txt_hrd=''):
        """
        Write data as npy
        """
        np.savez(file_path, data=data, ax1=ax1, time=time, txt_hrd=txt_hrd)
        
        
    def cleardata(self):
        """
        Clear data
        """
        self.data.clear()









