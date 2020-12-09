# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:26:36 2020

@author: roder

CODE ADAPTED FROM EXISTING CODE BY DR. TIMOTHY A. LIVENGOOD
COPIED COMMENTS/CODE IS NOT EXPLICITELY REFERENCED
"""

import numpy as np  # for calculations
import matplotlib.pyplot as plt  # for plot

import pandas as pd # to save the data

#%%

def ir_spectra(year, observations):
    filepath = '../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_combined_spec''.pkl'
    epoxi_data_spec = pd.read_pickle(filepath)
    epoxi_data_spec['target'] = "Earth Observation 1"
    epoxi_data_spec['units']  = "W/[m^2 um]"
    # ill_frac = 0.5*(1 + np.cos(np.deg2rad(phase_angle)))
    ####### interpolate emphemeris data to add diameter to epoxi_data_spec
   
    duration_rates = [1999.64, 1435.64]
    for i in duration_rates:
        epoxi_data_spec_temp = epoxi_data_spec[epoxi_data_spec['duration'] == i].reset_index(drop=True)
        if i ==1999.64:
            mode = 'slow'
            rate = 300.00e-06   # scan rate in radian/sec
            time = i*1e-3       # duration of frame, in sec
        elif i==1435.64:
            mode = 'fast'
            rate = 1724.00e-06  # scan rate in radian/sec
            time = i*1e-3       # duration of frame, in sec
        else:
            print('duration/speed mistake')
    
        # Solid angle subtended by individual pixels of the detector.
        pixel_width = 10e-6 #* 10e-6 # ster/pixel, (pixel size in radians)^2
        # # Scan rates across the Earth, in radians per second.
        # rate_fast      = 1724.00e-06   # scan rate in radian/sec
        # time_fast      = 1435.64e-03   # duration of frame, in sec
        # rate_slow      =  300.00e-06   # scan rate in radian/sec
        # time_slow      = 1999.64e-03   # duration of frame, in sec
    
        ## Scale fluxes by distance. The surface brightness of Earth varies as
        ## helicoentric distance squared. Scale distances to 1 AU range.
        factor = pixel_width * rate * time * epoxi_data_spec_temp.loc[:,'range_SC']**2 * epoxi_data_spec_temp.loc[:,'range_Sun']**2
    
        epoxi_data_spec_temp.loc[:,'image'] *=  factor
        epoxi_data_spec_temp.loc[:,'cross_short']   *=  factor
        epoxi_data_spec_temp.loc[:,'cross_long']    *=  factor
        epoxi_data_spec_temp.loc[:,'spectrum']      *=  factor    
    
        # Every instance has 3 scans, find the best one and keep it
        first_scan = np.arange(0,epoxi_data_spec_temp.shape[0],3)
        indexes_max = []
        indexes_close = []
        for j in first_scan:
            sums = [np.sum(epoxi_data_spec_temp.at[j,'image']),np.sum(epoxi_data_spec_temp.at[j+1,'image']),np.sum(epoxi_data_spec_temp.at[j+2,'image'])]
            max_signal = np.max(sums)
            difference = sums-max_signal
            #indexes_max.append(np.where(sums==max_signal))
            indexes_max.append(int(j+sums.index(max_signal)))
            # with 1.5e-11, only get some that have 1 close, with 2e-11 all 3 can sometimes be close
            indexes_close.append(j+np.where(np.logical_and(0<np.abs(difference), np.abs(difference)<1.5e-6))[0]) #DIFF
        indexes_max.append(int(np.median(indexes_max)))
        spec = epoxi_data_spec_temp.loc[indexes_max,:].reset_index(drop = True)
        for idx,j in enumerate(indexes_close): 
            if j.shape[0] == 1: # the ones that are not max, but close can be lower index than max so low:high indexing not possible
                spec.at[idx,'duration'] = np.sum(spec.at[idx,'duration'] + epoxi_data_spec_temp.at[j[0],'duration']) * 0.5
                spec.at[idx,'minimum']  = np.minimum(spec.at[idx,'minimum'], epoxi_data_spec_temp.at[j[0],'minimum'])
                spec.at[idx,'maximum']  = np.maximum(spec.at[idx,'maximum'], epoxi_data_spec_temp.at[j[0],'maximum'])
                spec.at[idx,'median']   = np.median((spec.at[idx,'median'], epoxi_data_spec_temp.at[j[0],'median']))
                spec.at[idx,'one_sigma'] = np.sqrt(np.sum(spec.at[idx,'one_sigma']**2 + epoxi_data_spec_temp.at[j[0],'one_sigma']**2) * 0.5)
                spec.at[idx,'RA']       = np.sum(spec.at[idx,'RA'] + epoxi_data_spec_temp.at[j[0],'RA']) * 0.5
                spec.at[idx,'DEC']      = np.sum(spec.at[idx,'DEC'] + epoxi_data_spec_temp.at[j[0],'DEC']) * 0.5
                spec.at[idx,'range_SC'] = np.sum(spec.at[idx,'range_SC'] + epoxi_data_spec_temp.at[j[0],'range_SC']) * 0.5
                # diameter is still 0
                spec.at[idx,'diameter'] = np.sum(spec.at[idx,'diameter'] + epoxi_data_spec_temp.at[j[0],'diameter']) * 0.5
                spec.at[idx,'range_Sun'] = np.sum(spec.at[idx,'range_Sun'] + epoxi_data_spec_temp.at[j[0],'range_Sun']) * 0.5
                spec.at[idx,'nppa']     = np.sum(spec.at[idx,'nppa'] + epoxi_data_spec_temp.at[j[0],'nppa']) * 0.5
                #spec.at[indexes_max[idx],'oblateness'] = np.sum(spec.at[indexes_max[idx],'oblateness'] + epoxi_data_spec_temp.at[i[0],'oblateness']) * 0.5
                # illum is still 100%
                spec.at[idx,'illum']    = np.sum(spec.at[idx,'illum'] + epoxi_data_spec_temp.at[j[0],'illum']) * 0.5
                spec.at[idx,'phase_angle'] = np.sum(spec.at[idx,'phase_angle'] + epoxi_data_spec_temp.at[j[0],'phase_angle']) * 0.5
                spec.at[idx,'sub_SC']   = list((np.array(spec.at[idx,'sub_SC']) +  np.array(epoxi_data_spec_temp.at[j[0],'sub_SC'])) * 0.5)
                spec.at[idx,'sub_Sun']  = list((np.array(spec.at[idx,'sub_Sun']) +  np.array(epoxi_data_spec_temp.at[j[0],'sub_Sun'])) * 0.5)
                spec.at[idx,'north_angle'] = np.sum(spec.at[idx,'north_angle'] + epoxi_data_spec_temp.at[j[0],'north_angle']) * 0.5
                spec.at[idx,'Sun_angle'] = np.sum(spec.at[idx,'Sun_angle'] + epoxi_data_spec_temp.at[j[0],'Sun_angle']) * 0.5
                spec.at[idx,'image']    = (spec.at[idx,'image'] + epoxi_data_spec_temp.at[j[0],'image']) * 0.5
                spec.at[idx,'snr']      = np.sqrt((spec.at[idx,'snr']**2 + epoxi_data_spec_temp.at[j[0],'snr']**2) * 0.5)
                spec.at[idx,'spectrum'] = (spec.at[idx,'spectrum'] + epoxi_data_spec_temp.at[j[0],'spectrum']) * 0.5
                spec.at[idx,'spec_snr'] = np.sqrt((spec.at[idx,'spec_snr']**2 + epoxi_data_spec_temp.at[j[0],'spec_snr']**2) * 0.5)
                spec.at[idx,'cross_short'] = (spec.at[idx,'cross_short'] + epoxi_data_spec_temp.at[j[0],'cross_short']) * 0.5
                spec.at[idx,'cross_long'] = (spec.at[idx,'cross_long'] + epoxi_data_spec_temp.at[j[0],'cross_long']) * 0.5
                            
        # last index still needs to be done
        idx = spec.shape[0] -1 # index of the last row, diurnal average
        spec.at[idx,'duration'] = np.sum(spec.loc[0:idx-1,'duration'])/idx
        spec.at[idx,'minimum']  = np.min(spec.loc[0:idx-1,'minimum'])
        spec.at[idx,'maximum']  = np.max(spec.loc[0:idx-1,'maximum'])
        spec.at[idx,'median']   = np.sum(spec.loc[0:idx-1,'median'])/idx # average median
        spec.at[idx,'one_sigma'] = np.sqrt(np.sum(spec.loc[0:idx-1,'duration']**2)/idx)
        spec.at[idx,'RA']       = np.sum(spec.loc[0:idx-1,'RA'])/idx
        spec.at[idx,'DEC']      = np.sum(spec.loc[0:idx-1,'DEC'])/idx
        spec.at[idx,'range_SC'] = np.sum(spec.loc[0:idx-1,'range_SC'])/idx
        # diameter is still 0
        spec.at[idx,'diameter'] = np.sum(spec.loc[0:idx-1,'diameter'])/idx
        spec.at[idx,'range_Sun'] = np.sum(spec.loc[0:idx-1,'range_Sun'])/idx
        spec.at[idx,'nppa']     = np.sum(spec.loc[0:idx-1,'nppa'])/idx
        # illum is still 100%
        spec.at[idx,'illum']    = np.sum(spec.loc[0:idx-1,'illum'])/idx
        spec.at[idx,'phase_angle'] = np.sum(spec.loc[0:idx-1,'phase_angle'])/idx
        spec.at[idx,'sub_SC']   = [270, np.sum(np.sum(spec.loc[0:idx-1,'sub_SC'])[1::2])/idx] # difficult to make a sum with the list stuff, this works
        spec.at[idx,'sub_Sun']  = [0, np.sum(np.sum(spec.loc[0:idx-1,'sub_Sun'])[1::2])/idx]
        spec.at[idx,'north_angle'] = np.sum(spec.loc[0:idx-1,'north_angle'])/idx
        spec.at[idx,'Sun_angle'] = np.sum(spec.loc[0:idx-1,'Sun_angle'])/idx
        
        spec.at[idx,'image']    = np.sum(spec.loc[0:idx-1,'image'])/idx
        spec.at[idx,'snr']      = np.sqrt(np.sum(spec.loc[0:idx-1,'snr']**2)/idx)
        spec.at[idx,'spectrum'] = np.sum(spec.loc[0:idx-1,'spectrum'])/idx
        spec.at[idx,'spec_snr'] = np.sqrt(np.sum(spec.loc[0:idx-1,'spec_snr']**2)/idx)
        spec.at[idx,'cross_short'] = np.sum(spec.loc[0:idx-1,'cross_short'])/idx
        spec.at[idx,'cross_long'] = np.sum(spec.loc[0:idx-1,'cross_long'])/idx
        
        spec.to_pickle('../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_spec_duration_'+mode+'.pkl')
    
        fig,ax = plt.subplots()
        ax.set_title('Infrared data Earth Observation '+str(observations[0])+' '+mode)
        ax.set_ylabel(r'signal ($10^{-7} W/m^2/\mu m$)')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength ($\mu m$)')
        ax.set_xlim([1.05,2.7])
        # plt.plot(spec.at[idx,'spec_wave'], spec.at[idx,'spectrum']*1e7)
        mask_short = np.where(spec.at[idx,'spec_wave']<2.7)
        ax.plot(spec.at[idx,'spec_wave'][mask_short], spec.at[idx,'spectrum'][mask_short]*1e7)
        
        fig2,ax2 = plt.subplots()
        ax2.set_title('Infrared data Earth Observation '+str(observations[0])+' '+mode)
        ax2.set_ylabel(r'signal ($10^{-7} W/m^2/\mu m$)')
        ax2.set_yscale('log')
        ax2.set_xlabel('Wavelength ($\mu m$)')
        ax2.set_xlim([2.5,4.5])
        # plt.plot(spec.at[idx,'spec_wave'], spec.at[idx,'spectrum']*1e7)
        mask_long = np.where(spec.at[idx,'spec_wave']>2.5)
        ax2.plot(spec.at[idx,'spec_wave'][mask_long], spec.at[idx,'spectrum'][mask_long]*1e7)  
    return
        
#%%

if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    year = '2008'
    observations = ['078','079'] 
    #observations = ['149','150'] 
    #observations = ['156','157'] 
    #year = '2009'
    #observations = ['086','087'] 
    #observations = ['277','278']
    ir_spectra(year,observations)
        
        