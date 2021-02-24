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

from photutils.background import BackgroundBase, Background2D, MedianBackground
from astropy.stats import SigmaClip

#%%

def ir_spectra(year, observations, remove_background = True):
    filepath = '../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_combined_spec.pkl'
    #filepath = '../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_combined_spec_box_fixed.pkl'
    epoxi_data_spec = pd.read_pickle(filepath)
    epoxi_data_spec['target'] = "Earth Observation 1"
    epoxi_data_spec['units']  = "W/[m^2 um]"
    # ill_frac = 0.5*(1 + np.cos(np.deg2rad(phase_angle)))
    ####### interpolate emphemeris data to add diameter to epoxi_data_spec
   
    duration_rates = [1435.64, 1999.64]
    for i in duration_rates:
        epoxi_data_spec_temp = epoxi_data_spec[epoxi_data_spec['duration'] == i].reset_index(drop=True)
        if i ==1999.64:
            mode = 'slow'
            # Scan rates across the Earth, in radians per second.
            rate = 300.00e-06   # scan rate in radian/sec
            time = i*1e-3       # duration of frame, in sec
        elif i==1435.64:
            mode = 'fast'
            rate = 1724.00e-06  # scan rate in radian/sec
            time = i*1e-3       # duration of frame, in sec
        else:
            print('duration/speed mistake')
        
        # for j in np.arange(epoxi_data_spec_temp.shape[0]):
        #     #background = Background2D(epoxi_data_spec_temp.at[j,'image'],(64,256),exclude_percentile=5,sigma_clip=None)
        #     sigma_clip = SigmaClip(sigma=3.)
        #     bkg_estimator = MedianBackground()
        #     background = Background2D(epoxi_data_spec_temp.at[j,'image'][:,26:97],(32,64),filter_size=(3, 3),sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
        #     epoxi_data_spec_temp.at[j,'image'][:,26:97] = epoxi_data_spec_temp.at[j,'image'][:,26:97] - background.background
        #     #print(np.max(background.background))
        
        # Solid angle subtended by individual pixels of the detector.
        pixel_width = 10e-6 #* 10e-6 # ster/pixel, (pixel size in radians)^2
    
        ## Scale fluxes by distance. The surface brightness of Earth varies as
        ## helicoentric distance squared. Scale distances to 1 AU range.
        ###### *2 works better (wrt Livengood) for EarthObs1 but not for the others, fast scan is closer to Livengood results
        factor = pixel_width * rate * time * epoxi_data_spec_temp.loc[:,'range_SC']**2 * epoxi_data_spec_temp.loc[:,'range_Sun']**2
    
        epoxi_data_spec_temp.loc[:,'image'] *=  factor
        epoxi_data_spec_temp.loc[:,'cross_short']   *=  factor
        epoxi_data_spec_temp.loc[:,'cross_long']    *=  factor
        epoxi_data_spec_temp.loc[:,'spectrum']      *=  factor    
        
        if i ==1999.64:
            for j in np.arange(4):
                fig7,ax7 = plt.subplots()
                ax7.grid(True)
                ax7.set_title('Infrared data cross_long '+str(observations[0])+'BEFORE REMOVE BACKGROUND')
                ax7.plot(np.arange(0,128),epoxi_data_spec_temp.at[j,'cross_long'])
                ax7.set_ylabel(r'signal times wavelength spacing ($10^{-7} W/m^2$)')
                ax7.set_xlabel('Row of the image')  
                
                # fig8,ax8 = plt.subplots()
                # ax8.grid(True)
                # ax8.set_title('Infrared data cross_short '+str(observations[0])+'BEFORE REMOVE BACKGROUND')
                # ax8.plot(np.arange(0,128),epoxi_data_spec_temp.at[j,'cross_short'])
                # ax8.set_ylabel(r'signal times wavelength spacing ($10^{-7} W/m^2$)')
                # ax8.set_xlabel('Row of the image')  
        
        if remove_background==True:
            ## too much background subtracted needs to be corrected, this is found by analysing the cross-long (and cross-short)
            # rows can be found from those plots where the spectrum is negative
            #              0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
            ### for observation 1:
            # the rows that include the background:            
            lim_a_fast = [83,26,70,78,26,63,78,26,63,78,26,65,80,26,65,83,26,68,85,26,70,85,26,73,85,26,70,87,26,70,85,26,70,85,26,70,82,26,69]
            lim_b_fast = [96,42,96,96,39,96,96,40,96,96,39,96,96,42,96,96,45,96,96,45,96,96,46,96,96,47,96,96,45,96,96,46,96,96,46,96,96,43,96]
            # the rows that include the signal: NOT GOOD FOR OBS1
            lim_c_fast = [83,26,70,78,26,63,78,26,63,78,26,65,80,26,65,83,26,68,85,26,70,85,26,73,85,26,70,87,26,70,85,26,70,85,26,70,82,26,69]
            lim_d_fast = [96,42,96,96,39,96,96,40,96,96,39,96,96,42,96,96,45,96,96,45,96,96,46,96,96,47,96,96,45,96,96,46,96,96,46,96,96,43,96]
            
            if observations[0] == '149':
                #              0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
                lim_a_fast = [73,26,60,26,26,65,26,26,65,26,26,62,26,26,80,75,26,61,67,26,53,70,26,56,74,87,58,67,26,51,64,79,48,69,26,54,74,26,60]
                lim_b_fast = [96,52,96,45,58,96,47,60,90,45,60,77,45,58,96,96,49,96,96,45,96,96,42,96,96,96,96,96,45,96,96,96,96,96,46,96,96,50,96]
                
                #              0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
                lim_c_fast = [35,50,26,45,55,30,45,60,30,45,55,30,40,57,28,35,50,20,30,45,18,40,56,26,40,53,26,32,47,16,27,41,11,32,48,16,38,52,26]
                lim_d_fast = [75,90,60,80,95,65,85,95,65,80,95,61,78,92,63,73,87,58,65,80,50,70,84,54,72,86,56,65,79,50,62,75,45,67,82,51,72,87,57]
            
            if observations[0] == '156':
                #              0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
                lim_a_fast = [78,27,60,27,27,68,75,27,60,72,27,60,72,27,75,83,27,60,75,27,58,70,27,55,78,27,65,75,27,65,70,27,55,70,27,55,75,27,55]
                lim_b_fast = [95,55,95,45,60,95,95,55,95,95,55,95,95,50,95,95,55,95,95,50,95,95,50,95,95,57,95,95,55,95,95,50,95,95,50,95,95,45,95]
            
            for j in np.arange(epoxi_data_spec_temp.shape[0]):
                lim_0 = lim_a_fast[j]
                lim_1 = lim_b_fast[j]
                background = np.copy(epoxi_data_spec_temp.at[j,'image'][lim_0,:])
                # background is the average signal of all the rows that were negative in cross-long
                for k in np.arange(lim_0+1, lim_1+1): # loop through the rows that contain the background 
                    background += epoxi_data_spec_temp.at[j,'image'][k,:] 
                background = background/(lim_1-lim_0+1)
                
                lim_0 = lim_c_fast[j]
                lim_1 = lim_d_fast[j]
                ################# Why min to max general signal ones and not the specific for every measurement?
                #for k in np.arange(np.min(lim_c_fast),np.max(lim_d_fast)+1): # loop through the rows that need to be updated 
                for k in np.arange(26,96+1): # loop through the rows that need to be updated                 
                    epoxi_data_spec_temp.at[j,'image'][k,:]  = epoxi_data_spec_temp.at[j,'image'][k,:] - background
                #for k in np.arange(lim_0+1, lim_1+1): # loop through the rows that have the signal in them
                ignore_this = True
                for k in np.arange(26,96+1): # this is needed
                    epoxi_data_spec_temp.at[j,'spectrum']  += epoxi_data_spec_temp.at[j,'image'][k,:]
                    epoxi_data_spec_temp.at[j,'spec_wave'] += epoxi_data_spec_temp.at[j,'wavelength'][k,:]
                    epoxi_data_spec_temp.at[j,'spec_delta_wave'] += epoxi_data_spec_temp.at[j,'delta_wave'][k,:]
                    epoxi_data_spec_temp.at[j,'spec_snr'] += epoxi_data_spec_temp.at[j,'snr'][k,:]**2
                #n_rows = lim_1+1 - lim_0
                n_rows = 96-26 +1
                epoxi_data_spec_temp.at[j,'spec_wave'] = epoxi_data_spec_temp.at[j,'spec_wave'] /(n_rows)
                epoxi_data_spec_temp.at[j,'spec_delta_wave'] = epoxi_data_spec_temp.at[j,'spec_delta_wave'] /(n_rows)
                epoxi_data_spec_temp.at[j,'spec_snr'] = np.sqrt(epoxi_data_spec_temp.at[j,'spec_snr'])
                
            #####
            # for j in np.arange(epoxi_data_spec_temp.shape[0]):
            #     background = Background2D(epoxi_data_spec_temp.at[j,'image'],(64,256),exclude_percentile=5,sigma_clip=None)
            #     epoxi_data_spec_temp.at[j,'image'] = epoxi_data_spec_temp.at[j,'image'] - background.background
        
        spec_wave_arr_mask = np.array(epoxi_data_spec_temp['spec_wave'].tolist())
        mask_short = np.where(np.logical_and(spec_wave_arr_mask<=2.7, spec_wave_arr_mask >=1.05))
        mask_short = zip(mask_short[0],mask_short[1])
        for j in mask_short:
            # add image*delta, all rows of the corresponding column
            # To get the cross-dispersion (parallel to slit) distribution of signal.
            epoxi_data_spec_temp.at[j[0],'cross_short'] +=  epoxi_data_spec_temp.at[j[0],'image'][:,j[1]] * epoxi_data_spec_temp.at[j[0],'delta_wave'][:,j[1]]
        mask_long = np.where(np.logical_and(spec_wave_arr_mask>=2.7, spec_wave_arr_mask <=4.6))
        mask_long = zip(mask_long[0],mask_long[1])
        for j in mask_long:
            epoxi_data_spec_temp.at[j[0],'cross_long'] +=  epoxi_data_spec_temp.at[j[0],'image'][:,j[1]] * epoxi_data_spec_temp.at[j[0],'delta_wave'][:,j[1]]        
                    
        # Every instance has 3 scans, find the best one and keep it
        first_scan = np.arange(0,epoxi_data_spec_temp.shape[0],3)
        indexes_max = []
                
        indexes_close = []
        for j in first_scan:
            sums = [np.sum(epoxi_data_spec_temp.at[j,'image']),np.sum(epoxi_data_spec_temp.at[j+1,'image']),np.sum(epoxi_data_spec_temp.at[j+2,'image'])]
            max_signal = np.max(sums)
            difference = sums-max_signal
            indexes_max.append(int(j+sums.index(max_signal)))
            # with 1.5e-11, only get some that have 1 close, with 2e-11 all 3 can sometimes be close
            indexes_close.append(j+np.where(np.logical_and(0<np.abs(difference), np.abs(difference)<5.0e-6))[0]) #DIFF
        indexes_max.append(int(np.median(indexes_max)))
        
        #SPEC = SPEC_SLOW[[ 1, 4, 7,10,13,15,18,21,24,27,30,33,37,18]]
        #  The other scan   0     6  9 12                      36
        #indexes_max = [1, 4, 7,10,13,15,18,21,24,27,30,33,37,18]
        #indexes_close = [0,-1,6,9,12,-1,-1,-1,-1,-1,-1,-1,36]
        spec = epoxi_data_spec_temp.loc[indexes_max,:].reset_index(drop = True)
        for idx,j in enumerate(indexes_close): # the ones that are not max, but close can be lower index than max so low:high indexing not possible
            if j.shape[0] >= 1:
            #if j >= 0:
                division_factor = (j.shape[0]+1)
                #division_factor = 2
                spec.at[idx,'duration'] = (spec.at[idx,'duration'] + np.sum(epoxi_data_spec_temp.loc[j,'duration']) ) /division_factor
                spec.at[idx,'minimum']  = np.minimum(spec.at[idx,'minimum'], np.min(epoxi_data_spec_temp.loc[j,'minimum']))
                spec.at[idx,'maximum']  = np.maximum(spec.at[idx,'maximum'], np.max(epoxi_data_spec_temp.loc[j,'maximum']))
                spec.at[idx,'median']   = np.median((spec.at[idx,'median'], np.median(epoxi_data_spec_temp.loc[j,'median'])))
                spec.at[idx,'one_sigma'] = np.sqrt((spec.at[idx,'one_sigma']**2 + np.sum(epoxi_data_spec_temp.loc[j,'duration']**2)) /division_factor)
                spec.at[idx,'RA']       = (spec.at[idx,'RA'] + np.sum(epoxi_data_spec_temp.loc[j,'RA'])) /division_factor 
                spec.at[idx,'DEC']      = (spec.at[idx,'DEC'] + np.sum(epoxi_data_spec_temp.loc[j,'DEC'])) /division_factor
                spec.at[idx,'range_SC'] = (spec.at[idx,'range_SC'] + np.sum(epoxi_data_spec_temp.loc[j,'range_SC'])) /division_factor
                # diameter is still 0
                spec.at[idx,'diameter'] = (spec.at[idx,'diameter'] + np.sum(epoxi_data_spec_temp.loc[j,'diameter'])) /division_factor
                spec.at[idx,'range_Sun'] = (spec.at[idx,'range_Sun'] + np.sum(epoxi_data_spec_temp.loc[j,'range_Sun'])) /division_factor
                spec.at[idx,'nppa']     = (spec.at[idx,'nppa'] + np.sum(epoxi_data_spec_temp.loc[j,'nppa'])) /division_factor
                #spec.at[indexes_max[idx],'oblateness'] = np.sum(spec.at[indexes_max[idx],'oblateness'] + epoxi_data_spec_temp.at[i[0],'oblateness']) * 0.5
                # illum is still 100%
                spec.at[idx,'illum']    = (spec.at[idx,'illum'] + np.sum(epoxi_data_spec_temp.loc[j,'illum'])) /division_factor
                spec.at[idx,'phase_angle'] = (spec.at[idx,'phase_angle'] + np.sum(epoxi_data_spec_temp.loc[j,'phase_angle'])) /division_factor
                ############### put back
                #spec.at[idx,'sub_SC']   = list((np.array(spec.at[idx,'sub_SC']) + np.sum(np.array(epoxi_data_spec_temp.loc[j,'sub_SC'].tolist()),axis = 0)) /division_factor)
                #spec.at[idx,'sub_Sun']  = list((np.array(spec.at[idx,'sub_Sun']) + np.sum(np.array(epoxi_data_spec_temp.loc[j,'sub_Sun'].tolist()),axis = 0)) /division_factor)
                spec.at[idx,'north_angle'] = (spec.at[idx,'north_angle'] + np.sum(epoxi_data_spec_temp.loc[j,'north_angle'])) /division_factor
                spec.at[idx,'Sun_angle'] = (spec.at[idx,'Sun_angle'] + np.sum(epoxi_data_spec_temp.loc[j,'Sun_angle'])) /division_factor
                spec.at[idx,'image']    = (spec.at[idx,'image'] + np.sum(epoxi_data_spec_temp.loc[j,'image'])) /division_factor
                spec.at[idx,'snr']      = np.sqrt((spec.at[idx,'snr']**2 + np.sum(epoxi_data_spec_temp.loc[j,'snr']**2)) /division_factor)
                spec.at[idx,'spectrum'] = (spec.at[idx,'spectrum'] + np.sum(epoxi_data_spec_temp.loc[j,'spectrum'])) /division_factor
                spec.at[idx,'spec_snr'] = np.sqrt((spec.at[idx,'spec_snr']**2 + np.sum(epoxi_data_spec_temp.loc[j,'spec_snr']**2)) /division_factor)
                spec.at[idx,'cross_short'] = (spec.at[idx,'cross_short'] + np.sum(epoxi_data_spec_temp.loc[j,'cross_short'])) /division_factor
                spec.at[idx,'cross_long'] = (spec.at[idx,'cross_long'] + np.sum(epoxi_data_spec_temp.loc[j,'cross_long'])) /division_factor
                            
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
        
        ##### SAVING
        ###spec.to_pickle('../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_spec_duration_'+mode+'.pkl')
    
        fig,ax = plt.subplots()
        ax.set_title('Infrared data Earth Observation '+str(observations[0])+' '+mode)
        ax.set_ylabel(r'signal ($10^{-7} W/m^2/\mu m$)')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength ($\mu m$)')
        ax.set_xlim([1.05,2.7])
        ax.grid(True)
        # plt.plot(spec.at[idx,'spec_wave'], spec.at[idx,'spectrum']*1e7)
        mask_short = np.where(spec.at[idx,'spec_wave']<2.7)
        ax.plot(spec.at[idx,'spec_wave'][mask_short], spec.at[idx,'spectrum'][mask_short]*1e7)
        
        fig2,ax2 = plt.subplots()
        ax2.set_title('Infrared data Earth Observation '+str(observations[0])+' '+mode)
        ax2.set_ylabel(r'signal ($10^{-7} W/m^2/\mu m$)')
        ax2.set_yscale('log')
        ax2.set_xlabel('Wavelength ($\mu m$)')
        ax2.set_xlim([2.5,4.5])
        ax2.grid(True)
        # plt.plot(spec.at[idx,'spec_wave'], spec.at[idx,'spectrum']*1e7)
        mask_long = np.where(spec.at[idx,'spec_wave']>2.5)
        ax2.plot(spec.at[idx,'spec_wave'][mask_long], spec.at[idx,'spectrum'][mask_long]*1e7)  
        
        fig3,ax3 = plt.subplots()
        ax3.grid(True)
        ax3.set_title('Infrared data cross_long '+str(observations[0])+' '+mode)
        ax3.plot(np.arange(0,128),spec.at[idx,'cross_long'])
        
        fig4,ax4 = plt.subplots()
        ax4.grid(True)
        ax4.set_title('Infrared data cross_short '+str(observations[0])+' '+mode)
        ax4.plot(np.arange(0,128),spec.at[idx,'cross_short'])
        
    return epoxi_data_spec_temp,spec
        
#%%

if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    year = '2008'
    observations = ['078','079'] 
    #observations = ['149','150'] 
    #observations = ['156','157'] 
    #year = '2009'
    #observations = ['086','087'] 
    #observations = ['277','278']
    df,df2 = ir_spectra(year,observations, remove_background=True)
        
    #%%
    
#for j in np.arange(df.shape[0]):
for j in np.arange(4):
    fig5,ax5 = plt.subplots()
    ax5.grid(True)
    ax5.set_title('Infrared data cross_long '+str(observations[0]))
    ax5.plot(np.arange(0,128),df.at[j,'cross_long'])
    ax5.set_ylabel(r'signal times wavelength spacing ($10^{-7} W/m^2$)')
    ax5.set_xlabel('Row of the image')    
    
    # fig6,ax6 = plt.subplots()
    # ax6.grid(True)
    # ax6.set_title('Infrared data cross_short '+str(observations[0]))
    # ax6.plot(np.arange(0,128),df.at[j,'cross_short'])