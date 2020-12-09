# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:48:58 2020

@author: roder

CODE ADAPTED FROM EXISTING CODE BY DR. TIMOTHY A. LIVENGOOD
COPIED COMMENTS/CODE IS NOT EXPLICITELY REFERENCED
"""
import numpy as np  # for calculations
import matplotlib.pyplot as plt  # for plot

from astropy.io import fits   # For reading the files, opening and closing
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

import glob  # to acces the files on pc

import pandas as pd # to save the data
import scipy.ndimage as ndim # to use the median filter


#%%
def load_epoxi_IR(year,observations,background_frames = [3,4,8],signal_frames = [5,6,7]):
    background_list = []
    background_snr_list = []
    for i in background_frames:
        filepath = '../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_frame'+str(i)+'.pkl'
        epoxi_temp = pd.read_pickle(filepath)
        if i==4:
            # We prepare the BACKGROUND array by simply reading one of the frame
            # series. I choose to use frame 4, because it is in the middle of
            # each set and thus likely to best represent midpoint behavior in
            # the descriptive data and any variability in the dark frames.        
            background = epoxi_temp.copy()
        image_arr   = np.array(epoxi_temp['image'].tolist()) # df.to_array didn't work, so first to list and then array
        snr_arr     = np.array(epoxi_temp['snr'].tolist())
        epoxi_temp['image'] = list(ndim.median_filter(image_arr,5)) # array does not work so turn back into a list 
        epoxi_temp['snr']   = list(ndim.median_filter(snr_arr,5))
        background_list.append(epoxi_temp['image'].tolist())
        background_snr_list.append(epoxi_temp['snr'].tolist())
    # equivalent to IMGS3 and SNR_3 in Livengood
    background_arr = np.array(background_list)
    background_snr_arr = np.array(background_snr_list)
    # list only turn first axis into list, arr.tolist() turns all dimensions to lists
    background['image'] = list(np.median(background_arr,axis = 0)) 
    background['snr'] = list(np.median(background_snr_arr,axis = 0))
    
    mask_short = np.where(np.array(background['spec_wave'].tolist())<2.7*128)
    mask_short = zip(mask_short[0],mask_short[1])
    for i in mask_short:
        # add image*delta, all rows of the corresponding column
        # To get the cross-dispersion (parallel to slit) distribution of signal.
        background.at[i[0],'cross_short'] +=  background.at[i[0],'image'][:,i[1]] * background.at[i[0],'delta_wave'][:,i[1]]
    mask_long = np.where(np.array(background['spec_wave'].tolist())>2.7*128)
    mask_long = zip(mask_long[0],mask_long[1])
    for i in mask_long:
        background.at[i[0],'cross_long'] +=  background.at[i[0],'image'][:,i[1]] * background.at[i[0],'delta_wave'][:,i[1]]
    
    combined_spec_list = []
    combined_spec_snr_list = []
    for i in signal_frames:
        filepath = '../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_frame'+str(i)+'.pkl'
        epoxi_temp = pd.read_pickle(filepath)
        if i ==6:
            combined_spec = epoxi_temp.copy()
        
        epoxi_temp['image'] = epoxi_temp['image'] - background['image']
        # where background SNR==0 set image to 0
        epoxi_temp['image'] = list(np.where(np.array(list(background['snr'])) == 0,0,np.array(list(epoxi_temp['image']))))
        
        image_arr   = np.array(epoxi_temp['image'].tolist()) # df.to_array didn't work, so first to list and then array
        snr_arr     = np.array(epoxi_temp['snr'].tolist())
        epoxi_temp['image'] = list(ndim.median_filter(image_arr,3)) # array does not work so turn back into a list 
        epoxi_temp['snr']   = list(ndim.median_filter(snr_arr,3))
        combined_spec_list.append(epoxi_temp['image'].tolist())
        combined_spec_snr_list.append(epoxi_temp['snr'].tolist())
        
        naxis2 = epoxi_temp.at[0,'naxis2']
        spec_arr = np.zeros((78,512))
        spec_wave_arr = np.zeros((78,512))
        spec_delta_wave_arr = np.zeros((78,512))
        spec_snr_arr = np.zeros((78,512))
        for i in np.arange(0,naxis2): # loop through the rows
            # sum of all rows for every column 
            spec_arr        += np.array(list(epoxi_temp['image']))[:,i,:]  # (SPATIAL) SUM OF THE SPECTRUM NOT AVERAGE
            spec_wave_arr   += np.array(list(epoxi_temp['wavelength']))[:,i,:]
            spec_delta_wave_arr += np.array(list(epoxi_temp['delta_wave']))[:,i,:]
            spec_snr_arr    += np.array(list(epoxi_temp['snr']))[:,i,:]**2
        epoxi_temp['spectrum']  = list(spec_arr)
        epoxi_temp['spec_wave'] = list(spec_wave_arr /naxis2) # AVERAGE IS NOT DONE IN FILEREADER AND FOR BACKGROUND
        epoxi_temp['spec_delta_wave'] = list(spec_delta_wave_arr /naxis2)
        epoxi_temp['spec_snr']  = list(np.sqrt(spec_snr_arr))   
        
        mask_short = np.where(np.array(epoxi_temp['spec_wave'].tolist())<2.7*128)
        mask_short = zip(mask_short[0],mask_short[1])
        for i in mask_short:
            # add image*delta, all rows of the corresponding column
            # To get the cross-dispersion (parallel to slit) distribution of signal.
            epoxi_temp.at[i[0],'cross_short'] +=  epoxi_temp.at[i[0],'image'][:,i[1]] * epoxi_temp.at[i[0],'delta_wave'][:,i[1]]
        mask_long = np.where(np.array(epoxi_temp['spec_wave'].tolist())>2.7*128)
        mask_long = zip(mask_long[0],mask_long[1])
        for i in mask_long:
            epoxi_temp.at[i[0],'cross_long'] +=  epoxi_temp.at[i[0],'image'][:,i[1]] * epoxi_temp.at[i[0],'delta_wave'][:,i[1]]
        
        epoxi_temp['oblateness'] = 1.0/298.257 # From Livengood taken from ephemerisis
    
    combined_spec_arr = np.array(combined_spec_list)
    combined_spec_snr_arr = np.array(combined_spec_snr_list)
    combined_spec['image'] = list(np.sum(combined_spec_arr,axis = 0)) 
    combined_spec['snr'] = list(np.sqrt(np.sum(combined_spec_snr_arr*combined_spec_snr_arr,axis = 0))) # squared sum, then sqrt
    
    naxis2 = combined_spec.at[0,'naxis2']
    spec_arr = np.zeros((78,512))
    spec_wave_arr = np.zeros((78,512))
    spec_delta_wave_arr = np.zeros((78,512))
    spec_snr_arr = np.zeros((78,512))
    for i in np.arange(0,naxis2): # loop through the rows
        # sum of all rows for every column 
        spec_arr        += np.array(list(combined_spec['image']))[:,i,:]  # (SPATIAL) SUM OF THE SPECTRUM NOT AVERAGE
        spec_wave_arr   += np.array(list(combined_spec['wavelength']))[:,i,:]
        spec_delta_wave_arr += np.array(list(combined_spec['delta_wave']))[:,i,:]
        spec_snr_arr    += np.array(list(combined_spec['snr']))[:,i,:]**2
    combined_spec['spectrum']  = list(spec_arr)
    combined_spec['spec_wave'] = list(spec_wave_arr /naxis2) # AVERAGE IS NOT DONE IN FILEREADER AND FOR BACKGROUND
    combined_spec['spec_delta_wave'] = list(spec_delta_wave_arr /naxis2)
    combined_spec['spec_snr']  = list(np.sqrt(spec_snr_arr))   
    
    mask_short = np.where(np.array(combined_spec['spec_wave'].tolist())<2.7*128)
    mask_short = zip(mask_short[0],mask_short[1])
    for i in mask_short:
        # add image*delta, all rows of the corresponding column
        # To get the cross-dispersion (parallel to slit) distribution of signal.
        combined_spec.at[i[0],'cross_short'] +=  combined_spec.at[i[0],'image'][:,i[1]] * combined_spec.at[i[0],'delta_wave'][:,i[1]]
    mask_long = np.where(np.array(combined_spec['spec_wave'].tolist())>2.7*128)
    mask_long = zip(mask_long[0],mask_long[1])
    for i in mask_long:
        combined_spec.at[i[0],'cross_long'] +=  combined_spec.at[i[0],'image'][:,i[1]] * combined_spec.at[i[0],'delta_wave'][:,i[1]]
    combined_spec.to_pickle('../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_combined_spec''.pkl')
    return 

#%%    
if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    # INPUT
    year = '2008'
    observations = ['078','079'] 
    #observations = ['149','150'] 
    #observations = ['156','157'] 
    year = '2009'
    #observations = ['086','087'] 
    observations = ['277','278']
    
    for idx,i in enumerate(observations):
        epoxi_data_temp = pd.read_hdf('../output/IR_RAD_'+year+'_'+i+'_dictionary_info.h5')
        #epoxi_data_temp = pd.read_hdf('../output/RADREV_'+year+'_'+i+'_min_aper_150_dictionary_info.h5')
        if idx ==0:
            epoxi_data = epoxi_data_temp
        else:
            epoxi_data = pd.concat([epoxi_data,epoxi_data_temp], ignore_index=True)
    
    used_frames = [3,4,5,6,7,8]
    for i in used_frames:
        epoxi_temp = epoxi_data[epoxi_data['image_number']==i]
        epoxi_temp = epoxi_temp.reset_index(drop=True)
        epoxi_temp.to_pickle('../output/IR_RAD_'+year+'_'+observations[0]+'_'+observations[1]+'_frame'+str(i)+'.pkl')
    #%%
    load_epoxi_IR(year,observations)

