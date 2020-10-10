# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:34:29 2020

@author: roder

CODE ADAPTED FROM EXISTING CODE BY DR. TIMOTHY A. LIVENGOOD
COPIED COMMENTS/CODE IS NOT EXPLICITELY REFERENCED
"""

import pandas as pd # to save the data

import numpy as np

import matplotlib.pyplot as plt  # for plot

import scipy.signal as sig # to use the median filter

from filereader import aper_photom  # python runs the code of filereader  when importing the if __name__ == "__main__": prevents this



#%%
def update_signal(epoxi_data_filter):
    for i in np.arange(epoxi_data_filter.shape[0]):
        centre = epoxi_data_filter.at[i,'target_center']
        width_trim = 6 # should correspond to the width trim that was originally applied
        image_prim = epoxi_data_filter.at[i,'image'][width_trim:512-width_trim,width_trim:512-width_trim] # elimanating the zeros at the edge
        weight = epoxi_data_filter.at[i,'weight'][width_trim:512-width_trim,width_trim:512-width_trim]
        med_image = sig.medfilt(image_prim*weight,3)
        
        aper_radius = np.minimum(1.01*epoxi_data_filter.at[i,'earth_radius_pxl'],image_prim.shape[0])
        aper_finish = np.minimum(4.0*epoxi_data_filter.at[i,'earth_radius_pxl'],image_prim.shape[0]) 
        
        done = False
        prev_signal = 0
        while done == False:
            signal_aperture, final, patch, annulus_radius  = aper_photom(med_image, centre = centre, radius = aper_radius)
            ### figure
            # fig2, ax2 = plt.subplots()
            # plt.title('Median data')
            # plt.imshow(med_image, cmap='gray')
            # circle1 = plt.Circle(centre+255.5, aper_radius, color='r', fill=False)
            # ax2.add_artist(circle1)
            # plt.scatter(centre[0]+255.5, centre[1]+255.5, s=10)
            # plt.colorbar()
    
            epoxi_data_filter.at[i,'signal']         = final[0]
            epoxi_data_filter.at[i,'signal_rms']     = final[3]
            epoxi_data_filter.at[i,'background']     = final[2]
            epoxi_data_filter.at[i,'background_rms'] = final[5]
            epoxi_data_filter.at[i,'aperture']       = final[6] * 2
      
            aper_radius += 2
            if aper_radius>aper_finish or np.abs((signal_aperture - prev_signal)/signal_aperture) < 5e-4: #1e-3s
                done = True
            prev_signal = signal_aperture
        # fig2, ax2 = plt.subplots()
        # plt.title('Median data')
        # plt.imshow(med_image, cmap='gray')
        # circle1 = plt.Circle(centre+255.5, aper_radius, color='r', fill=False)
        # ax2.add_artist(circle1)
        # plt.scatter(centre[0]+255.5, centre[1]+255.5, s=10)
        # plt.colorbar()
        
        # print(final)
    #print(np.sum(epoxi_data_filter['signal_rms']/epoxi_data_filter['signal'])/epoxi_data_filter['signal'].shape[0])
    return epoxi_data_filter

#%%
##################### 

def scale_to_range(df_epoxi_data, normalise = False): # dataFrame with epoxi data
    signal = df_epoxi_data['signal']  # np.array(epoxi_data['signal']) to make it into an array istead of series
    
    # average_range_SC_true = np.sum(df_epoxi_data['range_SC'])/df_epoxi_data['range_SC'].size
    # average_range_Sun_true = np.sum(df_epoxi_data['range_Sun'])/df_epoxi_data['range_Sun'].size
    # print(average_range_SC_true)
    # print(average_range_Sun_true)
    # average_range_SC is 
    
    # the values are in AU, scaled to 1 AU equivalent range from the spacecraft
    average_range_SC = 1.0 # check this!!! With other observations
    average_range_Sun = 1.0
    # Correct summed signal for physical effects -- 1/r^2 for distance,
    # scale back to a fully-illuminated disc not necessary as illumination of disc is not known.
    
    signal = signal * ((df_epoxi_data['range_SC'] /average_range_SC ) * (df_epoxi_data['range_Sun']/average_range_Sun))**2   
    df_epoxi_data['scaled signal'] = signal
    
    if normalise == True:
        # Divide signal by its own average value.    
        average_signal = np.sum(signal) / signal.size # needed for the lightcurves, not for values 
        signal     = signal / average_signal
        df_epoxi_data['normalised signal'] = signal
    return signal

#%%

def lightcurves_plot(year,observations,wavelengths,colours, pixel_solid_angle):
    plt.figure()
    idx = 0
    for i in wavelengths:
        filepath = r'../output/'+year+'_'+observations[0]+'_'+observations[1]+'_df_epoxi_data_filtered_'+str(i)+'.pkl'
        epoxi_data_filter = pd.read_pickle(filepath)
        epoxi_data_filter = update_signal(epoxi_data_filter)
        epoxi_data_filter['signal'] = scale_to_range(epoxi_data_filter)
        diurnally_averaged_signal = np.sum(epoxi_data_filter['signal']*pixel_solid_angle)/epoxi_data_filter['signal'].shape[0]
        print(i,diurnally_averaged_signal)
        epoxi_data_filter['diurnally averaged signal'] = diurnally_averaged_signal
        #print(np.sum(epoxi_data_filter['signal_rms']/epoxi_data_filter['signal'])/epoxi_data_filter['signal'].shape[0])
        di_avg_signal_std = np.std(epoxi_data_filter['signal']*pixel_solid_angle)
        print('std',di_avg_signal_std) # standard deviation
        epoxi_data_filter['diurnally averaged signal std'] = di_avg_signal_std
        
        if i ==450 or i==550 or i==650 or i==850:
            normalised_signal_filter = scale_to_range(epoxi_data_filter,normalise=True)
            
            ### imaging
            filtered_longtitudes = []
            for j in np.arange(normalised_signal_filter.shape[0]):
                filtered_longtitudes.append(epoxi_data_filter.at[j,'sub_SC_nom'][0])
                # nominal sub_SC as the other sub_SC has faulty values
            
            x = np.array(filtered_longtitudes)%360
            y = np.array(normalised_signal_filter)
        
            minimum_index = np.where(x==np.min(x))[0][0] # where it goes from 360 to 0 degrees 
            # allows to plot the observations separatly so no line from 360 to zero 
            # and no line between first and last observation
            
            plt.plot(x[0:minimum_index],y[0:minimum_index], label = 'CW'+str(i),color = colours[idx])
            plt.plot(x[minimum_index:],y[minimum_index:],color = colours[idx])
            idx += 1
        epoxi_data_filter.to_pickle('../output/'+year+'_'+observations[0]+'_'+observations[1]+'_'+'df_epoxi_data_filtered_'+str(i)+'.pkl')
    plt.xlim(0,360)
    plt.ylim(0.8,1.2)
    plt.grid(True)
    plt.legend()
    return
#%%
### RERUN EVERYTHING
# Done: 
# 2008 078,079 149,150 156, 157
# 2009 086,087

if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    # INPUT
    year = '2008'
    #observations = ['078','079'] 
    #observations = ['149','150'] 
    observations = ['156','157'] 
    #observations = ['086','087']
    
    wavelengths = [350,450,550,650,750,850,950]
    colours = ['b','g','r','y'] # plots are 2 lines combined so manually assign colours
    # CONSTANT
    pixel_solid_angle = 2.0e-06 * 2.0e-06

    lightcurves_plot(year, observations, wavelengths, colours, pixel_solid_angle)
    


