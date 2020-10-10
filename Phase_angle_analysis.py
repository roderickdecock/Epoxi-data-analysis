# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:34:00 2020

@author: roder
"""
import pandas as pd # to save the data

import numpy as np

import matplotlib.pyplot as plt  # for plot

import scipy.signal as sig # to use the median filter
#%%
year = '2009'
#observations = ['078','079'] 
#observations = ['149','150'] 
#observations = ['156','157'] 
observations = ['086','087']
i = 350

filepath = r'../output/'+year+'_'+observations[0]+'_'+observations[1]+'_df_epoxi_data_filtered_'+str(i)+'.pkl'
epoxi_data_filter = pd.read_pickle(filepath)


#%%



year = ['2008','2009']
plt.figure()

for i in year:
    if i == '2008':
        observations = [('078','079'), ('149','150'), ('156','157')]
    else:
        observations = [('086','087')]
    for j in observations:
        print(j)
        idx = 0
        wavelengths = [350,450,650,850]
        colours = ['y','b','g','r']
        for k in wavelengths:
            filepath = r'../output/'+i+'_'+j[0]+'_'+j[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
            epoxi_data_filter = pd.read_pickle(filepath)
            pixel_solid_angle = 2.0e-06 * 2.0e-06
            phase_angles = epoxi_data_filter['phase_angle']
            scaled_signal = epoxi_data_filter['signal']*pixel_solid_angle
            plt.plot(phase_angles,scaled_signal, color = colours[idx])            
            idx +=1

plt.grid(True)
plt.xlim(0,90)
#plt.ylim(0,5.5e-7)








