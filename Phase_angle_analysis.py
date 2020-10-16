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

year = ['2008','2009']
fig, ax = plt.subplots()

for i in year:
    if i == '2008':
        observations = [('078','079'), ('149','150'), ('156','157')]
    else:
        observations = [('086','087'),('277','278')]
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
            scaled_signal = epoxi_data_filter['scaled signal']*pixel_solid_angle
            plt.plot(phase_angles,scaled_signal, color = colours[idx])            
            idx +=1
        if j[0]=='078':
            fig.legend(['350','450','650','850'])

fig.text(58,1.5e-7,'EarthObs1',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(75,1e-7,'EarthObs4',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(76.6,0.7e-7,'EarthObs5',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(87,0.5e-7,'PolarObs1North',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(86,0.3e-7,'PolarObs2South',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
plt.grid(True)
ax.set_xlim(0,90)
ax.set_ylim(0,5.5e-7)
ax.set_ylabel('unscaled signal W/$m^2$/$\mu m$')
ax.set_xlabel('phase angle')








