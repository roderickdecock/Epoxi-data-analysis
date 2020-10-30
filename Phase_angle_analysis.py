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
            if j[0] == '149' or j[0]== '086':
                filepath = r'../output/RADREV_'+i+'_'+j[0]+'_'+j[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
            epoxi_data_filter = pd.read_pickle(filepath)
            pixel_solid_angle = 2.0e-06 * 2.0e-06
            phase_angles = epoxi_data_filter['phase_angle']
            scaled_signal = epoxi_data_filter['scaled signal']*pixel_solid_angle
            plt.scatter(phase_angles,scaled_signal, color = colours[idx],s=1)            
            idx +=1
        if j[0]=='078':
            fig.legend(['350','450','650','850'])

fig.text(58,1.5e-7,'EarthObs1',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(75,1e-7,'EarthObs4',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(76.6,0.7e-7,'EarthObs5',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(85,0.5e-7,'PolarObs1North',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
fig.text(87,0.3e-7,'PolarObs2South',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
plt.grid(True)
ax.set_xlim(0,90)
ax.set_ylim(0,5.5e-7)
ax.set_ylabel('scaled signal W/$m^2$/$\mu m$')
ax.set_xlabel('phase angle')


#%%

earth_diam = 1.2756e04         #Earth diameter in km
astronomical_unit = 149.597870691e06 # km

SOLAR_350 = 1.033249e+03  #W/m^2/um at 1 AU from Sun
SOLAR_450 = 1.878244e+03  #W/m^2/um at 1 AU from Sun
SOLAR_550 = 1.850052e+03  #W/m^2/um at 1 AU from Sun
SOLAR_650 = 1.595328e+03  #W/m^2/um at 1 AU from Sun
SOLAR_750 = 1.277213e+03  #W/m^2/um at 1 AU from Sun
SOLAR_850 = 1.027796e+03  #W/m^2/um at 1 AU from Sun
SOLAR_950 = 8.278461e+02  #W/m^2/um at 1 AU from Sun

scaling   = (astronomical_unit / (earth_diam*0.5))**2
SOLAR_350 = SOLAR_350 / scaling
SOLAR_450 = SOLAR_450 / scaling
SOLAR_550 = SOLAR_550 / scaling
SOLAR_650 = SOLAR_650 / scaling
SOLAR_750 = SOLAR_750 / scaling
SOLAR_850 = SOLAR_850 / scaling
SOLAR_950 = SOLAR_950 / scaling

radius = earth_diam*0.5
distance = 1*astronomical_unit
solid_angle = 2.0e-06 * 2.0e-06

phase_angle = np.arange(0,91,1)

ill_frac = 0.5*(1 + np.cos(np.deg2rad(phase_angle)))
lamber_phase_func = ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
                     np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi

#%%
### Mallama uses data from Goode
# angles from Goode and the corresponding effective albedo
# Earth phase = 180-lunar phase, lunar phase is shown on graph
phase_angle_goode = [0,15,30,60,90]
effective_albedo_astr =  np.array([1.15,1.3,1.45,1.55,1.3])
effective_albedo =  10**(effective_albedo_astr*-1/2.5)
# for phase_angle = 0, effective albedo = 1.15
# for phase_angle = 60, effective albedo = 1.55
# for phase_angle = 30, effective albedo = 1.45
incident_irradiance_list = []
for idx,i in enumerate(phase_angle_goode):
    # Lambert phase function of Goode
    lamber_phase_func_2 = ( (np.pi - np.abs(np.deg2rad(i)) ) * np.cos(np.deg2rad(i)) + \
                     np.sin(np.abs(np.deg2rad(i))) )/np.pi
    
    # Mallama says that ''Multiplying their effective albedo by their Lambert phase law 
    # representation and by two-thirds retrieves the phase function''
    phase_func_mallama = 2/3*lamber_phase_func_2*effective_albedo[idx]
    print(i)
    print(phase_func_mallama)
    # values are off for 0 and 15
    # phase function mallama =/= lambert phase function    
    incident_irradiance_list.append(phase_func_mallama)
    
plt.plot(phase_angle_goode,incident_irradiance_list)   #, color = colours[idx]
     
# table 1 mallama contains j(alpha) normally, but he normalises with the incident flux
# so for alpha = 0 this is the geometric albedo
# for the others j(alpha) = Phi(alpha)*j(0), this is divided by pi*F to normalise
# so this results in j(alpha) = Phi(alpha) * Ag

#%%
mallama_angles = [0,15,30,60,90]
mallama_points = np.array([0.2,0.18,0.15,0.1,0.06])

#plt.figure()
list_wl = [SOLAR_350,SOLAR_450,SOLAR_650,SOLAR_850]
for idx, i in enumerate(list_wl):
    incident_irradiance = 2/3*i*lamber_phase_func * 0.2 # times 0.2 as that is the geometric albedo
    incident_irradiance = 2/3*i*lamber_phase_func 
    #plt.plot(phase_angle,incident_irradiance, color = colours[idx])    
    plt.plot(mallama_angles,mallama_points*i,  color = colours[idx])
    ### def j(alpha) in analytic models
    # j(alpha) = 2/3 * omega * pi*F * lamber_phase_func  #omega = 1 for Lambert
    incident_irradiance = 2/3*i*lamber_phase_func
    # rescaling is done in the incoming flux, part of SOLAR
    power_lamb = incident_irradiance * (radius/distance)**2 * solid_angle
    # These values are too large
    plt.plot(phase_angle,incident_irradiance, color = colours[idx]) 
    
    # geometric albedo, Ag = 0.2 = j(0)/(pi*F) with j(0) flux at phase = 0 and pi*F is the incoming/incident flux
    # phase curve, Phi(alpha) = j(alpha)/j(0) with phase angle alpha
    # Ag * Phi(alpha) =  j(alpha)/(pi*F) # This is what Mallama does normalised j(alpha)
    # we don't have an observation at j(0) so we use Ag*(pi*F)
    # j(alpha) = Ag * Phi(alpha)*(pi*F)
    # geometric albedo should be 2/3*omega with omega=1 for a lambertian surface
    # using a geometic albedo = 0.2 from Mallama works better
    test = 0.2*lamber_phase_func*i
    plt.plot(phase_angle,test, color = colours[idx])
    
plt.legend()





