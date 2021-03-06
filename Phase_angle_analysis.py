# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:34:00 2020

@author: roder
"""
import pandas as pd # to save the data

import numpy as np

import matplotlib.pyplot as plt  # for plot

#from pylab import legend, gca

#%%

earth_diam = 1.2756e04         #Earth diameter in km
astronomical_unit = 149.597870691e06 # km

# Values from ASTM 2000 Standard Extraterrestrial Spectrum Reference E-490-00
# Values retrieved by Dr. Livengood
# These values are adpated for the wavelength range of the instrument
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
#phase_angle = np.arange(0,181,1)

# ill_frac = 0.5*(1 + np.cos(np.deg2rad(phase_angle)))
lamber_phase_func = ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
                     np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi

def lamber_phase_function(phase_angle):
    return ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
                     np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi
        
def lamber_phase_func_sphere(phase_angle):
    scattering_angle = np.deg2rad(180 - phase_angle)
    return 8/(3*np.pi) * (np.sin(scattering_angle) - scattering_angle*np.cos(scattering_angle))
    
#%%

def phase_curve(year,list_wl, mallama = False, scale_to_E1 = False, scale_to_E1_E4_E5 = False, scale_to_P1_P2 = False):
    fig, ax = plt.subplots()
    
    averages_signal_obs = []
    averages_reflectance_obs = []
    geometric_albedo_list = []
    geometric_albedo_std_list = []
    geometric_albedo_avg_list = []
    
    if mallama == True:
        error_file = open('../output/errors_mallama.txt','w')
    elif scale_to_E1 == True:
        error_file = open('../output/errors_scale_to_E1.txt','w')
    elif scale_to_E1_E4_E5 == True:
        error_file = open('../output/errors_scale_to_E1_E4_E5.txt','w')
    elif scale_to_P1_P2 == True:
        error_file = open('../output/errors_scale_to_P1_P2.txt','w')
    
    for i in year:
        if i == '2008':
            observations = [('078','079'), ('149','150'), ('156','157')]
            #observations = [('156','157')]
        else:
            observations = [('086','087'),('277','278')]
        for j in observations:
            print(j)
            error_file.write(str(j)+'\n')
            wavelengths = [350,450,550,650,750,850,950]
            colours = ['y','b','c','g','m','r','k']
            average_signal = []
            average_reflectance = []
            geometric_albedo_temp = []
            geometric_albedo_temp_std = []            
            
            for idx,k in enumerate(wavelengths):
                filepath = r'../output/'+i+'_'+j[0]+'_'+j[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                if j[0] == '149' or j[0]== '086': #using different background method for these observations
                    filepath = r'../output/RADREV_'+i+'_'+j[0]+'_'+j[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                epoxi_data_filter = pd.read_pickle(filepath)
                pixel_solid_angle = 2.0e-06 * 2.0e-06
                phase_angles_obs = epoxi_data_filter['phase_angle']
                scaled_signal = epoxi_data_filter['scaled signal']*pixel_solid_angle
                avg_phase_angles_obs = int(np.round(np.mean(phase_angles_obs)))
                #print(np.mean(phase_angles_obs))
                if mallama ==True:
                    geometric_albedo = 0.2
                    geometric_albedo_avg_list.append(geometric_albedo)
                    signal = geometric_albedo*lamber_phase_func*list_wl[idx]
                    if j == ('078','079'): #only need to plot this once, the first time
                        plt.plot(phase_angle,signal, color = colours[idx])                      
                
                elif scale_to_E1 == True:
                    if j == ('078','079'):                    
                        lamber_phase_func_alpha = lamber_phase_function(np.array(phase_angles_obs, dtype=np.float64))
                        geometric_albedo = scaled_signal * 1/(list_wl[idx]*lamber_phase_func_alpha)
                        geometric_albedo_avg = np.mean(geometric_albedo)
                        geometric_albedo_avg_list.append(geometric_albedo_avg)
                        signal = geometric_albedo_avg*lamber_phase_func*list_wl[idx]
                        plt.plot(phase_angle,signal, color = colours[idx])
                        #print(signal[0],signal[5],signal[10])                         
                        
                elif scale_to_E1_E4_E5 == True:
                    if i =='2008':
                        albedo_list = []
                        for l in observations: # loop through E1,E4,E5 to get the data for the Lambert phase
                            filepath_temp = r'../output/'+i+'_'+l[0]+'_'+l[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                            if l[0] == '149':
                                filepath_temp = r'../output/RADREV_'+i+'_'+l[0]+'_'+l[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                            epoxi_data_filter_temp = pd.read_pickle(filepath_temp)
                            phase_angles_obs_temp = epoxi_data_filter_temp['phase_angle']
                            scaled_signal_temp = epoxi_data_filter_temp['scaled signal']*pixel_solid_angle
                            # now find the albedo
                            lamber_phase_func_alpha = lamber_phase_function(np.array(phase_angles_obs_temp, dtype=np.float64))
                            geometric_albedo = scaled_signal_temp * 1/(list_wl[idx]*lamber_phase_func_alpha)
                            albedo_list.append(geometric_albedo)
                        geometric_albedo_avg = np.mean(albedo_list)
                        geometric_albedo_avg_list.append(geometric_albedo_avg)
                        signal = geometric_albedo_avg*lamber_phase_func*list_wl[idx]
                        plt.plot(phase_angle,signal, color = colours[idx]) 
                
                elif scale_to_P1_P2 == True:
                    if i =='2008': # don't want to loop trough it twice for no reason
                        year_temp = '2009' 
                        albedo_list = []
                        observations_temp = [('086','087'),('277','278')]
                        for l in observations_temp: # loop trhough P1,P2 to get the data for the Lambert phase
                            filepath_temp = r'../output/'+year_temp+'_'+l[0]+'_'+l[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                            if l[0] == '086':
                                filepath_temp = r'../output/RADREV_'+year_temp+'_'+l[0]+'_'+l[1]+'_df_epoxi_data_filtered_'+str(k)+'.pkl'
                            epoxi_data_filter_temp = pd.read_pickle(filepath_temp)
                            phase_angles_obs_temp = epoxi_data_filter_temp['phase_angle']
                            scaled_signal_temp = epoxi_data_filter_temp['scaled signal']*pixel_solid_angle
                            # now find the albedo
                            lamber_phase_func_alpha = lamber_phase_function(np.array(phase_angles_obs_temp, dtype=np.float64))
                            geometric_albedo = scaled_signal_temp * 1/(list_wl[idx]*lamber_phase_func_alpha)
                            albedo_list.append(geometric_albedo)
                        geometric_albedo_avg = np.mean(albedo_list)
                        geometric_albedo_avg_list.append(geometric_albedo_avg)
                        signal = geometric_albedo_avg*lamber_phase_func*list_wl[idx]
                        plt.plot(phase_angle,signal, color = colours[idx]) 
                else:
                    signal = np.zeros(scaled_signal.shape)
                                
                ####### RECALCULATE FOR SCALE TO MALLAMA, E1_4_5 and P1_P2
                ####### RECALCULATE FOR SCALE TO MALLAMA, E1_4_5 and P1_P2 and P_1 with stand_error_mean
                ### It is working now, recalculate the signal for each wavelength
                
                signal = geometric_albedo_avg_list[idx]*lamber_phase_func*list_wl[idx]
                #print(signal[0],signal[5],signal[10])  
                error = scaled_signal - signal[avg_phase_angles_obs]
                mean_error = np.mean(np.abs(error))                
                # standard deviations represent the variation due to the Earth rotation. It is not a representation
                # of the standard deviation of the error.
                std_error = np.std(error)
                # standard error of the mean, to get an estimate for the error of the mean, so how
                # accurately the mean represents sample data
                stand_error_mean = np.std(error, ddof=1) / np.sqrt(np.size(error))
                #print(k,'mean error',mean_error,'std error', std_error, 'st error mean', stand_error_mean)
                L = [str(k),' mean error ',str(mean_error),' std error ', str(std_error), ' stand_error_mean ', str(stand_error_mean),  '\n']
                error_file.writelines(L)
                
                lamber_phase_func_alpha = lamber_phase_function(np.array(phase_angles_obs, dtype=np.float64))
                geometric_albedo = scaled_signal * 1/(list_wl[idx]*lamber_phase_func_alpha)
                geometric_albedo_temp.append(np.mean(geometric_albedo))
                geometric_albedo_temp_std.append(np.std(geometric_albedo))
                average_signal.append(np.mean(scaled_signal))
                average_reflectance.append(np.mean(scaled_signal)/list_wl[idx])
                plt.scatter(phase_angles_obs,scaled_signal, color = colours[idx],s=1) 
            averages_signal_obs.append(average_signal)
            averages_reflectance_obs.append(average_reflectance)
            geometric_albedo_list.append(geometric_albedo_temp)
            geometric_albedo_std_list.append(geometric_albedo_temp_std)
            if j[0]=='078':
                # Use the next line as legend
                #fig.legend(['350','450','550','650','750','850','950'], bbox_to_anchor=(0.13, 0.12),loc = 'lower left')
                
                # This is specific for scale_to_E1, to have the wavelength at every line
                l1 = plt.legend(['350'], loc = 'upper left', bbox_to_anchor=(0, 5.4e-7), bbox_transform = ax.transData)
                ax.add_artist(l1)
                l2 = plt.legend(['450'], loc = 'upper left', bbox_to_anchor=(58, 5.5e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l2)
                l2.get_lines()[0].set_color(colours[1])
                l3 = plt.legend(['550'], loc = 'upper left', bbox_to_anchor=(34, 5.5e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l3)
                l3.get_lines()[0].set_color(colours[2])
                l4 = plt.legend(['650'], loc = 'upper left', bbox_to_anchor=(0, 4.69e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l4)
                l4.get_lines()[0].set_color(colours[3])
                l5 = plt.legend(['750'], loc = 'upper left', bbox_to_anchor=(0, 3.95e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l5)
                l5.get_lines()[0].set_color(colours[4])
                l6 = plt.legend(['850'], loc = 'upper left', bbox_to_anchor=(0, 3.4e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l6)
                l6.get_lines()[0].set_color(colours[5])
                l7 = plt.legend(['950'], loc = 'upper left', bbox_to_anchor=(0, 2.0e-7), bbox_transform = ax.transData) #, bbox_transform = ax.transData
                ax.add_artist(l7)
                l7.get_lines()[0].set_color(colours[6])
    error_file.close()
    
    fig.text(58,1e-7,'EarthObs1',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(75,4e-7,'EarthObs4',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(76.7,0.7e-7,'EarthObs5',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(84.5,0.5e-7,'PolarObs',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(84.5,0.3e-7,'1North',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(84.8,4.4e-7,'PolarObs',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    fig.text(85.5,4.2e-7,'2South',  horizontalalignment='center', verticalalignment='center', transform=ax.transData)
    plt.grid(True)
    ax.set_title('Earth\'s phase curve')
    ax.set_xlim(0,90)
    #ax.fill_between(phase_angle, 0, 1, where=phase_angle > 90,color='grey')
    ax.set_ylim(0,5.5e-7)
    ax.set_ylabel('scaled signal W/$m^2$/$\mu m$')
    ax.set_xlabel('phase angle [deg]') 
    
    return np.array(geometric_albedo_list), np.array(geometric_albedo_std_list)

mallama = False
scale_to_E1 = False
scale_to_E1_E4_E5 = False
scale_to_P1_P2 = True

year = ['2008','2009']
#year = ['2008']

list_wl = [SOLAR_350,SOLAR_450,SOLAR_550,SOLAR_650,SOLAR_750,SOLAR_850,SOLAR_950]
if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    geometric_albedo, geometric_albedo_std = phase_curve(year, list_wl, scale_to_E1=True)

#%%   BELOW THIS NOT IMPORTANT 
# averages = np.array(averages_signal_obs)
# omega = np.zeros(averages.shape)
# list_wl = [SOLAR_350,SOLAR_450,SOLAR_550,SOLAR_650,SOLAR_750,SOLAR_850,SOLAR_950]
# phase_angles = np.array([58,75,76.6,85,87])
# for idx2, j in enumerate(averages):
#     phase_angle = phase_angles[idx2]
#     for idx, i in enumerate(list_wl):
#         lamber_phase_func = ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
#                          np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi
#         omega[idx2,idx] = j[idx] * 3/2 * 1/(i*lamber_phase_func)
# #%%
# phase_angle = np.arange(0,91,1)
# lamber_phase_func = ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
#                      np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi
# for idx, i in enumerate(list_wl): 
#     for j in np.arange(omega.shape[0]):
#         test = 2/3*omega[j,idx]*lamber_phase_func*i
#         plt.plot(phase_angle,test, color = colours[idx])
# #%%    
# phase_angle = np.arange(0,91,1)
# lamber_phase_func = ((np.pi - np.abs(np.deg2rad(phase_angle))) * np.cos(np.deg2rad(phase_angle)) + \
#                      np.sin(np.abs(np.deg2rad(phase_angle))) )/np.pi
# for idx, i in enumerate(list_wl): 
#     for j in np.arange(omega.shape[0]):
#         average_omega = np.mean(omega[j,:])
#         test = 2/3*average_omega*lamber_phase_func*i
#         plt.plot(phase_angle,test, color = colours[idx])
# #%%
# ### Mallama uses data from Goode
# # angles from Goode and the corresponding effective albedo
# # Earth phase = 180-lunar phase, lunar phase is shown on graph
# phase_angle_goode = [0,15,30,60,90]
# effective_albedo_astr =  np.array([1.15,1.3,1.45,1.55,1.3])
# effective_albedo =  10**(effective_albedo_astr*-1/2.5)
# # for phase_angle = 0, effective albedo = 1.15
# # for phase_angle = 60, effective albedo = 1.55
# # for phase_angle = 30, effective albedo = 1.45
# incident_irradiance_list = []
# for idx,i in enumerate(phase_angle_goode):
#     # Lambert phase function of Goode
#     lamber_phase_func_2 = ( (np.pi - np.abs(np.deg2rad(i)) ) * np.cos(np.deg2rad(i)) + \
#                      np.sin(np.abs(np.deg2rad(i))) )/np.pi
    
#     # Mallama says that ''Multiplying their effective albedo by their Lambert phase law 
#     # representation and by two-thirds retrieves the phase function''
#     phase_func_mallama = 2/3*lamber_phase_func_2*effective_albedo[idx]
#     print(i)
#     print(phase_func_mallama)
#     # values are off for 0 and 15
#     # phase function mallama =/= lambert phase function    
#     incident_irradiance_list.append(phase_func_mallama)
    
# plt.plot(phase_angle_goode,incident_irradiance_list)   #, color = colours[idx]
     
# # table 1 mallama contains j(alpha) normally, but he normalises with the incident flux
# # so for alpha = 0 this is the geometric albedo
# # for the others j(alpha) = Phi(alpha)*j(0), this is divided by pi*F to normalise
# # so this results in j(alpha) = Phi(alpha) * Ag

# #%%
# mallama_angles = [0,15,30,60,90]
# mallama_points = np.array([0.2,0.18,0.15,0.1,0.06])

# #plt.figure()
# list_wl = [SOLAR_350,SOLAR_450,SOLAR_650,SOLAR_850]
# for idx, i in enumerate(list_wl):
#     incident_irradiance = 2/3*i*lamber_phase_func * 0.2 # times 0.2 as that is the geometric albedo
#     incident_irradiance = 2/3*i*lamber_phase_func 
#     #plt.plot(phase_angle,incident_irradiance, color = colours[idx])    
#     plt.plot(mallama_angles,mallama_points*i,  color = colours[idx])
#     ### def j(alpha) in analytic models
#     # j(alpha) = 2/3 * omega * pi*F * lamber_phase_func  #omega = 1 for Lambert
#     incident_irradiance = 2/3*i*lamber_phase_func
#     # rescaling is done in the incoming flux, part of SOLAR
#     power_lamb = incident_irradiance * (radius/distance)**2 * solid_angle
#     # These values are too large
#     plt.plot(phase_angle,incident_irradiance, color = colours[idx]) 
    
#     # geometric albedo, Ag = 0.2 = j(0)/(pi*F) with j(0) flux at phase = 0 and pi*F is the incoming/incident flux
#     # phase curve, Phi(alpha) = j(alpha)/j(0) with phase angle alpha
#     # Ag * Phi(alpha) =  j(alpha)/(pi*F) # This is what Mallama does normalised j(alpha)
#     # we don't have an observation at j(0) so we use Ag*(pi*F)
#     # j(alpha) = Ag * Phi(alpha)*(pi*F)
#     # geometric albedo should be 2/3*omega with omega=1 for a lambertian surface
#     # using a geometic albedo = 0.2 from Mallama works better
#     test = 0.2*lamber_phase_func*i
#     plt.plot(phase_angle,test, color = colours[idx])
#     # geometric albedo https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
#     # too high values
#     #test2 = 0.434*lamber_phase_func*i
#     #plt.plot(phase_angle,test2, color = colours[idx])    
    
# plt.legend()








