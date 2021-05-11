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

#import scipy.signal as sig # to use the median filter
import scipy.ndimage as ndim # to use the median filter

from filereader import aper_photom  # python runs the code of filereader  when importing the if __name__ == "__main__": prevents this

### ILUMINATION FRACTION IS NOT UPDATED, STILL ON 100%

def moon_centroids_finder(filter_wavelength):
    if filter_wavelength == 350:
        centroids = np.zeros((2,25))
        centroids[:, 0] = [-250,-51] ; centroids[:, 1] = [-250,-48] ; centroids[:, 2] = [-250,-42]
        centroids[:, 3] = [-250,-37] ; centroids[:, 4] = [-250,-28] ; centroids[:, 5] = [-249,-22]
        centroids[:, 6] = [-207,-13] ; centroids[:, 7] = [-167, 15] ; centroids[:, 8] = [-134, 25]
        centroids[:, 9] = [ -80, 29] ; centroids[:,10] = [ -24, 35] ; centroids[:,11] = [  34, 36]
        centroids[:,12] = [  72, 38] ; centroids[:,13] = [  99, 27] ; centroids[:,14] = [ 120, 24]
        centroids[:,15] = [ 147, 11] ; centroids[:,16] = [ 177,-13] ; centroids[:,17] = [ 224,-16]
        centroids[:,18] = [ 250,  5] ; centroids[:,19] = [ 250, 19] ; centroids[:,20] = [ 250, 33]
        centroids[:,21] = [ 250, 36] ; centroids[:,22] = [ 250, 34] ; centroids[:,23] = [ 250, 48]
        centroids[:,24] = [ 250, 56]
        
    elif filter_wavelength == 450:
        centroids = np.zeros((2,97))
        centroids[:, 0] = [-250,-50] ; centroids[:, 1] = [-250,-51] ; centroids[:, 2] = [-250,-48]
        centroids[:, 3] = [-250,-45] ; centroids[:, 4] = [-250,-46] ; centroids[:, 5] = [-250,-46]
        centroids[:, 6] = [-250,-42] ; centroids[:, 7] = [-250,-39] ; centroids[:, 8] = [-250,-41]
        centroids[:, 9] = [-250,-40] ; centroids[:,10] = [-250,-40] ; centroids[:,11] = [-250,-39]
        centroids[:,12] = [-250,-36] ; centroids[:,13] = [-250,-35] ; centroids[:,14] = [-250,-33]
        centroids[:,15] = [-250,-30] ; centroids[:,16] = [-250,-28] ; centroids[:,17] = [-250,-29]
        centroids[:,18] = [-250,-27] ; centroids[:,19] = [-250,-24] ; centroids[:,20] = [-249,-22]
        centroids[:,21] = [-239,-19] ; centroids[:,22] = [-230,-15] ; centroids[:,23] = [-220,-13]
        centroids[:,24] = [-209,-13] ; centroids[:,25] = [-197,-10] ; centroids[:,26] = [-186, -3]
        centroids[:,27] = [-176,  5] ; centroids[:,28] = [-168, 15] ; centroids[:,29] = [-158, 19]
        centroids[:,30] = [-153, 22] ; centroids[:,31] = [-145, 24] ; centroids[:,32] = [-136, 25]
        centroids[:,33] = [-123, 26] ; centroids[:,34] = [-113, 25] ; centroids[:,35] = [ -97, 28]
        centroids[:,36] = [ -82, 29] ; centroids[:,37] = [ -67, 32] ; centroids[:,38] = [ -55, 32]
        centroids[:,39] = [ -37, 34] ; centroids[:,40] = [ -26, 35] ; centroids[:,41] = [ -11, 33]
        centroids[:,42] = [   2, 32] ; centroids[:,43] = [  17, 35] ; centroids[:,44] = [  31, 37]
        centroids[:,45] = [  45, 36] ; centroids[:,46] = [  55, 37] ; centroids[:,47] = [  65, 40]
        centroids[:,48] = [  69, 38] ; centroids[:,49] = [  76, 36] ; centroids[:,50] = [  84, 34]
        centroids[:,51] = [  92, 31] ; centroids[:,52] = [  97, 27] ; centroids[:,53] = [ 100, 23]
        centroids[:,54] = [ 107, 23] ; centroids[:,55] = [ 112, 23] ; centroids[:,56] = [ 118, 24]
        centroids[:,57] = [ 126, 20] ; centroids[:,58] = [ 132, 19] ; centroids[:,59] = [ 139, 16]
        centroids[:,60] = [ 146, 11] ; centroids[:,61] = [ 156,  5] ; centroids[:,62] = [ 165,  0]
        centroids[:,63] = [ 172, -5] ; centroids[:,64] = [ 175,-12] ; centroids[:,65] = [ 188,-16]
        centroids[:,66] = [ 195,-18] ; centroids[:,67] = [ 209,-17] ; centroids[:,68] = [ 222,-16]
        centroids[:,69] = [ 243, -8] ; centroids[:,70] = [ 250, -4] ; centroids[:,71] = [ 250, -1]
        centroids[:,72] = [ 250,  5] ; centroids[:,73] = [ 250,  8] ; centroids[:,74] = [ 250, 10]
        centroids[:,75] = [ 250, 13] ; centroids[:,76] = [ 250, 18] ; centroids[:,77] = [ 250, 20]
        centroids[:,78] = [ 250, 24] ; centroids[:,79] = [ 250, 27] ; centroids[:,80] = [ 250, 32]
        centroids[:,81] = [ 250, 35] ; centroids[:,82] = [ 250, 37] ; centroids[:,83] = [ 250, 37]
        centroids[:,84] = [ 250, 35] ; centroids[:,85] = [ 250, 30] ; centroids[:,86] = [ 250, 26]
        centroids[:,87] = [ 250, 29] ; centroids[:,88] = [ 250, 32] ; centroids[:,89] = [ 250, 33]
        centroids[:,90] = [ 250, 35] ; centroids[:,91] = [ 250, 42] ; centroids[:,92] = [ 250, 46]
        centroids[:,93] = [ 250, 48] ; centroids[:,94] = [ 250, 52] ; centroids[:,95] = [ 250, 54]
        centroids[:,96] = [ 250, 54]
    elif filter_wavelength == 550:
        centroids = np.zeros((2,97))
        centroids[:, 0] = [-250,-47] ; centroids[:, 1] = [-250,-48] ; centroids[:, 2] = [-250,-45]
        centroids[:, 3] = [-250,-43] ; centroids[:, 4] = [-250,-45] ; centroids[:, 5] = [-250,-45]
        centroids[:, 6] = [-250,-41] ; centroids[:, 7] = [-250,-38] ; centroids[:, 8] = [-250,-38]
        centroids[:, 9] = [-250,-38] ; centroids[:,10] = [-250,-39] ; centroids[:,11] = [-250,-39]
        centroids[:,12] = [-250,-35] ; centroids[:,13] = [-250,-33] ; centroids[:,14] = [-250,-31]
        centroids[:,15] = [-250,-28] ; centroids[:,16] = [-250,-26] ; centroids[:,17] = [-250,-28]
        centroids[:,18] = [-250,-26] ; centroids[:,19] = [-250,-23] ; centroids[:,20] = [-248,-21]
        centroids[:,21] = [-237,-20] ; centroids[:,22] = [-225,-15] ; centroids[:,23] = [-219,-14]
        centroids[:,24] = [-205,-13] ; centroids[:,25] = [-195,-10] ; centroids[:,26] = [-183, -3]
        centroids[:,27] = [-174,  5] ; centroids[:,28] = [-164, 15] ; centroids[:,29] = [-157, 18]
        centroids[:,30] = [-150, 22] ; centroids[:,31] = [-144, 23] ; centroids[:,32] = [-131, 25]
        centroids[:,33] = [-123, 26] ; centroids[:,34] = [-108, 25] ; centroids[:,35] = [ -96, 27]
        centroids[:,36] = [ -79, 29] ; centroids[:,37] = [ -67, 31] ; centroids[:,38] = [ -52, 32]
        centroids[:,39] = [ -37, 34] ; centroids[:,40] = [ -21, 35] ; centroids[:,41] = [  -9, 33]
        centroids[:,42] = [   7, 32] ; centroids[:,43] = [  20, 35] ; centroids[:,44] = [  34, 37]
        centroids[:,45] = [  46, 36] ; centroids[:,46] = [  59, 37] ; centroids[:,47] = [  66, 39]
        centroids[:,48] = [  74, 38] ; centroids[:,49] = [  79, 35] ; centroids[:,50] = [  87, 34]
        centroids[:,51] = [  94, 30] ; centroids[:,52] = [ 101, 27] ; centroids[:,53] = [ 103, 23]
        centroids[:,54] = [ 112, 22] ; centroids[:,55] = [ 113, 22] ; centroids[:,56] = [ 123, 24]
        centroids[:,57] = [ 129, 20] ; centroids[:,58] = [ 135, 19] ; centroids[:,59] = [ 141, 15]
        centroids[:,60] = [ 150, 11] ; centroids[:,61] = [ 156,  5] ; centroids[:,62] = [ 169,  0]
        centroids[:,63] = [ 174, -5] ; centroids[:,64] = [ 179,-13] ; centroids[:,65] = [ 187,-16]
        centroids[:,66] = [ 198,-18] ; centroids[:,67] = [ 211,-18] ; centroids[:,68] = [ 226,-16]
        centroids[:,69] = [ 243, -9] ; centroids[:,70] = [ 250, -4] ; centroids[:,71] = [ 250, -1]
        centroids[:,72] = [ 250,  5] ; centroids[:,73] = [ 250,  7] ; centroids[:,74] = [ 250,  9]
        centroids[:,75] = [ 250, 14] ; centroids[:,76] = [ 250, 19] ; centroids[:,77] = [ 250, 21]
        centroids[:,78] = [ 250, 25] ; centroids[:,79] = [ 250, 28] ; centroids[:,80] = [ 250, 33]
        centroids[:,81] = [ 250, 36] ; centroids[:,82] = [ 250, 38] ; centroids[:,83] = [ 250, 38]
        centroids[:,84] = [ 250, 36] ; centroids[:,85] = [ 250, 30] ; centroids[:,86] = [ 250, 27]
        centroids[:,87] = [ 250, 29] ; centroids[:,88] = [ 250, 33] ; centroids[:,89] = [ 250, 34]
        centroids[:,90] = [ 250, 36] ; centroids[:,91] = [ 250, 42] ; centroids[:,92] = [ 250, 47]
        centroids[:,93] = [ 250, 49] ; centroids[:,94] = [ 250, 53] ; centroids[:,95] = [ 250, 54]
        centroids[:,96] = [ 250, 55]
    elif filter_wavelength == 650:
        centroids = np.zeros((2,97))
        centroids[:, 0] = [-250,-47] ; centroids[:, 1] = [-250,-48] ; centroids[:, 2] = [-250,-45]
        centroids[:, 3] = [-250,-43] ; centroids[:, 4] = [ 250,-45] ; centroids[:, 5] = [-250,-45]
        centroids[:, 6] = [-250,-41] ; centroids[:, 7] = [-250,-38] ; centroids[:, 8] = [-250,-38]
        centroids[:, 9] = [-250,-38] ; centroids[:,10] = [-250,-39] ; centroids[:,11] = [-250,-39]
        centroids[:,12] = [-250,-35] ; centroids[:,13] = [-250,-33] ; centroids[:,14] = [-250,-31]
        centroids[:,15] = [-250,-28] ; centroids[:,16] = [ 250,-26] ; centroids[:,17] = [-250,-28]
        centroids[:,18] = [-250,-26] ; centroids[:,19] = [-250,-23] ; centroids[:,20] = [-249,-22]
        centroids[:,21] = [-240,-20] ; centroids[:,22] = [-230,-15] ; centroids[:,23] = [-221,-13]
        centroids[:,24] = [-207,-13] ; centroids[:,25] = [-197,-10] ; centroids[:,26] = [-187, -3]
        centroids[:,27] = [-176,  5] ; centroids[:,28] = [-169, 15] ; centroids[:,29] = [-159, 19]
        centroids[:,30] = [-154, 22] ; centroids[:,31] = [-146, 24] ; centroids[:,32] = [-136, 25]
        centroids[:,33] = [-125, 26] ; centroids[:,34] = [-114, 25] ; centroids[:,35] = [ -98, 28]
        centroids[:,36] = [ -81, 29] ; centroids[:,37] = [ -68, 32] ; centroids[:,38] = [ -56, 32]
        centroids[:,39] = [ -39, 34] ; centroids[:,40] = [ -24, 35] ; centroids[:,41] = [ -11, 33]
        centroids[:,42] = [   1, 32] ; centroids[:,43] = [  17, 35] ; centroids[:,44] = [  33, 37]
        centroids[:,45] = [  44, 37] ; centroids[:,46] = [  56, 37] ; centroids[:,47] = [  64, 40]
        centroids[:,48] = [  70, 38] ; centroids[:,49] = [  76, 36] ; centroids[:,50] = [  83, 33]
        centroids[:,51] = [  91, 31] ; centroids[:,52] = [  98, 27] ; centroids[:,53] = [ 102, 23]
        centroids[:,54] = [ 107, 22] ; centroids[:,55] = [ 111, 23] ; centroids[:,56] = [ 119, 24]
        centroids[:,57] = [ 126, 20] ; centroids[:,58] = [ 131, 19] ; centroids[:,59] = [ 139, 16]
        centroids[:,60] = [ 146, 11] ; centroids[:,61] = [ 154,  5] ; centroids[:,62] = [ 163,  0]
        centroids[:,63] = [ 171, -5] ; centroids[:,64] = [ 175,-13] ; centroids[:,65] = [ 186,-16]
        centroids[:,66] = [ 194,-18] ; centroids[:,67] = [ 208,-17] ; centroids[:,68] = [ 224,-16]
        centroids[:,69] = [ 243, -9] ; centroids[:,70] = [ 250, -4] ; centroids[:,71] = [ 250, -1]
        centroids[:,72] = [ 250,  5] ; centroids[:,73] = [ 250,  7] ; centroids[:,74] = [ 250,  9]
        centroids[:,75] = [ 250, 14] ; centroids[:,76] = [ 250, 19] ; centroids[:,77] = [ 250, 21]
        centroids[:,78] = [ 250, 25] ; centroids[:,79] = [ 250, 28] ; centroids[:,80] = [ 250, 33]
        centroids[:,81] = [ 250, 36] ; centroids[:,82] = [ 250, 38] ; centroids[:,83] = [ 250, 38]
        centroids[:,84] = [ 250, 36] ; centroids[:,85] = [ 250, 30] ; centroids[:,86] = [ 250, 27]
        centroids[:,87] = [ 250, 29] ; centroids[:,88] = [ 250, 33] ; centroids[:,89] = [ 250, 34]
        centroids[:,90] = [ 250, 36] ; centroids[:,91] = [ 250, 42] ; centroids[:,92] = [ 250, 47]
        centroids[:,93] = [ 250, 49] ; centroids[:,94] = [ 250, 53] ; centroids[:,95] = [ 250, 54]
        centroids[:,96] = [ 250, 55]
    if filter_wavelength == 750:
        centroids = np.zeros((2,25))
        centroids[:, 0] = [ 250,-51] ; centroids[:, 1] = [ 250,-48] ; centroids[:, 2] = [ 250,-42]
        centroids[:, 3] = [ 250,-37] ; centroids[:, 4] = [ 250,-28] ; centroids[:, 5] = [-249,-22]
        centroids[:, 6] = [-208,-13] ; centroids[:, 7] = [-168, 15] ; centroids[:, 8] = [-136, 25]
        centroids[:, 9] = [ -82, 29] ; centroids[:,10] = [ -24, 35] ; centroids[:,11] = [  34, 36]
        centroids[:,12] = [  72, 38] ; centroids[:,13] = [  99, 27] ; centroids[:,14] = [ 119, 23]
        centroids[:,15] = [ 146, 11] ; centroids[:,16] = [ 176,-13] ; centroids[:,17] = [ 223,-16]
        centroids[:,18] = [ 250,  5] ; centroids[:,19] = [ 250, 19] ; centroids[:,20] = [ 250, 33]
        centroids[:,21] = [ 250, 36] ; centroids[:,22] = [ 250, 34] ; centroids[:,23] = [ 250, 48]
        centroids[:,24] = [ 250, 56]
    elif filter_wavelength == 850:
        centroids = np.zeros((2,97))
        centroids[:, 0] = [-250,-47] ; centroids[:, 1] = [-250,-48] ; centroids[:, 2] = [-250,-45]
        centroids[:, 3] = [-250,-43] ; centroids[:, 4] = [ 250,-45] ; centroids[:, 5] = [-250,-45]
        centroids[:, 6] = [-250,-41] ; centroids[:, 7] = [-250,-38] ; centroids[:, 8] = [-250,-38]
        centroids[:, 9] = [-250,-38] ; centroids[:,10] = [-250,-39] ; centroids[:,11] = [-250,-39]
        centroids[:,12] = [-250,-35] ; centroids[:,13] = [-250,-33] ; centroids[:,14] = [-250,-31]
        centroids[:,15] = [-250,-28] ; centroids[:,16] = [ 250,-26] ; centroids[:,17] = [-250,-28]
        centroids[:,18] = [-250,-26] ; centroids[:,19] = [-250,-23] ; centroids[:,20] = [-249,-22]
        centroids[:,21] = [-243,-20] ; centroids[:,22] = [-231,-15] ; centroids[:,23] = [-224,-14]
        centroids[:,24] = [-208,-13] ; centroids[:,25] = [-201,-10] ; centroids[:,26] = [-188, -3]
        centroids[:,27] = [-180,  5] ; centroids[:,28] = [-168, 15] ; centroids[:,29] = [-162, 19]
        centroids[:,30] = [-155, 22] ; centroids[:,31] = [-150, 23] ; centroids[:,32] = [-135, 25]
        centroids[:,33] = [-128, 26] ; centroids[:,34] = [-115, 25] ; centroids[:,35] = [-101, 28]
        centroids[:,36] = [ -82, 29] ; centroids[:,37] = [ -71, 32] ; centroids[:,38] = [ -60, 32]
        centroids[:,39] = [ -39, 34] ; centroids[:,40] = [ -24, 35] ; centroids[:,41] = [ -11, 33]
        centroids[:,42] = [   1, 32] ; centroids[:,43] = [  17, 35] ; centroids[:,44] = [  33, 37]
        centroids[:,45] = [  44, 37] ; centroids[:,46] = [  56, 37] ; centroids[:,47] = [  64, 40]
        centroids[:,48] = [  70, 38] ; centroids[:,49] = [  76, 36] ; centroids[:,50] = [  83, 33]
        centroids[:,51] = [  91, 31] ; centroids[:,52] = [  98, 27] ; centroids[:,53] = [ 102, 23]
        centroids[:,54] = [ 107, 22] ; centroids[:,55] = [ 111, 23] ; centroids[:,56] = [ 119, 23]
        centroids[:,57] = [ 122, 20] ; centroids[:,58] = [ 130, 19] ; centroids[:,59] = [ 135, 16]
        centroids[:,60] = [ 146, 11] ; centroids[:,61] = [ 151,  5] ; centroids[:,62] = [ 163,  0]
        centroids[:,63] = [ 167, -5] ; centroids[:,64] = [ 176,-13] ; centroids[:,65] = [ 183,-16]
        centroids[:,66] = [ 192,-18] ; centroids[:,67] = [ 204,-17] ; centroids[:,68] = [ 222,-16]
        centroids[:,69] = [ 243, -9] ; centroids[:,70] = [ 250, -4] ; centroids[:,71] = [ 250, -1]
        centroids[:,72] = [ 250,  5] ; centroids[:,73] = [ 250,  7] ; centroids[:,74] = [ 250,  9]
        centroids[:,75] = [ 250, 14] ; centroids[:,76] = [ 250, 19] ; centroids[:,77] = [ 250, 21]
        centroids[:,78] = [ 250, 25] ; centroids[:,79] = [ 250, 28] ; centroids[:,80] = [ 250, 33]
        centroids[:,81] = [ 250, 36] ; centroids[:,82] = [ 250, 38] ; centroids[:,83] = [ 250, 38]
        centroids[:,84] = [ 250, 36] ; centroids[:,85] = [ 250, 30] ; centroids[:,86] = [ 250, 27]
        centroids[:,87] = [ 250, 29] ; centroids[:,88] = [ 250, 33] ; centroids[:,89] = [ 250, 34]
        centroids[:,90] = [ 250, 36] ; centroids[:,91] = [ 250, 42] ; centroids[:,92] = [ 250, 47]
        centroids[:,93] = [ 250, 49] ; centroids[:,94] = [ 250, 53] ; centroids[:,95] = [ 250, 54]
        centroids[:,96] = [ 250, 55]
    elif filter_wavelength == 950:
        centroids = np.zeros((2,25))  
        centroids[:, 0] = [ 250,-51] ; centroids[:, 1] = [ 250,-48] ; centroids[:, 2] = [ 250,-42]
        centroids[:, 3] = [ 250,-37] ; centroids[:, 4] = [ 250,-28] ; centroids[:, 5] = [-249,-22]
        centroids[:, 6] = [-208,-13] ; centroids[:, 7] = [-168, 15] ; centroids[:, 8] = [-135, 25]
        centroids[:, 9] = [ -81, 29] ; centroids[:,10] = [ -24, 35] ; centroids[:,11] = [  34, 36]
        centroids[:,12] = [  72, 38] ; centroids[:,13] = [  99, 27] ; centroids[:,14] = [ 119, 23]
        centroids[:,15] = [ 147, 10] ; centroids[:,16] = [ 176,-13] ; centroids[:,17] = [ 224,-16]
        centroids[:,18] = [ 250,  5] ; centroids[:,19] = [ 250, 19] ; centroids[:,20] = [ 250, 33]
        centroids[:,21] = [ 250, 36] ; centroids[:,22] = [ 250, 34] ; centroids[:,23] = [ 250, 48]
        centroids[:,24] = [ 250, 56]
    return centroids.T

#from photutils import CircularAperture, aperture_photometry, CircularAnnulus
#from photutils.background import Background2D

#%%
def update_signal(epoxi_data_filter, moon=False):
    if moon ==True:
        filter_cw = epoxi_data_filter.at[0,'filter_cw']
        moon_centroids = moon_centroids_finder(filter_cw) 
    for i in np.arange(epoxi_data_filter.shape[0]):
        centre = epoxi_data_filter.at[i,'target_center']
        width_trim = 6 # should correspond to the width trim that was originally applied
        image_prim = epoxi_data_filter.at[i,'image'][width_trim:512-width_trim,width_trim:512-width_trim] # elimanating the zeros at the edge
        weight = epoxi_data_filter.at[i,'weight'][width_trim:512-width_trim,width_trim:512-width_trim]
        med_image = ndim.median_filter(image_prim*weight,3)
        
        aper_radius = np.minimum(1.01*epoxi_data_filter.at[i,'earth_radius_pxl'],image_prim.shape[0]) #*1.01, *1.5 for EarthObs4
        aper_finish = np.minimum(4.0*epoxi_data_filter.at[i,'earth_radius_pxl'],image_prim.shape[0]) 
        
        done = False
        prev_signal = 0  
        
        if moon ==True:   
            moon_centroid_loc = moon_centroids[i,:]
            if np.sqrt(np.sum((moon_centroid_loc-centre)**2)) >= aper_radius:
                # new med image is the image without the Moon
                moon_signal, moon_final, med_image = aper_photom(med_image, centre = moon_centroid_loc, radius = aper_radius*0.27)
        while done == False:
            signal_aperture, final, patch  = aper_photom(med_image, centre = centre, radius = aper_radius)
            
            # centre_adjusted = centre + 255.5
            # aperture = CircularAperture(centre_adjusted, r=aper_radius)
            # annulus_aperture = CircularAnnulus(centre_adjusted, r_in=aper_radius, r_out=aper_radius*1.5)
            # apers = [aperture, annulus_aperture]
            # phot_table = aperture_photometry(med_image, apers)
            # for col in phot_table.colnames:
            #     phot_table[col].info.format = '%.8g'  # for consistent table output
            # #print(phot_table)
            # bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
            # bkg_sum = bkg_mean * aperture.area
            # final_sum = phot_table['aperture_sum_0'] - bkg_sum
            # phot_table['residual_aperture_sum'] = final_sum
            # phot_table['residual_aperture_sum'].info.format = '%.8g'  # for consistent table output
            # #print(phot_table)
            # signal_aperture = final_sum[0]
            
            ### figure
            # fig2, ax2 = plt.subplots()
            # plt.title('Median data')
            # plt.imshow(med_image, cmap='gray')
            # circle1 = plt.Circle(centre+255.5, aper_radius, color='r', fill=False)
            # ax2.add_artist(circle1)
            # plt.scatter(centre[0]+255.5, centre[1]+255.5, s=10)
            # plt.colorbar()
            
            # print(final)
            
            epoxi_data_filter.at[i,'signal']         = final[0]
            if moon == True:
                epoxi_data_filter.at[i,'signal']     = final[0] + moon_signal
                #epoxi_data_filter.at[i,'signal']     = signal_aperture
            epoxi_data_filter.at[i,'signal_rms']     = final[3]
            epoxi_data_filter.at[i,'background']     = final[2]
            #epoxi_data_filter.at[i,'background']     = bkg_sum[0]
            epoxi_data_filter.at[i,'background_rms'] = final[5]
            epoxi_data_filter.at[i,'aperture']       = final[6] * 2
            #print(signal_aperture,final[0] + moon_signal)
            #print(final[2],bkg_sum)
            aper_radius += 2
            
            if aper_radius>aper_finish or np.abs((signal_aperture - prev_signal)/signal_aperture) < 5e-4:
                                           #and final[3]<75): #1e-3ss                
                done = True
            prev_signal = signal_aperture

        # fig2, ax2 = plt.subplots()
        # plt.title('Median data, index:'+str(i))
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
    average_range_SC = 1.0 
    average_range_Sun = 1.0
    # Correct summed signal for physical effects -- 1/r^2 for distance,
    # scale back to a fully-illuminated disc not necessary
    
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
    plt.figure() #is here when using all wavelengths, messes up when plotting in update signal
    idx = 0
    for i in wavelengths:
        filepath = r'../output/'+year+'_'+observations[0]+'_'+observations[1]+'_df_epoxi_data_filtered_'+str(i)+'.pkl'
        if observations == ['149','150'] or observations == ['086','087']: # they have different background subtraction
            filepath = r'../output/RADREV_'+year+'_'+observations[0]+'_'+observations[1]+'_df_epoxi_data_filtered_'+str(i)+'.pkl'
        epoxi_data_filter = pd.read_pickle(filepath)
        if observations == ['149','150']:
            epoxi_data_filter = update_signal(epoxi_data_filter, moon=True)
        else:
            epoxi_data_filter = update_signal(epoxi_data_filter)
        scaled_signal_filter = scale_to_range(epoxi_data_filter)
        diurnally_averaged_signal = np.sum(scaled_signal_filter*pixel_solid_angle)/scaled_signal_filter.shape[0]
        print(i,diurnally_averaged_signal)
        epoxi_data_filter['diurnally averaged signal'] = diurnally_averaged_signal
        #print(np.sum(epoxi_data_filter['signal_rms']/epoxi_data_filter['signal'])/epoxi_data_filter['signal'].shape[0])
        di_avg_signal_std = np.std(epoxi_data_filter['scaled signal']*pixel_solid_angle)
        print('std',di_avg_signal_std) # standard deviation
        epoxi_data_filter['diurnally averaged signal std'] = di_avg_signal_std
        
        if i ==450 or i==550 or i==650 or i==850:
            #plt.figure() # is here because I am only 1 wavelength
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
        ####################################################
        ### SAVING IS OFF !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #epoxi_data_filter.to_pickle('../output/'+year+'_'+observations[0]+'_'+observations[1]+'_'+'df_epoxi_data_filtered_'+str(i)+'.pkl')
        #epoxi_data_filter.to_pickle('../output/RADREV_'+year+'_'+observations[0]+'_'+observations[1]+'_'+'df_epoxi_data_filtered_'+str(i)+'.pkl')
    plt.title('AAAAAA') ###################### change manually
    plt.xlabel('Central Meridian (West Longtitude)')
    plt.ylabel('Normalised signal')
    plt.xlim(0,360)
    plt.ylim(0.8,1.2)
    plt.grid(True)
    plt.legend()
    return epoxi_data_filter
#%%
### RERUN  update signal doesn't overwrite anymore with scaled signal
# 2009 086,087, 277,278
# EDIT: I think I did this? I redid them anyway

if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    # INPUT
    #year = '2008'
    year = '2009'
    #observations = ['078','079'] 
    #observations = ['149','150'] 
    #observations = ['156','157'] 
    #observations = ['086','087']
    observations = ['277','278']
    
    wavelengths = [350,450,550,650,750,850,950]
    #wavelengths = [450]
    colours = ['b','g','r','y'] # plots are 2 lines combined so manually assign colours
    # CONSTANT
    pixel_solid_angle = 2.0e-06 * 2.0e-06

    # The uncertainty in conversion to absolute
    # radiometric units is estimated to be 5% for HRI except for the 950-nm
    # filter, where the uncertainty is ~10%.    -email Lori Feaga 07.12.2020

    df = lightcurves_plot(year, observations, wavelengths, colours, pixel_solid_angle)
    ignore_this = True


