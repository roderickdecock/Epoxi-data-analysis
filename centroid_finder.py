# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:33:46 2020

@author: roder

CODE ADAPTED FROM EXISTING CODE BY DR. TIMOTHY A. LIVENGOOD
COPIED COMMENTS/CODE IS NOT EXPLICITELY REFERENCED
"""

# import glob  # to acces the files on pc

import pandas as pd # to save the data

import numpy as np

import matplotlib.pyplot as plt  # for plot
import matplotlib.lines as lns

#import scipy.signal as sig # to use the median filter
import scipy.ndimage as ndim # to use the median filter

from scipy import signal

#%%
# MAKE_POLAR : function to construct arrays for the polar coordinate
#  values of elements in a 2D array, relative to an arbitrary
#  (non-integral pixel) center. The coordinate arrays are returned as
#  floating-point.
def make_polar(naxis1_in,naxis2_in, centre = None):
    def_centre = (np.array([naxis1_in,naxis2_in])-1)/2.
    
    if np.all(centre) == None:
        centre = def_centre
    x_array = np.zeros([naxis1_in,naxis2_in])
    y_array = np.zeros([naxis1_in,naxis2_in])
    for i in np.arange(0,naxis2_in):
        x_array[:,i] = i
    for i in np.arange(0,naxis1_in):
        y_array[i,:] = i
    x_array = x_array - centre[1]
    y_array = y_array - centre[0]
    
    radius = np.sqrt(x_array*x_array + y_array*y_array)
    
    # The angle derived from inverse trig functions is ambiguous. Subtract
    # angles for Y < 0 from 2pi in order to get a continuous variation of
    # angle from 0 to 2pi.
    phi = np.zeros([naxis1_in,naxis2_in])
    mask = np.where(radius>0)
    phi[mask] = np.arccos(x_array[mask]/radius[mask])
    # small values of radius could lead to a value outside of the -1 to +1 arccosine range, these could be set to zero 
    # but this does not happen as radius is either = or bigger than the corresponding value in the x_array    
    # clockwise is positive, phi goes from 0 to 2pi. Original file states that CCW is positive
    mask = np.where(y_array<0)
    phi[mask] = 2*np.pi - phi[mask]
    return radius, phi

#%%
# BRUTE_REGISTER : function to perform a brute-force auto-registration
#  calculation. The REGISTER image is simply moved around (by
#  single pixel-level shifting) until every available offset has been
#  tested and the auto-correlation coefficient between the REGISTER
#  image and the KEY image (pixel-by-pixel product of the two images)
#  has been calculated and the maximum shift found.

def brute_register(key_frame,registering_frame,max_shift = None):
    # axis1_key = key_frame.shape[0]
    # axis2_key = key_frame.shape[1]
    axis1_registering = registering_frame.shape[0]
    axis2_registering = registering_frame.shape[1]
    max_shift_default = np.round(0.5*np.minimum(axis1_registering,axis2_registering))
    if max_shift == None:
        max_shift = max_shift_default
    else:
        max_shift = np.minimum(np.round(max_shift),max_shift_default)
    # We won't keep a history of every shift we try, we'll just check
    # to see whether we have found the biggest correlation product
    # yet, and keep the shift values that go with that correlation product.
    # However, it's possible that the entered MAXIMUM is too small to
    # accomplish the goal of achieving registration. We will execute
    # multiple loops, increasing the search area by a factor of 2 (multiply
    # MAX_SHIFT by sqrt(2)) until MAX_SHIFT has exceeded the default
    # value, or the greater of the returned best-shift values is less
    # than the most recent value of MAX_SHIFT.
    best_coefficient = 0
    best_shifts = np.array([0,0])
    done = False
    while done == False:
        steps = np.arange(-max_shift, max_shift+1,1)
        for i in steps: # shift in rows
            for j in steps: # shift in columns
                # multiplication is element-wise not matrix multiplication
                correlation_coefficient = np.sum(key_frame*np.roll(registering_frame,(int(i),int(j)),axis = (0,1)))
                if correlation_coefficient > best_coefficient:
                    best_coefficient = correlation_coefficient
                    best_shifts = np.array([i,j])
        if np.max(abs(best_shifts)) < max_shift-1:
            done = True
        elif  max_shift >= max_shift_default:           
            done = True
        else:
            max_shift = np.minimum(np.round(max_shift * np.sqrt(2) ),max_shift_default)
    
    return best_shifts

#%% IMAGE CENTERING
def image_centering(epoxi_data,filter_wavelength,earth_diam_km = 1.2756e04,astronomical_unit = 149.597870691e06, image = False):
    #earth_diam_km = 1.2756e04         #Earth diameter in km
    #astronomical_unit = 149.597870691e06 #Astronomical UNIT in km.
    
    # Calculate Earth diameter for each image, in units of microradians. Construct
    # a model image for the limb of the Earth, and construct an image proportional
    # to the gradient across the EPOXI image, which will emphasize the limb and
    # mostly ignore the body of the Earth. Use auto-correlation to find the best
    # center position for aligning the two images.
    
    print('starting')
    epoxi_data['earth_radius_pxl'] = np.zeros(epoxi_data.shape[0])    
    epoxi_data['diameter'] = earth_diam_km / (epoxi_data['range_SC'] * astronomical_unit) * 1.0e06 # small angle approximation
        
    # np.where(epoxi_data['filter_cw']==i) find the indixes, [0] to acces those indixes    
    for j in np.where(epoxi_data['filter_cw']==filter_wavelength)[0]:
        #  epoxi_data.at[j,'diameter'] preferred, epoxi_data.loc[j,'diameter'] also possible 
        # epoxi_data['diameter'][j] not good

        earth_radius_pxl      = (epoxi_data.at[j,'diameter']/2.0) / 2.0 # /2.0 (1st diameter to radius)  
        # 2nd: 2.0 microradian on a side per pixel so divided by 2.0 to get per pixel
        centroid_last = np.array([25,13]) + (512-1)/2. # [25,13] is a choice that is close to the final found value,
        #centroid_last = np.array([-10,20]) + (512-1)/2. # [25,13] is a choice that is close to the final found value,
        # anything close to the middle of the picture whould work (if the object is close to that)
        naxis1 = epoxi_data.at[j,'naxis1']
        naxis2 = epoxi_data.at[j,'naxis2']
        radius, phi = make_polar(naxis1,naxis2, centre = centroid_last)
        med_image_prim = ndim.median_filter(epoxi_data.at[j,'image'],3)
        # https://en.wikipedia.org/wiki/Image_gradient :
        # "The derivative of an image can be approximated by finite differences. If central difference is used, to calculate 
        # \frac {\partial f}{\partial y}}} / {\frac {\partial f}{\partial y}} we can apply a 1-dimensional filter to the image A
        #  by convolution: {\frac {\partial f}{\partial y}}={\begin{bmatrix}-1\\+1\end{bmatrix}}*A
        # where * denotes the 1-dimensional convolution operation. This 2×1 filter will shift the image by half a pixel."
        # see https://en.wikipedia.org/wiki/Kernel_(image_processing) for extra info
        # image_gradient = (np.roll(med_image_prim,(0,1),axis = (0,1)) - np.roll(med_image_prim,(0,-1),axis = (0,1)))**2 \
        #     + (np.roll(med_image_prim,(1,0),axis = (0,1)) - np.roll(med_image_prim,(-1,0),axis = (0,1)))**2
        image_gradient_components = np.gradient(med_image_prim)
        image_gradient = image_gradient_components[0]*image_gradient_components[0] + image_gradient_components[1] * image_gradient_components[1]
        image_disk = np.zeros([naxis1,naxis2])
        # within the radius of the Earth and on the right of the centre so 1st and 4th quadrant
        mask_disk = np.where(np.logical_and(radius <= earth_radius_pxl, np.logical_or(phi <= 0.5*np.pi, phi>= 1.5*np.pi)))
        image_ring = np.zeros([naxis1,naxis2])
        # a ring of width 3 around the Earth radius
        mask_ring = np.where(np.logical_and(radius >= earth_radius_pxl, radius <= earth_radius_pxl+3.))
        # set these locations equal to a chosen constant value
        image_disk[mask_disk] = np.max(med_image_prim)*0.25
        image_ring[mask_ring] = np.max(med_image_prim)*0.25
        # get the maximum value, either from the ring or from the disk (which is scaled)  SCALING FACTOR???
        image_combined = np.maximum(image_ring, image_disk * 0.5 * np.max(image_gradient)/np.max(med_image_prim)) # SCALING FACTOR UNCLEAR
        ### This is too slow, is similar to brute register but over full image
        # corr = signal.correlate2d(image_gradient,image_combined, boundary='symm', mode='same')
        # y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
        # centroid_offset = np.array([y-255.,x-255.]) ### manually subtract half the image size
        
        ### correlation, chooses automatically the fastest option between convolution or fourier transform
        # (I think it chooses fourier transform) as the normal calculation is done in brute register 
        # and that took too much time.
        # The [::-1,::-1] rotates the image by 180 degrees, this makes it correlation and not convolution
        corr = signal.convolve(image_gradient,image_combined[::-1,::-1], mode='same')
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        centroid_offset = np.array([y-255.,x-255.]) ### manually subtract half the image size, 
        # additional -1 for better result (matches brute register result), has been removed due to 
        # sometimes not including the entire Earth on the right side which has more signal than the left side
        # probably due to handling of position in correlation
        
        # centroid_last2 = np.array([25,13]) + (512-1)/2. # [25,13] is a choice that is close to the final found value,
        # centroid_last2 += centroid_offset2
        # centre2 = np.round(centroid_last2 - (naxis1-1)*0.5) 
        # centre_reversed2 = np.array([centre2[1],centre2[0]])            
        #print(centroid_offset)
        
        print('starting with brute') 
        ### original method:
        #centroid_offset = brute_register(image_gradient, image_combined, max_shift = 60) ### choose max shift
        centroid_last += centroid_offset
        #print(centroid_offset)
        centre = np.round(centroid_last - (naxis1-1)*0.5) 
        #print(centre)
        # centre needs to be reversed, unknown where it goes wrong and the reason for it
        # reversing leads to good results
        centre_reversed = np.array([centre[1],centre[0]])
        # 'target_center' was the initial estimated centre
        epoxi_data.at[j,'target_center']  = centre_reversed 
        epoxi_data.at[j,'earth_radius_pxl'] = earth_radius_pxl
        #print(centre_reversed)
        
        if image == True:
            # Before I realised I had to reverse the centroid, I tried to correct it, this can be removed
            centre_corrected = np.array([13,25]) + centroid_offset 
            fig2, ax2 = plt.subplots()
            plt.title('Median filtered primary image')
            plt.imshow(med_image_prim, cmap='gray')
            #circle1 = plt.Circle(centre_corrected+255.5, earth_radius_pxl, color='r', fill=False, label = 'earth radius')
            #ax2.add_artist(circle1)
            #leg_1 = ax2.legend(circle1, 'earth radius', loc='lower left')
            #plt.scatter(centre_corrected[0]+255.5, centre_corrected[1]+255.5, s=10,color = 'b', label = 'final centroid')
            #plt.scatter(255.5, 255.5, s=10, color = 'r', label = 'centre image')
        
            #ax2.add_artist(leg_1)   
            centre_fig = ax2.scatter(centre_reversed[0]+255.5, centre_reversed[1]+255.5, s=10,color = 'r', label = 'Final centroid')
            ax2.add_artist(centre_fig)
            circle2 = plt.Circle(centre_reversed+255.5, earth_radius_pxl, color='r', fill=False, label = 'Earth radius')
            ax2.add_artist(circle2)
            #ax2.legend((centre_fig,circle2), ('Centroid','Earth radius'))
            line1 = lns.Line2D(range(1), range(1), linewidth=0, color="white", marker='o', markeredgecolor="red",markerfacecolor="gray")
            ax2.legend((centre_fig,line1), ('Centroid','Earth edge'))
            # plt.scatter(centre_reversed2[0]+255.5, centre_reversed2[1]+255.5, s=10,color = 'y', label = 'reversed final centroid')
            # circle3 = plt.Circle(centre_reversed2+255.5, earth_radius_pxl, color='g', fill=False, label = 'earth radius')
            # ax2.add_artist(circle3)
            #leg_1 = ax2.legend(circle2, 'Earth radius', loc='lower left')
            #ax2.add_artist(leg_1)
            #plt.legend()
            #plt.colorbar()
            plt.grid()
        ignore_this = True

    # SAVING ################
    df_epoxi_data_filter = epoxi_data[epoxi_data['filter_cw']==filter_wavelength]
    df_epoxi_data_filter = df_epoxi_data_filter.reset_index(drop = True)   
    #df_epoxi_data_filter.to_pickle('../output/RADREV_'+year+'_'+observations[0]+'_'+observations[1]+'_'+'df_epoxi_data_filtered_'+str(filter_wavelength)+'.pkl')
    return


#%%    
if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    # INPUT
    year = '2008'
    observations = ['078','079'] 
    #observations = ['149','150'] 
    #observations = ['156','157'] 
    #observations = ['086','087'] 
    #observations = ['277','278'] # WHAT TO DO WITH THE 270 OBSERVATION?
    
################### -1 removed at centroid offset, does not include entire Earth
### RE-RUN because of image gradient and correlation function instead of brute register and -1 removed
# 2008 078,079 all wavelengths
# 2008 149,150 350,450,550
# 2009 086,087 350

# DONE RERUNNING and verified in lightcurves
# 2008 078,079: all wavelengths !!! Difference for 950 !!!

# I don't remember but I think I did them all
# DONE
# 2008 149,150 350,450
# 2009 277, 278

    #%%
    for idx,i in enumerate(observations):
        epoxi_data_temp = pd.read_hdf('../output/'+year+'_'+i+'_min_aper_150_dictionary_info.h5')
        #epoxi_data_temp = pd.read_hdf('../output/RADREV_'+year+'_'+i+'_min_aper_150_dictionary_info.h5')
        if idx ==0:
            epoxi_data = epoxi_data_temp
        else:
            epoxi_data = pd.concat([epoxi_data,epoxi_data_temp], ignore_index=True)
    
    #%%
    filter_wavelength = 450 # one wavelength at the time, long runtime is improved  
    image_centering(epoxi_data, filter_wavelength, image = True)



    