# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:45:01 2020

@author: roder

CODE ADAPTED FROM EXISTING CODE BY DR. TIMOTHY A. LIVENGOOD
COPIED COMMENTS/CODE IS NOT EXPLICITELY REFERENCED
"""
#####
##### Rad uses Livengood background method
##### Radrev is used to try the photutils background, for 149,150 size was 128,128 no disk 
#####     Not a perfect result, small jumps and zig-zag still present
#####   for 086,087 size is 256,128 and exclude_percentile = 0 and disk of radius 150 is applied as mask
#####     good result

#### IDL first index column, 2nd row  Python first index row, end column
#### IDL slicing includes the last number: 1:3 includes index 3. Python does not
#### IDL < minimum operator
#### http://mathesaurus.sourceforge.net/idl-numpy.html idl to python operations

import numpy as np  # for calculations
import matplotlib.pyplot as plt  # for plot

from astropy.io import fits   # For reading the files, opening and closing
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

import glob  # to acces the files on pc
#import scipy.signal as sig # to use the median filter
import scipy.ndimage as ndim # to use the median filter

import pandas as pd # to save the data

from centroid_finder import make_polar

from photutils.background import Background2D

# FUNCTIONS
def rotate_image(original_image):
    rotated_image = np.rot90(original_image, k=1, axes=(1,0))
    return rotated_image

def trim_edges(original_image,width):
    original_image[:,0:width] = 0 
    original_image[0:width,:] = 0
    original_image[:,original_image.shape[1]-width:original_image.shape[1]] = 0 
    original_image[original_image.shape[0]-width:original_image.shape[0],:] = 0
    return original_image

def repair_middle_func(original_image):
    middle_column = int(original_image.shape[1]*0.5)
    original_image[:,middle_column] = (original_image[:,middle_column-1] + original_image[:,middle_column+1])*0.5
    return original_image

def centroid_func(image,weight):
    n_rows = image.shape[0]
    row_indexes = np.arange(n_rows)
    row_vector = np.zeros(n_rows)
    for i in np.arange(0,n_rows):
        row_vector += image[i,:]*weight[i,:]
    x_centroid = np.sum(row_indexes*row_vector)/np.sum(row_vector)
    # Modify the centroid position to be a value relative to the center of the
    # image and use that as the target center.
    x_centroid -= (n_rows-1)*0.5 # -1 to get index position
    n_columns = image.shape[1]  # n_columns are actually row, should be swaped
    col_indexes = np.arange(n_columns)
    col_vector = np.zeros(n_columns)
    for i in np.arange(0,n_columns):
        col_vector += image[:,i]*weight[:,i]
    y_centroid = np.sum(col_indexes*col_vector)/np.sum(col_vector)
    y_centroid -= (n_columns-1)*0.5
    return x_centroid, y_centroid

##### Not needed to find the first non zero row, take first row outside of trimmed edge manually
# find the first zero row encountered, width depends on the trimed edges, reverse is to get the first or last row
# def zero_row_func(med_image,width,reverse):
#     non_zero_row = False
#     if reverse == False:
#         row_indexes =  np.arange(width,med_image.shape[0])    
#     else:    
#         row_indexes = np.arange(med_image.shape[0]-1,width,-1)
#     for i in row_indexes:  
#         for j in np.arange(width,med_image.shape[1]-width):
#             if med_image_prim[i,j] !=0:
#                 non_zero_row = True
#             else:
#                 break
#         if non_zero_row ==True:
#             break
#     return i
# def zero_col_func(med_image,width,reverse):
#     non_zero_col = False
#     if reverse == False:
#         col_indexes =  np.arange(width,med_image.shape[1])    
#     else:    
#         col_indexes = np.arange(med_image.shape[1]-1,width,-1)
#     for i in col_indexes:  
#         for j in np.arange(width,med_image.shape[0]-width):
#             if med_image_prim[j,i] !=0:
#                 non_zero_col = True
#             else:
#                 break
#         if non_zero_col ==True:
#             break
#     return i

def background_average(med_image,med_weight,non_zero_row,top,rows):
    if top == True:
        sign = +1
    else:
        sign = -1
    if rows == True:
        rows_imag = med_image[non_zero_row,:]*med_weight[non_zero_row,:] + \
                         med_image[non_zero_row+sign,:]*med_weight[non_zero_row+sign,:] + \
                         med_image[non_zero_row+sign*2,:]*med_weight[non_zero_row+sign*2,:] 
        rows_weight = med_weight[non_zero_row,:] + med_weight[non_zero_row+sign,:] + med_weight[non_zero_row+sign*2,:]
        weight_gt0 = np.where(rows_weight>0) # only keeping the >0 indixes as you can not devide by 0
        rows_imag[weight_gt0] = rows_imag[weight_gt0]/rows_weight[weight_gt0] # weighted average of the 3 bottom rows filtered >0
        return rows_imag
    else: # non_zero_row are columns here
        cols_imag = med_image[:,non_zero_row]*med_weight[:,non_zero_row] + \
                         med_image[:,non_zero_row+sign]*med_weight[:,non_zero_row+sign] + \
                         med_image[:,non_zero_row+sign*2]*med_weight[:,non_zero_row+sign*2] 
        cols_weight = med_weight[:,non_zero_row] + med_weight[:,non_zero_row+sign] + med_weight[:,non_zero_row+sign*2]
        weight_gt0 = np.where(cols_weight>0) # only keeping the >0 indixes as you can not devide by 0
        cols_imag[weight_gt0] = cols_imag[weight_gt0]/cols_weight[weight_gt0] # weighted average of the 3 bottom rows filtered >0
        return cols_imag

def aper_photom(image, centre = None,radius = None ):
    ############## 2 dimension image as input
    imag_x_axis_size = image.shape[1] # number of columns
    imag_y_axis_size = image.shape[0] # number of rows
    x_origin = (imag_x_axis_size-1)*0.5
    y_origin = (imag_y_axis_size-1)*0.5
    
    # The radius must be a positive scalar number. If it is entered with a
    # non-numeric value, assume a default radius equal to half the smaller
    # of the two dimensions of the image array. If it is numeric, use the
    # absolute value of the first element of the entered value.    
    if radius == None:
        aper_radius = int(np.min([imag_x_axis_size,imag_y_axis_size])*0.5)
    else:
        aper_radius = radius
    
    annulus_radius = np.max([aper_radius+1,aper_radius*np.sqrt(np.sqrt(2))])
    
    # The center position must be defined by a numeric 2-element vector (or
    # at least the first 2 elements of a vector). If it is entered with a
    # non-numeric value, or a scalar, assume a default center position at
    # the center of the array.
    if np.any(centre) == None:
        x_centre = x_origin
        y_centre = y_origin
    else:
        x_centre = centre[0]
        y_centre = centre[1]
    
    x_diff = np.zeros([imag_x_axis_size,imag_y_axis_size])
    y_diff = np.zeros([imag_x_axis_size,imag_y_axis_size])
    for i in np.arange(0,imag_x_axis_size):
        x_diff[:,i] = i
    for i in np.arange(0,imag_y_axis_size):
        y_diff[i,:] = i
    x_diff = x_diff - x_origin - x_centre 
    y_diff = y_diff - y_origin - y_centre 
    radial_distr = x_diff*x_diff + y_diff*y_diff
    
    # Identify the region of the aperture over which to sum, and the region
    # of the surrounding annulus to use in estimating the background signal
    # level. Compute the total signal in the aperture, the total signal in
    # the annulus, the total number of elements contributing to each signal,
    # the variance of the signal in the annulus, and use that to estimate
    # the variance of the signal in the summed regions.
    aper_wh = np.where(radial_distr <= aper_radius*aper_radius)  ########### not sure what happens here
    aper_num = np.float(aper_wh[0].shape[0]) # number that fulfil the above condition
    annulus_wh = np.where(np.logical_and(annulus_radius*annulus_radius >= radial_distr, radial_distr >= aper_radius*aper_radius))
    annulus_num = np.float(annulus_wh[0].shape[0])
    aper_signal = np.sum(image[aper_wh])
    #print(aper_signal)
    if annulus_num >0:
        annulus_signal = np.sum(image[annulus_wh])
        average_background = annulus_signal/annulus_num
        variance_background = np.sum((image[annulus_wh]-average_background)**2) / annulus_num
    else:
        annulus_signal = 0
        average_background = 0
        variance_background = 0
    
    aper_background = average_background * aper_num
    variance_background = variance_background * aper_num
    variance_signal = variance_background
    
    photom = aper_signal - aper_background
    photom_unc = np.sqrt(variance_signal + variance_background)
    final = [photom, aper_signal, aper_background, photom_unc, np.sqrt(variance_signal), np.sqrt(variance_background), aper_radius, \
              x_centre, y_centre]  # should be y_centre, x_centre as python has the order [row,column]
    # not [column,row] as in IDL, but left like this for now
    # PATCHED_IMAGE - a replica of the initial IMAGE, but with the
    # pixel values in the aperture region replaced by the background
    # pixel value.    
    patched_image = np.copy(image)
    patched_image[aper_wh] = average_background
    return photom, final, patched_image
    


#%%

def epoxi_vis_read(folder,year,observations,trim_primary,repair_middle,remove_background,\
                   min_aperature,max_aperature,width_trim, astronomical_unit = 149.597870691e06):
    df = pd.DataFrame()
    # MAIN
    ### Get filepath of all fit files and access all files one by one
    # * causes it to access every file within the folder
    for filepath in glob.iglob(r'../../DATA/dif-e-hriv-3_4-epoxi-earth-v2.0/data/'+folder+'/'+year+'/'+observations+'/*.fit'):
    # for filepath in glob.iglob(r'./verification/*.fit'): ### verification
    #for filepath in glob.iglob(r'../DATA/dif-e-hriv-3_4-epoxi-earth-v2.0/data/rad/2008/078/*.fit'):   
        print(filepath)
        fits_inf = fits.open(filepath)
        #fits_inf.info()  
        ### NAN values not filtered out for flags and snr and destripe
        # np.nan_to_num(fits_inf[1].data, nan = 1.0) can be used for image_flags
        image_prim = np.nan_to_num(rotate_image(fits_inf[0].data)) # Rotate the image to get North up, 90 deg CW !!!CHECK IF THIS IS NORTH UP OR SOUTH UP!!!
        image_flags = rotate_image(fits_inf[1].data) # Order should be the same for all visible light files 
        # image_snr = rotate_image(fits_inf[2].data)
        # image_destripe = rotate_image(fits_inf[3].data)
        ########################### verification only prim and flags
        # image_prim = np.nan_to_num(fits_inf[0].data) # Rotate the image to get North up, 90 deg CW !!!CHECK IF THIS IS NORTH UP OR SOUTH UP!!!
        # image_flags = fits_inf[1].data # Order should be the same for all visible light files 
    
        
        # Subtract the quality-flag image from unity. If the quality flag is 0,
        # then the resulting weight is 1. If the quality flag is greater than 1,
        # the resulting weight is negative. Retain only values greater than 0.
        weight = np.ones(image_flags.shape) - image_flags      #################### if NAN values = 0 will result in 1 weight!!!!!!!!!!!!!
        weight = np.where(weight<0,0,weight) # setting negative values to zero
        # Set the weight at the edge of the image to zero. The weight of this region
        # is zero, regardless of whether the region is trimmed off before returning
        # the image from the function.
        weight = trim_edges(weight,width_trim)  
        
        # If the keyword is set to trim the edges of the image array, then zero the
        # values of rows and columns in the outer edge, to a width of 5 pixels.
        if trim_primary == True:
            image_prim = trim_edges(image_prim,width_trim)
            
        # The middle column of the image (actually, one to the right of that) is
        # overly amplified by the flat-field correction. If the keyword REPAIR_MIDDLE is
        # set, replace this column by the average of the neighboring two columns.
        # (Technically, the middle is at (NAXIS1-1)/2 = for instance, 255.5. 256 is
        # the column whose value needs to be corrected).
        if repair_middle == True:
            image_prim = repair_middle_func(image_prim)
        
        # If the keyword REMOVE_BACKGROUND is set, then we need to estimate an
        # image of the background that can be subtracted from the image. We will
        # do this by interpolating across the image array from an average of a
        # few rows or columns at the edge of the non-zero-weighted image region.
        # The interpolation step runs from the middle of the 3 rows or columns
        # that are averaged, so it extrapolates slightly to include the outermost
        # of the averaged rows/columns.
        if remove_background == True:
            med_image_prim = ndim.median_filter(image_prim,3)  # 3, default value,  sig.medfilt alternative but slower
            med_weight = ndim.median_filter(weight,3)
            
            earth_radius_pxl = 150 #60 150 for polar 1
            centroid_last = np.array([20,10]) + (512-1)/2. #np.array([10,30]) + (512-1)/2. for polar 1
            
            radius, phi = make_polar(512,512, centre = centroid_last)
            
            image_disk = np.zeros([512,512])
            mask_disk = np.where(np.logical_and(radius <= earth_radius_pxl, np.logical_or(phi <= 0.5*np.pi, phi>= 1.5*np.pi)))
            image_disk[mask_disk] = True
            
            width_trim = 6
            bkg = Background2D(med_image_prim,(256,128),mask = image_disk, exclude_percentile=0,sigma_clip=None) # (128,128) (ny,nx) (256,128)
            #bkg = Background2D(med_image_prim,(128,128)) # doesn't work exclude_percentile fixes it
            result_imag = med_image_prim - bkg.background
            result_imag = trim_edges(result_imag, 6)
            
            first_non_zero_row = width_trim
            last_non_zero_row = med_image_prim.shape[0] - width_trim -1 # -1 for index
            # Added image of the top 3 rows and added weight of the bottom 3 rows
            rows_top_imag = background_average(med_image_prim, med_weight, first_non_zero_row, True, True) # weighted average of the 3 bottom rows filtered >0
            rows_bot_imag = background_average(med_image_prim, med_weight, last_non_zero_row, False, True)
            for i in np.arange(first_non_zero_row,last_non_zero_row+1): # +1 as last is not included in np.arange
                # weigthed average of the background subtracted from all rows, starting at the middle of the averaged edge (3 rows thick)
                # so first_non_zero_row+1 = 7 from the top and bottom.
                # -1 for the index, -7 for the edge, -i to let rows closer to the top have a higher average background from the top background
                middle_averaged = first_non_zero_row +1
                image_prim[i,:] -= (rows_top_imag*(image_prim.shape[0]-1-middle_averaged-i) + rows_bot_imag*(i-middle_averaged))\
                    /(image_prim.shape[0]-1-middle_averaged-middle_averaged)
            # Need to calculate a new median image from the now-modified image.
            med_image_prim = ndim.median_filter(image_prim,3)
            first_non_zero_col = width_trim
            last_non_zero_col = med_image_prim.shape[1] - width_trim -1 #-1 for indexs
            cols_left_imag = background_average(med_image_prim, med_weight, first_non_zero_col, True, False)
            cols_right_imag = background_average(med_image_prim, med_weight, last_non_zero_col, False, False)
            # for the columns, left half subtract left average column, the right half the right average column
            for i in np.arange(first_non_zero_col, int((med_image_prim.shape[1]-1)*0.5)):
                image_prim[:,i] -= cols_left_imag
            for i in np.arange(int(med_image_prim.shape[1]*0.5),last_non_zero_col+1):
                image_prim[:,i] -= cols_right_imag
            ##### NEW BACKGROUND METHOD
            image_prim = result_imag
            
            
            
        naxis1 = image_prim.shape[1]
        naxis2 = image_prim.shape[0]
        epoxi_hrivis = {'file': 'no file processed',    \
                      'exposure_ID':   0,   'image_number':  0,           \
                      'date_calendar': ' ',       'date_julian':   0.0,       \
                      'calibration':   'unknown', 'units':         'arbitrary', \
                      'duration':      0.0,                                   \
                      'filter_name':   'unknown',  'filter_cw':      0.0,     \
                      'minimum':       0.0,       'maximum':       0.0,       \
                      'median':        0.0,       'one_sigma':     0.0,       \
                      'num_saturated': 0,                                     \
                      'mode_number':   0,         'mode':          'unknown', \
                      'mission':       'supposed to be EPOXI',                \
                      'platform':      'supposed to be fly-by',               \
                      'instrument':    'supposed to be HRIIR',   'cpu': 'none', \
                      'target':        'unknown',                             \
                      'RA':            0.0,      'DEC':           0.0,        \
                      'range_SC':      0.0,      'diameter':      0.0,        \
                      'range_Sun':     0.0,                                   \
                      'nppa':          0.0,       'oblateness':    0.0,       \
                      'illum':         100.0,     'phase_angle':   0.0,       \
                      'sub_SC':        np.zeros(2),  'sub_Sun':       np.zeros(2),  \
                      'sub_SC_nom':    np.zeros(2),  'sub_Sun_nom':   np.zeros(2),  \
                      'north_angle':   0.0,       'Sun_angle':     0.0,       \
                      'image':         np.zeros((naxis1,naxis2)),             \
                      'weight':        np.zeros((naxis1,naxis2)),             \
                      'naxis1':        naxis1,    'naxis2':        naxis2,    \
                      'target_center': np.zeros(2),                           \
                      'signal':        0.0,       'signal_rms':    0.0,       \
                      'background':    0.0,       'background_rms':0.0,       \
                      'aperture':      0 }        
        
        # get all info from the header 
        image_prim_header = fits_inf[0].header
        epoxi_hrivis['image'] = image_prim
        epoxi_hrivis['weight'] = weight
      
        epoxi_hrivis['naxis1']      = image_prim_header['NAXIS1']
        epoxi_hrivis['naxis2']      =  image_prim_header['NAXIS2']
    
        epoxi_hrivis['file'] = image_prim_header['FILESDC']      # image file name
        epoxi_hrivis['exposure_ID']   = image_prim_header['EXPID']    # exposure ID
        epoxi_hrivis['image_number']  =  image_prim_header['IMGNUM']   # image number within exposure set
        epoxi_hrivis['date_calendar'] =  image_prim_header['OBSMIDDT'] # image midpoint date-time
        epoxi_hrivis['date_julian']   =  image_prim_header['OBSMIDJD'] # image midpoint Julian date
        epoxi_hrivis['calibration']   =  image_prim_header['CALTYPE']  # calibration type
        epoxi_hrivis['units']         =  image_prim_header['BUNIT']    # calibrated units
        epoxi_hrivis['duration']      =  image_prim_header['INTTIME']  # image integration time
        epoxi_hrivis['filter_name']   =  image_prim_header['FILTER']   # name of filter
        epoxi_hrivis['filter_cw']     =  image_prim_header['FILTERCW'] # filter center wavelength
        epoxi_hrivis['minimum']       =  image_prim_header['DATAMIN']  # minimum data value in image
        epoxi_hrivis['maximum']       =  image_prim_header['DATAMAX']  # maximum data value in image
        epoxi_hrivis['median']        =  image_prim_header['MEDPVAL']  # median data value in image
        epoxi_hrivis['one_sigma']     =  image_prim_header['STDPVAL']  # standard deviation of image pixels
        epoxi_hrivis['num_saturated'] =  image_prim_header['PSATNUM']  # number saturated pixels in image
        epoxi_hrivis['mode_number']   =  image_prim_header['IMGMODE']  # image-acquisition mode number
        epoxi_hrivis['mode']          =  image_prim_header['IMGMODEN'] # image-acquisition mode name
        
        epoxi_hrivis['mission']       = image_prim_header['MISSION']  # name of spacecraft mission
        epoxi_hrivis['platform']      = image_prim_header['OBSERVAT'] # 'observatory' name
        epoxi_hrivis['instrument']    = image_prim_header['INSTRUME'] # instrument name
        epoxi_hrivis['cpu']           = image_prim_header['SCPROCU']  # which spacecraft CPU
        
        epoxi_hrivis['target']        = image_prim_header['OBJECT']   # target of image
        epoxi_hrivis['RA']            = image_prim_header['BORERA']   # RA of instrument bore-sight
        epoxi_hrivis['DEC']           = image_prim_header['BOREDEC']  # DEC of instrument bore-sight
        epoxi_hrivis['range_SC']      = image_prim_header['RANGECEN']/astronomical_unit  # target range to spacecraft, AU
        epoxi_hrivis['range_Sun']     = image_prim_header['TARSUNR']/astronomical_unit  # target range to Sun, AU
        epoxi_hrivis['phase_angle']   = image_prim_header['PHANGLE']  # target phase angle, degrees
        epoxi_hrivis['north_angle']   = image_prim_header['CELESTN']  # J2000 Equat. North angle, CW from up, deg
        epoxi_hrivis['Sun_angle']     = image_prim_header['SOLARCLK'] # Sun clock angle wrt bore, CW from up, deg
        epoxi_hrivis['Sun_angle']     = epoxi_hrivis['north_angle'] - epoxi_hrivis['Sun_angle'] # Sun clock angle, CCW from Celestial north
        epoxi_hrivis['nppa']          = image_prim_header['RECPPAZ']  # Body North pole, CW from up, degs
        epoxi_hrivis['nppa']          = epoxi_hrivis['north_angle'] - epoxi_hrivis['nppa'] # Body North pole, CCW from Celestial north
        epoxi_hrivis['north_angle']   = 360. - epoxi_hrivis['north_angle']  # Equatorial North angle, CCW from 'up' on image
        
        # Correct the direction of north for rotation of the image.
        epoxi_hrivis['north_angle']  = (epoxi_hrivis['north_angle'] + 90.0) % 360
        
        # [W Longitude, latitude] of sub-spacecraft and sub-Solar points.
        # The header actually stores East longitude and thus must be converted
        # by subtraction from 360 degrees.
        # nom is also kept as REC sometimes has nan values (-1399,-999)
    
        epoxi_hrivis['sub_SC']       = [360 - image_prim_header['RECSCLON'], image_prim_header['RECSCLAT']]
        epoxi_hrivis['sub_SC_nom']       = [360 - image_prim_header['NOMSCLON'], image_prim_header['NOMSCLAT']]
        epoxi_hrivis['sub_Sun']      = [360 - image_prim_header['RECSOLON'], image_prim_header['RECSOLAT']]
        epoxi_hrivis['sub_Sun_nom']      = [360 - image_prim_header['NOMSOLON'], image_prim_header['NOMSOLAT']]
        
        
        # Centroid the image as an approximation to the center of the target image.
        # Collapse the rows and columns to create two vectors that can be multiplied
        # by index vectors in order to compute the centroid position.
        centroid = centroid_func(image_prim,weight)
        initial_centroid = centroid
        
        # Employ a succession of circular apertures of increasing dimension to
        # estimate the total signal in the image. This is just a crude measure
        # to get an initial estimate; however, for images with the Earth fully
        # within the field of view, it ought to be pretty accurate. Stop
        # tweaking the aperture size when the total signal within the aperture
        # (ignore the background-subtracted signal, at this point) ceases to vary.
        # The largest permissible aperture diameter is the smaller of the image
        # dimensions. However, if APER_FINISH reaches this point with a value of
        # zero, then it has not yet been set and needs to be set to the default
        # value of the smallest of the two image dimensions.
    
        # Do an initial estimate of the background, then subtract the image with
        # the aperture replaced by the background in order to recalculate the
        # centroid. This may help in rejecting other sources and background
        # signal that may distort the identification of the centroid position.
        #################
        
        aper_start = np.around(min_aperature)
        aper_finish = np.around(max_aperature)
        if aper_finish<aper_start: 
            aper_finish = np.min([image_prim.shape[0],image_prim.shape[1]])
        aper_finish = np.min([aper_finish,image_prim.shape[0],image_prim.shape[1]])
        aper_diam = np.min([aper_start*1.5,image_prim.shape[0],image_prim.shape[1]]) # *1.5 here
        
        if aper_start >0: 
            signal_aperture, final, patch  = aper_photom(image_prim, centre = centroid,radius = aper_diam*0.5)
        else:
            patch = image_prim*0 + np.sum(image_prim)/np.float(image_prim.size)
        test = image_prim - patch  
        ########################################################    
        n_columns = image_prim.shape[1]
        row_indexes = np.arange(n_columns)
        row_vector = np.zeros(n_columns)
        for i in np.arange(0,n_columns):
            row_vector += test[i,:]*weight[i,:]
        x_centroid = np.sum(row_indexes*row_vector)/np.sum(row_vector)
        # Modify the centroid position to be a value relative to the center of the
        # image and use that as the target center.
        x_centroid -= (n_columns-1)*0.5 # -1 to get index position
        n_rows = image_prim.shape[0]
        col_indexes = np.arange(n_rows)
        col_vector = np.zeros(n_rows)
        for i in np.arange(0,n_rows):
            col_vector += image_prim[:,i]*weight[:,i]  ##### WHY IMAGE AND NOT TEST? the difference is small, image slightly higher,
            # which is closer to the true middle of the Earth
        y_centroid = np.sum(col_indexes*col_vector)/np.sum(col_vector)
        y_centroid -= (n_rows-1)*0.5
        centroid = np.array([x_centroid,y_centroid])
        # Now we finally are going to calculate the actual aperture photometry.
        # Work on a lightly median-filtered version of the image, to resist cosmic-ray
        # hits. This is an unwise choice for stellar photometry, but it should
        # work just fine for the Earth. It also eliminates the pesky influence
        # of scattered zero-weighted pixels.
        med_image = ndim.median_filter(image_prim*weight,3) 
        aper_diam = np.min([aper_start,image_prim.shape[0],image_prim.shape[1]]) # aper_start*1.5 works better
        # The verification of the code is successful however, it does not work (dividing by zero error) when the background is zero 
        # as the annulus signal is then zero. Furthermore, if the aperture radius is too small, the average background from 
        # the annulus will be equal to the signal within the aperture which will also lead to dividing by zero. 
        # This problem occurs due to the fact that when you find the centroid position, you multiply the starting radius 
        # by 1.5, but you do not do that when you do the iterations when finding the correct aperture. 
        # With the real data it did not lead to problems as there are no real constant values as in the verification data 
        # (10.0 values in a circle, 0.5 outside of the circle).
        
        if aper_start >0:
            done = False
            prev_signal = 0
            while done == False:
                signal_aperture, final, patch  = aper_photom(med_image, centre = centroid, radius = aper_diam*0.5)
                ### figure
                # fig2, ax2 = plt.subplots()
                # plt.title('Median data')
                # plt.imshow(med_image, cmap='gray')
                # circle1 = plt.Circle(centroid+255.5, aper_diam*0.5, color='r', fill=False)
                # ax2.add_artist(circle1)
                # plt.scatter(centroid[0]+255.5, centroid[1]+255.5, s=10)
                # plt.colorbar()
        
                epoxi_hrivis['signal']         = final[0]
                epoxi_hrivis['signal_rms']     = final[3]
                epoxi_hrivis['background']     = final[2]
                epoxi_hrivis['background_rms'] = final[5]
                epoxi_hrivis['aperture']       = final[6] * 2
                epoxi_hrivis['target_center']  = final[7:8+1] 
          
                aper_diam = aper_diam*1.05
                if aper_diam>aper_finish or np.abs((signal_aperture - prev_signal)/signal_aperture) < 1e-3: #1e-3
                    done = True
                prev_signal = signal_aperture
            
        else:
            signal_aperture = np.sum(med_image)
            n_pixels = np.float(med_image.size)
            # final = [signal_aperture, signal_aperture, 0, np.sqrt(np.sum((med_image-signal_aperture/n_pixels)**2)/n_pixels),0,]
            
            epoxi_hrivis['signal']         = signal_aperture
            epoxi_hrivis['signal_rms']     = np.sqrt(np.sum((med_image-signal_aperture/n_pixels)**2)/n_pixels)
            epoxi_hrivis['background']     = 0
            epoxi_hrivis['background_rms'] = np.copy(epoxi_hrivis['signal_rms'])
            epoxi_hrivis['aperture']       = np.sqrt(n_pixels)
            epoxi_hrivis['target_center']  = [0,0]
        #print(final)
        
        ### imaging
        # fig, ax = plt.subplots()
        # plt.title('Primary data')
        # plt.imshow(image_prim, cmap='gray') 
        # circle1 = plt.Circle(centroid+255.5, aper_diam*0.5, color='r', fill=False)
        # circle2 = plt.Circle(centroid+255.5, aper_diam*0.5/1.05, color='g', fill=False)
        # circle3 = plt.Circle(centroid+255.5, annulus_radius, color='b', fill=False)
        # ax.add_artist(circle1)
        # ax.add_artist(circle2)
        # ax.add_artist(circle3)
        # plt.scatter(centroid[0]+255.5, centroid[1]+255.5, s=10)
        # plt.colorbar()
        
        fig2, ax2 = plt.subplots()
        plt.title('Polar Observation 2: South')
        plt.imshow(med_image, cmap='gray')
        # circle1 = plt.Circle(centroid+255.5, aper_diam*0.5, color='r', fill=False, label = 'final aperture')
        # circle2 = plt.Circle(centroid+255.5, aper_diam*0.5/1.05, color='g', fill=False, label = 'previous aperture')
        # circle3 = plt.Circle(centroid+255.5, annulus_radius, color='b', fill=False, label = 'annulus')
        # ax2.add_artist(circle1)
        # ax2.add_artist(circle2)
        # ax2.add_artist(circle3)
        # leg_1 = ax2.legend((circle1,circle2,circle3), ('final aperture','previous aperture','annulus'), loc='lower left')
        # plt.scatter(centroid[0]+255.5, centroid[1]+255.5, s=10,color = 'b', label = 'final centroid')
        # plt.scatter(255.5, 255.5, s=10, color = 'r', label = 'centre image')
        # plt.scatter(initial_centroid[0]+255.5, initial_centroid[1]+255.5, s=10,color = 'g', label = 'initial centroid')
        # plt.legend()
        plt.colorbar()
        # ax2.add_artist(leg_1)    
    
        
        # ### saving
        df_temp = pd.DataFrame.from_dict(epoxi_hrivis,orient='index')
        df_temp = df_temp.transpose()
        df=df.append(df_temp,ignore_index=True)
        
        fits_inf.close() 
        ########################################################################## SAVING

    #df.to_hdf('../output/RADREV_'+ filepath.split('/')[-2]+'_'+observations+'_'+'min_aper'+'_'+str(min_aperature)+'_'+'dictionary_info.h5','epoxi_hrivis') ### manually change this
    return df
############

#%%
# astronomical_unit = 149.597870691e06 # astronomical unit, in km

# # INPUTS
# trim_primary = True
# repair_middle = True
# remove_background = True  # Flase for verification
# min_aperature = 150 # 38 for verification 300
# max_aperature = 0
# width_trim = 6 

verification = False
#%%
if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    df = epoxi_vis_read('rad','2009','277',True,True,True,150,0,6)
    # No difference in rad and radrev as far as I can see

# df = pd.DataFrame()

# observations = '150'

# folder = 'rad'
#%%
    

#%% VERIFICATION
    if verification==True:
        fits_inf = fits.open("../verification/veri_circle_30.0_radius.fit")
        data = fits_inf[0].data
        veri_weight = np.ones((512,512))
        veri_centroid = centroid_func(data,veri_weight)
        veri_aper_diam = 60.
        veri_signal_aperture, veri_final, veri_patch  = aper_photom(data, centre = veri_centroid, radius = veri_aper_diam*0.5)
        
        
        
        fits_inf.close()  
        #%%
        fits_inf = fits.open("../verification/veri_circle_60.0_radius.fit")
        data = fits_inf[0].data
        veri_weight = np.ones((512,512))
        veri_centroid = centroid_func(data,veri_weight)
        veri_aper_diam = 120.
        veri_signal_aperture, veri_final, veri_patch  = aper_photom(data, centre = veri_centroid, radius = veri_aper_diam*0.5)
        fits_inf.close()  
        
        #%%
        fits_inf = fits.open("../verification/veri_circle_30.0_radius_signal_strength_10.0.fit")
        data = fits_inf[0].data
        veri_weight = np.ones((512,512))
        veri_centroid = centroid_func(data,veri_weight)
        veri_aper_diam = 115.
        veri_signal_aperture, veri_final, veri_patch  = aper_photom(data, centre = veri_centroid, radius = veri_aper_diam*0.5)
        
        bkg = Background2D(data,(128,128))
        result_imag = data - bkg.background
        
        fits_inf.close()  
        
        print(veri_final)






