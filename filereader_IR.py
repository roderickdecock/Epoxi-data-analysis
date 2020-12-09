# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:51:08 2020

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

#%%
def epoxi_ir_read(folder,year,observations, astronomical_unit = 149.597870691e06):
    df = pd.DataFrame()
    for filepath in glob.iglob(r'../../DATA/dif-e-hrii-3_4-epoxi-earth-v2.0/data/'+folder+'/'+year+'/'+observations+'/*.fit'):
        print(filepath)
        fits_inf = fits.open(filepath)
        #fits_inf.info()
        image_prim = np.nan_to_num(fits_inf[0].data)
        image_flags = np.nan_to_num(fits_inf[1].data, nan = 1.0)
        image_wavelength = fits_inf[2].data # wavelength_map
        image_delta_wavelength = fits_inf[3].data # spectral_bandwidth_map
        image_snr = fits_inf[4].data # signal_to_noise
        
        # get all info from the header 
        image_prim_header = fits_inf[0].header
        
        naxis1      = image_prim_header['NAXIS1']
        naxis2      = image_prim_header['NAXIS2']
        
        epoxi_hriir = {'file': 'no file processed',    \
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
                      #'flags':         np.zeros((naxis1,naxis2)),             \
                      'wavelength':    np.zeros((naxis1,naxis2)),             \
                      'delta_wave':    np.zeros((naxis1,naxis2)),             \
                      'snr':           np.zeros((naxis1,naxis2)),             \
                      'spectrum':      np.zeros(naxis1),                      \
                      'reflect':       np.zeros(naxis1),                      \
                      'spec_wave':     np.zeros(naxis1),                      \
                      'spec_delta_wave': np.zeros(naxis1),                    \
                      'spec_snr':      np.zeros(naxis1),                      \
                      'cross_short':   np.zeros(naxis2),                      \
                      'cross_long':    np.zeros(naxis2),                      \
                      'naxis1':        naxis1,    'naxis2':        naxis2,    }          
        
        # filter bad (flagged) pixel out of SNR
        mask_flags = np.where(image_flags != 0)
        image_snr[mask_flags] = 0
        # filter pixels outside of wavelength range (1.05-4.55) out of the prim, delta wavelength and SNR
        # 1.05 to 4.83 micron is the spectral range of the instrument
        mask_wavelength = np.where(np.logical_or(image_wavelength <1.05,image_wavelength > 4.55))
        image_prim[mask_wavelength] = 0
        image_delta_wavelength[mask_wavelength] = 0
        image_snr[mask_wavelength] = 0
        
        epoxi_hriir['image'] = image_prim
        #epoxi_hriir['flags'] = image_flags
        epoxi_hriir['wavelength'] = image_wavelength
        epoxi_hriir['delta_wave'] = image_delta_wavelength
        epoxi_hriir['snr'] = image_snr
        
        epoxi_hriir['file'] = image_prim_header['FILESDC']      # image file name
        epoxi_hriir['exposure_ID']   = image_prim_header['EXPID']     # exposure ID
        epoxi_hriir['image_number']  =  image_prim_header['IMGNUM']   # image number within exposure set
        epoxi_hriir['date_calendar'] =  image_prim_header['OBSMIDDT'] # image midpoint date-time
        epoxi_hriir['date_julian']   =  image_prim_header['OBSMIDJD'] # image midpoint Julian date
        epoxi_hriir['calibration']   =  image_prim_header['CALTYPE']  # calibration type
        epoxi_hriir['units']         =  image_prim_header['BUNIT']    # calibrated units
        epoxi_hriir['duration']      =  image_prim_header['INTTIME']  # image integration time
        epoxi_hriir['filter_name']   =  'HRI_IR'                      # name of filter
        epoxi_hriir['filter_cw']     = np.sum(image_wavelength)/image_wavelength.size # filter center wavelength
        epoxi_hriir['minimum']       =  image_prim_header['DATAMIN']  # minimum data value in image
        epoxi_hriir['maximum']       =  image_prim_header['DATAMAX']  # maximum data value in image
        epoxi_hriir['median']        =  image_prim_header['MEDPVAL']  # median data value in image
        epoxi_hriir['one_sigma']     =  image_prim_header['STDPVAL']  # standard deviation of image pixels
        epoxi_hriir['num_saturated'] =  image_prim_header['PSATNUM']  # number saturated pixels in image
        epoxi_hriir['mode_number']   =  image_prim_header['IMGMODE']  # image-acquisition mode number
        epoxi_hriir['mode']          =  image_prim_header['IMGMODEN'] # image-acquisition mode name
        
        epoxi_hriir['mission']       = image_prim_header['MISSION']  # name of spacecraft mission
        epoxi_hriir['platform']      = image_prim_header['OBSERVAT'] # 'observatory' name
        epoxi_hriir['instrument']    = image_prim_header['INSTRUME'] # instrument name
        epoxi_hriir['cpu']           = image_prim_header['SCPROCU']  # which spacecraft CPU
        
        epoxi_hriir['target']        = image_prim_header['OBJECT']   # target of image
        epoxi_hriir['RA']            = image_prim_header['BORERA']   # RA of instrument bore-sight
        epoxi_hriir['DEC']           = image_prim_header['BOREDEC']  # DEC of instrument bore-sight
        epoxi_hriir['range_SC']      = image_prim_header['TARSCR']/astronomical_unit  # target range to spacecraft, AU
        epoxi_hriir['range_Sun']     = image_prim_header['TARSUNR']/astronomical_unit  # target range to Sun, AU
        epoxi_hriir['phase_angle']   = image_prim_header['PHANGLE']  # target phase angle, degrees
        epoxi_hriir['north_angle']   = image_prim_header['CELESTN']  # J2000 Equat. North angle, CW from up, deg
        epoxi_hriir['Sun_angle']     = image_prim_header['SOLARCLK'] # Sun clock angle wrt bore, CW from up, deg
        epoxi_hriir['Sun_angle']     = epoxi_hriir['north_angle'] - epoxi_hriir['Sun_angle'] # Sun clock angle, CCW from Celestial north
        epoxi_hriir['nppa']          = image_prim_header['RECPPAZ']  # Body North pole, CW from up, degs
        epoxi_hriir['nppa']          = epoxi_hriir['north_angle'] - epoxi_hriir['nppa'] # Body North pole, CCW from Celestial north
        epoxi_hriir['north_angle']   = 360. - epoxi_hriir['north_angle']  # Equatorial North angle, CCW from 'up' on image
        
        # [W Longitude, latitude] of sub-spacecraft and sub-Solar points.
        # The header actually stores East longitude and thus must be converted
        # by subtraction from 360 degrees.
        # nom is also kept as REC sometimes has nan values (-1399,-999) this was the case for visible
    
        epoxi_hriir['sub_SC']       = [360 - image_prim_header['RECSCLON'], image_prim_header['RECSCLAT']]
        epoxi_hriir['sub_SC_nom']       = [360 - image_prim_header['NOMSCLON'], image_prim_header['NOMSCLAT']]
        epoxi_hriir['sub_Sun']      = [360 - image_prim_header['RECSOLON'], image_prim_header['RECSOLAT']]
        epoxi_hriir['sub_Sun_nom']      = [360 - image_prim_header['NOMSOLON'], image_prim_header['NOMSOLAT']]
        
        for i in np.arange(0,naxis2): # loop through the rows
            # sum of all rows for every column 
            epoxi_hriir['spectrum'] += epoxi_hriir['image'][i,:]  # (SPATIAL) SUM OF THE SPECTRUM NOT AVERAGE
            epoxi_hriir['spec_wave'] += epoxi_hriir['wavelength'][i,:]
            epoxi_hriir['spec_delta_wave'] += epoxi_hriir['delta_wave'][i,:]
            epoxi_hriir['spec_snr'] += epoxi_hriir['snr'][i,:]**2
        epoxi_hriir['spec_snr'] = np.sqrt(epoxi_hriir['spec_snr'])
        
        mask_short = np.where(epoxi_hriir['spec_wave']<2.7*128) # originally 2.7 micron here, but since it is the sum *128
        for i in mask_short[0]:
            # add image*delta, all rows of the corresponding column
            # To get the cross-dispersion (parallel to slit) distribution of signal.
            epoxi_hriir['cross_short'] +=  epoxi_hriir['image'][:,i] * epoxi_hriir['delta_wave'][:,i]
        mask_long = np.where(epoxi_hriir['spec_wave']>2.7*128)
        for i in mask_long[0]:
            epoxi_hriir['cross_long'] +=  epoxi_hriir['image'][:,i] * epoxi_hriir['delta_wave'][:,i]
        # Saving
        df_temp = pd.DataFrame.from_dict(epoxi_hriir,orient='index')
        df_temp = df_temp.transpose()
        df=df.append(df_temp,ignore_index=True)
        
        fits_inf.close() 
    df.to_hdf('../output/IR_RAD_'+ filepath.split('/')[-2]+'_'+observations+'_'+'dictionary_info.h5','epoxi_hrivis') ### manually change this
    return df


#%%
if __name__ == "__main__": # to prevent this code from running when importing functions elsewhere
    df = epoxi_ir_read('rad','2009','278')







