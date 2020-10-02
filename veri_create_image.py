# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:20:39 2020

@author: roder
"""

from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

x = np.arange(0,512)
y = np.arange(0,512)
array = np.zeros((y.size,x.size))
flags = np.zeros((y.size,x.size))

cx = 255.5
cy = 255.5
r = 30.0

mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
strength =  10.
background_strength = 0.5
array[mask] = strength
array[mask==False] = background_strength
signal = np.count_nonzero(mask)
background = np.count_nonzero(mask==False)
total_signal = signal*strength
total_background = background*background_strength

# plt.figure()
# plt.pcolormesh(x, y, array)
# plt.colorbar()
# plt.show()

print(total_signal)


#%%
hdu1 = fits.PrimaryHDU(array)
hdu2 = fits.ImageHDU(flags)
new_hdul = fits.HDUList([hdu1, hdu2])
new_hdul.writeto('verification/veri_circle_'+str(r)+'_radius_signal_strength_'+str(strength)+'.fit',overwrite = True)

