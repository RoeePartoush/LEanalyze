#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:43:49 2020

@author: roeepartoush
"""

import numpy as np
from scipy import signal as sp
from scipy import interpolate as intp
# Set up matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from astropy.io import fits
from astropy import wcs

from astropy import units as u
from astropy.coordinates import Angle

from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import SkyCoord
from astropy.time import Time

from scipy.ndimage import gaussian_filter

import pandas as pd

# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb

# %%
plt.close('all')
x=np.arange(4200)
y=np.arange(2200)
Xmat, Ymat = np.meshgrid(x,y)

point = np.array([1200,1000])
uvec = LEtb.normalize(np.array([0.1,0.1]))
#t_uvec = LEtb.normalize(np.array([0,1]))

v_X = Xmat - point[0]
v_Y = Ymat - point[1]
    
dotprod_mat = v_X*uvec[0] + v_Y*uvec[1]

mat = dotprod_mat*1
mat = 100.0*(np.abs(mat)<0.5)
#func = DFit.GenConvLC(LCtable=LCtable,dm15=0.83)
#mat = func(mat,0,100)
gss_FWHM = 200
mat = gaussian_filter(mat, sigma=gss_FWHM/(2*np.sqrt(2*np.log(2))),mode='constant',cval=0)
#distvec_X = v_X - dotprod_mat*FP_pa_pix_uv[0]
#distvec_Y = v_Y - dotprod_mat*FP_pa_pix_uv[1]
#
#dist_mat = np.sqrt(distvec_X**2 + distvec_Y**2)



w = wcs.WCS(hdu_list[ind][0].header)
fig = plt.figure()
fig.add_subplot(111,projection=w)
clim = mat.max()#100
plt.imshow(mat, vmin=-0*clim, vmax=clim, origin='lower', cmap='gray', interpolation='none')