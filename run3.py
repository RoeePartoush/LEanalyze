#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:03:20 2020

@author: roeepartoush
"""


# base imports
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib
from scipy.optimize import curve_fit
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
# astropy imports
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import ascii
from astropy.time import Time
from astropy import wcs
from astropy.visualization import ZScaleInterval

from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb
import LEplots

#%%
# ========== LOAD LIGHT CURVES ==========
LChome_dir = '/Users/roeepartoush/Documents/Research/Data/light_curves'
LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
LCtable = F2D.LightCurves(LCs_file)

# %%
# ========== LOAD FITS FILES ==========
files=[]

# roots=['/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/']

# files1=['plhstproc/tyc4419/1/tyc4419.VR.20120118.216797_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120118.216798_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20120624.24651_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120624.24652_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20120626.24973_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120626.24974_stch_5.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20130109t.1692969_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20130109t.1692977_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20130109t.1692994_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20130609.32243_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20130609.32244_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20130826.33217_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20130826.33218_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.ut131005.DEIMOS.0045_5.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.ut131005.DEIMOS.0045_6.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.r.20131202.135755_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.r.20131202.135756_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.r.20131202.135757_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.g.b131228_0044_3.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.R.r131228_0038_4.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.g.b131228_0045_3.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4419_1.R.r131228_0039_4.sw',
#         'plhstproc/tyc4419/1/tyc4419.r.20140226.023317_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.r.20140226.023748_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.r.20140226.030626_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.r.20140226.032132_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20140530.111433_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20140531.105244_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20140531.105550_stch_5.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20140621.103511_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20140621.103815_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20140827.48447_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20140827.48448_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20140923.149397_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20140923.149398_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20141222.54623_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20141222.54624_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20141223.54713_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20141223.54714_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150211.157395_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150211.157399_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150211.157400_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150212.257541_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150212.257542_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20150213.357726_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150213.357727_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20150814.64481_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150814.64482_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150817.564744_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150817.564745_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150911.165398_stch_1.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20150911.165399_stch_1.sw']


roots=['/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/']

files1=['KECK/tyc4419_1_DEIMOS_5_6_coadd',
        '20131202/tyc4419_tyc4519_20131202_coadd',
        'KECK/tyc4419_1_R_LRIS_131228_coadd',
        '20140226/tyc4419_tyc4519_20140226_coadd']

for file in files1:
    files.append(roots[0]+file)

DIFF_df_BU = F2D.FitsDiff(files)

# %%

# row_i = 1 # FITS INDEXING: ORIGIN = 1
# col_i = 1 # FITS INDEXING: ORIGIN = 1
image_inds = [0,1,2,3]
Org=SkyCoord([[12.39251174, 58.76486208]],frame='fk5',unit=(u.deg, u.deg))[0]

fig = plt.figure()
ax = fig.add_subplot()
mjd_ls=[]
for image_i in image_inds:
    HDU = DIFF_df_BU.iloc[image_i]['Diff_HDU']
    # PLTSCL = HDU.header['SW_PLTSC']
    w_i = DIFF_df_BU.iloc[image_i]['WCS_w']
    
    print(w_i.wcs.cunit)
    print(wcs.utils.proj_plane_pixel_scales(w_i))
    PLTSCL = Angle(wcs.utils.proj_plane_pixel_scales(w_i)[0],'deg').arcsec
    # PLTSCL = LEtb.pix2ang(w_i,1).arcsec
    print(str(PLTSCL)+' [arcsec]')
    
    Org_pix = Org.to_pixel(w_i,origin=1,mode='wcs')
    col_i = np.round(float(Org.to_pixel(w_i,origin=1,mode='wcs')[0])).astype(int)
    row_i = np.round(float(Org.to_pixel(w_i,origin=1,mode='wcs')[1])).astype(int)
    print(col_i)
    
    mjd = HDU.header['MJD-OBS']
    mjd_ls.append(mjd)
    MAT = HDU.data
    flux_prof = MAT[:,col_i]
    arcsec_axis = (np.arange(flux_prof.size).astype(float)-row_i*1.0)*PLTSCL
    ax.scatter(arcsec_axis,flux_prof,label=mjd)
    plt.xlabel('[arcsec]')
    
plt.legend()