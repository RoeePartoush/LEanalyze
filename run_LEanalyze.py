#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 00:41:12 2021

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
from tkinter.filedialog import askopenfiles
import tkinter
from IPython import get_ipython
from copy import deepcopy

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





# ========== LOAD LIGHT CURVES ==========
LChome_dir = '/Users/roeepartoush/Documents/Research/Data/light_curves'
LCs_file = os.path.join(LChome_dir,'SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desg_ab.txt')
LCtable = F2D.LightCurves(LCs_file)

# %%
# ========== LOAD FITS FILES ==========

# files=[]

# root=''#'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/'
# get_ipython().run_line_magic('gui', 'tk')
# filenames = askopenfiles(
#             initialdir=root,
#             title='Choose a fits file',
#             filetypes=[("fits files", "*.fits")],
#             mode="r"
#             )

# root = tkinter.Tk()
# root.destroy()
# get_ipython().run_line_magic('gui','')


# for flnm in filenames:
#     files.append(flnm.name[:-5])
#     flnm.close()


root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
    
filenames =['tyc4419_1_DEIMOS_5_6_coadd.fits',
            'tyc4419_tyc4519_20131202_coadd.fits',
            'tyc4419_1_R_LRIS_131228_coadd.fits']

files = []
for file in filenames:
    files.append(os.path.join(root, file[:-5]))


DIFF_df = F2D.FitsDiff(files)

### MANUAL PSF
DIFF_df.iloc[:,8] = pd.Series(data=np.zeros((len(DIFF_df),)))
for i in np.arange(len(DIFF_df)): DIFF_df.iloc[i,10]=LEtb.pix2ang(DIFF_df.iloc[i]['WCS_w'],DIFF_df.iloc[i,8])


# %%
# ========== PLOT FITS IMAGES ==========
get_ipython().run_line_magic('matplotlib', 'qt')
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]

global coord_list
coord_list=[]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures)

# %%
# ========== DEFINE PROFILE SHAPE ==========
# LBV_sc = SkyCoord(["5:34:12.1529 +37:36:58.879"],frame='fk5',unit=(u.hourangle, u.deg))
# Orgs = SkyCoord([LBV_sc])

# Orgs=SkyCoord([[13.607 ,59.073]],frame='fk5',unit=(u.deg, u.deg))
# Orgs=SkyCoord([Orgs[0].directional_offset_by(Angle(135,'deg'),Angle(3,u.arcsec))])

Orgs=SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 172 deg, "A1" in tyc4419_1.ut131228.slits.png


PA = Angle([Angle(172,'deg') for Org in Orgs])
Ln = Angle([100  for Org in Orgs],u.arcsec)
Wd = Angle([1  for Org in Orgs],u.arcsec)


clmns = ['Orig', 'PA', 'Length','WIDTH']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

# ========== EXTRACT PROFILE FROM IMAGE ==========
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec))

# %%
# ========== PLOT IMAGES & PROFILES ==========
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')

w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*1.4,slitFPdf.iloc[0]['Length']*2.4)

