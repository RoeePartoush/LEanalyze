#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:02:52 2021

@author: roeepartoush
"""


# ========== IMPORT PACKAGES ==========

# base imports
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')
# import matplotlib
# matplotlib.use('GTK3Agg') 
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
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb
import LEplots

from lmfit import Model
import lmfit
import corner

# %
# ========== LOAD LIGHT CURVES ==========
LChome_dir = '/Users/roeepartoush/Documents/Research/Data/light_curves'
LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
LCtable = F2D.LightCurves(LCs_file,extrapolate=True)
dm15 = 1.0
SN_f = LCtable['func_L'][LCtable['dm15']==dm15][0]

[X,Y] = np.meshgrid(np.linspace(-50,250,301),np.linspace(-100,100,201))

rot_ang = Angle(45,u.deg)
X_rot = X*np.cos(rot_ang.rad) -Y*np.sin(rot_ang.rad)
Y_rot = X*np.sin(rot_ang.rad) +Y*np.cos(rot_ang.rad)

Z = SN_f(X)
Z = Z/Z.max()
# plt.figure()
# ax = plt.pcolormesh(X_rot,Y_rot,np.zeros(Z.shape),alpha=Z)

# plt.xlim([-100,100])
# plt.ylim([-100,100])

# for til
t = np.linspace(-50,250,301)

axs=[]

scale_factors = [1/np.cos(Angle(ang,u.deg).rad) for ang in [0,45]]
for factor in scale_factors:
    grad = 10.0 # [days/arcsec]
    fig = plt.figure()
    ax = fig.add_subplot()
    axs.append(ax)
    plt.plot(t*factor/grad,SN_f(t),linestyle='-',label='t1',c='#1f77b4ff')
    plt.plot((t-10)*factor/grad,SN_f(t),'-',label='t2',c='#1f77b4ff')
    plt.plot((t+20)*factor/grad,SN_f(t),'-',label='t0',c='#1f77b4ff')
    # plt.plot(t*factor/grad,SN_f(t),linestyle='--',label='t1',c='#1f77b4ff')
    # plt.plot((t-10)*factor/grad,SN_f(t),linestyle=(0, (1, 1)),label='t2',c='#1f77b4ff')
    # plt.plot((t+20)*factor/grad,SN_f(t),label='t0',c='#1f77b4ff')
    plt.xlabel('[arcsec]')
    plt.xlim([-5,10])
    plt.ylim([-0.0,1.1])
    
    ax.spines['bottom'].set_position('zero')
    
    ax_phs = ax.twiny()
    plt.xlabel('[days]')
    
    xlims_arcsec = np.array(ax.get_xlim())
    peak_loc = 0.0
    ax_phs.set_xlim((xlims_arcsec-peak_loc)*grad/factor)



#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('viridis', 1)
newcolors = viridis(np.array([1]))#np.linspace(0, 1, 256))
newcolors[0] = np.array([0x5f/255,0x9c/255,0xff/255,0xff/255])
newcmp = ListedColormap(newcolors)


def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
n_bins = 1000
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)

[X,Y] = np.meshgrid(xx,yy)

# Generate the blobs. The range of the values is roughly -.0002 to .0002
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]

gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])

gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])

weights = 0.0*SN_f(X+Y-50)#(np.outer(gauss_y_high, gauss_x_high)
           #- np.outer(gauss_y_low, gauss_x_low))

# We'll also create a grey background into which the pixels will fade
greys = np.full((*weights.shape, 3), 255, dtype=np.uint8)

# First we'll plot these blobs using ``imshow`` without transparency.
vmax = np.abs(weights).max()
imshow_kwargs = {
    'vmax': vmax,
    'vmin': -vmax,
    'cmap': newcmp,#'jet',#'RdYlBu',
    'extent': (xmin, xmax, ymin, ymax),
}

fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, **imshow_kwargs)
ax.set_axis_off()
# Create an alpha channel of linearly increasing values moving to the right.
alphas = SN_f(X+Y-50)#np.ones(weights.shape)
# alphas[:, 30:] = np.linspace(1, 0, 70)

# Create the figure and image
# Note that the absolute values may be slightly different
fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, alpha=alphas, **imshow_kwargs)
ax.set_axis_off()

#%%
# ==== online example ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
n_bins = 100
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)

# Generate the blobs. The range of the values is roughly -.0002 to .0002
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]

gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])

gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])

weights = (np.outer(gauss_y_high, gauss_x_high)
           - np.outer(gauss_y_low, gauss_x_low))

# We'll also create a grey background into which the pixels will fade
greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)

# First we'll plot these blobs using ``imshow`` without transparency.
vmax = np.abs(weights).max()
imshow_kwargs = {
    'vmax': vmax,
    'vmin': -vmax,
    'cmap': 'jet',#'RdYlBu',
    'extent': (xmin, xmax, ymin, ymax),
}

fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, **imshow_kwargs)
ax.set_axis_off()
# Create an alpha channel of linearly increasing values moving to the right.
alphas = np.ones(weights.shape)
alphas[:, 30:] = np.linspace(1, 0, 70)

# Create the figure and image
# Note that the absolute values may be slightly different
fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, alpha=alphas, **imshow_kwargs)
ax.set_axis_off()