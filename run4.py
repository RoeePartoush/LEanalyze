#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:14:12 2020

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

# files1=['20120118/tyc4419_tyc4519_20120118_coadd',
#         '20120626/tyc4419_tyc4519_20120626_coadd',
#         'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd',
#         '20130109/tyc4419_tyc4519_20130109_coadd',
#         '20130609/tyc4419_tyc4519_20130609_coadd',
#         '20130826/tyc4419_tyc4519_20130826_coadd',
#         'KECK/tyc4419_1_DEIMOS_5_6_coadd',
#         '20131202/tyc4419_tyc4519_20131202_coadd',
#         'KECK/tyc4419_1_g_LRIS_131228_coadd',
#         'KECK/tyc4419_1_R_LRIS_131228_coadd',
#         '20140226/tyc4419_tyc4519_20140226_coadd',
#         '20140531/tyc4419_tyc4519_20140531_coadd',
#         '20140621/tyc4419_tyc4519_20140621_coadd',
#         '20140827/tyc4419_tyc4519_20140827_coadd',
#         '20140923/tyc4419_tyc4519_20140923_coadd']


for file in files1:
    files.append(roots[0]+file)

DIFF_df_BU = F2D.FitsDiff(files)

# %%
# ========== PLOT FITS IMAGES ==========
DIFF_df = DIFF_df_BU.copy()
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]

global coord_list
coord_list=[]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures)

# %%
# ========== DEFINE PROFILE SHAPE ==========
A1_sc = SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg))
Orgs = SkyCoord([A1_sc.directional_offset_by(Angle(172,'deg'),Angle(27,u.arcsec)),A1_sc])
# Orgs=SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 172 deg, "A1" in tyc4419_1.ut131228.slits.png
# Orgs=SkyCoord([[12.36304697 , 58.76167572]],frame='fk5',unit=(u.deg, u.deg)) # PA = 114 deg, "A2" in tyc4419_1.ut131228.slits.png
# Orgs=SkyCoord(["0:49:29.5278 +58:45:45.734"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 135
# Orgs=SkyCoord([[12.39123064, 58.76446428]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# Orgs=SkyCoord([[12.39251174, 58.76486208]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180 ### LATEST
# Orgs=SkyCoord([[12.37197527, 58.76040537]],frame='fk5',unit=(u.deg, u.deg)) # PA = 90
# Orgs=SkyCoord([[12.39073779, 58.76434532]],frame='fk5',unit=(u.deg, u.deg)) # PA = 130
# Orgs=SkyCoord([[12.37100268, 58.76017914]],frame='fk5',unit=(u.deg, u.deg)) # PA = 156
# Orgs=SkyCoord([[12.39087967, 58.76356894]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# Orgs=SkyCoord([[12.40778143, 58.76893596]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# ra = Longitude(np.array(coord_list).reshape((len(coord_list),2))[:,0],unit=u.deg)
# dec = Latitude(np.array(coord_list).reshape((len(coord_list),2))[:,1],unit=u.deg)
# Orgs = SkyCoord(ra, dec, frame='fk5')

# # SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5')
# SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5')
# PA = Angle([Org.position_angle(SN_sc)+Angle(180,'deg') for Org in Orgs])

# PA = Angle([Angle(180,'deg') for Org in Orgs])
# Ln = Angle([18  for Org in Orgs],u.arcsec)
PA = Angle([172,172],'deg')
Ln = Angle([18,100],u.arcsec)
Wd = Angle([7.5,1],u.arcsec)
clmns = ['Orig', 'PA', 'Length','WIDTH']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])
Wid = Angle(7.5,u.arcsec)
# %%
# ========== EXTRACT PROFILE FROM IMAGE ==========
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, width=Wid, REF_image=None, N_bins=18)

#%%
# ========== PLOT IMAGES & PROFILES ==========
plt.close('all')
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=None,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN', crest_lines=CLs)#,peaks_locs=b
# %
figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*4,slitFPdf.iloc[0]['Length']*2)
# LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'].directional_offset_by(slitFPdf.iloc[0]['PA'],Angle(25,u.arcsec)),slitFPdf.iloc[0]['Length']/2,slitFPdf.iloc[0]['Length']/4)


# %%
# ========== PLOT PROFILES 3D ==========

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xy'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xy'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# SN_f = LCtable['func_L'][LCtable['dm15']==1.0][0]
# fig=plt.figure()
# ax=fig.add_subplot(projection='3d')
# fig=plt.figure()
# ax2=fig.add_subplot()
inds=[0,1,2,3]
x_cutoff = [[-8,7.5]]*4#[[-11,2.5],   [-2.5,10],  [-2,10]]
y_cutoff = [[-1.2,30]]*4#[[-5,7.5],   [-7.5,10],  [-7.5,10]]
# star_masks = [[[-7.309,6.631],1.0], [[-2.217,8.295],1.0], [[0.022,4.964],1.0], [[-1.817,5.066],1.0], [[2.213,-0.652],1.0], [[-7.621,-2.471],1.0], [[-6.133,-2.835],1.0]]
star_masks = [[[-8.861,3.128],1.6], [[-4.050,5.485],1.2], [[-1.370,2.498],1.2], [[-3.205,2.342],1.2], [[1.581,-2.760],1.2], [[-7.903,-5.929],1.2], [[-6.380,-6.082],1.2]]
# star_masks = [[[-9.713,-0.867],1.0], [[-5.801,2.794],1.0], [[-2.364,0.721],1.0], [[-4.074,0.036],1.0], [[1.995,-3.444],1.0]]
# star_masks = [[[-2.321,9.508],1.0], [[-0.798,3.375],1.0], [[-0.648,9.610],1.0], [[3.164,-8.330],1.0]]
# star_masks = [[[-8.64,1.03],1.2],  [[-6.76,-1.56],1.2],    [[-5.05,-0.67],1.2],   [[-0.35,-4.46],1.2],   [[9.7,-3.13],1.2], [[-12.24,-2.95],1]]
input_arr_s, z_s = [], []
for ind in inds:
    iii=inds.index(ind)
    fig=plt.figure()
    if iii!=0:
        ax=fig.add_subplot(sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot()
    # if iii!=0:
    #     ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    # else:
    #     ax=fig.add_subplot(projection='3d')
    x=FP_df_lst[0].iloc[ind]['ProjAng'].arcsec
    y=FP_df_lst[0].iloc[ind]['PerpAng'].arcsec
    z=FP_df_lst[0].iloc[ind]['FluxProfile_ADU']
    mjd_curr = DIFF_df.iloc[ind]['Idate'    ]
    PSF_curr = DIFF_df.iloc[ind]['FWHM_ang'].arcsec
    
    # x_inds = np.logical_and(x>x_cutoff[iii][0], x<x_cutoff[iii][1])
    # x = x[x_inds]
    # y = y[x_inds]
    # z = z[x_inds]
    
    # y_inds = np.logical_and(y>y_cutoff[iii][0], y<y_cutoff[iii][1])
    # x = x[y_inds]
    # y = y[y_inds]
    # z = z[y_inds]
    
    for star_mask in star_masks:
        st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(star_mask[1]*PSF_curr)
        # st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(1.2*PSF_curr)
        x = x[st_mk_inds]
        y = y[st_mk_inds]
        z = z[st_mk_inds]
    
    shape = (len(x),1)
    input_arr_s.append(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1))
    z_s.append(z.reshape(shape))

    ax.scatter(x,y,cmap='jet',s=7,c=z)#,s=2,c=z)#,vmax=200)
    # ax.view_init(elev=90,azim=0)
    plt.xlim([-11,11])
    axisEqual3D(ax)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.xlabel('x')
    plt.ylabel('y')
    


# %%
# ========== Independent Light Curve ==========
# ======== FITTING FUNCTION DEFINITION ========

def LCgrid_eval(LCfunc,phs_wid,phs_res,phases_p):
    FWHMoSIG = 2*np.sqrt(2*np.log(2))
    
    phs_sigma = phs_wid/FWHMoSIG
    max_ph = np.nanmax(phases_p)+phs_sigma*3
    min_ph = np.nanmin(phases_p)-phs_sigma*3
    
    x_p = np.linspace(min_ph,max_ph,np.ceil((max_ph-min_ph)/phs_res).astype(int)+1)
    res = np.nanmean(np.diff(x_p))
    y_p_unscaled = gaussian_filter(LCfunc(x_p), sigma=phs_sigma/res, mode='constant', cval=0)
    
    return y_p_unscaled, x_p


SN_f = LCtable['func_L'][LCtable['dm15']==1.0][0]
phs_res = 0.1
def ConvLC(phase, phs_wid, PeakFlux, add_const):
    phs_min = np.max([np.nanmin(phase),-1e3])
    phs_max = np.min([np.nanmax(phase),4e3])
    flux_p_unscaled, phase_p = LCgrid_eval(SN_f,phs_wid,phs_res,np.array([phs_min,phs_max]))    
    const = PeakFlux/SN_f(0) # normalize maximun luminosity to PeakFlux
    phase_arr=np.array(phase)
    flux = np.interp(phase_arr, phase_p, const*flux_p_unscaled, left=0, right=0) + add_const
    # flux = SN_f(phase)*const
    return flux


def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06,add_const):
    x, y, mjd, PSFs = input_array[:,0], input_array[:,1], input_array[:,2], input_array[:,3]
    
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y# +mjd
    xy_phases = np.cos(p05)*p06*(x-p04) +np.sin(p05)*p06*y
    
    phase_grad = p06#np.hypot(polyphase_params[0,1],polyphase_params[1,0])
    
    eff_phs_wid = np.hypot(p01,phase_grad*PSFs[0])
    ConvLC_params = [eff_phs_wid,p02,add_const]
    output_array = ConvLC(xy_phases, *ConvLC_params)
    
    return output_array

## == TRASH ==
# polyphase_params = np.array([[p04,              np.sin(p05)*p06,    0,  0],
#                              [np.cos(p05)*p06,  0,                  0,  0],
#                              [0,                0,                  0,  0],
#                              [0,                0,                  0,  0]])
# xy_base_phases = np.polynomial.polynomial.polyval2d(x, y, polyphase_params)
# xy_phases = xy_base_phases + mjd
# %%
# ========== Independent Light Curve ==========
# ============== FITTING & PLOTS ==============
    
bnd = ([-200.0,     1e1,            -1e2,   -50.0*np.pi/180,  -4e1, -5e1 ],
        [200.0,    1e3,             1e2,    0.0*np.pi/180,   -1.3e-1, 5e1])
    
# bnd = ([0.0,     1e1,            -5.68e4,   -50.0*np.pi/180,   -5e1 ],
#         [200.0,    1e3,           -5.65e4,    50.0*np.pi/180,   -1.3e-1])

# bnd = ([0.0,     1e1,             -56800,    -27.0*np.pi/180,   -18 ],
#         [5.0,    1e3,             -56500,    -26.0*np.pi/180,   -17])

# bnd = ([0.0,     1e1,             -56633,    -27.0*np.pi/180,   -15 ],
#         [200.0,    1e3,             -56632,    -26.0*np.pi/180,   -14])

# x_arcsec origin version
def plot_local_phase_plane(ax, x_orig, ang_rad, slope, y_span, phase_span, mjd):
    y_span = np.array(y_span)
    phase_span = np.array(phase_span)
    y_msh, phs_msh = np.meshgrid(y_span,phase_span)
    x_msh = x_orig -(np.sin(ang_rad)*slope*y_span + 0*mjd+phs_msh)/(np.cos(ang_rad)*slope)
    
    # X_mesh, Y_mesh = np.meshgrid(np.linspace(x_msh.flatten().min(),x_msh.flatten().max(),10), np.linspace(y_msh.flatten().min(),y_msh.flatten().max(),10))
    
    # Z_mesh = 
    ax.plot_surface(x_msh,y_msh,mjd+phs_msh,alpha=0.2)
    return

# mjd origin version
# def plot_local_phase_plane(ax, mjd_orig, ang_rad, slope, y_span, phase_span, mjd):
#     y_span = np.array(y_span)
#     phase_span = np.array(phase_span)
#     y_msh, phs_msh = np.meshgrid(y_span,phase_span)
#     x_msh = -(np.sin(ang_rad)*slope*y_span + mjd_orig+0*mjd+phs_msh)/(np.cos(ang_rad)*slope)
    
#     # X_mesh, Y_mesh = np.meshgrid(np.linspace(x_msh.flatten().min(),x_msh.flatten().max(),10), np.linspace(y_msh.flatten().min(),y_msh.flatten().max(),10))
    
#     # Z_mesh = 
#     ax.plot_surface(x_msh,y_msh,mjd+phs_msh,alpha=0.2)
#     return

F_SCALE=100
# plt.figure()
# ax2 = plt.subplot() 
fig = plt.figure()
ax3 = fig.add_subplot(projection='3d')
CLs = np.zeros((len(input_arr_s),3,3))
popt_lst=[]
errs_lst=[]
for i in np.arange(len(input_arr_s)):
    z = z_s[i].flatten()
    inp_arr = input_arr_s[i]
    mjd = inp_arr[:,2][0]
    popt, pcov = curve_fit(ConvLC_2Dpolyphase,inp_arr,z,bounds=bnd,absolute_sigma=True,sigma=np.ones(inp_arr[:,0].shape)*20)
    errs = np.sqrt(np.diag(pcov))
    popt_lst.append(popt.copy())
    errs_lst.append(errs.copy())
    # popt = [20.0,     10e1,            -0.0,   -34.0*np.pi/180,   -14 ]
    # popt = [  3.09876149, 116.87448495,   8.91707928 +(mjd-56654.2),  -0.47029305,  -5.38910457, -14.00613042]
    print(popt)
    p01,p02,p04,p05,p06, add_const = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    PSF = inp_arr[:,3][0] # arcsec
    grad = popt[4] # day/arcsec
    # total_wid = popt[0] # days
    # dust_wid = np.sqrt(total_wid**2 - (grad*PSF)**2)
    # print('Total width: '+'{:.3f}'.format(total_wid)+' [days] <--> '+'{:.3f}'.format(total_wid/grad)+' [arcsec]')
    # print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
    # print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/grad)+' [arcsec]\n')
    dust_wid = popt[0] # days
    print('Total width: '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF))+' [days] <--> '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF)/grad)+' [arcsec]')
    print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
    print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/grad)+' [arcsec]\n')
    # print(np.sqrt(np.diag(pcov)))
    fig=plt.figure()
    if i!=0:
        ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(projection='3d')
    x, y = inp_arr[:,0], inp_arr[:,1]    
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y +mjd
    # ax.scatter(x,y,xy_phases,c='b',s=1)
    ax.scatter(x,y,ConvLC_2Dpolyphase(inp_arr,*popt),cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c='r',s=1)
    ax.scatter(x,y,z,cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,s=1)
    # im=ax.scatter(x,y,cmap='jet',c=z-ConvLC_2Dpolyphase(inp_arr,*popt),s=1)
    # fig.colorbar(im,ax=ax,orientation="horizontal")
    plt.xlabel('x [arcsec]')
    plt.ylabel('y [arcsec]')
    plt.title('{:.0f}'.format(mjd))
    plt.ylim([-1.5,4])
    axisEqual3D(ax)
    
    # ax2.scatter(p04/p06,mjd)
    # zero intersect: cos(p05)*p06*X + sin(p05)*p06*Y + p04 +mjd = 0 --> X = -(sin(p05)*p06*Y + p04 +mjd)/cos(p05)*p06
    zero_y = np.array([-0.5,0.5,0])*8
    zero_x = p04 -(np.sin(p05)*p06*zero_y + 0*mjd)/(np.cos(p05)*p06)
    # zero_x = -(np.sin(p05)*p06*zero_y + p04+0*mjd)/(np.cos(p05)*p06)
    ax3.plot(zero_x,zero_y,mjd)
    
    plot_local_phase_plane(ax3, p04, p05, p06, [-0.5,0.5], [-10,0,10], mjd)
    
    CLs[i,0,0], CLs[i,1,0], CLs[i,2,0] = zero_x*1.0
    CLs[i,0,1], CLs[i,1,1], CLs[i,2,1] = zero_y*1.0
    CLs[i,0,2], CLs[i,1,2], CLs[i,2,2] = mjd*np.ones(len(zero_y))

axisEqual3D(ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')

popts = np.stack(popt_lst)
errss = np.stack(errs_lst)

mjds = DIFF_df.iloc[:,1].to_numpy().astype(float).reshape(len(DIFF_df),1)
peaks = popts[:,2].reshape(len(DIFF_df),1)
peak_errs = errss[:,2].reshape(len(DIFF_df),1)
b = np.concatenate([mjds,peaks,peak_errs],axis=1)

# %%
# x = np.linspace(0, 1, 20)
# y = np.linspace(0, 1, 20)
# X, Y = np.meshgrid(x, y, copy=False)
# Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01

X = CLs[:,:,0].flatten()
Y = CLs[:,:,1].flatten()
Z = CLs[:,:,2].flatten()

A = np.array([X*0+1, X, Y]).T#, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
B = Z.flatten()

coeff, r, rank, s = np.linalg.lstsq(A, B)

X_mesh, Y_mesh = np.meshgrid(np.linspace(X.min(),X.max(),10), np.linspace(Y.min(),Y.max(),10))
Z_lstsq = coeff[0] + coeff[1]*X_mesh + coeff[2]*Y_mesh
ax3.plot_surface(X_mesh,Y_mesh,Z_lstsq,alpha=0.2)
# %%
# ========== GAUSSIAN FITTING & DISPLAY ==========

from scipy.stats import norm
def Gauss_model(input_array, mean, std, amp, angle_rad):
    x, y = input_array[:,0], input_array[:,1]
    polyarcsec_params = np.array([[0,              np.sin(angle_rad)],
                                  [np.cos(angle_rad),  0]])
    xy_arcsec = np.polynomial.polynomial.polyval2d(x, y, polyarcsec_params)
    output = amp*norm.pdf(xy_arcsec,mean,std)
    return output

bnd = ([-50, 0, 0, -50.0*np.pi/180],
       [50, 50, 1e3, +50.0*np.pi/180])

F_SCALE=100
plt.figure()
ax2 = plt.subplot()
for i in np.arange(len(input_arr_s)):
    z = z_s[i].flatten()
    inp_arr = input_arr_s[i]
    popt, pcov = curve_fit(Gauss_model,inp_arr,z,bounds=bnd)
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    fig=plt.figure()
    if i!=0:
        ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(projection='3d')
    x, y = inp_arr[:,0], inp_arr[:,1]
    ax.scatter(x,y,Gauss_model(inp_arr,*popt),cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c='b')
    ax.scatter(x,y,z,cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c=z,s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    mjd = inp_arr[:,2][0]
    plt.title(mjd)
    axisEqual3D(ax)
    ax2.scatter(popt[0],mjd)


# %%
# # ========== DEFINE FITTING FUNCTION ==========
# SN_f = LCtable['func_L'][LCtable['dm15']==1.0][0]
# phs_res = 0.1
# def ConvLC(phase, phs_wid, PeakFlux, add_const):
#     phs_min = np.max([np.nanmin(phase),-1e3])
#     phs_max = np.min([np.nanmax(phase),4e3])
#     flux_p_unscaled, phase_p = DFit.LCgrid_eval(SN_f,phs_wid,phs_res,np.array([phs_min,phs_max]))    
#     const = PeakFlux/SN_f(0) # normalize maximun luminosity to PeakFlux
#     phase_arr=np.array(phase)
#     flux = np.interp(phase_arr, phase_p, const*flux_p_unscaled, left=0, right=0) + add_const
#     return flux


# def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06):
#     pxx,pxy,pyy = 0,0,0
# # def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06,             pxx,pxy,pyy):
# # def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06,  p08,             pxx,pxy,pyy):
#     polyphase_params = np.array([[p04,              np.sin(p05)*p06,    pyy,  0],
#                                  [np.cos(p05)*p06,  pxy,                  0,  0],
#                                  [pxx,              0,                  0,  0],
#                                  [0,              0,                  0,  0]])
#     x, y, mjd, PSFs = input_array[:,0], input_array[:,1], input_array[:,2], input_array[:,3]
#     inds_umjd = []
#     for umjd in np.unique(mjd):
#         inds_umjd.append(np.argwhere(mjd==umjd)[:,0])
#     xy_base_phases = np.polynomial.polynomial.polyval2d(x, y, polyphase_params)
#     xy_phases = xy_base_phases + mjd#-mjd[0]
    
#     output_array = np.zeros(x.shape)
#     phase_grad = p06#np.hypot(polyphase_params[0,1],polyphase_params[1,0])
    
#     eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[0][0]])
#     ConvLC_params = [eff_phs_wid,p02,0]
#     # print('1: '+str(phase_grad*PSFs[inds_umjd[0][0]]))
#     output_array[inds_umjd[0]] = ConvLC(xy_phases[inds_umjd[0]], *ConvLC_params)
    
#     # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[1][0]])
#     # ConvLC_params = [eff_phs_wid,p08,0]
#     # # print('2: '+str(phase_grad*PSFs[inds_umjd[1][0]]))
#     # output_array[inds_umjd[1]] = ConvLC(xy_phases[inds_umjd[1]], *ConvLC_params)
    
#     # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[2][0]])
#     # ConvLC_params = [eff_phs_wid,p09,0]
#     # # print('3: '+str(phase_grad*PSFs[inds_umjd[2][0]]))
#     # output_array[inds_umjd[2]] = ConvLC(xy_phases[inds_umjd[2]], *ConvLC_params)
    
#     # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[3][0]])
#     # ConvLC_params = [eff_phs_wid,p10,0]
#     # # print('4: '+str(phase_grad*PSFs[inds_umjd[3][0]]))
#     # output_array[inds_umjd[3]] = ConvLC(xy_phases[inds_umjd[3]], *ConvLC_params)
    
#     return output_array

# # %%
# # ========== Light Curve Model FITTING ==========
    
# input_arr = np.concatenate(input_arr_s,axis=0)
# z = np.concatenate(z_s,axis=0).flatten()
# # input_arr = np.concatenate([input_arr_s[0],input_arr_s[2]],axis=0)
# # z = np.concatenate([z_s[0],z_s[2]],axis=0).flatten()

# # bnd = ([0.0,    1e1,            -5.68e4,   -50.0*np.pi/180,  -2e1,              1e1,1e1,1e1,             -1e-10, -1e-10, -1e-10],
# #         [10.0,   1e3,           -5.64e4,    50.0*np.pi/180,   -1e1,              1e3,1e3,1e3,             +1e-10, +1e-10, +1e-10])

# # bnd = ([0.0,    1e1,            -5.68e4,   -50.0*np.pi/180,  -2e1,              1e1,             -1e-10, -1e-10, -1e-10],
# #         [10.0,   1e3,           -5.64e4,    50.0*np.pi/180,   -1e1,              1e3,             +1e-10, +1e-10, +1e-10])

# # bnd = ([0.0,    1e1,            -5.68e4,   -50.0*np.pi/180,  -4e1,                           -1e-10, -1e-10, -1e-10],
# #         [80.0,   1e3,           -5.65e4,    50.0*np.pi/180,   -1e-2,                           +1e-10, +1e-10, +1e-10])

# bnd = ([0.0,    1e1,            -5.68e4,   -50.0*np.pi/180,  -4e1],
#         [80.0,   1e3,           -5.65e4,    50.0*np.pi/180,   -1e-2])

# # p_init = [ 2.0e-04,  1.1e+02, -5.6e+04, -3.7e-01, -7.3e+00,  5.0e+02,  5.050000e+02,  5.050000e+02]
# popt, pcov = curve_fit(ConvLC_2Dpolyphase,input_arr,z,bounds=bnd)#,p0 = p_init)
# print(popt)

# # %%
# # ========== DISPLAY FITTING RESULT ==========

# F_SCALE=1e2
# u_mjds = list(np.unique(input_arr[:,2]))
# for mjd in u_mjds:
#     indsss = np.argwhere(input_arr[:,2]==mjd)[:,0]
#     # print(indsss[0:10])
#     iii=u_mjds.index(mjd)
#     fig=plt.figure()
#     if iii!=0:
#         ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
#     else:
#         ax=fig.add_subplot(projection='3d')
#     # ppp = popt.copy()
#     zz=ConvLC_2Dpolyphase(input_arr[:,:], *popt)
#     x=input_arr[:,0]
#     y=input_arr[:,1]
#     PSF = input_arr[indsss,3][0] # arcsec
#     grad = popt[4] # day/arcsec
#     dust_wid = popt[0] # days
#     print('Total width: '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF))+' [days]')
#     print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]')
#     polyphase_params = np.array([[popt[2],                  np.sin(popt[3])*grad,    0,  0],
#                                   [np.cos(popt[3])*grad,          0,                  0,  0],
#                                   [0,                             0,                  0,  0],
#                                   [0,                                   0,                  0,  0]])
#     # polyphase_params = np.array([[popt[2],                  np.sin(popt[3])*grad,    popt[7],  0],
#     #                               [np.cos(popt[3])*grad,          popt[6    ],                  0,  0],
#     #                               [popt[5],                             0,                  0,  0],
#     #                               [0,                                   0,                  0,  0]])
#     # polyphase_params = np.array([[popt[2],                  np.sin(popt[3])*grad,    popt[10],  0],
#     #                              [np.cos(popt[3])*grad,          popt[9],                  0,  0],
#     #                              [popt[8],                             0,                  0,  0],
#     #                              [0,                                   0,                  0,  0]])
    
#     polyphase = np.polynomial.polynomial.polyval2d(x, y, polyphase_params) + mjd
#     # ax.scatter(x[indsss],y[indsss],polyphase[indsss],cmap='jet',vmax=F_SCALE/3,vmin=-F_SCALE/3,s=2,c=polyphase[indsss])
#     # ax.scatter(x[indsss],y[indsss],z[indsss]-zz[indsss],cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c=np.abs(z[indsss]-zz[indsss]),s=1)
#     ax.scatter(x[indsss],y[indsss],zz[indsss],cmap='bwr',vmax=1,vmin=0,s=2,c=polyphase[indsss]>0)
#     ax.scatter(x[indsss],y[indsss],z[indsss],cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c=z[indsss],s=5)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(mjd)
#     axisEqual3D(ax)

# %%
###########################
######## S T A S H ########
###########################
### INLINE RUNS TO KEEP ###
###########################

### MANUAL PSF
DIFF_df_BU.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
for i in np.arange(len(DIFF_df_BU)): DIFF_df_BU.iloc[i,10]=LEtb.pix2ang(DIFF_df_BU.iloc[i]['WCS_w'],DIFF_df_BU.iloc[i,8])

#%%
### STAR MASK PRINT
sss='['
SClst = [SkyCoord(coord,frame='fk5',unit=(u.deg, u.deg))[0] for coord in coord_list]
for SC in SClst:
    Sep_as = Orgs[0].separation(SC).arcsec
    PA_a = Orgs[0].position_angle(SC)-PA[0]
    x=Sep_as*np.cos(PA_a.rad)
    y=Sep_as*np.sin(PA_a.rad)
    print('x = '+str(x))
    print('y = '+str(y)+'\n')
    sss += '[['+'{:.3f}'.format(x)+','+'{:.3f}'.format(y)+'],1.2]'
    if not (SClst.index(SC)==(len(SClst)-1)):
        sss += ', '
sss += ']'
print(sss)

#%%
def remove_most_frequent(mat):
    mat = mat*1.0
    a,b=np.unique(mat.flatten(),return_counts=True)
    
    ind=np.argmax(b)
    value = a[ind]
    #print(value)
    
    mat[mat==value] = np.nan
    return mat
# # %
Zscale = ZScaleInterval()
figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for fig in tqdm(figures):
    # print('Fig. no.: '+str(fig.number))
    if fig.number>=1:#6e6:
        for ax in fig.get_axes():
            # print(ax.title.get_text())
            for im in ax.get_images():
                mat = im.get_array()*1.0
                clim = Zscale.get_limits(remove_most_frequent(mat))
                # mat[mat<0.1*np.mean(clim)] = np.nan
                # im.set_data(mat)
                # clim = np.array([-1*0,1])*1000
                # clim = np.array([0,25e3])
                # print(im.get_clim())
    #            cvmin, cvmax = im.get_clim()
    #            inc = float(10)
    #            if pchr=='a':
    #                inc = -inc
    #            elif pchr=='d':
    #                inc = inc
                span=clim[1]-clim[0]
                im.set_clim(clim[0],clim[1]+0.6 *span)
            fig.canvas.draw()
# %%
plt.figure()
phases = np.linspace(-50,100,1501)
plt.plot(phases,SN_f(phases))
#%%
coord_list=[np.array([[12.38895946, 58.76689237]]),
  np.array([[12.38806794, 58.76547782]]),
  np.array([[12.38985258, 58.76485593]]),
  np.array([[12.38979816, 58.7653667 ]]),
  np.array([[12.39286126, 58.7642474 ]]),
  np.array([[12.39383564, 58.76697893]]),
  np.array([[12.3940303 , 58.76656578]])]