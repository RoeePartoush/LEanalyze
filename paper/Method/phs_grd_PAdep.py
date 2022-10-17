#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:29:44 2022

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

import run_global_vars

# ========== LOAD LIGHT CURVES ==========
LChome_dir = '/Users/roeepartoush/Documents/Research/Data/light_curves'
LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
run_global_vars.LCtable = F2D.LightCurves(LCs_file)


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

dm15 = 1.0
SN_f = run_global_vars.LCtable['func_L'][run_global_vars.LCtable['dm15']==dm15][0]

def impulse(t):
    return (np.abs(t)<0.5)*10.0

run_global_vars.light_curve_fn = SN_f

# phs_res = 0.1
def ConvLC(phase, phs_wid, PeakFlux, add_const, phs_res):
    phs_min = np.max([np.nanmin(phase),-1e3])
    phs_max = np.min([np.nanmax(phase),4e3])
    flux_p_unscaled, phase_p = LCgrid_eval(run_global_vars.light_curve_fn,phs_wid,phs_res,np.array([phs_min,phs_max]))    
    const = PeakFlux/run_global_vars.light_curve_fn(0) # normalize maximun luminosity to PeakFlux
    phase_arr=np.array(phase)
    flux = np.interp(phase_arr, phase_p, const*flux_p_unscaled, left=0, right=0) + add_const
    return flux

arcs_res = 0.1
def LEprof_model( input_array, sig_dust,  peak_flux,   peak_loc, phs_grad,      add_const,      PSF,    ignore_PSF=False, ignore_dust=False):
    #               [arcsec],   [days],        [1],    [arcsec], [days/arcsec], [arbittrary],  [arcsec] 
    # ignore_PSF=False, ignore_dust=False - dummy parameters for API compatibility
    x = input_array
    phases = phs_grad*(x-peak_loc)
    
    if ignore_dust:
        sig_dust=0.0
    if ignore_PSF:
        PSF=0.0
    # eff_phs_wid = np.hypot(phs_grad*sig_dust,phs_grad*PSF)
    eff_phs_wid = np.hypot(sig_dust*phs_grad,phs_grad*PSF)
    phs_res = np.abs(arcs_res*phs_grad)
    ConvLC_params = [eff_phs_wid,peak_flux,add_const, phs_res]
    output_array = ConvLC(phases, *ConvLC_params)
    
    return output_array


def plot_local_phase_plane(ax, x_orig, ang_rad, slope, y_span, phase_span, mjd):
    y_span = np.array(y_span)
    phase_span = np.array(phase_span)
    y_msh, phs_msh = np.meshgrid(y_span,phase_span)
    x_msh = x_orig -(np.sin(ang_rad)*slope*y_span + 0*mjd+phs_msh)/(np.cos(ang_rad)*slope)
    
    # X_mesh, Y_mesh = np.meshgrid(np.linspace(x_msh.flatten().min(),x_msh.flatten().max(),10), np.linspace(y_msh.flatten().min(),y_msh.flatten().max(),10))
    
    # Z_mesh =  
    ax.plot_surface(x_msh,y_msh,mjd+phs_msh,alpha=0.2)
    return

# ======== AUXILIARY PLOTTING FUNCTIONS ========
def phase_line(phase,zero_loc,angle_rad,grad,bbox):
    x1,y1,x2,y2 = None, None, None, None
    x_min, x_max = bbox[0,0], bbox[0,1]
    y_min, y_max = bbox[1,0], bbox[1,1]
    
    x_ymin = (phase-np.sin(angle_rad)*grad*y_min)/(np.cos(angle_rad)*grad) +zero_loc
    x_ymax = (phase-np.sin(angle_rad)*grad*y_max)/(np.cos(angle_rad)*grad) +zero_loc
    x1,y1,x2,y2 = x_ymin, y_min*1.0, x_ymax, y_max*1.0
    
    if x1>x_max:
        x1 = x_max*1.0
        y1 = (phase-np.cos(angle_rad)*grad*(x1-zero_loc))/(np.sin(angle_rad)*grad)
    elif x1<x_min:
        x1 = x_min*1.0
        y1 = (phase-np.cos(angle_rad)*grad*(x1-zero_loc))/(np.sin(angle_rad)*grad)
    
    if x2>x_max:
        x2 = x_max*1.0
        y2 = (phase-np.cos(angle_rad)*grad*(x1-zero_loc))/(np.sin(angle_rad)*grad)
    elif x2<x_min:
        x2 = x_min*1.0
        y2 = (phase-np.cos(angle_rad)*grad*(x1-zero_loc))/(np.sin(angle_rad)*grad)        
    
    return x1,y1,x2,y2

def plot_proj_phaseplane(ax,zero_loc,angle_rad,grad,bbox):
    x_min, x_max = bbox[0,0], bbox[0,1]
    y_min, y_max = bbox[1,0], bbox[1,1]
    x_b = np.array([bbox[0,0],bbox[0,1],bbox[0,1],bbox[0,0],bbox[0,0]])
    y_b = np.array([bbox[1,0],bbox[1,0],bbox[1,1],bbox[1,1],bbox[1,0]])
    
    if str(type(ax))=="<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
        ax.plot(x_b,y_b,np.zeros(x_b.shape),c='black',linewidth=2)
        ax.plot(np.array([x_min,x_max]),np.array([0,0]),np.array([0,0]),c='black',linewidth=1)
    
    xy_phases_b = np.cos(angle_rad)*grad*(x_b-zero_loc) +np.sin(angle_rad)*grad*y_b
    phase_min = xy_phases_b.min()
    phase_max = xy_phases_b.max()
    phase_int = 10.0
    
    N_min = np.ceil(phase_min/phase_int)
    N_max = np.floor(phase_max/phase_int)
    phases = np.arange(N_min,N_max+1)*phase_int
    for phase in phases:
        if phase==0.0:
            c='r'
        else:
            c='black'
        x1,y1,x2,y2 = phase_line(phase,zero_loc,angle_rad,grad,bbox)
        if str(type(ax))=="<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
            x_l = np.array([x1,x2])
            y_l = np.array([y1,y2])
            ax.plot(x_l,y_l,np.zeros(x_l.shape),c=c,linewidth=1)
        else:
            ax.axvline(x=x1,ymin=0,ymax=200,c=c,linewidth=1)
    return


# ======== AUXILIARY FUNCTIONS ========

def coord2ofs(SC,Org,PA):
    Sep_as = Org.separation(SC).arcsec
    PA_a = Org.position_angle(SC)-PA
    x=Sep_as*np.cos(PA_a.rad)
    y=Sep_as*np.sin(PA_a.rad)
    return x,y

### STAR MASK PRINT
def coord2ofs_list(coord_list,Org,PA,verbose=False):
    sss='['
    SClst = [SkyCoord(coord,frame='fk5',unit=(u.deg, u.deg))[0] for coord in coord_list]
    xy_list = []
    for SC in SClst:
        x,y = coord2ofs(SC,Org,PA)
        psf=1.2
        xy_list.append([[x,y],psf])
        # print('x = '+str(x))
        # print('y = '+str(y)+'\n')
        sss += '[['+'{:.3f}'.format(x)+','+'{:.3f}'.format(y)+'],1.2]'
        if not (SClst.index(SC)==(len(SClst)-1)):
            sss += ', '
    sss += ']'
    if verbose:
        print(sss)
    return xy_list

def coord_corners2xy_lims(corners_unzipped,Org,PA):
    corners_SClst = [SkyCoord(coord,frame='fk5',unit=(u.deg, u.deg))[0] for coord in coord_list]
    x_lims = []
    y_lims = []
    for corner1, corner2 in zip(corners_SClst[0::2],corners_SClst[1::2]):
        x1,y1 = coord2ofs(corner1,Org,PA)
        x2,y2 = coord2ofs(corner2,Org,PA)
        x_lims.append([min(x1,x2), max(x1,x2)])
        y_lims.append([min(y1,y2), max(y1,y2)])
    return x_lims, y_lims


def cov2corr(cov):
    corr = np.zeros(cov.shape)
    for i in np.arange(cov.shape[0]):
        for j in np.arange(cov.shape[1]):
            corr[i,j] = cov[i,j]/(np.sqrt(cov[i,i])*np.sqrt(cov[j,j]))
    return corr
# # %%

# ax1 = plt.figure().add_subplot()
# ax2 = plt.figure().add_subplot()


# # def LEprof_model( input_array, sig_dust,  peak_flux,   peak_loc, phs_grad,      add_const,      PSF):
# #     #               [arcsec],   [days],        [1],    [arcsec], [days/arcsec], [arbittrary],  [arcsec] 

# xx = np.linspace(-250,250,6001)
# sig_d = np.power(10,np.linspace(1,20,120)/10)
# peak_f = 1
# peak_l = 0
# phs_g = 1
# add_const = 0
# PSF = 0

# z_samps=[]
# peak_shfts=[]
# for sig in sig_d:
#     zz = LEprof_model(xx, sig,  peak_f,   peak_l, phs_g,      add_const,      PSF)
#     ax1.plot(xx, zz)
#     peak_shft = xx[np.nanargmax(zz)]
#     peak_shfts.append(peak_shft)
#     ax2.scatter(sig, peak_shft)
#     z_samps.append(zz.copy())

# x_s = xx.copy()
# peak_shfts=np.array(peak_shfts)

# def peak_shft_fun(sigma):
#     shft = np.interp(sigma,sig_d[1:],peak_shfts[1:],left=peak_shfts[1],right=peak_shfts[-1])
#     return shft # [days]

# sig_d_test = np.linspace(0,400,3000)
# ax2.plot(sig_d_test,peak_shft_fun(sig_d_test))
# # %%
# pre_calc_LE_model_DF = pd.DataFrame(np.stack(z_samps).T,index=xx)
# pre_calc_LE_model_DF.to_csv('/Users/roeepartoush/Documents/Research/Code/LEanalyze/pre_calc_LE_model/pre_calc_LE_model_DF_dm15_'+str(int(dm15*100))+'.csv')
# # %%

# ppp = pd.read_csv('/Users/roeepartoush/Documents/Research/Code/LEanalyze/pre_calc_LE_model/pre_calc_LE_model_DF_dm15_'+str(int(dm15*100))+'.csv',index_col=0)
# f = interpolate.interp2d(ppp.index.to_numpy(),ppp.columns.to_numpy().astype(int),ppp.values.T,kind='cubic')
# # quick version: use pre-calculated lookup table instead of gaussian filtering in real time
# def LEprof_model_quick( input_array, sig_dust,  peak_flux,   peak_loc, phs_grad,      add_const,      PSF, ignore_PSF=False, ignore_dust=False):
#     #                   [arcsec],   [days],        [1],    [arcsec], [days/arcsec], [arbittrary],  [arcsec] 
#     x = input_array
#     phases = phs_grad*(x-peak_loc)
    
#     if ignore_dust:
#         sig_dust=0.0
#     if ignore_PSF:
#         PSF=0.0
    
#     eff_phs_wid = np.hypot(sig_dust,phs_grad*PSF) # [days]
#     output_array = peak_flux*f(x, eff_phs_wid) + add_const
    
#     return output_array

# # %%
# # corrected version: peak location here refers to actual measured flux profile, not hidden ("clean") flux profile
# def LEprof_model_truPeak( input_array, sig_dust,  peak_flux,   peak_loc, phs_grad,      add_const,      PSF, ignore_PSF=False, ignore_dust=False):
#     #               [arcsec],   [days],        [1],    [arcsec], [days/arcsec], [arbittrary],  [arcsec] 
#     x = input_array
#     eff_phs_wid = np.hypot(sig_dust,phs_grad*PSF) # [days]
#     hidden_peak_loc = peak_loc -peak_shft_fun(eff_phs_wid)/phs_grad
#     phases = phs_grad*(x-hidden_peak_loc)
    
#     if ignore_dust:
#         sig_dust=0.0

#     if ignore_PSF:
#         PSF=0.0
    
#     eff_phs_wid = np.hypot(sig_dust,phs_grad*PSF) # [days]
#     ConvLC_params = [eff_phs_wid,peak_flux,add_const]
#     output_array = ConvLC(phases, *ConvLC_params)
    
#     return output_array

# %%
# == DEFAULT VALUES ==
global coord_list
coord_list=[]

x_cutoff = []

y_cutoff = []
star_list = [np.array([[16.41070624, 59.46283456]]), np.array([[16.41075895,59.46280093]])]
star_list = [np.array([[12.36635052, 58.75915926]]),
             np.array([[12.36616822, 58.75952169]]),
             np.array([[12.37062784, 58.75877992]]),
             np.array([[12.37021811, 58.7589931 ]]),
             np.array([[12.36964233, 58.7591566 ]]),
             np.array([[12.36907532, 58.75933821]]),
             np.array([[12.36856058, 58.75947901]]),
             np.array([[12.36781835, 58.75946603]]),
             np.array([[12.36690952, 58.75921769]]),
             np.array([[12.36664831, 58.75946697]])] # list of "stars" that cover a cosmic ray on Keck R 131228

phs_grd_lst = []
peak_loc_lst = []
popt_lst = []
result_lst = []
estimate_flux_bias = False
estimate_peak = False
# ====================

# ========= USER INPUT =========
# == COPY AND PASTE OVER HERE ==
# ------------------------------

# # == Time Series #1 ==
# x_cutoff = np.sort(-np.array([[-7.5,3],[-1,8],[-1,10]])).tolist()
# Orgs=SkyCoord([(12.38828633, 58.77037112)],frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([Angle(160+180,'deg') for Org in Orgs])
# Ln = Angle([20  for Org in Orgs],u.arcsec)
# Wd = Angle([2  for Org in Orgs],u.arcsec)

# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames = ['20130609/tyc4419_tyc4519_20130609_coadd.fits',
#             '20130826/tyc4419_tyc4519_20130826_coadd.fits',
#             'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits']


# == Angle Dependence - Phase grad ==
Orgs=SkyCoord([(12.37159974, 58.75964257)]*5,frame='fk5',unit=(u.deg, u.deg))
PA = Angle([155,125,95,85,75],'deg')
Ln = Angle([30]*len(Orgs),u.arcsec)
Wd = Angle([2]*len(Orgs),u.arcsec)
x_cutoff = [[-30,8.5]]*len(Orgs)
root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
filenames = ['KECK/tyc4419_1_R_LRIS_131228_coadd.fits']#,'KECK/tyc4419_1_g_LRIS_131228_coadd.fits']

# # == Time Series #8 ==
# Orgs=SkyCoord([(12.37159974, 58.75964257)]*1,frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([155,125,95,85,75],'deg')
# Ln = Angle([20]*len(Orgs),u.arcsec)
# Wd = Angle([24]*len(Orgs),u.arcsec)
# x_cutoff = [[-30,8.5]]*5#len(Orgs)
# Orgs = SkyCoord([Orgs[0].directional_offset_by(PA[0]-Angle(90,u.deg),Angle(-7,u.arcsec))])
# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames = [#'20120118/tyc4419_tyc4519_20120118_coadd.fits',
# #             '20120626/tyc4419_tyc4519_20120626_coadd.fits',
# #             'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
# #             '20130109/tyc4419_tyc4519_20130109_coadd.fits', 
# #             '20130609/tyc4419_tyc4519_20130609_coadd.fits', 
#             '20130826/tyc4419_tyc4519_20130826_coadd.fits', 
#             'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits',         
#             # '20131202/tyc4419_tyc4519_20131202_coadd.fits', 
# #             'KECK/tyc4419_1_g_LRIS_131228_coadd.fits',      
#             'KECK/tyc4419_1_R_LRIS_131228_coadd.fits']      
#             # '20140226/tyc4419_tyc4519_20140226_coadd.fits']
# #             '20140531/tyc4419_tyc4519_20140531_coadd.fits', 
# #             '20140621/tyc4419_tyc4519_20140621_coadd.fits', 
# #             '20140827/tyc4419_tyc4519_20140827_coadd.fits',
# #             '20140923/tyc4419_tyc4519_20140923_coadd.fits']
# x_cutoff, y_cutoff = ([[-10.204883014137323, -1.6173234894024995],
#   [-7.07217728730872, 1.3376197891436623],
#   # [-9.892837604294177, 4.104778198720775],
#   [-10, 7.67500758374971]],
#   [[-0.5994765492901245, 7.506798052469106],
#   [-5.866260498335004, 4.55927148261596],
#   # [-10.7319729185079, -6.527917265994323],
#   [-7.495580882446901, -4.518850357447092]])

# # == Time Series #7 (tyc4820/5) ==
# Orgs=SkyCoord([(16.455, 59.462 )]*7,frame='fk5',unit=(u.deg, u.deg)) # ,(16.42939, 59.45377)
# # Orgs=SkyCoord([(16.44543051, 59.45420532 )]*7,frame='fk5',unit=(u.deg, u.deg)) # ,(16.42939, 59.45377)
# PA = Angle([170,160,150,140,130,120,110],'deg')
# Ln = Angle([150]*len(PA),u.arcsec)
# Wd = Angle([1]*len(PA),u.arcsec)

# # Orgs = SkyCoord([Orgs[0].directional_offset_by(PA[0]-Angle(0,u.deg),Angle(-0,u.arcsec))])
# # Orgs2 = SkyCoord([Orgs[0].directional_offset_by(PA[1]-Angle(90,u.deg),Angle(5,u.arcsec))])[0]
# # Orgs = SkyCoord([Orgs[0], Orgs2])
# root = ''
# filenames = [#'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.090915.2086_5.sw.fits',
# # '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.091217.441_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20101110.102797_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20110922.11223_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120117.116607_stch_5.sw.fits',
# # '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120220.1526891_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120621.24306_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130109t.1692991_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130610.32389_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130826.33227_stch_5.sw.fits',
# # '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20140621.102501_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20140827.48444_stch_5.sw.fits',
# # '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20150213.357723_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20150814.64478_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.r.20131202.135752_stch_5.sw.fits']
# # w = DIFF_df.iloc[0].WCS_w
# # # aaa=F2D.dcmp2df(filenames[0][:-5]+'.dcmp')
# # # ggg=aaa[(aaa.Xpos<1260) & (aaa.Ypos>1120)]
# # # star_list = list(w.wcs_pix2world(np.array([ggg.Xpos,ggg.Ypos]).T,1))
# # # for i in np.arange(len(star_list)): star_list[i] = star_list[i].reshape((1,2))
# # # peak_loc_lst = np.array([-66, -61,-40.4, -17.3, -10.4, 0, 13.51, 24.8, 27, 36, 55]).tolist()

# # peak_loc_lst = [-42.240561714866516,
# #                   -19.59303553820837,
# #                   -8.107308586551339,
# #                   1.3175300146907896,
# #                   15.976285757840389,
# #                   25.685951558370057,
# #                   31.645932488668347,
# #                   38.403183883032504]

# # peak_loc_lst = [-51.65585454037466,
# #                   -27.05199966637916,
# #                   -17.557500290758103,
# #                   -0.2874178643171132,
# #                   19.198689732547116,
# #                   32.930237858062874,
# #                   42.324037553285336,
# #                   55.69818220987464]

# # filenames = ['KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits']#'KECK/tyc4419_1_R_LRIS_131228_coadd.fits']

# # root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'

# # == Time Series #3 ==
# Orgs=SkyCoord([(12.37191576, 58.76229830)],frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([Angle(150,'deg') for Org in Orgs])+Angle(180,u.deg)
# Ln = Angle([40  for Org in Orgs],u.arcsec)
# Wd = Angle([2  for Org in Orgs],u.arcsec)

# # y_cutoff = [[-10,-7],[-10,-6],[-10,-5],    [-7,-3]]
# # x_cutoff = [[-19,-7],[-10.5,2],[0,7],  [3,15]]
# # x_cutoff = [[-10.5,2],[0,7],  [3,15]]

# x_cutoff = np.sort(-np.array([[-19,-10],[-13.5,-0],[3,-3],  [0,12]])).tolist()


# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames = ['20120118/tyc4419_tyc4519_20120118_coadd.fits',
#     '20120626/tyc4419_tyc4519_20120626_coadd.fits',
#         'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
#     '20130109/tyc4419_tyc4519_20130109_coadd.fits']
# img_titles = ['KP, VR, 20120118',
#               'KP, VR, 20120626',
#               'KECK, R, 20120918',
#               'KP, VR, 20130109']
    

# # == Time Series #5 ==
# Orgs=SkyCoord([(12.36521799, 58.75615434)],frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([Angle(150,'deg') for Org in Orgs])
# Ln = Angle([50  for Org in Orgs],u.arcsec)
# Wd = Angle([50  for Org in Orgs],u.arcsec)
# # old:
# # x_cutoff = [[-16.2, -9.0], [-12.8, -6.2], [-12.6, -3.3], [-10.4, -1.7], [-11.2, 1.5], [-5.7, 7.6], [-3.8, 8.7], [1.0, 10.6]]
# # y_cutoff = [[-11.3, -3.3], [-15.6, -4.9], [-21.3, -14.0], [-17.5, -9.1], [-4.2, 3.0], [-2.6, 5.4], [-1.2, 3.8], [0.1, 5.0]]
# x_cutoff = [ [-12.0, -4.2], [-12.6, -3.3], [-7.0, -0.0], [-11.2, 1.5], [-5.7, 7.6], [-3.8, 8.7], [1.0, 10.6]]
# y_cutoff = [[-14.5, -11.0], [-21.3, -17.0], [-17.5, -15.5], [-3.2, 0.0], [-1.0, 2.5], [-0.5, 2.5], [0.1, 5.0]]
# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames=[#'20130826/tyc4419_tyc4519_20130826_coadd.fits',
#  'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits',
#  '20131202/tyc4419_tyc4519_20131202_coadd.fits',
#  'KECK/tyc4419_1_g_LRIS_131228_coadd.fits',
#  '20140226/tyc4419_tyc4519_20140226_coadd.fits',
#  '20140531/tyc4419_tyc4519_20140531_coadd.fits',
#  '20140621/tyc4419_tyc4519_20140621_coadd.fits',
#  '20140827/tyc4419_tyc4519_20140827_coadd.fits']


# == Time Series #6 ==
# Orgs=SkyCoord([(12.36968543, 58.75960183)],frame='fk5',unit=(u.deg, u.deg))
# # Orgs=SkyCoord([(12.38236381, 58.76515572)],frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([Angle(155,'deg') for Org in Orgs])
# Ln = Angle([60  for Org in Orgs],u.arcsec)
# Wd = Angle([20  for Org in Orgs],u.arcsec)

# # x_cutoff = [[-28.6, -18.1], [-21.0, -8.5], [-15.5, -3.3], [-5.8, 4.1], [2.3, 14.4], [7.0, 15.4], [13.9, 21.6], [13.4, 24.1]]
# # y_cutoff = [[-10.0, -4.0], [-9.0, -4.0], [-4.0, 2.0], [-3.5, 0.0], [-5.1, -2.0], [-5.2, -2.6], [0.5, 5.6], [-4.0, -0.5]]
# # # x_cutoff = [[-28.6, -18.1], [-21.0, -8.5], [-9.5, -3.3], [-5.8, 4.1], [2.3, 14.4], [7.0, 15.4], [13.9, 21.6], [13.4, 24.1]]
# # # y_cutoff = [[-10.0, -4.0], [-9.0, -4.0], [-9.7, -0.1], [-3.5, 0.0], [-5.1, -2.0], [-5.2, -2.6], [0.5, 5.6], [-4.0, -0.5]]

# x_cutoff, y_cutoff = ([[-29.57156499625225, -16.995702771944256],
#   [-20.117160531587867, -7.89460810072578],
#   [-10.813109764772264, -2.5520542682004983],
#   [-5.980850026874898, 5.9266111527008345],
#   [0.821755086833356, 13.556149700501114],
#   [7.23481543806444, 16.14679643642008],
#   [13.695721637080746, 23.22740703011048],
#   [13.806411825595724, 24.725569350220717]],
#  [[-9.71172799237423, -4.157876654034165],
#   [-9.411073078777145, -3.9851830425152146],
#   [-9.291905274949828, -0.07647660496754645],
#   [-3.5262707944175338, 0.28679906895718554],
#   [-7.2886665010970235, -3.276011869975013],
#   [-5.928936882422306, -3.500508338074843],
#   [-0.9998824512913977, 2.752038284395801],
#   [-5.1393687787950775, -2.18380880527846]])

# # # x_cutoff = [[-10,10],[-8,5]]
# # # y_cutoff = [[-10,10],[-0,5]]
# # #	<dust_width, PeakFlux, arcs_shift, phs_grad, add_const>
# # # popt_lst = [np.array([97.944, 46.816, -22.056, -36.584, 3.746]),
# # # np.array([200.0, 208.404, -13.17, -60.726, 0.285]),
# # # np.array([21.743, 90.199, -6.657, -15.296, -0.118]),
# # # np.array([43.557, 48.488, 0.886, -19.698, 0.526]),
# # # np.array([81.254, 75.89, 8.646, -26.628, -0.951]),
# # # np.array([49.317, 41.728, 12.106, -28.681, -0.639]),
# # # np.array([181.039, 89.14, 18.385, -99.989, 1.775]),
# # # np.array([46.294, 90.103, 20.144, -22.949, 1.203])]
# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames = ['20120118/tyc4419_tyc4519_20120118_coadd.fits',
#     '20120626/tyc4419_tyc4519_20120626_coadd.fits',
#       'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
#     '20130109/tyc4419_tyc4519_20130109_coadd.fits']#,
#     '20130609/tyc4419_tyc4519_20130609_coadd.fits',
#     '20130826/tyc4419_tyc4519_20130826_coadd.fits',
#     'KECK/tyc4419_1_R_LRIS_131228_coadd.fits',
#     '20140226/tyc4419_tyc4519_20140226_coadd.fits']
# filenames = ['20120118/tyc4419_tyc4519_20120118_coadd.fits',
#             '20120626/tyc4419_tyc4519_20120626_coadd.fits',
#             'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
#             '20130109/tyc4419_tyc4519_20130109_coadd.fits', 
#             '20130609/tyc4419_tyc4519_20130609_coadd.fits', 
#             '20130826/tyc4419_tyc4519_20130826_coadd.fits', 
#             'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits',         
#             '20131202/tyc4419_tyc4519_20131202_coadd.fits', 
#             'KECK/tyc4419_1_g_LRIS_131228_coadd.fits',      
#             'KECK/tyc4419_1_R_LRIS_131228_coadd.fits',      
#             '20140226/tyc4419_tyc4519_20140226_coadd.fits', 
#             '20140531/tyc4419_tyc4519_20140531_coadd.fits', 
#             '20140621/tyc4419_tyc4519_20140621_coadd.fits', 
#             '20140827/tyc4419_tyc4519_20140827_coadd.fits',
#             '20140923/tyc4419_tyc4519_20140923_coadd.fits']

# peak_loc_lst = np.array([-21.08682347,
#                             -12.06180804,
#                             -6.06379279,
#                               0.8328446 ,
#                               8.9746946 ,
#                             12.86815938,
#                             18.34886525,
#                             20.86194638]).tolist()


# # == Single Image #2 ==
# Orgs=SkyCoord([(12.37159974, 58.75964257)]*3,frame='fk5',unit=(u.deg, u.deg))
# PA = Angle([160,100,130],'deg')+Angle(180,u.deg)
# Ln = Angle([30]*3,u.arcsec)
# Wd = Angle([2,2,2],u.arcsec)
# x_cutoff = np.sort(-np.array([[-30,5]])).tolist()

# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
# filenames = ['KECK/tyc4419_1_g_LRIS_131228_coadd.fits']





# phs_grd_lst = [16.2]*len(filenames)
# estimate_flux_bias = True # estimate additive constant
# estimate_peak = True # peak flux amplitude 
# # before fitting, and fix the respective parameters during fitting
# ------------------------------


# %
# ========== LOAD FITS FILES ==========
print('\n\n=== Loading Fits Files... ===')
files=[]
# root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'

for file in filenames:
    files.append(os.path.join(root, file[:-5]))

DIFF_df = F2D.FitsDiff(files)

# x_cutoff, y_cutoff = coord_corners2xy_lims(coord_list,Orgs[0],PA[0])
if not x_cutoff:
    x_cutoff = [[-Ln[0].arcsec/2, Ln[0].arcsec/2]]*len(DIFF_df)
if not y_cutoff:
    y_cutoff = [[-Wd[0].arcsec/2, Wd[0].arcsec/2]]*len(DIFF_df)

# w = DIFF_df.iloc[0].WCS_w
# aaa=F2D.dcmp2df(filenames[0][:-5]+'.dcmp')
# ggg=aaa[(aaa.Xpos<1260) & (aaa.Ypos>1120)]
# star_list = star_list + list(w.wcs_pix2world(np.array([ggg.Xpos,ggg.Ypos]).T,1))
# for i in np.arange(len(star_list)): star_list[i] = star_list[i].reshape((1,2))


# ### MANUAL PSF
# DIFF_df.iloc[:,8] = pd.Series(data=np.zeros((len(DIFF_df),)))
# # DIFF_df.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
# for i in np.arange(len(DIFF_df)): DIFF_df.iloc[i,10]=LEtb.pix2ang(DIFF_df.iloc[i]['WCS_w'],DIFF_df.iloc[i,8])

### MANUAL PSF
# DIFF_df.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
# for i in np.arange(len(DIFF_df)): DIFF_df.iloc[i,10]=LEtb.pix2ang(DIFF_df.iloc[i]['WCS_w'],DIFF_df.iloc[i,8])

# %


# %
# # ========== PLOT FITS IMAGES ==========
# plt.close('all')
# figures=[plt.figure() for i in DIFF_df.index]

# managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
# for mng in managers: mng.window.showMaximized()
# LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures)

# %
clmns = ['Orig', 'PA', 'Length','WIDTH']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

# ========== EXTRACT PROFILE FROM IMAGE ==========
print('\n\n=== Extracting Flux Profiles... ===')
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec),uniform_wcs=False)


# ========== PLOT IMAGES & PROFILES ==========
print('\n\n=== Plotting Images... ===')
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]
# figures = [figures[0]]*8
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
# for mng in managers: mng.window.showMaximized()
# LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')#, crest_lines=CLs)#,peaks_locs=b)

# img_axs = [figures[1].add_subplot(141+ii,projection=DIFF_df.iloc[ii].WCS_w) for ii in np.arange(len(figures))]
# figures = [figures[0]]*len(figures)
# axs = LEplots.imshows(DIFF_df,plot_Fprofile=True, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, prof_crop=(x_cutoff,y_cutoff), popts=popt_lst, g_cl_arg=coord_list, FullScreen=False, figs=figures, img_axs=img_axs)
# for ii in np.arange(len(axs)): axs[ii].title.set_text(img_titles[ii])
axs = LEplots.imshows(DIFF_df,plot_Fprofile=True, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, prof_crop=(x_cutoff,y_cutoff), popts=popt_lst, g_cl_arg=coord_list, FullScreen=False, figs=figures)

# axs = LEplots.imshows(DIFF_df, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, prof_crop=None, popts=popt_lst, g_cl_arg=coord_list, FullScreen=False, figs=figures)
# %

w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(axs,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*0.7,slitFPdf.iloc[0]['Length']*1.0)
# LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'].directional_offset_by(slitFPdf.iloc[0]['PA'],Angle(25,u.arcsec)),slitFPdf.iloc[0]['Length']*0.35,slitFPdf.iloc[0]['Length']*0.3)


# %
# ========== PLOT PROFILES 3D ==========

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xy'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xy'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


inds=[i for i in np.arange(len(DIFF_df))]#[0,1,2,3]#,4,5,6,7]
# slit_inds = [0]
slit_inds = np.arange(len(Orgs)).tolist()


plot_flux_samps = False
input_arr_s_allslits, z_s_allslits = [], []
input_arr_s_uc_allslits, z_s_uc_allslits = [], []
for slit_ind in slit_inds:
    input_arr_s, z_s = [], []
    input_arr_s_uc, z_s_uc = [], []
    pa = PA[slit_ind]
    for ind in inds:
        iii=inds.index(ind)
        if plot_flux_samps:
            fig=plt.figure()
            if iii!=0 or slit_ind!=0:
                ax=fig.add_subplot(sharex=ax,sharey=ax)
            else:
                ax=fig.add_subplot()
        # if iii!=0:
        #     ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
        # else:
        #     ax=fig.add_subplot(projection='3d')
        x=FP_df_lst[slit_ind].iloc[ind]['ProjAng'].arcsec
        y=FP_df_lst[slit_ind].iloc[ind]['PerpAng'].arcsec
        z=FP_df_lst[slit_ind].iloc[ind]['FluxProfile_ADU']
        mjd_curr = DIFF_df.iloc[ind]['Idate']
        PSF_curr = DIFF_df.iloc[ind]['FWHM_ang'].arcsec
        
        # uncropped data:
        shape = (len(x),1)
        input_arr_s_uc.append(deepcopy(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1)))
        z_s_uc.append(deepcopy(z.reshape(shape)))
        
        x_inds = np.logical_and(x>(x_cutoff[iii][0]), x<(x_cutoff[iii][1]))
        x = x[x_inds]
        y = y[x_inds]
        z = z[x_inds]
        
        y_inds = np.logical_and(y>y_cutoff[iii][0], y<y_cutoff[iii][1])
        x = x[y_inds]
        y = y[y_inds]
        z = z[y_inds]
        
        if star_list:
            star_masks = coord2ofs_list(star_list,Orgs[slit_ind],PA[slit_ind],verbose=False)
            for star_mask in star_masks:
                st_mk_inds = np.hypot((x-star_mask[0][0])*1.0,y-star_mask[0][1])>(0.5)#3.0*PSF_curr)
                # st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(1.2*PSF_curr)
                x = x[st_mk_inds]
                y = y[st_mk_inds]
                z = z[st_mk_inds]
        
        # == SLIT WIDTH CORRECTION ==
        # x = x - np.tan((pa-PA[0]).rad)*y # EXPERIMENTAL - slit width effect cancellation
        # x_SCALED = x*np.cos(PA[slit_ind].rad-PA[0].rad)
        # x_inds = np.logical_and(x_SCALED>(-0.0), x_SCALED<(4.0))
        # x = x[x_inds]
        # y = y[x_inds]
        # z = z[x_inds]
        # x = x*np.cos(PA[slit_ind].rad-PA[0].rad)
        
        shape = (len(x),1)
        input_arr_s.append(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1))
        z_s.append(z.reshape(shape))
    
        # ax.scatter(x,y,cmap='jet',s=7,c=z)#,s=2,c=z)#,vmax=200)
        # if popt_lst:
        #     ax.axvline(x=popt_lst[ind][2])
        # # ax.view_init(elev=90,azim=0)
        # if plot_flux_samps:
        #     fig.set_figwidth(10)
        #     fig.set_figheight(10)
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.xlim([np.min(np.stack(x_cutoff).flatten()),np.max(np.stack(x_cutoff).flatten())])
        #     plt.ylim([np.min(np.stack(y_cutoff).flatten()),np.max(np.stack(y_cutoff).flatten())])
        #     axisEqual3D(ax)
    
    input_arr_s_allslits.append(input_arr_s)
    z_s_allslits.append(z_s)
    input_arr_s_uc_allslits.append(input_arr_s_uc)
    z_s_uc_allslits.append(z_s_uc)

# %

popt_lst_BU = deepcopy(popt_lst)
result_lst_BU = deepcopy(result_lst)
phs_grd_lst_BU = deepcopy(phs_grd_lst)
peak_loc_lst_BU = deepcopy(peak_loc_lst)
# ========== Independent Light Curve ==========
# ============== FITTING & PLOTS ==============
bnd = ([0.0,       0.0,        -16,        1,         -500,      -np.inf],
        [20.0,     1e3,        16,         40,         5e3,       np.inf])
#      sig_dust,  peak_flux,   peak_loc,   phs_grad,    add_const,      PSF
#      [days],        [1],    [arcsec],  [days/arcsec], [arbittrary],  [arcsec] 
bnds=[bnd]*len(DIFF_df)*len(slit_inds)
# result_lst_BU = []
# result_lst = []

F_SCALE=100
# plt.figure()
# ax2 = plt.subplot() 
fig = plt.figure()
ax3 = fig.add_subplot(projection='3d')
CLs = np.zeros((len(input_arr_s),3,3))
popt_lst=[]
errs_lst=[]
phs_grads=[]

pcov_lst=[]
pcorr_lst=[]

LE_model = Model(LEprof_model)
# LE_model = Model(LEprof_model_quick)
# LE_model = Model(LEprof_model_truPeak)

for slit_ind in slit_inds:
    input_arr_s, z_s = input_arr_s_allslits[slit_ind], z_s_allslits[slit_ind]
    input_arr_s_uc, z_s_uc = input_arr_s_uc_allslits[slit_ind], z_s_uc_allslits[slit_ind]
    FP = FP_df_lst[slit_ind]
    # for i in np.arange(len(input_arr_s)):
    for i in inds:
        z = deepcopy(z_s[i].flatten())
        inp_arr = deepcopy(input_arr_s[i])
        z_uc = deepcopy(z_s_uc[i].flatten())
        inp_arr_uc = deepcopy(input_arr_s_uc[i])
        x_uc, y_uc = inp_arr_uc[:,0], inp_arr_uc[:,1]
        
        x_FP_bin = FP.iloc[i].FP_Ac_bin_x
        z_FP_bin = FP.iloc[i].FP_Ac_bin_y
            
        x, y, PSF = inp_arr[:,0], inp_arr[:,1], inp_arr[0,3]
        x_argsort=x.argsort()
        x_sorted = x[x_argsort]
        inp_arr_sorted = inp_arr[x_argsort]
        z_sorted = z[x_argsort]
        x_span = x.max()-x.min()
        
        x_bnd = (inp_arr[:,0].min(),inp_arr[:,0].max())
        mjd = inp_arr[:,2][0]
        bnd_tmp = deepcopy(bnds[i])
        bnd_tmp[0][2] = x_bnd[0]
        bnd_tmp[1][2] = x_bnd[1]
        
        paramss = LE_model.make_params()
        for ii in np.arange(len(paramss)):
            p = paramss[list(paramss)[ii]]
            p.min=bnd_tmp[0][ii]
            p.max=bnd_tmp[1][ii]
            p.value = (p.min+p.max)/2
        
        
        if estimate_flux_bias:
            args = (x.max()-x)<(x_span/10)
            bias_est = np.mean(z[args])
            paramss['add_const'].value = bias_est # estimate flux bias
            # paramss['add_const'].vary = False # fix flux bias
            
            if estimate_peak:
                light_curve_IPR = 35
                # intg_est = np.trapz(z_sorted-bias_est,x_sorted)
                # paramss['add_const'].min = z.min()
                # paramss['add_const'].max = intg_est/x_span
                # paramss['add_const'].value = (paramss['add_const'].max+paramss['add_const'].min)/2
                # intg_est = np.trapz(z_sorted-bias_est,x_sorted)
                # paramss['peak_flux'].expr = str('{:.3f}'.format(intg_est)+'*abs(phs_grad)/'+'{:.3f}'.format(light_curve_IPR)) # estimate peak flux
                
                paramss._asteval.symtable['x_sorted'] = x_sorted
                paramss._asteval.symtable['z_sorted'] = z_sorted
                paramss['peak_flux'].value = np.trapz(z_sorted-bias_est,x_sorted)*np.abs(phs_grad)/(light_curve_IPR)
                # paramss['peak_flux'].expr = str('trapz(z_sorted-add_const,x_sorted)'+'*abs(phs_grad)/'+'{:.3f}'.format(light_curve_IPR)) # estimate peak flux
        
        if phs_grd_lst_BU:
            if phs_grd_lst_BU[i]:
                phs_grd = deepcopy(phs_grd_lst_BU[i])
                paramss['phs_grad'].value = phs_grd # fix phase gradient
                paramss['phs_grad'].vary = False # fix phase gradient
        
        if peak_loc_lst_BU:
            peak_loc = deepcopy(peak_loc_lst_BU[i])
            paramss['peak_loc'].value = peak_loc # fix peak location
            paramss['peak_loc'].vary = False # fix peak location
        
        # paramss['sig_dust'].value = 1.19
        # paramss['sig_dust'].vary = False
        
        paramss['PSF'].value = PSF
        paramss['PSF'].vary = False
        # == FITTING == 
        args = (x.max()-x)<(x_span/10)
        data_std_est = np.std(z[args])
        z_stds = np.ones(z.shape)*1.0*data_std_est
        if result_lst_BU:
            result = deepcopy(result_lst[i])
        else:
            result=LE_model.fit(z,params=paramss,input_array=x, weights=1.0/z_stds)
            result_lst.append(deepcopy(result))
        popt = []
        for ii in np.arange(len(paramss)):
            p = result.params[list(paramss)[ii]]
            popt.append(p.value)
        pcov = deepcopy(result.covar)
        # ============
        
        # errs = np.sqrt(np.diag(pcov))
        popt_lst.append(popt.copy())
        pcov_lst.append(deepcopy(pcov))
        # pcorr_lst.append(cov2corr(pcov))
        # errs_lst.append(errs.copy())
        # popt = [20.0,     10e1,            -0.0,   -34.0*np.pi/180,   -14 ]
        # popt = [  3.09876149, 116.87448495,   8.91707928 +(mjd-56654.2),  -0.47029305,  -5.38910457, -14.00613042]
        print('dust_width, PeakFlux, arcs_shift, phs_grad, add_const = '+str(popt))
        sig_dust,peak_flux,peak_loc,p05,phs_grad, add_const = popt[0], popt[1], popt[2], 0, popt[3], popt[4]
        phs_grads.append(np.array(phs_grad).copy())
        # sig_dust,peak_flux,peak_loc,p05,phs_grad, add_const = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        # PSF = inp_arr[:,3][0] # arcsec
        grad = phs_grad # day/arcsec
        # total_wid = popt[0] # days
        # dust_wid = np.sqrt(total_wid**2 - (grad*PSF)**2)
        # print('Total width: '+'{:.3f}'.format(total_wid)+' [days] <--> '+'{:.3f}'.format(total_wid/grad)+' [arcsec]')
        # print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
        # print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/grad)+' [arcsec]\n')
        dust_wid = popt[0] # days # NEW: [arcsec] (July 2021)
        # print('Total width: '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF))+' [days] <--> '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF)/np.abs(grad))+' [arcsec]')
        # print('PSF width: '+'{:.3f}'.format(np.abs(grad)*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
        # print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/np.abs(grad))+' [arcsec]\n')
        total_width_arcsec = np.hypot(dust_wid,PSF)
        print('Total width: '+'{:.3f}'.format(total_width_arcsec*np.abs(grad))+' [days] <--> '+'{:.3f}'.format(total_width_arcsec)+' [arcsec]')
        print('PSF width: '+'{:.3f}'.format(np.abs(grad)*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
        print('Dust width: '+'{:.3f}'.format(np.abs(grad)*dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid)+' [arcsec]\n')
        # print(np.sqrt(np.diag(pcov)))
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig=figures[i]
        plt.figure(fig.number);
        if i!=0:
            # ax=fig.add_subplot(222)#,sharex=ax)
            ax = plt.subplot2grid((3,3),(0,1),rowspan=1,colspan=2)
            # ax=fig.add_subplot(224,projection='3d',sharex=ax,sharey=ax)
        else:
            # ax=fig.add_subplot(222)
            ax = plt.subplot2grid((3,3),(0,1),rowspan=1,colspan=2)
            # ax=fig.add_subplot(224,projection='3d')
        
        const_offs = 50*slit_ind
        # # x, y = inp_arr[:,0], inp_arr[:,1]
        # # xy_phases = peak_loc +np.cos(p05)*phs_grad*x +np.sin(p05)*phs_grad*y +mjd
        # # ax.scatter(x,y,xy_phases,c='b',s=1)
        bbox=np.array([[x.min(),x.max()],[y.min(),y.max()]])
        # plot_proj_phaseplane(ax,peak_loc,p05,grad,bbox)
        
        x_scaling = 1.0#np.cos(PA[slit_ind].rad-PA[0].rad)
        # # ax.plot(x_sorted,LEprof_model(x_sorted,0.0,peak_flux,peak_loc,phs_grad, add_const, 0.0),c='k',linewidth=0.5,label='clean light curve')
        # # ax.plot(x_sorted,LEprof_model(x_sorted,0.0,peak_flux,peak_loc,phs_grad, add_const, PSF),c='b',linewidth=0.5,label='PSF only')
        ax.plot(x_sorted,LEprof_model(x_sorted,*popt,ignore_dust=True,ignore_PSF=True)-result.params['add_const'],c='k',linewidth=1,label='light curve')
        # ax.plot(x_sorted,LEprof_model(x_sorted,*popt,ignore_dust=True),c='b',linewidth=0.5,label='PSF only')
        ax.plot(x_sorted*x_scaling,LEprof_model(x_sorted,*popt)+const_offs-result.params['add_const'],c='r',linewidth=2,label='fit (PSF+dust)')
        # ax.axhline(y=add_const*0+data_std_est,c='c')
        # ax.axhline(y=add_const*0-data_std_est,c='c')
        # ax.axvline(x=peak_loc,c='r')
        
        # ax.plot(x_sorted,z_sorted,linewidth=0.2,label='data')
        # ax.scatter(x,LEprof_model(inp_arr_zeroPSF,0,peak_flux,peak_loc,phs_grad, add_const),c='k',s=0.2,label='zero PSF & zero dust width')
        # ax.scatter(x,LEprof_model(inp_arr,0,peak_flux,peak_loc,phs_grad, add_const),c='b',s=0.2,label='zero dust width')
        # ax.scatter(x,LEprof_model(inp_arr,*popt),c='r',s=1,label='fit')
        ax.scatter(x*x_scaling,z+const_offs,s=0.5, label='data')
        # ax.plot(x_FP_bin*x_scaling,z_FP_bin+const_offs, label='binned')
        # ax.errorbar(x,z,yerr=z_stds,fmt='.',label='data')
        # ax.scatter(x_uc,z_uc,c='k',s=0.3, label='')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        # place a text box in upper left in axes coords
        # ax.text(0.05, 0.95, result.fit_report(), transform=ax.transAxes, fontsize=7,verticalalignment='top', bbox=props)
        
        # im=ax.scatter(x,y,cmap='jet',c=z-LEprof_model(inp_arr,*popt),s=1)
        # fig.colorbar(im,ax=ax,orientation="horizontal")
        ax.set_xlabel('x [arcsec]')
        ax.set_ylabel('Flux [ADU]')
        ax.set_title('Phase Gradient: '+'{:.1f}'.format(grad)+' [days/arcsec]')
        # ax.set_title('MJD: '+'{:.0f}'.format(mjd))
        
        if str(type(ax))=="<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
            axisEqual3D(ax)
        
        # ax2.scatter(peak_loc/phs_grad,mjd)
        # zero intersect: cos(p05)*phs_grad*X + sin(p05)*phs_grad*Y + peak_loc +mjd = 0 --> X = -(sin(p05)*phs_grad*Y + peak_loc +mjd)/cos(p05)*phs_grad
        zero_y = np.array([-0.5,0.5,0])*Wd[0].arcsec*1.2
        zero_x = peak_loc -(np.sin(p05)*phs_grad*zero_y + 0*mjd)/(np.cos(p05)*phs_grad)
        # zero_x = -(np.sin(p05)*phs_grad*zero_y + peak_loc+0*mjd)/(np.cos(p05)*phs_grad)
        ax3.plot(zero_x,zero_y,mjd)
        
        plot_local_phase_plane(ax3, peak_loc, p05, phs_grad, [-0.5,0.5], [-10,0,10], mjd)
        
        CLs[i,0,0], CLs[i,1,0], CLs[i,2,0] = zero_x
        CLs[i,0,1], CLs[i,1,1], CLs[i,2,1] = zero_y
        CLs[i,0,2], CLs[i,1,2], CLs[i,2,2] = mjd*np.ones(len(zero_y))
        
        # ax.legend()
        
        ax_img = axs[i]
        prof_orig = slitFPdf['Orig'][0]
        prof_PA_x = slitFPdf['PA'][0]
        prof_PA_y = slitFPdf['PA'][0] + Angle(90,u.deg)
        prof_width = slitFPdf['WIDTH'][0]
        
        peak_arcsec_shift = Angle(popt[2],u.arcsec)
        corner1 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y,-prof_width/2)
        corner2 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y, prof_width/2)
        xy = np.array([[corner1.ra.degree, corner1.dec.degree], [corner2.ra.degree, corner2.dec.degree]])
        # ax_img.plot(xy[:,0], xy[:,1], transform = ax_img.get_transform('world'), linewidth=0.5, color='r')
        
        # === July 30 2021 ===
        # ax_phs = ax.twiny()
        # xlims_arcsec = np.array(ax.get_xlim())
        # ax_phs.set_xlim((xlims_arcsec-peak_loc)*grad)
        # if np.sign(grad)<0: ax_phs.invert_xaxis()
        
        # == SLITLET ==
        FWHMoSIG = 2*np.sqrt(2*np.log(2))
        sltltDF = pd.DataFrame(columns=['cntr_arcsec','length_arcsec'],data=np.array([[0*peak_loc,2]]))
        x_arcsec = np.linspace(x.min()-total_width_arcsec*4,x.max()+total_width_arcsec*4,500) # [arcsec]
        x_days = (x_arcsec-peak_loc)*grad
        arcsec_res = np.mean(np.diff(x_arcsec))
        rect_win = np.abs(np.abs(x_arcsec-sltltDF.cntr_arcsec[0])<sltltDF.length_arcsec[0]/2)*1.0
        conv_win = ndimage.gaussian_filter(rect_win,sigma=(total_width_arcsec/FWHMoSIG)/arcsec_res)
        conv_win = conv_win/conv_win.max()
        
        delta_win = np.abs(np.abs(x_arcsec-sltltDF.cntr_arcsec[0])<np.diff(x_arcsec).min())*1.0
        conv_ker = ndimage.gaussian_filter(delta_win,sigma=(total_width_arcsec/FWHMoSIG)/arcsec_res)
        conv_ker = conv_ker/conv_ker.max()
        
        clean_LC = SN_f((x_arcsec-peak_loc)*grad)/SN_f(0)
        eff_LC = clean_LC*conv_win
        
        ax_wf_days = plt.subplot2grid((3,3),(1,1),rowspan=1,colspan=2)
        ax_wf_days.plot(x_days,rect_win,':',c='k',linewidth=1,label='slitlet aperture')
        ax_wf_days.plot(x_days,conv_ker,c='r',linewidth=1,label='conv. kernel')
        ax_wf_days.plot(x_days,conv_win,c='b',linewidth=2,label='window function')
        plt.ylabel('Weight [1]')
        plt.xlabel('Phase [days]')
        plt.legend()
        
        ax_wf_days = plt.subplot2grid((3,3),(2,1),rowspan=1,colspan=2,sharex=ax_wf_days)
        ax_wf_days.plot(x_days,clean_LC,'--',c='k',linewidth=1,label='light curve')
        ax_wf_days.plot(x_days,eff_LC,c='r',linewidth=2,label='eff. light curve')
        
        # ax_wf_days.plot(x_days,signal.gaussian(x_days.shape[0],(total_width_arcsec/FWHMoSIG)/arcsec_res),c='m',linewidth=1,label='gaussian')
        plt.ylabel('Weight [1]')
        plt.xlabel('Phase [days]')
        plt.legend()
        
        # ax.plot(x_arcsec,rect_win,':',c='k',linewidth=1,label='slitlet aperture')
        ax.fill_between(x_arcsec,0,rect_win*result.params['peak_flux'].value,color='r',alpha=0.3,label='slitlet aperture')
        plt.legend()
        
        sltlt_wid = slitFPdf.iloc[slit_ind].WIDTH.arcsec
        sltlt_len = sltltDF.length_arcsec[0]#Angle(sltltDF.length_arcsec[0],u.arcsec)
        sltlt_cntr = sltltDF.cntr_arcsec[0]#Angle(sltltDF.cntr_arcsec[0],u.arcsec)
        
        prof_orig = slitFPdf['Orig'][0]
        prof_PA_x = slitFPdf['PA'][0]
        prof_PA_y = slitFPdf['PA'][0] + Angle(90,u.deg)
        x_lims = Angle(sltlt_cntr+np.array([-0.5,0.5])*sltlt_len,u.arcsec)
        y_lims = Angle(0.0+np.array([-0.5,0.5])*sltlt_wid,u.arcsec)
        corner1 = prof_orig.directional_offset_by(prof_PA_x,x_lims[0]).directional_offset_by(prof_PA_y,y_lims[0])
        corner2 = prof_orig.directional_offset_by(prof_PA_x,x_lims[1]).directional_offset_by(prof_PA_y,y_lims[0])
        corner3 = prof_orig.directional_offset_by(prof_PA_x,x_lims[1]).directional_offset_by(prof_PA_y,y_lims[1])
        corner4 = prof_orig.directional_offset_by(prof_PA_x,x_lims[0]).directional_offset_by(prof_PA_y,y_lims[1])
        xy = np.array([[corner1.ra.degree, corner1.dec.degree], [corner2.ra.degree, corner2.dec.degree], 
                       [corner3.ra.degree, corner3.dec.degree], [corner4.ra.degree, corner4.dec.degree], [corner1.ra.degree, corner1.dec.degree]])
        ax_img.plot(xy[:,0], xy[:,1], transform = ax_img.get_transform('world'), linewidth=1, color='r')
        # ax_wf_days = ax_wf_arcsec.twiny()#sharex=ax_phs)
        # xlims_arcsec = np.array(ax_wf_arcsec.get_xlim())
        # ax_wf_days.set_xlim((xlims_arcsec-peak_loc)*grad)
        # if np.sign(grad)<0: ax_wf_days.invert_xaxis()
        
        
        

axisEqual3D(ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')

popts = np.stack(popt_lst)
# errss = np.stack(errs_lst)

mjds = DIFF_df.iloc[:,1].to_numpy().astype(float).reshape(len(DIFF_df),1)
peaks = popts[:,2].reshape(len(slitFPdf),len(DIFF_df)).T
peaks = peaks[:,0]
# peak_errs = errss[:,2].reshape(len(DIFF_df),1)
# b = np.concatenate([mjds,peaks,peak_errs],axis=1)

# %
slope, intercept, r_value, p_value, std_err = stats.linregress(mjds.flatten(),peaks.flatten())
plt.figure()
plt.title('Apparent motion and Phase gradients')
plt.scatter(peaks.flatten(),mjds.flatten(),label='Peaks')

plt.plot(intercept+slope*mjds.flatten(),mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(1/slope))+' [days/arcsec]')

# mjds_tilt = np.array([mjds.min(),mjds.mean(),mjds.max()])
# tilt1 = slope+std_err
# tilt2 = slope-std_err
# plt.plot(intercept+slope*mjds.mean()+tilt1*(mjds_tilt-mjds.mean()),mjds_tilt,label='Regression: '+'{:.1f}'.format(np.abs(1/tilt1))+' [days/arcsec]',c='r')
# plt.plot(intercept+slope*mjds.mean()+tilt2*(mjds_tilt-mjds.mean()),mjds_tilt,label='Regression: '+'{:.1f}'.format(np.abs(1/tilt2))+' [days/arcsec]',c='b')
for pg,mjd,pk in zip(phs_grads,mjds.flatten(),peaks.flatten()):
    pnts_x = pk + np.array([-1,1])
    pnts_y = mjd - np.array([-1,1])*pg
    plt.plot(pnts_x,pnts_y,c='r',label='local_phs_grad: '+'{:.1f}'.format(np.abs(pg))+' [days/arcsec]')
plt.legend()
plt.xlabel('Peak brightness location [arcsec]')
plt.ylabel('Obs. date (MJD)')


# %%
# == peak location for MANUALLY clicked peak locations ==
plt.figure(100)
slopes = []
for i in np.arange(len(slitFPdf)):
    mjds = DIFF_df.iloc[:,1].to_numpy().astype(float).reshape(len(DIFF_df),1)
    slit_coords_lst = [DIFF_df.iloc[dd].coords_list[i] for dd in np.arange(len(DIFF_df))]
    peaks = np.array([c[0][0] for c in coord2ofs_list(slit_coords_lst,Orgs[0],PA[i],verbose=False)])
    # peaks = peaks[:,0]
    # peak_errs = errss[:,2].reshape(len(DIFF_df),1)
    # b = np.concatenate([mjds,peaks,peak_errs],axis=1)
    
    # %
    slope, intercept, r_value, p_value, std_err = stats.linregress(mjds.flatten(),peaks.flatten())
    slopes.append(1/slope)
    plt.title('Peak location vs. MJD')
    plt.scatter(mjds.flatten(),peaks.flatten(),label='Peaks')
    plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(1/slope))+' [arcsec/day]')
plt.ylabel('Peak location [arcsec]')
plt.xlabel('MJD')

plt.figure(101)
plt.scatter(PA.deg,slopes)
center_ind=4
theta_rad = np.linspace(PA.rad.min(),PA.rad.max(),51)
theta_deg = np.linspace(PA.deg.min(),PA.deg.max(),51)
plt.plot(theta_deg,slopes [center_ind]*np.cos(theta_rad-PA[center_ind].rad))
plt.ylabel('Apparent Motion ^-1 [days/arcsec]')
plt.xlabel('Position Angle [deg]')
# %%

peaks_stderr = []
for res in result_lst:
    peaks_stderr.append(0.5)#$res.params['peak_loc'].stderr.copy())
peaks_stderr = np.array(peaks_stderr)

peaks_stderr = peaks_stderr.reshape(len(slitFPdf),len(DIFF_df)).T
peaks_stderr = peaks_stderr[:,0]

plt.figure()
plt.title('Apparent motion')
plt.errorbar(mjds.flatten()-56e3,peaks.flatten(),yerr=peaks_stderr,fmt='o',linestyle='None',label='Peaks')
plt.plot(mjds.flatten()-56e3,intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Peak location [arcsec]')
plt.xlabel('Obs. date (MJD - 56000)')

import seaborn as sns
data = {'x':mjds.flatten(), 'y':peaks.flatten()}
frame = pd.DataFrame(data, columns=['x', 'y'])
sns.lmplot('x', 'y', frame, ci=95,markers='.')
plt.title('Apparent motion')
plt.errorbar(mjds.flatten(),peaks.flatten(),yerr=peaks_stderr,markersize=1,fmt='o',linestyle='None',label='Peaks')
plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Peak location [arcsec]')
plt.xlabel('Obs. date (MJD)')


# %%
peaks_stderr = []
for res in result_lst:
    peaks_stderr.append(0.5)#res.params['peak_loc'].stderr.copy())
peaks_stderr = np.array(peaks_stderr)

peaks_stderr = peaks_stderr.reshape(len(slitFPdf),len(DIFF_df)).T
peaks_stderr = peaks_stderr[:,0]

MJDoffset = 56100+525

LinModel = lmfit.models.LinearModel()
LinParams = LinModel.make_params()
LinResult = LinModel.fit(peaks.flatten(),params=LinParams,x=mjds.flatten()-MJDoffset, weights=1.0/peaks_stderr)
plt.figure()
plt.title('Apparent motion (linear model by lmfit)')
plt.errorbar(mjds.flatten()-MJDoffset,peaks.flatten(),yerr=peaks_stderr,fmt='o',linestyle='None',label='Peaks')
slope = LinResult.params['slope'].value
intercept = LinResult.params['intercept'].value
plt.plot(mjds.flatten()-MJDoffset,intercept+slope*(mjds.flatten()-MJDoffset),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Peak location [arcsec]')
plt.xlabel('Obs. date (MJD - %i)'%MJDoffset)



# RESAMPLING by standard error
Nsamp = 100
mjds_resamp = np.zeros((len(mjds),Nsamp))
peaks_resamp = np.zeros((len(peaks),Nsamp))


LinModel = lmfit.models.LinearModel()
LinParams = LinModel.make_params()

for ii in np.arange(len(peaks)):
    sig_peak = 2*peaks_stderr[ii]/2*np.sqrt(2*np.log(2))
    peaks_resamp[ii,:] = peaks[ii] + sig_peak*(np.random.rand(Nsamp)-0.5)
    # peaks_resamp[ii,:] = np.random.normal(peaks[ii],sig_peak,(Nsamp,))
    mjds_resamp[ii,:] = mjds.flatten()[ii]

slopes = []
intercepts = []
for ii in np.arange(Nsamp):
    LinResult = LinModel.fit(peaks_resamp[:,ii].flatten(),params=LinParams,x=mjds_resamp[:,ii].flatten()-MJDoffset)#, weights=1.0/peaks_stderr)
    slopes.append(LinResult.params['slope'].value)
    intercepts.append(LinResult.params['intercept'].value)

slopes = np.array(slopes)
mjds_resamp = mjds_resamp.flatten()
peaks_resamp = peaks_resamp.flatten()

LinResult = LinModel.fit(peaks_resamp.flatten(),params=LinParams,x=mjds_resamp.flatten()-MJDoffset)#, weights=1.0/peaks_stderr)
plt.figure()
plt.title('Apparent motion (linear model by lmfit)')
# plt.scatter(mjds_resamp.flatten()-MJDoffset,peaks_resamp.flatten(),s=1,linestyle='None',label='Peaks')

slope = LinResult.params['slope'].value
intercept = LinResult.params['intercept'].value
plt.plot(mjds_resamp.flatten()-MJDoffset,intercept+slope*(mjds_resamp.flatten()-MJDoffset),label='Regression: '+'{:.1f}'.format(1/np.abs(365/365*slope))+' [days/arcsec]')

# plt.plot(mjds_resamp.flatten()-MJDoffset,intercept+slopes.mean()*(mjds_resamp.flatten()-MJDoffset),label='Resampling Mean slope: '+'{:.1f}'.format(1/np.abs(365/365*slopes.mean()))+' [days/arcsec]')
plt.plot(mjds_resamp.flatten()-MJDoffset,intercept+(slopes.mean()-slopes.std())*(mjds_resamp.flatten()-MJDoffset),label='Resampling Mean-std slope: '+'{:.1f}'.format(1/np.abs(365/365*(slopes.mean()-slopes.std())))+' [days/arcsec]')
plt.plot(mjds_resamp.flatten()-MJDoffset,intercept+(slopes.mean()+slopes.std())*(mjds_resamp.flatten()-MJDoffset),label='Resampling Mean+std slope: '+'{:.1f}'.format(1/np.abs(365/365*(slopes.mean()+slopes.std())))+' [days/arcsec]')

plt.errorbar(mjds.flatten()-MJDoffset,peaks.flatten(),yerr=peaks_stderr,fmt='',linestyle='None',label='Peaks')

plt.legend()
plt.ylabel('Peak location [arcsec]')
plt.xlabel('Obs. date (MJD - %i)'%MJDoffset)

# import seaborn as sns
# data = {'x':mjds_resamp.flatten(), 'y':peaks_resamp.flatten()}
# frame = pd.DataFrame(data, columns=['x', 'y'])
# sns.lmplot('x', 'y', frame, ci=95,markers='.')
# plt.title('Apparent motion')
# plt.scatter(mjds_resamp.flatten()-MJDoffset*0.0,peaks_resamp.flatten(),s=1,linestyle='None',label='Peaks')
# # plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
# plt.legend()
# plt.ylabel('Peak location [arcsec]')
# plt.xlabel('Obs. date (MJD)')
# %%
phs_grads_stderr = []
phs_grads = []
for res in result_lst:
    phs_grads_stderr.append(res.params['phs_grad'].stderr.copy())
    phs_grads.append(res.params['phs_grad'].value)
phs_grads_stderr = np.array(phs_grads_stderr)
phs_grads = np.array(phs_grads)

phs_grads_stderr = phs_grads_stderr.reshape(len(slitFPdf),len(DIFF_df)).T
phs_grads_stderr = phs_grads_stderr[:,0]

phs_grads = phs_grads.reshape(len(slitFPdf),len(DIFF_df)).T
phs_grads = phs_grads[:,0]

plt.figure()
plt.title('Phase gradient vs. Apperent motion')
plt.errorbar(mjds.flatten(),phs_grads,yerr=phs_grads_stderr,linestyle='',marker='.',label='individual phase gradients')
plt.axhline(y=-1/slope,c='r',label='Regression')
plt.axhline(y=-1/(slope+std_err),linewidth=0.5,c='r')
plt.axhline(y=-1/(slope-std_err),linewidth=0.5,c='r')
# plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Phase gradient [days/arcsec]')
plt.xlabel('Obs. date (MJD)')

# %%
phs_grads_stderr = []
phs_grads = []
for res in result_lst:
    phs_grads_stderr.append(res.params['phs_grad'].stderr.copy())
    phs_grads.append(res.params['phs_grad'].value)
phs_grads_stderr = np.array(phs_grads_stderr)
phs_grads = np.array(phs_grads)

phs_grads_stderr = phs_grads_stderr.reshape(len(slitFPdf),len(DIFF_df))
phs_grads_stderr = phs_grads_stderr[:,0]

phs_grads = phs_grads.reshape(len(slitFPdf),len(DIFF_df))
phs_grads = phs_grads[:,0]

plt.figure()
plt.title('Phase gradient vs. Position Angle')
plt.errorbar(-(PA.deg-PA[0].deg),phs_grads,yerr=phs_grads_stderr*1.0,linestyle='',marker='.',label='individual phase gradients')

pa_samps = np.linspace(PA[0].rad,PA[-1].rad,100)-PA[0].rad
plt.plot(-pa_samps*180/np.pi,phs_grads[0]*np.cos(-pa_samps),label='phs_grad[0]*cos(PA)')
# plt.axhline(y=-1/slope,c='r',label='Regression')
# plt.axhline(y=-1/(slope+std_err),linewidth=0.5,c='r')
# plt.axhline(y=-1/(slope-std_err),linewidth=0.5,c='r')
# plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Phase gradient [days/arcsec]')
plt.xlabel('Position angle rel. to perp. profile [deg]')


# %%
max_diff = []
fig = plt.figure()
for slit_ind in slit_inds:
    input_arr_s, z_s = input_arr_s_allslits[slit_ind], z_s_allslits[slit_ind]
    input_arr_s_uc, z_s_uc = input_arr_s_uc_allslits[slit_ind], z_s_uc_allslits[slit_ind]
    FP = FP_df_lst[slit_ind]
    # for i in np.arange(len(input_arr_s)):
    for i in inds:
        z = deepcopy(z_s[i].flatten())
        inp_arr = deepcopy(input_arr_s[i])
        z_uc = deepcopy(z_s_uc[i].flatten())
        inp_arr_uc = deepcopy(input_arr_s_uc[i])
        x_uc, y_uc = inp_arr_uc[:,0], inp_arr_uc[:,1]
        
        x_FP_bin = FP.iloc[i].FP_Ac_bin_x
        z_FP_bin = FP.iloc[i].FP_Ac_bin_y
            
        x, y, PSF = inp_arr[:,0], inp_arr[:,1], inp_arr[0,3]
        x_argsort=x.argsort()
        x_sorted = x[x_argsort]
        inp_arr_sorted = inp_arr[x_argsort]
        z_sorted = z[x_argsort]
        x_span = x.max()-x.min()
        
        
        paramss = LE_model.make_params()
        
        result = deepcopy(result_lst[slit_ind])

        popt = []
        for ii in np.arange(len(paramss)):
            p = result.params[list(paramss)[ii]]
            popt.append(p.value)
        pcov = deepcopy(result.covar)

        if i!=0:
            ax=fig.add_subplot(121)#,sharex=ax)
            ax2=fig.add_subplot(122)#,sharex=ax)
            # ax=fig.add_subplot(224,projection='3d',sharex=ax,sharey=ax)
        else:
            ax=fig.add_subplot(121)
            ax2=fig.add_subplot(122)
            # ax=fig.add_subplot(224,projection='3d')
        
        const_offs = 100*slit_ind

        
        x_scaling = 1.0#np.cos(PA[slit_ind]-PA[0])
        # # ax.plot(x_sorted,LEprof_model(x_sorted,0.0,peak_flux,peak_loc,phs_grad, add_const, 0.0),c='k',linewidth=0.5,label='clean light curve')
        # # ax.plot(x_sorted,LEprof_model(x_sorted,0.0,peak_flux,peak_loc,phs_grad, add_const, PSF),c='b',linewidth=0.5,label='PSF only')
        # ax.plot(x_sorted,LEprof_model(x_sorted,*popt,ignore_dust=True,ignore_PSF=True),c='k',linewidth=0.5,label='clean light curve')
        # ax.plot(x_sorted,LEprof_model(x_sorted,*popt,ignore_dust=True),c='b',linewidth=0.5,label='PSF only')
        fit_z = LEprof_model(x_sorted,*popt)
        ax.plot(x_sorted*x_scaling,fit_z+const_offs,c='r',linewidth=1,label='fit (PSF+dust)')
        ax2.plot(x_sorted[1:]*x_scaling,np.diff(fit_z)/np.diff(x_sorted)/x_scaling)
        max_diff.append((np.diff(fit_z)/np.diff(x_sorted)).min())
        # ax.axhline(y=add_const+data_std_est,c='c')
        # ax.axhline(y=add_const-data_std_est,c='c')
        # ax.axvline(x=peak_loc,c='r')
        
        # ax.plot(x_sorted,z_sorted,linewidth=0.2,label='data')
        # ax.scatter(x,LEprof_model(inp_arr_zeroPSF,0,peak_flux,peak_loc,phs_grad, add_const),c='k',s=0.2,label='zero PSF & zero dust width')
        # ax.scatter(x,LEprof_model(inp_arr,0,peak_flux,peak_loc,phs_grad, add_const),c='b',s=0.2,label='zero dust width')
        # ax.scatter(x,LEprof_model(inp_arr,*popt),c='r',s=1,label='fit')
        ax.scatter(x*x_scaling,z+const_offs,s=0.5, label='data')
        # ax.plot(x_FP_bin*x_scaling,z_FP_bin+const_offs, label='binned')
        # ax.errorbar(x,z,yerr=z_stds,fmt='.',label='data')
        # ax.scatter(x_uc,z_uc,c='k',s=0.3, label='')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        # place a text box in upper left in axes coords
        # ax.text(0.05, 0.95, result.fit_report(), transform=ax.transAxes, fontsize=7,verticalalignment='top', bbox=props)
        
        # im=ax.scatter(x,y,cmap='jet',c=z-LEprof_model(inp_arr,*popt),s=1)
        # fig.colorbar(im,ax=ax,orientation="horizontal")
        ax.set_xlabel('x [arcsec]')
        ax.set_ylabel('Flux [ADU]')
        # ax.set_title('Phase Gradient: '+'{:.0f}'.format(grad)+' [days/arcsec]')
        ax2.set_xlabel('x [arcsec]')
        ax2.set_ylabel('d(Flux)/dx')
        ax2.set_title('Flux Slope')

plt.figure()
plt.title('Phase gradient vs. Position Angle')
plt.errorbar(-(PA.deg-PA[0].deg),np.array(max_diff),yerr=phs_grads_stderr*0.0,linestyle='',marker='.',label='individual phase gradients')

pa_samps = np.linspace(PA[0].rad,PA[-1].rad,100)-PA[0].rad
plt.plot(-pa_samps*180/np.pi,max_diff[0]*np.cos(-pa_samps),label='max_slope[0]*cos(PA)')
# plt.axhline(y=-1/slope,c='r',label='Regression')
# plt.axhline(y=-1/(slope+std_err),linewidth=0.5,c='r')
# plt.axhline(y=-1/(slope-std_err),linewidth=0.5,c='r')
# plt.plot(mjds.flatten(),intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(365*slope))+' [arcsec/year]')
plt.legend()
plt.ylabel('Phase gradient [days/arcsec]')
plt.xlabel('Position angle rel. to perp. profile [deg]')
# %%
sig_dusts_stderr = []
sig_dusts = []
for res in result_lst:
    sig_dusts_stderr.append(res.params['sig_dust'].stderr.copy())
    sig_dusts.append(res.params['sig_dust'].value)
sig_dusts_stderr = np.array(sig_dusts_stderr)
sig_dusts = np.array(sig_dusts)

plt.figure()
plt.title('Sigma Dust')
plt.errorbar(peaks.flatten(),sig_dusts,yerr=sig_dusts_stderr,linestyle='',marker='.',label='sig_dust')

plt.legend()
plt.ylabel('sig_dust [days]')
plt.xlabel('Obs. date (MJD)')
# %%
result_emcee_lst = []

LE_model = Model(LEprof_model)
for i in np.arange(len(input_arr_s)):
    print(i)
    z = deepcopy(z_s[i].flatten())
    inp_arr = deepcopy(input_arr_s[i])
    z_uc = deepcopy(z_s_uc[i].flatten())
    inp_arr_uc = deepcopy(input_arr_s_uc[i])
    x_uc, y_uc = inp_arr_uc[:,0], inp_arr_uc[:,1]
        
    x, y, PSF = inp_arr[:,0], inp_arr[:,1], inp_arr[0,3]
    args = (x.max()-x)<(x_span/10)
    data_std_est = np.std(z[args])
    
    # def residual(p):
    #     return (LE_model.eval(params=p,input_array=x) - z)/data_std_est
    
    def residual(params):
        pars = (params['sig_dust'].value, params['peak_flux'].value, params['peak_loc'].value, params['phs_grad'].value, params['add_const'].value, params['PSF'].value)
        out = LEprof_model(x,*pars)
        return (out - z)/data_std_est
    
    result_emcee = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, nwalkers=50, steps=1000, thin=20, params=result_lst[i].params.copy(), is_weighted=True, progress=True)
    # emcee_kws = dict(steps=1000, burn=300, thin=20, is_weighted=True,progress=True)
    # emcee_params = result_lst[i].params.copy()
    # result_emcee = LE_model.fit(data=z, input_array=x, params=emcee_params, method='emcee',nan_policy='omit', fit_kws=emcee_kws)
    result_emcee_lst.append(deepcopy(result_emcee))

# %%
for result in result_emcee_lst:
    emcee_corner = corner.corner(result.flatchain, labels=result.var_names,truths=list(result.params.valuesdict().values()))
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
# == STASH ==

figures[0].subplots_adjust(hspace=0,wspace=0)

for vv in figures[0].get_axes():
    vv.set_title('')
    for ll in vv.get_lines()[0:1]:
        ll.set_linewidth(0.2)
        ll.set_c('b')