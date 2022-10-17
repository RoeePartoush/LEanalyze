#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:41:03 2021

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



# == Time Series #7 (tyc4820/5) ==
Orgs=SkyCoord([(16.455, 59.462 )]*5+[(16.437940, 59.46777 )]*0,frame='fk5',unit=(u.deg, u.deg)) # ,(16.42939, 59.45377)
# Orgs=SkyCoord([(16.44543051, 59.45420532 )]*7,frame='fk5',unit=(u.deg, u.deg)) # ,(16.42939, 59.45377)
PA = Angle([150,130,110] + [200,65],'deg')
Ln = Angle([150]*3+[80]*2,u.arcsec)
Wd = Angle([1]*len(PA),u.arcsec)

# Orgs = SkyCoord([Orgs[0].directional_offset_by(PA[0]-Angle(0,u.deg),Angle(-0,u.arcsec))])
# Orgs2 = SkyCoord([Orgs[0].directional_offset_by(PA[1]-Angle(90,u.deg),Angle(5,u.arcsec))])[0]
# Orgs = SkyCoord([Orgs[0], Orgs2])
root = ''
filenames = [#'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.090915.2086_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.091217.441_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20101110.102797_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20110922.11223_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120117.116607_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120220.1526891_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20120621.24306_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130109t.1692991_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130610.32389_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20130826.33227_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20140621.102501_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20140827.48444_stch_5.sw.fits',
# '/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20150213.357723_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.VR.20150814.64478_stch_5.sw.fits',
'/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4820/5/tyc4820.r.20131202.135752_stch_5.sw.fits']

# [Time(mj, format='mjd', scale='utc').to_value(format='isot')[:10] for mj in DIFF_df.iloc[:,1]]


# w = DIFF_df.iloc[0].WCS_w
# # aaa=F2D.dcmp2df(filenames[0][:-5]+'.dcmp')
# # ggg=aaa[(aaa.Xpos<1260) & (aaa.Ypos>1120)]
# # star_list = list(w.wcs_pix2world(np.array([ggg.Xpos,ggg.Ypos]).T,1))
# # for i in np.arange(len(star_list)): star_list[i] = star_list[i].reshape((1,2))
# # peak_loc_lst = np.array([-66, -61,-40.4, -17.3, -10.4, 0, 13.51, 24.8, 27, 36, 55]).tolist()

# peak_loc_lst = [-42.240561714866516,
#                   -19.59303553820837,
#                   -8.107308586551339,
#                   1.3175300146907896,
#                   15.976285757840389,
#                   25.685951558370057,
#                   31.645932488668347,
#                   38.403183883032504]

# peak_loc_lst = [-51.65585454037466,
#                   -27.05199966637916,
#                   -17.557500290758103,
#                   -0.2874178643171132,
#                   19.198689732547116,
#                   32.930237858062874,
#                   42.324037553285336,
#                   55.69818220987464]


# phs_grd_lst = [-15.8]*len(filenames)
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

# # ========== EXTRACT PROFILE FROM IMAGE ==========
# print('\n\n=== Extracting Flux Profiles... ===')
# FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec),uniform_wcs=True)


# ========== PLOT IMAGES & PROFILES ==========
print('\n\n=== Plotting Images... ===')
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]

fig_fig = plt.figure()
inds_figfig = [1,3]
axs=[]
for i,ii in enumerate(inds_figfig):
    figures[ii] = fig_fig
    # axs.append(plt.subplot2grid((2,len(inds_figfig)),(0,i),rowspan=1,projection=DIFF_df.WCS_w[ii],fig=figures[ii]))

for i in DIFF_df.index:
    if i in inds_figfig:
        axs.append(plt.subplot2grid((2,len(inds_figfig)),(0,inds_figfig.index(i)),rowspan=1,projection=DIFF_df.WCS_w[i],fig=figures[i]))
    else:
        axs.append(plt.subplot2grid((1,1),(0,0),rowspan=1,projection=DIFF_df.WCS_w[i],fig=figures[i]))


# figures = [figures[0]]*8
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
# for mng in managers: mng.window.showMaximized()
# LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')#, crest_lines=CLs)#,peaks_locs=b)
axs = LEplots.imshows(DIFF_df,plot_Fprofile=True, profDF=slitFPdf, img_axs = axs,prof_sampDF_lst=None, prof_crop=(x_cutoff,y_cutoff), popts=popt_lst, g_cl_arg=coord_list, FullScreen=False, figs=figures)
# axs = LEplots.imshows(DIFF_df, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, prof_crop=None, popts=popt_lst, g_cl_arg=coord_list, FullScreen=False, figs=figures)
# %
img_titles = [Time(mj, format='mjd', scale='utc').to_value(format='isot')[:10]+'\nMJD: '+str(np.round(mj).astype(int)) for mj in DIFF_df.iloc[:,1]]
for i,ax in enumerate(axs):
    ax.set_title(img_titles[i])
    ax.coords[0].set_axislabel(' ')
    if i>1:
        ax.coords[1].set_axislabel(' ')
        ax.coords[1].set_ticklabel_visible(False)

w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(axs,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*1.20,slitFPdf.iloc[0]['Length']*1.2)
# LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'].directional_offset_by(slitFPdf.iloc[0]['PA'],Angle(25,u.arcsec)),slitFPdf.iloc[0]['Length']*0.35,slitFPdf.iloc[0]['Length']*0.3)



# %%

DIFF_df.coords_list[0] = [np.array([[16.4379904 , 59.47700486]]), np.array([[16.42941884, 59.47267454]]), np.array([[16.42138098, 59.46780024]])]
DIFF_df.coords_list[1] = [np.array([[16.44383534, 59.47116198]]), np.array([[16.43794487, 59.46859227]]), np.array([[16.43178679, 59.46575128]])]
DIFF_df.coords_list[2] = [np.array([[16.44676283, 59.46925809]]), np.array([[16.44167895, 59.46763718]]), np.array([[16.43739311, 59.46533653]])]
DIFF_df.coords_list[3] = [np.array([[16.44995316, 59.46667531]]), np.array([[16.44700604, 59.46505141]]), np.array([[16.44379291, 59.46356351]]),
                          np.array([[16.46173699, 59.47194907]]), np.array([[16.43574718, 59.45719753]])]
DIFF_df.coords_list[4] = [np.array([[16.45260809, 59.46395757]]), np.array([[16.45260254, 59.46300776]]), np.array([[16.45153119, 59.46246662]]),
                          np.array([[16.45634771, 59.46408756]]), np.array([[16.45152251, 59.46097406]])]
DIFF_df.coords_list[5] = [np.array([[16.45605806, 59.46028874]]), np.array([[16.45766267, 59.46069333]]), np.array([[16.45873407, 59.46123442]]),
                          np.array([[16.45042693, 59.45622662]]), np.array([[16.46596435, 59.46447948]])]
DIFF_df.coords_list[6] = [np.array([[16.45818451, 59.4586572 ]]), np.array([[16.46005855, 59.4594684 ]]), np.array([[16.4600676 , 59.46096096]]),
                          np.array([[16.4471983 , 59.45188938]]), np.array([[16.47291931, 59.46623204]])]
DIFF_df.coords_list[7] = [np.array([[16.45951056, 59.45716256]]), np.array([[16.46298828, 59.45824259]]), np.array([[16.46673665, 59.45986484]])]
DIFF_df.coords_list[8] = [np.array([[16.46641619, 59.451317 ]]), np.array([[16.47124397, 59.45483701]]), np.array([[16.47900451, 59.45753772]])]
DIFF_df.coords_list[9] = [np.array([[16.47438091, 59.44451948]]), np.array([[16.48215391, 59.449391 ]]), np.array([[16.49340372, 59.45479853]])]


plt.figure(100)
slopes = []
slope_errs = []
ax21 = plt.subplot2grid((2,2),(1,0),fig=figures[1])
for i in np.arange(len(slitFPdf)):
    mjds=[]
    # slit_coords_lst = [DIFF_df.iloc[dd].coords_list[i] for dd in np.arange(len(DIFF_df))]
    slit_coords_lst = []
    for dd in np.arange(len(DIFF_df)):
        lst = DIFF_df.iloc[dd].coords_list
        if len(lst)>i:
            slit_coords_lst.append(lst[i])
            mjds.append(DIFF_df.iloc[dd,1])
    peaks = np.array([c[0][0] for c in coord2ofs_list(slit_coords_lst,Orgs[i],PA[i],verbose=False)])
    mjds = np.array(mjds)#DIFF_df.iloc[:,1].to_numpy().astype(float).reshape(len(DIFF_df),1)[:len(slit_coords_lst)]
    # peaks = peaks[:,0]
    # peak_errs = errss[:,2].reshape(len(DIFF_df),1)
    # b = np.concatenate([mjds,peaks,peak_errs],axis=1)
    
    # %
    slope, intercept, r_value, p_value, std_err = stats.linregress(mjds.flatten(),peaks.flatten())
    slopes.append(1/slope)
    slope_errs.append(np.abs(1/(slope+std_err) - 1/slope))
    ax21.set_title('Peak location vs. MJD')
    # plt.scatter(mjds.flatten(),peaks.flatten(),label='Peaks')
    ax21.plot(mjds.flatten()-56250,intercept+slope*mjds.flatten(),label='Regression: '+'{:.1f}'.format(np.abs(1/slope))+' [arcsec/day]')
ax21.set_ylabel('Peak location [arcsec]')
ax21.set_xlabel('MJD - 56250 [days]')

ax22 = plt.subplot2grid((2,2),(1,1),fig=figures[1])

# plt.figure(101)
center_ind=2

from lmfit.models import SineModel
PAdep_model = SineModel()
PAdep_params = PAdep_model.make_params()
PAdep_params['frequency'].value = 1.0
PAdep_params['frequency'].vary = False
PAdep_params['shift'].value = -(PA[center_ind]-Angle(90,u.deg)).wrap_at(360 * u.deg).rad
result = PAdep_model.fit(np.array(slopes)[:7],PAdep_params,x=PA[:7].rad,weights = 1/np.array(slope_errs))
PAdep_params = result.params

# PAdep_params['amplitude'].value = slopes[center_ind]
# PAdep_params['shift'].value = -(PA[center_ind]-Angle(90,u.deg)).wrap_at(360 * u.deg).rad
# plt.scatter(PA.deg,slopes)
# theta_rad = np.linspace(PA.rad.min(),PA.rad.max(),51)
# theta_deg = np.linspace(PA.deg.min(),PA.deg.max(),51)
# plt.plot(theta_deg,slopes [center_ind]*np.cos(theta_rad-PA[center_ind].rad))

PAb = PA[0:7]
slopes_b = slopes[0:7]
slope_errs_b = slope_errs[0:7]
ax22.errorbar(PAb.deg,slopes_b,yerr=slope_errs_b,linestyle='',fmt='o')
# plt.scatter(PAb.deg,slopes_b)
theta_rad = np.linspace(PAb.rad.min(),PAb.rad.max(),51)
theta_deg = np.linspace(PAb.deg.min(),PAb.deg.max(),51)
ax22.plot(theta_rad*180/np.pi,PAdep_model.eval(params=PAdep_params,x=theta_rad))
# plt.plot(theta_deg,slopes[center_ind]*np.cos(theta_rad-PA[center_ind].rad))
ax22.set_title('Phase Gradient vs. Position Angle')
ax22.set_ylabel('P [days/arcsec]')
ax22.set_xlabel('θ [deg]')
# PAb = PA[7:]
# slopes_b = slopes[7:]
# plt.scatter(PAb.deg,slopes_b)
# theta_rad = np.linspace(PAb.rad.min(),PAb.rad.max(),51)
# theta_deg = np.linspace(PAb.deg.min(),PAb.deg.max(),51)
# plt.plot(theta_rad*180/np.pi,PAdep_model.eval(params=PAdep_params,x=theta_rad))
# # plt.plot(theta_deg,slopes[center_ind]*np.cos(theta_rad-PA[center_ind].rad))
# plt.ylabel('Apparent Motion ^-1 [days/arcsec]')
# plt.xlabel('Position Angle [deg]')
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


LinModel = lmfit.models.LinearModel()
LinParams = LinModel.make_params()
LinResult = LinModel.fit(peaks.flatten(),params=LinParams,x=mjds.flatten(), weights=1.0/peaks_stderr)
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