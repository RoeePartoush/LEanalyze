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

roots=['/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/']

files1=['20120118/tyc4419_tyc4519_20120118_coadd.fits',
        '20120626/tyc4419_tyc4519_20120626_coadd.fits',
        'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
        '20130109/tyc4419_tyc4519_20130109_coadd.fits',
        '20130609/tyc4419_tyc4519_20130609_coadd.fits',
        '20130826/tyc4419_tyc4519_20130826_coadd.fits',
        'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits',
        '20131202/tyc4419_tyc4519_20131202_coadd.fits',
        'KECK/tyc4419_1_g_LRIS_131228_coadd.fits',
        'KECK/tyc4419_1_R_LRIS_131228_coadd.fits',
        '20140226/tyc4419_tyc4519_20140226_coadd.fits',
        '20140531/tyc4419_tyc4519_20140531_coadd.fits',
        '20140621/tyc4419_tyc4519_20140621_coadd.fits',
        '20140827/tyc4419_tyc4519_20140827_coadd.fits',
        '20140923/tyc4419_tyc4519_20140923_coadd.fits']


for file in files1:
    files.append(roots[0]+file)

DIFF_df = F2D.FitsDiff(files)

# %%
# ========== PLOT FITS IMAGES ==========
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]

global coord_list
coord_list=[]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures)

# %%
# ========== DEFINE PROFILE SHAPE ==========

# Orgs=SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 172 deg, "A1" in tyc4419_1.ut131228.slits.png
# Orgs=SkyCoord([[12.36304697 , 58.76167572]],frame='fk5',unit=(u.deg, u.deg)) # PA = 114 deg, "A2" in tyc4419_1.ut131228.slits.png
# Orgs=SkyCoord(["0:49:29.5278 +58:45:45.734"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 135
Orgs=SkyCoord([[12.39123064, 58.76446428]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# Orgs=SkyCoord([[12.37100268, 58.76017914]],frame='fk5',unit=(u.deg, u.deg)) # PA = 156
# Orgs=SkyCoord([[12.39087967, 58.76356894]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# Orgs=SkyCoord([[12.40778143, 58.76893596]],frame='fk5',unit=(u.deg, u.deg)) # PA = 180
# ra = Longitude(np.array(coord_list).reshape((len(coord_list),2))[:,0],unit=u.deg)
# dec = Latitude(np.array(coord_list).reshape((len(coord_list),2))[:,1],unit=u.deg)
# Orgs = SkyCoord(ra, dec, frame='fk5')

# # SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5')
# SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5')
# PA = Angle([Org.position_angle(SN_sc)+Angle(180,'deg') for Org in Orgs])

PA = Angle([Angle(180+0,'deg') for Org in Orgs])
Ln = Angle([20  for Org in Orgs],u.arcsec)
clmns = ['Orig', 'PA', 'Length']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i]) for i in np.arange(len(Orgs))])
Wid = Angle(3,u.arcsec)
# %%
# ========== EXTRACT PROFILE FROM IMAGE ==========
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, width=Wid, REF_image=None, N_bins=20)

#%%
# ========== PLOT IMAGES & PROFILES ==========
plt.close('all')
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=None,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')#,peaks_locs=b)
# %
figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(figures,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*3,slitFPdf.iloc[0]['Length']*1.3)
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

SN_f = LCtable['func_L'][LCtable['dm15']==1.0][0]
# fig=plt.figure()
# ax=fig.add_subplot(projection='3d')
# fig=plt.figure()
# ax2=fig.add_subplot()
inds=[0]
x_cutoff = [[-10,0]]*4#[[-11,2.5],   [-2.5,10],  [-2,10]]
y_cutoff = [[-3,1.3]]*4#[[-5,7.5],   [-7.5,10],  [-7.5,10]]
star_masks = [[[-2.321,9.508],1.0], [[-0.798,3.375],1.0], [[-0.648,9.610],1.0], [[3.164,-8.330],1.0]]
# star_masks = [[[-8.64,1.03],1.2],  [[-6.76,-1.56],1.2],    [[-5.05,-0.67],1.2],   [[-0.35,-4.46],1.2],   [[9.7,-3.13],1.2], [[-12.24,-2.95],1]]
input_arr_s, z_s = [], []
for ind in inds:
    iii=inds.index(ind)
    fig=plt.figure()
    if iii!=0:
        ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(projection='3d')
    x=FP_df_lst[0].iloc[ind]['ProjAng'].arcsec
    y=FP_df_lst[0].iloc[ind]['PerpAng'].arcsec
    z=FP_df_lst[0].iloc[ind]['FluxProfile_ADU']
    mjd_curr = DIFF_df.iloc[ind]['Idate']
    PSF_curr = DIFF_df.iloc[ind]['FWHM_ang'].arcsec
    
    x_inds = np.logical_and(x>x_cutoff[iii][0], x<x_cutoff[iii][1])
    x = x[x_inds]
    y = y[x_inds]
    z = z[x_inds]
    
    y_inds = np.logical_and(y>y_cutoff[iii][0], y<y_cutoff[iii][1])
    x = x[y_inds]
    y = y[y_inds]
    z = z[y_inds]
    
    # for star_mask in star_masks:
    #     st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(1.4*PSF_curr)
    #     x = x[st_mk_inds]
    #     y = y[st_mk_inds]
    #     z = z[st_mk_inds]
    
    shape = (len(x),1)
    input_arr_s.append(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1))
    z_s.append(z.reshape(shape))

    ax.scatter(x,y,z,cmap='jet',s=2,c=z)#,s=2,c=z)#,vmax=200)
    ax.view_init(elev=90,azim=0)
    plt.xlim([-11,11])
    axisEqual3D(ax)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.xlabel('x')
    plt.ylabel('y')
    
# %%
# ========== DEFINE FITTING FUNCTION ==========
    
phs_res = 0.1
def ConvLC(phase, phs_wid, PeakFlux, add_const):
    phs_min = np.max([np.nanmin(phase),-1e3])
    phs_max = np.min([np.nanmax(phase),4e3])
    flux_p_unscaled, phase_p = DFit.LCgrid_eval(SN_f,phs_wid,phs_res,np.array([phs_min,phs_max]))    
    const = PeakFlux/SN_f(0) # normalize maximun luminosity to PeakFlux
    phase_arr=np.array(phase)
    flux = np.interp(phase_arr, phase_p, const*flux_p_unscaled, left=0, right=0) + add_const
    return flux


def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06,                  pxx,pxy,pyy):
    polyphase_params = np.array([[p04,              np.sin(p05)*p06,    pyy,  0],
                                 [np.cos(p05)*p06,  pxy,                  0,  0],
                                 [pxx,              0,                  0,  0],
                                 [0,              0,                  0,  0]])
    x, y, mjd, PSFs = input_array[:,0], input_array[:,1], input_array[:,2], input_array[:,3]
    inds_umjd = []
    for umjd in np.unique(mjd):
        inds_umjd.append(np.argwhere(mjd==umjd)[:,0])
    xy_base_phases = np.polynomial.polynomial.polyval2d(x, y, polyphase_params)
    xy_phases = xy_base_phases + mjd#-mjd[0]
    
    output_array = np.zeros(x.shape)
    phase_grad = p06#np.hypot(polyphase_params[0,1],polyphase_params[1,0])
    
    eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[0][0]])
    ConvLC_params = [eff_phs_wid,p02,0]
    # print('1: '+str(phase_grad*PSFs[inds_umjd[0][0]]))
    output_array[inds_umjd[0]] = ConvLC(xy_phases[inds_umjd[0]], *ConvLC_params)
    
    # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[1][0]])
    # ConvLC_params = [eff_phs_wid,p08,0]
    # # print('2: '+str(phase_grad*PSFs[inds_umjd[1][0]]))
    # output_array[inds_umjd[1]] = ConvLC(xy_phases[inds_umjd[1]], *ConvLC_params)
    
    # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[2][0]])
    # ConvLC_params = [eff_phs_wid,p09,0]
    # # print('3: '+str(phase_grad*PSFs[inds_umjd[2][0]]))
    # output_array[inds_umjd[2]] = ConvLC(xy_phases[inds_umjd[2]], *ConvLC_params)
    
    # eff_phs_wid = np.hypot(p01,phase_grad*PSFs[inds_umjd[3][0]])
    # ConvLC_params = [eff_phs_wid,p10,0]
    # # print('4: '+str(phase_grad*PSFs[inds_umjd[3][0]]))
    # output_array[inds_umjd[3]] = ConvLC(xy_phases[inds_umjd[3]], *ConvLC_params)
    
    return output_array

# %%
# ========== PERFORM FITTING ==========
    
input_arr = np.concatenate(input_arr_s,axis=0)
z = np.concatenate(z_s,axis=0).flatten()
# input_arr = np.concatenate([input_arr_s[0],input_arr_s[2]],axis=0)
# z = np.concatenate([z_s[0],z_s[2]],axis=0).flatten()

bnd = ([0.0,    1e1,            -5.68e4,   -50.0*np.pi/180,  -1e2,                           -1e-10, -1e-10, -1e-10],
       [80.0,   1e3,           -5.65e4,    50.0*np.pi/180,   -1e0,                           +1e-10, +1e-10, +1e-10])

# p_init = [ 2.0e-04,  1.1e+02, -5.6e+04, -3.7e-01, -7.3e+00,  5.0e+02,  5.050000e+02,  5.050000e+02]
popt, pcov = curve_fit(ConvLC_2Dpolyphase,input_arr,z,bounds=bnd)#,p0 = p_init)
print(popt)

# %%
# ========== DISPLAY FITTING RESULT ==========

F_SCALE=100
u_mjds = list(np.unique(input_arr[:,2]))
for mjd in u_mjds:
    indsss = np.argwhere(input_arr[:,2]==mjd)[:,0]
    # print(indsss[0:10])
    iii=u_mjds.index(mjd)
    fig=plt.figure()
    if iii!=0:
        ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(projection='3d')
    zz=ConvLC_2Dpolyphase(input_arr[:,:], *popt)
    x=input_arr[:,0]
    y=input_arr[:,1]
    PSF = input_arr[indsss,3][0] # arcsec
    grad = popt[4] # day/arcsec
    dust_wid = popt[0] # days
    print('Total width: '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF))+' [days]')
    print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]')
    polyphase_params = np.array([[popt[2],                  np.sin(popt[3])*grad,    popt[7],  0],
                                 [np.cos(popt[3])*grad,          popt[6],                  0,  0],
                                 [popt[5],                          0,                  0,  0],
                                 [0,                          0,                  0,  0]])
    polyphase = np.polynomial.polynomial.polyval2d(x, y, polyphase_params) + mjd
    # ax.scatter(x[indsss],y[indsss],polyphase[indsss],cmap='jet',vmax=F_SCALE/3,vmin=-F_SCALE/3,s=2,c='b')#,c=zz)
    ax.scatter(x[indsss],y[indsss],zz[indsss],cmap='jet',vmax=F_SCALE/3,vmin=-F_SCALE/3,s=2,c='b')#,c=zz)
    ax.scatter(x[indsss],y[indsss],z[indsss],cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c=z[indsss],s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(mjd)
    axisEqual3D(ax)
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
       [50, 50, 500, +50.0*np.pi/180])

F_SCALE=120
plt.figure()
ax2 = plt.subplot()
for i in np.arange(len(input_arr_s)):
    z = z_s[i].flatten()
    inp_arr = input_arr_s[i]
    popt, pcov = curve_fit(Gauss_model,inp_arr,z,bounds=bnd)
    print(popt)
    fig=plt.figure()
    if i!=0:
        ax=fig.add_subplot(projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(projection='3d')
    x, y = inp_arr[:,0], inp_arr[:,1]
    ax.plot_trisurf(x,y,Gauss_model(inp_arr,*popt),cmap='jet',vmax=F_SCALE/3,vmin=-F_SCALE/3)#,c=zz)
    ax.scatter(x,y,z,cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c=z,s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    mjd = inp_arr[:,2][0]
    plt.title(mjd)
    axisEqual3D(ax)
    ax2.scatter(popt[0],mjd)
    
# %%
###########################
######## S T A S H ########
###########################
### INLINE RUNS TO KEEP ###
###########################

### MANUAL PSF
# DIFF_df_BU.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))
# for i in np.arange(len(DIFF_df_BU)): DIFF_df_BU.iloc[i,10]=LEtb.pix2ang(DIFF_df_BU.iloc[i]['WCS_w'],DIFF_df_BU.iloc[i,8])

### STAR MASK PRINT
# sss='['
# SClst = [SkyCoord(coord,frame='fk5',unit=(u.deg, u.deg))[0] for coord in coord_list]
# for SC in SClst:
#     Sep_as = Orgs[0].separation(SC).arcsec
#     PA_a = Orgs[0].position_angle(SC)-PA[0]
#     x=Sep_as*np.cos(PA_a.rad)
#     y=Sep_as*np.sin(PA_a.rad)
#     print('x = '+str(x))
#     print('y = '+str(y)+'\n')
#     sss += '[['+'{:.3f}'.format(x)+','+'{:.3f}'.format(y)+'],1.0]'
#     if not (SClst.index(SC)==(len(SClst)-1)):
#         sss += ', '
# sss += ']'
# print(sss)