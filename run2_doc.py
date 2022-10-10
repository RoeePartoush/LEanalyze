#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 06:06:14 2021

@author: roeepartoush
"""

# ========== IMPORT PACKAGES ==========

# base imports
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt')
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
LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
LCtable = F2D.LightCurves(LCs_file)


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


def LEprof_model( input_array,    p01,        p02, p04,   p06,    add_const):
    #           (x,y,mjds,PSFs), dust_width, amp, shift, phs_grd, add_const
    p05=0*np.pi/180 # [rad]
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

star_list = []
phs_grd_lst = []
popt_lst = []
# ========= USER INPUT =========
# == COPY AND PASTE OVER HERE ==
# ------------------------------
# == Time Series #6 ==
Orgs=SkyCoord([(12.36968543, 58.75960183)],frame='fk5',unit=(u.deg, u.deg))
PA = Angle([Angle(150,'deg') for Org in Orgs])
Ln = Angle([60  for Org in Orgs],u.arcsec)
Wd = Angle([20  for Org in Orgs],u.arcsec)
x_cutoff = [[-28.6, -18.1], [-21.0, -8.5], [-10.9, -3.3], [-5.8, 4.1], [2.3, 12.4], [7.0, 15.4], [13.9, 21.6], [13.4, 24.1]]
y_cutoff = [[-10.0, -4.0], [-9.0, -4.0], [-9.7, -0.1], [-3.5, 0.0], [-5.1, -2.0], [-5.2, -2.6], [0.5, 5.6], [-4.0, -0.5]]

#	<dust_width, PeakFlux, arcs_shift, phs_grad, add_const>
# popt_lst = [np.array([97.944, 46.816, -22.056, -36.584, 3.746]),
# np.array([200.0, 208.404, -13.17, -60.726, 0.285]),
# np.array([21.743, 90.199, -6.657, -15.296, -0.118]),
# np.array([43.557, 48.488, 0.886, -19.698, 0.526]),
# np.array([81.254, 75.89, 8.646, -26.628, -0.951]),
# np.array([49.317, 41.728, 12.106, -28.681, -0.639]),
# np.array([181.039, 89.14, 18.385, -99.989, 1.775]),
# np.array([46.294, 90.103, 20.144, -22.949, 1.203])]

filenames = ['20120118/tyc4419_tyc4519_20120118_coadd.fits',
 '20120626/tyc4419_tyc4519_20120626_coadd.fits',
 'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
 '20130109/tyc4419_tyc4519_20130109_coadd.fits',
 '20130609/tyc4419_tyc4519_20130609_coadd.fits',
 '20130826/tyc4419_tyc4519_20130826_coadd.fits',
 'KECK/tyc4419_1_R_LRIS_131228_coadd.fits',
 '20140226/tyc4419_tyc4519_20140226_coadd.fits']



# phs_grd_lst = [-18.3]*len(filenames)
# ------------------------------


# %
# ========== LOAD FITS FILES ==========

files=[]
root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'

for file in filenames:
    files.append(os.path.join(root, file[:-5]))

DIFF_df = F2D.FitsDiff(files)

# ### MANUAL PSF
# DIFF_df.iloc[:,8] = pd.Series(data=np.zeros((len(DIFF_df),)))
# # DIFF_df.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
# for i in np.arange(len(DIFF_df)): DIFF_df.iloc[i,10]=LEtb.pix2ang(DIFF_df.iloc[i]['WCS_w'],DIFF_df.iloc[i,8])

### MANUAL PSF
# DIFF_df.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
# for i in np.arange(len(DIFF_df)): DIFF_df.iloc[i,10]=LEtb.pix2ang(DIFF_df.iloc[i]['WCS_w'],DIFF_df.iloc[i,8])

# %
global coord_list
coord_list=[]

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
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec))


# ========== PLOT IMAGES & PROFILES ==========
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]
# figures = [figures[0]]*8
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
# LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')#, crest_lines=CLs)#,peaks_locs=b)
axs = LEplots.imshows(DIFF_df, profDF=slitFPdf, prof_sampDF_lst=FP_df_lst, prof_crop=(x_cutoff,y_cutoff), popts=popt_lst, g_cl_arg=coord_list, FullScreen=True, figs=figures)
# %

w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(axs,w_s,slitFPdf.iloc[0]['Orig'],slitFPdf.iloc[0]['Length']*1.4,slitFPdf.iloc[0]['Length']*1.4)
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
slit_inds = [0]


# x_cutoff, y_cutoff = coord_corners2xy_lims(coord_list,Orgs[0],PA[0])
if not x_cutoff:
    x_cutoff = [-Ln[0].arcsec/2, Ln[0].arcsec/2]*len(DIFF_df)
if not y_cutoff:
    y_cutoff = [-Wd[0].arcsec/2, Wd[0].arcsec/2]*len(DIFF_df)


for slit_ind in slit_inds:
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
        x=FP_df_lst[slit_ind].iloc[ind]['ProjAng'].arcsec
        y=FP_df_lst[slit_ind].iloc[ind]['PerpAng'].arcsec
        z=FP_df_lst[slit_ind].iloc[ind]['FluxProfile_ADU']
        mjd_curr = DIFF_df.iloc[ind]['Idate']
        PSF_curr = DIFF_df.iloc[ind]['FWHM_ang'].arcsec
        
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
                st_mk_inds = np.hypot((x-star_mask[0][0])*1,y-star_mask[0][1])>(1.5)#PSF_curr)
                # st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(1.2*PSF_curr)
                x = x[st_mk_inds]
                y = y[st_mk_inds]
                z = z[st_mk_inds]
        
        shape = (len(x),1)
        input_arr_s.append(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1))
        z_s.append(z.reshape(shape))
    
        ax.scatter(x,y,cmap='jet',s=7,c=z)#,s=2,c=z)#,vmax=200)
        if popt_lst:
            ax.axvline(x=popt_lst[ind][2])
        # ax.view_init(elev=90,azim=0)
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([np.min(np.stack(x_cutoff).flatten()),np.max(np.stack(x_cutoff).flatten())])
        plt.ylim([np.min(np.stack(y_cutoff).flatten()),np.max(np.stack(y_cutoff).flatten())])
        axisEqual3D(ax)
    

# %

popt_lst_BU = deepcopy(popt_lst)
phs_grd_lst_BU = deepcopy(phs_grd_lst)
# ========== Independent Light Curve ==========
# ============== FITTING & PLOTS ==============
# bnd = ([1e-3,     1e1,            15,   -50.0*np.pi/180,  -4e1, -5e1 ],
#         [200.0,    1e3,             35,    50.0*np.pi/180,   -1e0, 5e1])
# bnd = ([1e-3,     1e1,            -15,   -10.0*np.pi/180,  -4e1, -500 ],
#         [200.0,    1e3,             15,    10.0*np.pi/180,   -1e0, 2e3])
bnd = ([1e-3,     1e1,            -16,     -200, -500 ],
        [200.0,    1e3,             16,      -1e-1, 2e3])
bnds=[bnd]*len(DIFF_df)


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
for i in np.arange(len(input_arr_s)):
    z = z_s[i].flatten()
    inp_arr = input_arr_s[i]
    x_bnd = (inp_arr[:,0].min(),inp_arr[:,0].max())
    mjd = inp_arr[:,2][0]
    bnd_tmp = deepcopy(bnds[i])
    bnd_tmp[0][2] = x_bnd[0]
    bnd_tmp[1][2] = x_bnd[1]
    
    if phs_grd_lst_BU:
        del (bnd_tmp[0][3],bnd_tmp[1][3])
        phs_grd = deepcopy(phs_grd_lst_BU[i])
        popt, pcov = curve_fit(lambda x, p01,p02,p04, add_const: LEprof_model(x, p01,p02,p04, phs_grd, add_const), inp_arr,z,bounds=bnd_tmp,absolute_sigma=True)
        popt = np.insert(popt,3,phs_grd)
        pcov = np.insert(pcov,2,np.zeros((4,)),0)
        pcov = np.insert(pcov,2,np.zeros((5,)),1)
    elif popt_lst_BU:
        popt = deepcopy(popt_lst_BU[i])
    else:
        popt, pcov = curve_fit(LEprof_model,inp_arr,z,bounds=bnd_tmp,absolute_sigma=True)#,sigma=np.ones(inp_arr[:,0].shape)*20)
    # errs = np.sqrt(np.diag(pcov))
    popt_lst.append(popt.copy())
    pcov_lst.append(deepcopy(pcov))
    pcorr_lst.append(cov2corr(pcov))
    # errs_lst.append(errs.copy())
    # popt = [20.0,     10e1,            -0.0,   -34.0*np.pi/180,   -14 ]
    # popt = [  3.09876149, 116.87448495,   8.91707928 +(mjd-56654.2),  -0.47029305,  -5.38910457, -14.00613042]
    print('dust_width, PeakFlux, arcs_shift, phs_grad, add_const = '+str(popt))
    p01,p02,p04,p05,p06, add_const = popt[0], popt[1], popt[2], 0, popt[3], popt[4]
    phs_grads.append(p06.copy())
    # p01,p02,p04,p05,p06, add_const = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    PSF = inp_arr[:,3][0] # arcsec
    grad = p06 # day/arcsec
    # total_wid = popt[0] # days
    # dust_wid = np.sqrt(total_wid**2 - (grad*PSF)**2)
    # print('Total width: '+'{:.3f}'.format(total_wid)+' [days] <--> '+'{:.3f}'.format(total_wid/grad)+' [arcsec]')
    # print('PSF width: '+'{:.3f}'.format(grad*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
    # print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/grad)+' [arcsec]\n')
    dust_wid = popt[0] # days
    print('Total width: '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF))+' [days] <--> '+'{:.3f}'.format(np.hypot(dust_wid,grad*PSF)/np.abs(grad))+' [arcsec]')
    print('PSF width: '+'{:.3f}'.format(np.abs(grad)*PSF)+' [days]  <--> '+'{:.3f}'.format(PSF)+' [arcsec]')
    print('Dust width: '+'{:.3f}'.format(dust_wid)+' [days]  <--> '+'{:.3f}'.format(dust_wid/np.abs(grad))+' [arcsec]\n')
    # print(np.sqrt(np.diag(pcov)))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig=figures[i]
    if i!=0:
        ax=fig.add_subplot(122)#,sharex=ax)
        # ax=fig.add_subplot(224,projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(122)
        # ax=fig.add_subplot(224,projection='3d')
    
    x, y = inp_arr[:,0], inp_arr[:,1]
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y +mjd
    # ax.scatter(x,y,xy_phases,c='b',s=1)
    bbox=np.array([[x.min(),x.max()],[y.min(),y.max()]])
    plot_proj_phaseplane(ax,p04,p05,grad,bbox)
    
    inp_arr_zeroPSF = deepcopy(inp_arr)
    inp_arr_zeroPSF[:,-1] = inp_arr_zeroPSF[:,-1]*0
    
    x_argsort=x.argsort()
    x_sorted = x[x_argsort]
    inp_arr_sorted = inp_arr[x_argsort]
    inp_arr_zeroPSF_sorted = inp_arr_zeroPSF[x_argsort]
    z_sorted = z[x_argsort]
    ax.plot(x_sorted,LEprof_model(inp_arr_zeroPSF_sorted,0,p02,p04,p06, add_const),c='k',linewidth=0.5,label='clean light curve')
    ax.plot(x_sorted,LEprof_model(inp_arr_sorted,0,p02,p04,p06, add_const),c='b',linewidth=0.5,label='PSF only')
    ax.plot(x_sorted,LEprof_model(inp_arr_sorted,*popt),c='r',linewidth=1,label='fit (PSF+dust)')
    # ax.plot(x_sorted,z_sorted,linewidth=0.2,label='data')
    # ax.scatter(x,LEprof_model(inp_arr_zeroPSF,0,p02,p04,p06, add_const),c='k',s=0.2,label='zero PSF & zero dust width')
    # ax.scatter(x,LEprof_model(inp_arr,0,p02,p04,p06, add_const),c='b',s=0.2,label='zero dust width')
    # ax.scatter(x,LEprof_model(inp_arr,*popt),c='r',s=1,label='fit')
    ax.scatter(x,z,cmap='jet',s=0.5, label='data')
    
    # im=ax.scatter(x,y,cmap='jet',c=z-LEprof_model(inp_arr,*popt),s=1)
    # fig.colorbar(im,ax=ax,orientation="horizontal")
    ax.set_xlabel('x [arcsec]')
    ax.set_ylabel('y [arcsec]')
    ax.set_title('Phase Gradient: '+'{:.0f}'.format(grad)+' [days/arcsec]')
    # ax.set_title('MJD: '+'{:.0f}'.format(mjd))
    
    if str(type(ax))=="<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
        axisEqual3D(ax)
    
    # ax2.scatter(p04/p06,mjd)
    # zero intersect: cos(p05)*p06*X + sin(p05)*p06*Y + p04 +mjd = 0 --> X = -(sin(p05)*p06*Y + p04 +mjd)/cos(p05)*p06
    zero_y = np.array([-0.5,0.5,0])*Wd[0].arcsec*1.2
    zero_x = p04 -(np.sin(p05)*p06*zero_y + 0*mjd)/(np.cos(p05)*p06)
    # zero_x = -(np.sin(p05)*p06*zero_y + p04+0*mjd)/(np.cos(p05)*p06)
    ax3.plot(zero_x,zero_y,mjd)
    
    plot_local_phase_plane(ax3, p04, p05, p06, [-0.5,0.5], [-10,0,10], mjd)
    
    CLs[i,0,0], CLs[i,1,0], CLs[i,2,0] = zero_x
    CLs[i,0,1], CLs[i,1,1], CLs[i,2,1] = zero_y
    CLs[i,0,2], CLs[i,1,2], CLs[i,2,2] = mjd*np.ones(len(zero_y))
    
    ax.legend()
    
    ax_img = axs[i]
    prof_orig = slitFPdf['Orig'][0]
    prof_PA_x = slitFPdf['PA'][0]
    prof_PA_y = slitFPdf['PA'][0] + Angle(90,u.deg)
    prof_width = slitFPdf['WIDTH'][0]
    
    peak_arcsec_shift = Angle(popt[2],u.arcsec)
    corner1 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y,-prof_width/2)
    corner2 = prof_orig.directional_offset_by(prof_PA_x,peak_arcsec_shift).directional_offset_by(prof_PA_y, prof_width/2)
    xy = np.array([[corner1.ra.degree, corner1.dec.degree], [corner2.ra.degree, corner2.dec.degree]])
    ax_img.plot(xy[:,0], xy[:,1], transform = ax_img.get_transform('world'), linewidth=0.5, color='r')

axisEqual3D(ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')

popts = np.stack(popt_lst)
# errss = np.stack(errs_lst)

mjds = DIFF_df.iloc[:,1].to_numpy().astype(float).reshape(len(DIFF_df),1)
peaks = popts[:,2].reshape(len(DIFF_df),1)
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