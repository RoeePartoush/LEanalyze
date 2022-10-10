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

# %%
# ========== LOAD FITS FILES ==========
x_cutoff = [[19,32.5]]*2 + [[19,32.5]]*2
y_cutoff = []
files=[]

# roots=['/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/']
# get_ipython().run_line_magic('gui', 'tk')
# filenames = askopenfiles(
#             initialdir=roots[0],
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

if not files:
    filenames =[#'20120118/tyc4419_tyc4519_20120118_coadd.fits',
                # '20120626/tyc4419_tyc4519_20120626_coadd.fits',
                # 'KECK/tyc4419_1.R.r120918_0182_4.hdrfix_CORRECTED_DEG_MJD.sw._NOT_REALLY_coadd.fits',
                # '20130109/tyc4419_tyc4519_20130109_coadd.fits',
                # '20130609/tyc4419_tyc4519_20130609_coadd.fits',
                # '20130826/tyc4419_tyc4519_20130826_coadd.fits',
                'KECK/tyc4419_1_DEIMOS_5_6_coadd.fits',
                '20131202/tyc4419_tyc4519_20131202_coadd.fits',
                'KECK/tyc4419_1_g_LRIS_131228_coadd.fits',
                'KECK/tyc4419_1_R_LRIS_131228_coadd.fits']
                # '20140226/tyc4419_tyc4519_20140226_coadd.fits']
                # '20140531/tyc4419_tyc4519_20140531_coadd.fits',
                # '20140621/tyc4419_tyc4519_20140621_coadd.fits',
                # '20140827/tyc4419_tyc4519_20140827_coadd.fits']
                # '20140923/tyc4419_tyc4519_20140923_coadd.fits'] 
    root = '/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
    
    files = []
    for file in filenames:
        files.append(os.path.join(root, file[:-5]))

DIFF_df_BU = F2D.FitsDiff(files)

### MANUAL PSF
DIFF_df_BU.iloc[:,8] = pd.Series(data=np.zeros((len(DIFF_df_BU),)))
# DIFF_df_BU.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
for i in np.arange(len(DIFF_df_BU)): DIFF_df_BU.iloc[i,10]=LEtb.pix2ang(DIFF_df_BU.iloc[i]['WCS_w'],DIFF_df_BU.iloc[i,8])

### MANUAL PSF
# DIFF_df_BU.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
# for i in np.arange(len(DIFF_df_BU)): DIFF_df_BU.iloc[i,10]=LEtb.pix2ang(DIFF_df_BU.iloc[i]['WCS_w'],DIFF_df_BU.iloc[i,8])

# ========== PLOT FITS IMAGES ==========
DIFF_df = DIFF_df_BU.copy()
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
# Orgs=SkyCoord([(12.38828633, 58.77037112)],frame='fk5',unit=(u.deg, u.deg))
# Orgs=SkyCoord([coord_list[0][0]],frame='fk5',unit=(u.deg, u.deg))
# Orgs=SkyCoord([Orgs[0].directional_offset_by(Angle(155,'deg'),Angle(1.5,u.arcsec))])

# A1_sc = SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg))
# Orgs = SkyCoord([A1_sc.directional_offset_by(Angle(172,'deg'),Angle(27,u.arcsec))])#,A1_sc,A1_sc])
Orgs=SkyCoord(["0:49:33.424 +58:46:19.21"],frame='fk5',unit=(u.hourangle, u.deg)) # PA = 172 deg, "A1" in tyc4419_1.ut131228.slits.png
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

# sc = SkyCoord([(12.37235314, 58.76254572)],frame='fk5',unit=(u.deg, u.deg))[0]
# Orgs=SkyCoord([(12.36968543, 58.75960183)],frame='fk5',unit=(u.deg, u.deg))
# Orgs=SkyCoord([Orgs[0].directional_offset_by(Angle(140-90,'deg'),Angle(3,u.arcsec))])
# Orgs = []
# for ofs in np.linspace(3,3,1):
#     Orgs.append(sc.directional_offset_by(Angle(150+180,'deg'),Angle(ofs,u.arcsec)))

# Orgs = SkyCoord(Orgs)
PA = Angle([Angle(172,'deg') for Org in Orgs])
# PA[0] = Angle(170,'deg')
Ln = Angle([100  for Org in Orgs],u.arcsec)
Wd = Angle([1  for Org in Orgs],u.arcsec)

# PA = Angle([150],'deg')
# Ln = Angle([16],u.arcsec)
# Wd = Angle([7],u.arcsec)

# PA = Angle([172,172,172],'deg')
# Ln = Angle([18,100,10],u.arcsec)
# Wd = Angle([7.5,1,10],u.arcsec)

clmns = ['Orig', 'PA', 'Length','WIDTH']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i],Wd[i]) for i in np.arange(len(Orgs))])

# ========== EXTRACT PROFILE FROM IMAGE ==========
FP_df_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf, REF_image=None, N_bins=int(Ln[0].arcsec))
# %%
if not x_cutoff:
    x_cutoff = [[-Ln[0].arcsec/2, Ln[0].arcsec/2]]*len(DIFF_df)
if not y_cutoff:
    y_cutoff = [[-Wd[0].arcsec/2, Wd[0].arcsec/2]]*len(DIFF_df)

# ========== PLOT IMAGES & PROFILES ==========
plt.close('all')
figures=[plt.figure() for i in DIFF_df.index]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
axs = LEplots.imshows(DIFF_df,REF_image=None,prof_crop=(x_cutoff,y_cutoff),g_cl_arg=coord_list,FullScreen=True,med_filt_size=None,figs=figures,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst,fluxSpace='LIN')#, crest_lines=CLs,peaks_locs=b)
# %

w_s = DIFF_df['WCS_w'].to_list()
LEplots.match_zoom_wcs(axs,w_s,slitFPdf.iloc[0]['Orig'].directional_offset_by(PA[0],Angle(10,u.arcsec)),slitFPdf.iloc[0]['Length']*0.3,slitFPdf.iloc[0]['Length']*0.6)
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
inds=np.arange(len(DIFF_df)).tolist()#,2,3,4,5,6,7]
slit_inds = [0]

# y_cutoff = [[-10,-7],[-10,-6],[-10,-5],    [-7,-3], [-7,-3], [-6,-1], [-1,4], [2,5], [4,7], [5,8]]
# x_cutoff = [[-16,-4],[-10.5,-1],[-7.3,1],  [-6.5,2],[-6.5,2],[-3.5,5.5], [0,10],  [2,10],  [5,13], [5,14]]
# x_cutoff, y_cutoff = coord_corners2xy_lims(coord_list,Orgs[0],PA[0])
for slit_ind in slit_inds:
    
    # star_masks = [[[-9.388,-0.419],1.2], [[-5.810,3.568],1.2], [[-2.205,1.803],1.2], [[-3.849,0.971],1.2], [[2.500,-1.966],1.2], [[-5.107,-8.458],1.2], [[-3.637,-8.029],1.2]]
    input_arr_s, z_s = [], []
    for ind in inds:
        # star_masks = coord2ofs_list(coord_list,Orgs[slit_ind],PA[slit_ind],verbose=False)
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
        mjd_curr = DIFF_df.iloc[ind]['Idate'    ]
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
        #     st_mk_inds = np.hypot((x-star_mask[0][0])*1,y-star_mask[0][1])>(1.5)#PSF_curr)
        #     # st_mk_inds = np.hypot(x-star_mask[0][0],y-star_mask[0][1])>(1.2*PSF_curr)
        #     x = x[st_mk_inds]
        #     y = y[st_mk_inds]
        #     z = z[st_mk_inds]
        
        shape = (len(x),1)
        input_arr_s.append(np.concatenate((x.reshape(shape),y.reshape(shape),mjd_curr*np.ones(shape),PSF_curr*np.ones(shape)),axis=1))
        z_s.append(z.reshape(shape))
    
        ax.scatter(x,y,cmap='jet',s=7,c=z)#,s=2,c=z)#,vmax=200)
        # ax.view_init(elev=90,azim=0)
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([np.min(np.stack(x_cutoff).flatten()),np.max(np.stack(x_cutoff).flatten())])
        plt.ylim([np.min(np.stack(y_cutoff).flatten()),np.max(np.stack(y_cutoff).flatten())])
        axisEqual3D(ax)
    

# %%
# ========== Independent Light Curve ==========
# ============== FITTING & PLOTS ==============
# bnd = ([1e-3,     1e1,            15,   -50.0*np.pi/180,  -4e1, -5e1 ],
#         [200.0,    1e3,             35,    50.0*np.pi/180,   -1e0, 5e1])
# bnd = ([1e-3,     1e1,            -15,   -10.0*np.pi/180,  -4e1, -500 ],
#         [200.0,    1e3,             15,    10.0*np.pi/180,   -1e0, 2e3])
bnd = ([1.0,     1e1,            -16,     -1e2, -500 ],
        [200.0,    1e3,             16,      -1e0, 2e3])
bnds=[bnd]*10

from copy import deepcopy
F_SCALE=100
# plt.figure()
# ax2 = plt.subplot() 
fig = plt.figure()
ax3 = fig.add_subplot(projection='3d')
CLs = np.zeros((len(input_arr_s),3,3))
popt_lst=[]
errs_lst=[]
phs_grads=[]
for i in np.arange(len(input_arr_s)):
    z = z_s[i].flatten()
    inp_arr = input_arr_s[i]
    x_bnd = (inp_arr[:,0].min(),inp_arr[:,0].max())
    mjd = inp_arr[:,2][0]
    bnd_tmp = deepcopy(bnds[i])
    bnd_tmp[0][2] = x_bnd[0]
    bnd_tmp[1][2] = x_bnd[1]
    ######### 2 PEAKS MODEL!!!! ####
    bnd_tmp[0].append(x_bnd[0])#0.5)
    bnd_tmp[1].append(29)#(x_bnd[1]-x_bnd[0])/2)
    bnd_tmp[0].append(bnd_tmp[0][1])
    bnd_tmp[1].append(bnd_tmp[1][1])
    # popt = deepcopy(popt_lst_BU[i])
    popt, pcov = curve_fit(SUM2_ConvLC_2Dpolyphase,inp_arr,z,bounds=bnd_tmp,absolute_sigma=True)#,sigma=np.ones(inp_arr[:,0].shape)*20)
    errs = np.sqrt(np.diag(pcov))
    popt_lst.append(popt.copy())
    errs_lst.append(errs.copy())
    # popt = [20.0,     10e1,            -0.0,   -34.0*np.pi/180,   -14 ]
    # popt = [  3.09876149, 116.87448495,   8.91707928 +(mjd-56654.2),  -0.47029305,  -5.38910457, -14.00613042]
    print('dust_width, PeakFlux, arcs_shift, phs_grad, add_const = '+str(popt))
    p01,p02,p04,p05,p06, add_const, p04_2,p02_2 = popt[0], popt[1], popt[2], 0, popt[3], popt[4], popt[5], popt[6]
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
        ax=fig.add_subplot(224,sharex=ax)
        # ax=fig.add_subplot(224,projection='3d',sharex=ax,sharey=ax)
    else:
        ax=fig.add_subplot(224)
        # ax=fig.add_subplot(224,projection='3d')
    
    x, y = inp_arr[:,0], inp_arr[:,1]
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y +mjd
    # ax.scatter(x,y,xy_phases,c='b',s=1)
    bbox=np.array([[x.min(),x.max()],[y.min(),y.max()]])
    plot_proj_phaseplane(ax,p04,p05,grad,bbox)
    ax.scatter(x,SUM2_ConvLC_2Dpolyphase(inp_arr,*popt),cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,c='r',s=1,label='fit')
    ax.plot(x,ConvLC_2Dpolyphase(inp_arr,p01,p02,p04,p06, add_const),c='b',linewidth=0.5,label='Sheet no. 1')
    ax.plot(x,ConvLC_2Dpolyphase(inp_arr,p01,p02_2,p04_2,p06, add_const),c='g',linewidth=0.5,label='Sheet no. 2')
    ax.scatter(x,z,cmap='jet',vmax=F_SCALE,vmin=-F_SCALE,s=1)
    
    # im=ax.scatter(x,y,cmap='jet',c=z-ConvLC_2Dpolyphase(inp_arr,*popt),s=1)
    # fig.colorbar(im,ax=ax,orientation="horizontal")
    ax.set_xlabel('x [arcsec]')
    ax.set_ylabel('y [arcsec]')
    ax.set_title('MJD: '+'{:.0f}'.format(mjd))
    
    if str(type(ax))=="<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
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
    
    ax.legend()

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
for pp in popt_lst: print('np.array('+str([ppp.round(decimals=3) for ppp in pp])+'),')

ss=''
for xl in x_cutoff: ss+=('['+str(np.array(xl).round(decimals=1)[0])+', '+str(np.array(xl).round(decimals=1)[1])+'], ')
print('x_cutoff = '+ss)

ss=''
for xl in y_cutoff: ss+=('['+str(np.array(xl).round(decimals=1)[0])+', '+str(np.array(xl).round(decimals=1)[1])+'], ')
print('y_cutoff = '+ss)
# %%
slope, intercept, r_value, p_value, std_err = stats.linregress(mjds.flatten(),peaks.flatten())
plt.figure()
plt.title('Apparent motion vs. Phase gradient')
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
plt.xlabel('[arcsec]')
plt.ylabel('MJD [days]')

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


# def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p05,p06,add_const):
def ConvLC_2Dpolyphase(input_array, p01,p02,p04,p06,add_const):
    p05=0*np.pi/180 # [rad]
    x, y, mjd, PSFs = input_array[:,0], input_array[:,1], input_array[:,2], input_array[:,3]
    
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y# +mjd
    xy_phases = np.cos(p05)*p06*(x-p04) +np.sin(p05)*p06*y
    
    phase_grad = p06#np.hypot(polyphase_params[0,1],polyphase_params[1,0])
    
    eff_phs_wid = np.hypot(p01,phase_grad*PSFs[0])
    ConvLC_params = [eff_phs_wid,p02,add_const]
    output_array = ConvLC(xy_phases, *ConvLC_params)
    
    return output_array

def SUM2_ConvLC_2Dpolyphase(input_array, p01,p02,p04,p06,add_const,p04_2,p02_2):
    p05=0*np.pi/180 # [rad]
    x, y, mjd, PSFs = input_array[:,0], input_array[:,1], input_array[:,2], input_array[:,3]
    
    phs_edges = SN_f(0,edge_query=True)
    
    # xy_phases = p04 +np.cos(p05)*p06*x +np.sin(p05)*p06*y# +mjd
    xy_phases1 = np.cos(p05)*p06*(x-p04) +np.sin(p05)*p06*y
    input_model_valid_ind = np.logical_and(xy_phases1>phs_edges[0],xy_phases1<phs_edges[1])
    
    
    xy_phases2 = np.cos(p05)*p06*(x-((p04_2))) +np.sin(p05)*p06*y
    input_model_valid_ind = np.logical_and(input_model_valid_ind,np.logical_and(xy_phases2>phs_edges[0],xy_phases2<phs_edges[1]))
    
    # xy_phases1[np.logical_not(input_model_valid_ind)] = 1000
    # xy_phases2[np.logical_not(input_model_valid_ind)] = 1000
    
    phase_grad = p06#np.hypot(polyphase_params[0,1],polyphase_params[1,0])
    
    eff_phs_wid = np.hypot(p01,phase_grad*PSFs[0])
    ConvLC_params = [eff_phs_wid,p02,add_const]
    output_array1 = ConvLC(xy_phases1, *ConvLC_params)
    
    ConvLC_params = [eff_phs_wid,p02_2,0]
    output_array2 = ConvLC(xy_phases2, *ConvLC_params)
    
    return output_array1 + output_array2

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
DIFF_df_BU.iloc[:,8] = pd.Series(data=np.zeros((len(DIFF_df_BU),)))
# DIFF_df_BU.iloc[:,8] = pd.Series(data=np.array([7.5,3.7,8.3,5.4]))#([8.4]*15))#
for i in np.arange(len(DIFF_df_BU)): DIFF_df_BU.iloc[i,10]=LEtb.pix2ang(DIFF_df_BU.iloc[i]['WCS_w'],DIFF_df_BU.iloc[i,8])

#%%

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
                im.set_clim(clim[0],clim[1]+0.9*span)
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

star_masks = world2pix_list(coord_list,Orgs[0],PA[0],verbose=False)


# %%

# ===== JUNK YARD =====

# files=[]

# roots=['/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/']

# files1=['plhstproc/tyc4419/1/tyc4419.VR.20120118.216797_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120118.216798_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20120624.24651_stch_1.sw',
#         'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120624.24652_stch_5.sw',
#         'plhstproc/tyc4419/1/tyc4419.VR.20120626.24973_stch_1.sw',
        # 'armin/leKeck/tyc4419a1/FIXED/tyc4519.VR.20120626.24974_stch_5.sw',
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


# roots=['/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/']
# roots=['/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4419/2/']
# files1=['tyc4419.VR.20140621.103815_stch_2.sw',
#         'tyc4419.VR.20140827.48448_stch_2.sw',
#         'tyc4419.VR.20140923.149398_stch_2.sw',
#         'tyc4419.VR.20141222.54623_stch_2.sw']
# files1=['crab3965.090916.3100_4',
#         'crab3965.VR.20111223.116410_stch_4']

# files1=['KECK/tyc4419_1_DEIMOS_5_6_coadd',
#         '20131202/tyc4419_tyc4519_20131202_coadd',
#         'KECK/tyc4419_1_R_LRIS_131228_coadd',
#         '20140226/tyc4419_tyc4519_20140226_coadd']

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


# roots=['/Users/roeepartoush/Documents/Research/Data/DOWNLOAD/plhstproc/tyc4620/7/']
# files1=['tyc4620.r.20140226.034427_stch_7.sw',
#         'tyc4620.VR.20140531.105857_stch_7.sw',
#         'tyc4620.VR.20140621.103205_stch_7.sw',
#         'tyc4620.VR.20140827.48446_stch_7.sw',
#         'tyc4620.VR.20140923.149418_stch_7.sw']

# files1=['tyc4820.090915.2086_5.sw',
#         'tyc4820.091217.441_5.sw',
#         'tyc4820.VR.20101110.102797_stch_5.sw',
#         'tyc4820.VR.20110922.11223_stch_5.sw',
#         'tyc4820.VR.20120117.116607_stch_5.sw',
#         'tyc4820.VR.20120220.1526891_stch_5.sw',
#         'tyc4820.VR.20120621.24306_stch_5.sw',
#         'tyc4820.VR.20130109t.1692991_stch_5.sw',
#         'tyc4820.VR.20130610.32389_stch_5.sw',
#         'tyc4820.VR.20130826.33227_stch_5.sw',
#         'tyc4820.r.20131202.135752_stch_5.sw',
#         'tyc4820.VR.20140621.102501_stch_5.sw',
#         'tyc4820.VR.20140827.48444_stch_5.sw',
#         'tyc4820.VR.20150213.357723_stch_5.sw',
#         'tyc4820.VR.20150814.64478_stch_5.sw']

# for file in files1:
#     files.append(roots[0]+file)