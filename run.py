#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:55:23 2019

@author: roeepartoush
"""

# base imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib
from scipy.optimize import curve_fit

# astropy imports
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.time import Time
from astropy import wcs
from astropy.visualization import ZScaleInterval

# local modules
import File2Data as F2D
import DustFit as DFit
import LEtoolbox as LEtb
import LEplots

# %%
## === LOAD DIFF IMAGES ===========
#home_dir = '/Users/roeepartoush/Downloads/Roee/'#2/'
#prefix = 'tyc4419.VR.20120118.216797_stch_'
#midfix = ''#_'#'_R.'
#sufix = ''#_2.diff'
#tmplt_img = ''#20140923.149398'#'20150211.157400'
#
#image_files = ['1','2','3','4','5','6','7','8']#['20120118.216797'    ,
##               '20120626.24973'     ,
##               '20130109t.1692969'  ,
##               '20130826.33218'     ,
##               '20140621.103815'    ,
##               '20141223.54713'     ,
##               '20150211.157395'    ,
##               '20150814.64482'     ,
##               '20150911.165399'     ]
##plt.close('all')
#flnm_lst=[]
#for ind in np.arange(8):#len(image_files)):
#    flnm_lst.append(home_dir+prefix+image_files[ind]+midfix+tmplt_img+sufix)
#DIFF_df = F2D.FitsDiff(flnm_lst)

#home_dir = 'https://stsci-transients.stsci.edu/atlas/cand/ROEE/CAS_a_DIFF/tyc3925/8/'
#image_files=urllib.request.urlopen(home_dir+'fileslist.txt').read().decode("utf-8").split('\n')
#image_files=image_files[0:-1]
#files = image_files
#for i in np.arange(len(files)): files[i]=home_dir+image_files[i]
home_dir='/Users/roeepartoush/Desktop/STSCI/tyc3923/4/'
files=open(home_dir+'fileslist.txt').read().split()
for i in np.arange(len(files)): files[i]=home_dir+files[i]
DIFF_df_BU = F2D.FitsDiff(files)
# %%
#inds=[manager.canvas.figure.number-1 for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
#DIFF_df=DIFF_df_BU.iloc[inds]
DIFF_df=DIFF_df_BU
plt.close('all')
global coord_list
coord_list=[]
figures=[plt.figure() for i in DIFF_df.index]
managers=[manager for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for mng in managers: mng.window.showMaximized()
LEplots.imshows(DIFF_df,REF_image=None,g_cl_arg=coord_list,FullScreen=True,med_filt_size=3,figs=figures)#,profDF=slitFPdf,prof_sampDF_lst=FP_df_lst2,fluxSpace='LIN')

# %%
Zscale = ZScaleInterval()
figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
for fig in figures:
    print(fig.number)
    for ax in fig.get_axes():
        print(ax.title.get_text())
        for im in ax.get_images():
            clim = Zscale.get_limits(im.get_array())
            print(im.get_clim())
#            cvmin, cvmax = im.get_clim()
#            inc = float(10)
#            if pchr=='a':
#                inc = -inc
#            elif pchr=='d':
#                inc = inc
            if fig.number>-6e6:
                im.set_clim(clim[0],clim[1])
                fig.canvas.draw()

# %%
ams = [0.07,0.10,0.13]
files=DIFF_df['filename'].to_list()
#global ev
#ev=[]
trkrs=[]
Ms_lst=[]
WCSs_lst=[]
for am in ams:
    Ms, WCSs = LEplots.imshows_shifted(DIFF_df,PA=None,app_mot=am*u.arcsec/u.day,ref_ind=0,med_filt_size=5,share_ax=None,plot_bl=False,downsamp=True)
    Ms_lst.append(Ms)
    WCSs_lst.append(WCSs)
# %%
for i in np.arange(len(ams)):
    plt.figure('App. motion = '+str(ams[i]))
    trkrs.append(LEplots.IndexTracker(ax=None, X=Ms_lst[i].copy(), titles=files, w_lst=WCSs_lst[i].copy()))

#%%
# xxxxz=== LOAD LIGHT CURVES ==========
#LChome_dir = '/Users/roeepartoush/Documents/Astropy/Data/light curves'
#LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
#LCtable = F2D.LightCurves(LCs_file)

# === DEFINE FLUX PROFILE AXIS ===
#ra = Longitude(['0h49m21.6s'], unit=u.hourangle)
#dec = Latitude(['58°43′3.5"'], unit=u.deg)
#ra = Longitude([345.65522446], unit=u.deg)
#dec = Latitude([56.81945503], unit=u.deg)
#ra = Longitude([12.26663719,12.31493338,12.24598815,12.20174356,12.15089583,12.18478501,12.21745521],unit=u.deg)#[12.21707640,12.22903355], unit=u.deg)
#dec = Latitude([58.71261933,58.71662884,58.70995621,58.69926131,58.68778571,58.68969553,58.70153964],unit=u.deg)#[58.70204724,58.70520108], unit=u.deg)
ra = Longitude(np.array(coord_list).reshape((len(coord_list),2))[:,0],unit=u.deg)
dec = Latitude(np.array(coord_list).reshape((len(coord_list),2))[:,1],unit=u.deg)
Orgs = SkyCoord(ra, dec, frame='fk5')
SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5')
#SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5')
PA = Angle([Org.position_angle(SN_sc)+Angle(180,'deg') for Org in Orgs])#Angle([62,0],u.degree)#Org.position_angle(SN_sc)+
Ln = Angle([12  for Org in Orgs],u.arcsec)
clmns = ['Orig', 'PA', 'Length']
slitFPdf = pd.DataFrame(index=np.arange(len(Orgs)), columns=clmns, data = [(Orgs[i],PA[i],Ln[i]) for i in np.arange(len(Orgs))])

#global DIFF_df
#DIFF_df= DIFF_df_BU_TEMP_3824_1.iloc[np.argwhere(boolinds_3824_1)[:,0]]
#DIFF_df_BU = F2D.FitsDiff(files)
#DIFF_df_BU = DIFF_df.copy()
#DIFF_df = DIFF_df_BU.iloc[51:55]
DIFF_df = DIFF_df_BU_4419_2.iloc[np.argwhere([item in moshe_t for item in DIFF_df_BU_4419_2['filename'].to_list()])[:,0]].copy()
#DIFF_df_tmp = DIFF_df_BU_TEMP_4419_3.iloc[np.argwhere([item in moshe_t for item in DIFF_df_BU_TEMP_4419_3['filename'].to_list()])[:,0]]#.copy()
#DIFF_df = DIFF_df_BU5.iloc[np.argwhere(boolinds5)[:,0]]
#DIFF_df = LEtb.mask_intersect(DIFF_df)
#DIFF_df = DIFF_df_BU5.iloc[moshe0[0:5]]#[20:25]]#[0:10]]
#DIFF_df = DIFF_df.append(DIFF_df_BU2.iloc[moshe0[10:20]])
#DIFF_df = DIFF_df.append(DIFF_df_BU2.iloc[27])
#DIFF_df = DIFF_df.append(DIFF_df_BU.iloc[13])
#DIFF_df = DIFF_df.append(DIFF_df_BU.iloc[40])


# === DETECT APPARENT MOTION =====
# currently assuming len(slitFPdf)=1
Wid = Angle(1,u.arcsec)
FP_df_lst2 = LEtb.getFluxProfile(DIFF_df, slitFPdf, width=Wid, REF_image=None)
over_fwhm = Angle(0.0,u.arcsec)
FP_df_PSFeq_lst = LEtb.getFluxProfile(DIFF_df, slitFPdf,PSFeq_overFWHM=over_fwhm,width=Wid,REF_image=stam)

#inds=np.array([13,14,15,16,18,19,21,24,28])#[26,27,28])-17#,29,30,31,33,34,36])-17#[8,9,10,11,12,13,15,16,18]
#inds=[8, 9, 10, 11, 12, 13, 18, 21, 25]
#FP_df_lst2_test = LEtb.getFluxProfile(DIFF_df.iloc[inds], slitFPdf, width=Wid, REF_image=None, N_bins=12)
stam = np.zeros((2200,4200))
#
#FP_df_lst4 = copy.deepcopy(FP_df_lst)
#FP_df_PSFeq_lst=LEtb.addPeakLoc(FP_df_PSFeq_lst)
# %%
FP_df_lst2 = LEtb.addPeakLoc(FP_df_lst2)
x = np.array(FP_df_lst2[0]['Peak_ProjAng_arcsec']).astype(float)
y = np.array(DIFF_df.iloc[inds]['Idate']).astype(float)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.figure()
plt.scatter(x,y,s=1)
plt.plot(x,slope*x+intercept)
plt.xlabel('Peak location [arcsec]')
plt.ylabel('Image date [mjd]')

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
FPdfLST = FP_df_lst2_test.copy()
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold', 'font.family':'serif'})
iii=7
y=FPdfLST[0].iloc[iii]['FluxProfile_ADUcal'].copy()
x=FPdfLST[0].iloc[iii]['ProjAng'].copy().arcsec*slope
xfit=-x.copy()#+43# +56
yfit=y.copy()
inds_nan=np.argwhere(np.isnan(yfit))[:,0]
xfit=np.delete(xfit,inds_nan)
yfit=np.delete(yfit,inds_nan)
phs_res = 0.1

tbl_ind=0
fL=LCtable['func_L'][tbl_ind]
phases=LCtable['phases'][tbl_ind]
phs_min = np.nanmin(phases)
phs_max = np.nanmax(phases)
def ConvLC(x, phs_wid, PeakFlux, phs_shft):
    y_p_unscaled, x_p = DFit.LCgrid_eval(fL,phs_wid,phs_res,np.array([phs_min,phs_max]))
    
    ref_FWHM_phshft = 0
#    if ref_FWHM_phs:
#        y_p_max_FWHM, x_p_max_FWHM = LCgrid_eval(LCfunc,np.sqrt(ref_FWHM_phs**2+phs_wid**2),phs_res,phases_p)
#        ref_FWHM_phshft = x_p_max_FWHM[np.nanargmax(y_p_max_FWHM)]
    
    const = PeakFlux/fL(0) # normalize maximun luminosity to PeakFlux
    x_arr=np.array(x)
    x_arr = x_arr + phs_shft
    x_arr = x_arr+0*x_p[np.nanargmax(y_p_unscaled)] +ref_FWHM_phshft
    y = np.interp(x_arr, x_p, const*y_p_unscaled, left=0, right=0)
#    plt.plot(np.array(x),y,label=('phs_wid='+str(int(phs_wid))+', PeakFlux='+str(int(PeakFlux))+', phs_shft='+str(int(phs_shft))+', ref_FWHM_phshft='+str(int(ref_FWHM_phshft))))
    return y
par_init = [0,140,0]
bnd = (np.array([   0,   0,-1000]),
       np.array([1000,1000, 1000]))

HDU=DIFF_df.iloc[inds[iii]]['Diff_HDU']
w = wcs.WCS(HDU.header)
popt,pcov = curve_fit(ConvLC,xfit,yfit,bounds=bnd,p0=par_init)
fig=plt.figure()
#ax_img.append(plt.subplot2grid((3,1),(0,0),rowspan=2,projection=w))
ax1=plt.subplot2grid((2,2),(0,0),rowspan=2,projection=w)
plt.imshow(HDU.data*1.0/float(HDU.header['KSUM00']),cmap='gray',vmin=-125,vmax=125)
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc=None, ec=None, lw=2)
t = ax1.text(1263, 1874, "Tycho", ha="center", va="center", rotation=45,
            size=20,
            bbox=bbox_props)
plt.xlim([1200,1300])
plt.ylim([1770,1900])
#divider = make_axes_locatable(ax1)
#cax = divider.append_axes("left", size="5%", pad=0.05)
#cax.get_yaxis().set_visible(False)
#plt.colorbar(ax1.get_images()[0], cax=cax)
#plt.colorbar()

plt.xlabel('RA')
plt.ylabel('DEC')
xy=FPdfLST[0].iloc[iii]['WrldCorners']
plt.plot(xy[:,0], xy[:,1], transform = ax1.get_transform('world'), linewidth=2, c='b')
xy=FPdfLST[0].iloc[iii]['WrldCntrs']
for i in np.arange(xy.shape[0]):
    plt.scatter(xy[i,0], xy[i,1], transform = ax1.get_transform('world'), s=75)

ax2=plt.subplot2grid((2,2),(0,1),rowspan=1)
plt.scatter(xfit+popt[2]-7.6,yfit,s=1,label='Flux profile')
par_init[1:2] = popt[1:2]
dust_wid=np.sqrt((popt[0]**2)-((DIFF_df.iloc[inds[iii]]['FWHM_ang'].arcsec*slope)**2))
plt.plot(np.sort(xfit+popt[2])-7.6,ConvLC(np.sort(xfit),*popt),label='fit, FWHM='+'{:2.0f}'.format(popt[0])+' [days]')#'\u221D'
plt.plot(phases[0:-1],ConvLC(np.sort(phases[0:-1]),*par_init),linewidth=1,c='r',label='model light curve')#, dm15='+str(LCtable['dm15'][tbl_ind]),c='r')
#plt.xlabel('time [days]')
plt.ylabel('Flux [ADU]')
xfit_lim=np.array([-50,150])
plt.xlim(xfit_lim)
plt.gca().legend()
ax3=ax2.twiny()
xxx = ((xfit+popt[2])-0*popt[2])/slope
xxx_lim = ((xfit_lim+popt[2])-1*popt[2])/slope
plt.xlim(xxx_lim)
plt.scatter(xxx,yfit,s=0,c='r')
plt.xlabel('Distance from peak [arcsec]')

ax4=plt.subplot2grid((2,2),(1,1),rowspan=1,sharex=ax2)
#ax5=ax4.twiny()
cntrs=FPdfLST[0].iloc[iii]['FP_Ac_bin_x'][::2] # arcsec
cntrs_phs = (-cntrs)*slope+popt[2]
bin_wid = np.mean(np.diff(cntrs))*slope # days
FWHMoSIG = 2*np.sqrt(2*np.log(2))
for i in np.arange(cntrs.shape[0]):
    start = cntrs_phs[i]-bin_wid-3*dust_wid
    stop = cntrs_phs[i]+bin_wid+3*dust_wid
    x=np.linspace(start,stop,num=int((stop-start)/phs_res)+1)
    y_rect = 1.0*(np.abs(x-cntrs_phs[i])<(bin_wid/2))
    y = ndimage.gaussian_filter(y_rect,sigma=(dust_wid/FWHMoSIG)/phs_res)
    plt.plot(x,y/y.max())
plt.xlabel('Time from peak [days]')
plt.ylabel('Weight [1]')

fig.subplots_adjust(hspace=0)
ax2.get_xaxis().set_visible(False)
#start = 0-bin_wid-3*dust_wid
#stop = 0+bin_wid+3*dust_wid
#x=np.linspace(start,stop,num=int((stop-start)/phs_res)+1)
#y_rect = 1.0*(np.abs(x-0*cntrs_phs[i])<(phs_res*2))
#y = ndimage.gaussian_filter(y_rect,sigma=(dust_wid/FWHMoSIG)/phs_res)
#plt.plot(x,y/y.max(),c='k',linewidth=1)
# %%
#from skimage.feature import masked_register_translation
#inds=
#detected_shift = masked_register_translation(src_image, target_image, src_mask, target_mask=None)#, overlap_ratio=0.3)
# %%
#inds=[11,12]
#colors=['r','b']
#fig=plt.figure()
#for ind in inds:
#    HDU = DIFF_df.iloc[ind]['Diff_HDU']
#    X,Y = np.meshgrid(np.arange(HDU.data.shape[1])[::10],np.arange(HDU.data.shape[0])[::10])
#    pix_vec = np.concatenate((X.reshape((X.size,1)),Y.reshape((Y.size,1))),axis=1)
#    w = wcs.WCS(HDU.header)
#    if ind==inds[0]:
#        fig.add_subplot(111,projection=w)
#    wrld_vec = w.wcs_pix2world(pix_vec,1)
#    plt.scatter(wrld_vec[:,0],wrld_vec[:,1],s=1,c=colors[np.mod(ind,len(colors))],transform = plt.gca().get_transform('world'))


# %%
DIFF_df.drop(columns='xyz_list',inplace=True)
SN_sc = SkyCoord('23h23m24s','+58°48.9′',frame='fk5')
SN_sc = SkyCoord('0h25.3m','+64° 09′',frame='fk5')
#yd=Time('2010-01-01T00:00:00')-Time('2009-01-01T00:00:00')
SN_time = Time('1572-11-16T00:00:00')# -300*yd
D_ly = 8900#11090
DIFF_df = LEtb.LE_DF_xyz(DIFF_df,SN_sc,SN_time,D_ly)

# %%
from scipy import ndimage
from PIL import  Image
from scipy import special
from scipy import stats
from astropy.visualization import ZScaleInterval
Zscale = ZScaleInterval()

sigma=1
N=300
mat = np.random.normal(0,sigma,(N,N))

hist_vals, hist_cnts = np.unique(mat,return_counts=True)
hist_cnts_cum = np.cumsum(hist_cnts)
fig = plt.figure()
fig.add_subplot(311)
plt.imshow(mat,cmap='gray')
#plt.plot(mat)
ax1=fig.add_subplot(312)
plt.scatter(hist_vals,hist_cnts_cum/hist_cnts_cum[-1],s=1)
#plt.plot(hist_vals[:-1],np.divide(np.diff(hist_cnts_cum),np.diff(hist_vals)))
plt.grid(color='black', ls='solid')
#plt.scatter(hist_vals,0.5+special.erf(hist_vals*(np.sqrt(0.5)/sigma))/2,s=1,c='r')
rr = 1#0.05
plt.gca().axvline(x=rr*sigma,color='m')
plt.gca().axvline(x=-rr*sigma,color='m')
#plt.gca().axhline(y=special.erf(rr*0.5)/2,color='m')
plt.gca().axhline(y=(0.5+special.erf(rr*np.sqrt(0.5))/2),color='m')
plt.gca().axhline(y=(0.5-special.erf(rr*np.sqrt(0.5))/2),color='m')
hist_res = 0.01
Nbins = 500#1+int((hist_vals.max()-hist_vals.min())/hist_res)
statistic, bin_edges, _ = stats.binned_statistic(hist_vals,hist_cnts_cum/hist_cnts_cum[-1],statistic='mean',bins=Nbins)
fig.add_subplot(313,sharex=ax1)#,sharey=ax1)
bin_cntrs = (bin_edges[:-1]+bin_edges[1:])/2
plt.scatter(bin_edges[1:-1],np.diff(statistic)/np.diff(bin_cntrs),s=1)
plt.grid(color='black', ls='solid')

mat_grad = np.sqrt(ndimage.sobel(mat/8,axis=1)**2 + ndimage.sobel(mat/8,axis=0)**2)
hist_vals, hist_cnts = np.unique(mat_grad,return_counts=True)
hist_cnts_cum = np.cumsum(hist_cnts)
fig = plt.figure()
fig.add_subplot(311)
plt.imshow(mat_grad,cmap='gray')
ax1=fig.add_subplot(312)
plt.scatter(hist_vals,hist_cnts_cum/hist_cnts_cum[-1],s=1)
plt.grid(color='black', ls='solid')
rr = 1
plt.gca().axvline(x=rr*sigma,color='m')
plt.gca().axvline(x=-rr*sigma,color='m')
plt.gca().axhline(y=(0.5+special.erf(rr*np.sqrt(0.5))/2),color='m')
plt.gca().axhline(y=(0.5-special.erf(rr*np.sqrt(0.5))/2),color='m')
hist_res = 0.05
Nbins = 500#1+int((hist_vals.max()-hist_vals.min())/hist_res)
statistic, bin_edges, _ = stats.binned_statistic(hist_vals,hist_cnts_cum/hist_cnts_cum[-1],statistic='mean',bins=Nbins)
fig.add_subplot(313,sharex=ax1)#,sharey=ax1)
bin_cntrs = (bin_edges[:-1]+bin_edges[1:])/2
pdf_y = np.diff(statistic)/np.diff(bin_cntrs)
pdf_x = bin_edges[1:-1]
plt.scatter(pdf_x,pdf_y,s=1,c='r')
pdf_y = ndimage.gaussian_filter(pdf_y,sigma=10)
plt.plot(pdf_x,pdf_y)
plt.grid(color='black', ls='solid')
peak_x = pdf_x[np.nanargmax(pdf_y)]
plt.gca().axvline(x=peak_x,color='m')
plt.xlabel('peak_x = '+'{:6.4f}'.format(peak_x/sigma)+'\u03C3')

#im=Image.open('/Users/roeepartoush/Downloads/WhatsApp Image 2019-10-06 at 08.13.22.jpeg')
#r,g,b=im.split()
#fig=plt.figure()
#ax_r = fig.add_subplot(131)
#plt.imshow(r,vmin=0,vmax=255,cmap='gray')
#fig.add_subplot(132,sharex=ax_r,sharey=ax_r)
#r_arr = np.array(r,dtype=float)
#mat = ndimage.filters.laplace(r_arr)
#clim = Zscale.get_limits(mat)
#plt.imshow(mat,vmin=clim[0],vmax=clim[1],cmap='gray')
#fig.add_subplot(133,sharex=ax_r,sharey=ax_r)
#mat = ndimage.sobel(r_arr,axis=1)
#clim = Zscale.get_limits(mat)
#plt.imshow(mat,vmin=clim[0],vmax=clim[1],cmap='gray')
# %%
from scipy import ndimage
N=1000
X,Y = np.meshgrid(np.arange(N)-N/2,np.arange(N)-N/2)
G1 = np.exp(-((X-20)**2+(Y)**2)/(100**2))
Z=G1-np.fliplr(G1)

plt.figure()
plt.imshow(Z)

Zx = ndimage.sobel(Z,axis=1)
Zy = ndimage.sobel(Z,axis=0)
plt.figure()
plt.imshow(np.sqrt(Zx**2+Zy**2))

# %%
inds = [4,5,6,7]#[1,2]#[14,15]#[9,10,11]#
maxFWHM = Angle(list(DIFF_df.iloc[inds]['FWHM_ang'])).max()+Angle(0.0,u.arcsec)
x_pix = [0,4200]#[[1700, 1740]#[2002, 2125]#
y_pix = [0,2200]#[2000, 2035]#[1598, 1725]#
#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
mean_mat = np.zeros((y_pix[1]-y_pix[0],x_pix[1]-x_pix[0]))
ax=[]
for ind in inds:
    mat = LEtb.matchFWHM(DIFF_df,ind,maxFWHM)[y_pix[0]:y_pix[1],x_pix[0]:x_pix[1]]*1.0/float(DIFF_df.iloc[ind]['Diff_HDU'].header['KSUM00'])
#    mat = DIFF_df.iloc[ind]['Diff_HDU'].data[y_pix[0]:y_pix[1],x_pix[0]:x_pix[1]]*1.0/float(DIFF_df.iloc[ind]['Diff_HDU'].header['KSUM00'])
    mean_mat = mean_mat + mat
    X,Y = np.meshgrid(np.arange(x_pix[1]-x_pix[0])+x_pix[0],np.arange(y_pix[1]-y_pix[0])+y_pix[0])
    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', LEplots.press)
    w = wcs.WCS(DIFF_df.iloc[ind]['Diff_HDU'].header)
    if ind==inds[0]:
        ax.append(fig.add_subplot(111,projection=w))
    else:
        ax.append(fig.add_subplot(111,sharex=ax[0],sharey=ax[0],projection=w))
    plt.imshow(mat,cmap='gray',vmin=-150,vmax=150)
#    ax=fig.add_subplot(111,projection='3d')
#    ax.scatter(X,Y,mat,s=1)
fig=plt.figure()
fig.canvas.mpl_connect('key_press_event', LEplots.press)
ax.append(fig.add_subplot(111,sharex=ax[0],sharey=ax[0],projection=w))
plt.imshow(mean_mat/len(inds),cmap='gray',vmin=-150,vmax=150)
#ax=fig.add_subplot(111,projection='3d')
#ax.scatter(X,Y,mean_mat/len(inds),s=1)
# %%
#inds_avg=[[0],[1],[2],[3,4,5,6],[7,8,9,10,11],[12,13,14,15],[16],[17],[18,19],[20],[21,22],[23],[24,25],[26,27],[28],[29,30],[31],[32],[33]]
inds_avg=[[42,43,44],[45],[46,47],[48]]
#maxFWHM = Angle(list(DIFF_df['FWHM_ang'])).max()
ax=[]
for inds_lst in inds_avg:
    mat = np.zeros((2200,4200))
    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', LEplots.press)
    maxFWHM = Angle(list(DIFF_df.iloc[inds_lst]['FWHM_ang'])).max()
    for ind in inds_lst:
        mat = mat + (LEtb.matchFWHM(DIFF_df,ind,maxFWHM)*1.0/float(DIFF_df.iloc[ind]['Diff_HDU'].header['KSUM00']))# - mean_image
    mat=mat/len(inds_lst)
    clim = [-150,150]#Zscale.get_limits(mat)
    w=wcs.WCS(DIFF_df.iloc[ind]['Diff_HDU'].header)
    if inds_lst==inds_avg[0]:
        ref_mat = mat.copy()
        ax.append(fig.add_subplot(111,projection=w))
    else:
        ax.append(fig.add_subplot(111,sharex=ax[0],sharey=ax[0],projection=w))
    plt.imshow(mat,vmin=clim[0],vmax=clim[1],cmap='gray')
#    plt.imshow(np.log10(np.abs(mat/ref_mat)),vmin=clim[0],vmax=clim[1],cmap='gray')
    plt.gca().title.set_text(DIFF_df.iloc[ind]['filename'])
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
# %%
inds = [8,9]
maxFWHM = Angle(list(DIFF_df.iloc[inds]['FWHM_ang'])).max()

#mat1 = (DIFF_df.iloc[inds[0]]['Diff_HDU'].data*1.0/float(DIFF_df.iloc[inds[0]]['Diff_HDU'].header['KSUM00']))
#mat2 = (DIFF_df.iloc[inds[1]]['Diff_HDU'].data*1.0/float(DIFF_df.iloc[inds[1]]['Diff_HDU'].header['KSUM00']))
mat1 = (LEtb.matchFWHM(DIFF_df,inds[0],maxFWHM)*1.0/float(DIFF_df.iloc[inds[0]]['Diff_HDU'].header['KSUM00'])) - mean_image
mat2 = (LEtb.matchFWHM(DIFF_df,inds[1],maxFWHM)*1.0/float(DIFF_df.iloc[inds[1]]['Diff_HDU'].header['KSUM00'])) - mean_image
mat_div = mat2/mat1
plt.figure()
plt.imshow(np.log10(np.abs(mat_div)),cmap='gray',vmin=-0.25,vmax=0.25)
# %%
files = ['2017jghCleaned.csv']#,'1993j.csv']#,'2016gkg.csv']
LC_df=pd.read_csv('/Users/roeepartoush/Documents/Astropy/Data/light curves/'+files[0]).rename(columns={'name': 'event'})
for file in files[1:]:
    LC_df = pd.concat((LC_df,pd.read_csv('/Users/roeepartoush/Documents/Astropy/Data/light curves/'+file)))
plt.figure()
for event in pd.unique(LC_df['event']):
    LC_df_event = LC_df[LC_df['event']==event]
    if event=='SN1993J':
        phs_ofs = 6 + (-3)
        mag_ofs = 10
    elif event=='SN2017jgh':
        phs_ofs = 20 + (-6.5)
        mag_ofs = 0
    elif event=='SN2016gkg':
        phs_ofs = 3067*0
        mag_ofs = 0
    for band in pd.unique(LC_df['band']):
#        if band=='V' or band=='R' or band=='kepler-clean':
        inds = LC_df_event['band']==band
        if (not any(inds)) or not (band=='V' or band=='R' or band=='kepler-clean'):#event!='SN1993J':# or band!='kepler-clean':
            continue
#        instr = pd.unique(LC_df.loc[inds]['instrument'])[0]
        phases = LC_df_event.loc[inds]['time'].to_numpy()
        phases = phases-phases[0]-phs_ofs
        mags = LC_df_event.loc[inds]['magnitude'].to_numpy()+mag_ofs
        fM,fL=F2D.LCdata2func(phases,mags,smooth_lin=True)
        LF_c = fL(phases)
#        LF_c = DFit.Mag2ADU(LF_c,inverse=True)
        plt.plot(phases,LF_c,label=str(event)+', filter='+str(band),c='r')
#        plt.scatter(phases,DFit.Mag2ADU(mags)/np.nanmax(DFit.Mag2ADU(mags)),s=1)
        plt.ylim([np.nanmin(LF_c),np.nanmax(LF_c)])
# %%
#y=FP_df_PSFeq_lst[0].iloc[8]['FP_Ac_bin_y']
y=FP_df_lst2[0].iloc[8]['FluxProfile_ADUcal']
y=y/100#np.nanmax(y)#/(1e10)
#x=slope*FP_df_PSFeq_lst[0].iloc[8]['FP_Ac_bin_x']+intercept
x=slope*(FP_df_lst2[0].iloc[8]['ProjAng'].arcsec-40)#+intercept
#x = x-x[np.nanargmax(y)]
xfit = -x[np.abs(x)<200]
yfit = y[np.abs(x)<200]
plt.scatter(xfit,yfit,label='LE',s=0.1)
plt.ylim([np.nanmin(y),np.nanmax(y)])


# %%
band='V'
mags = LC_df.loc[LC_df['band']==band]['magnitude'].to_numpy()
phases = LC_df.loc[LC_df['band']==band]['time'].to_numpy()
fM,fL=F2D.LCdata2func(phases,mags)
plt.figure()
plt.plot(phases-phases[np.nanargmin(mags)],1e6*fL(phases-phases[np.nanargmin(mags)]),c='m')
y=FP_df_lst2[0].iloc[8]['FP_Ac_bin_y']
x=slope*FP_df_lst2[0].iloc[8]['FP_Ac_bin_x']+intercept
x = x-x[np.nanargmax(y)]
plt.plot(-x,y)
# %%
def Xcorr_fit_fun(vec,a,b,c,k,j):
    x=vec[:,0]
    y=vec[:,1]
    
#    z = -a*((y-b*x-c)**2)
    xxx=y-b*x-c
    z = k*np.exp(-(xxx**2)/(2*(a**2)))+j
    return z

def Xcorr_fit_fun1d(x,a,b,k):
    xxx=x-b
    z = k*np.exp(-(xxx**2)/(2*(a**2)))
    return z

X,Y = np.meshgrid(np.arange(50),np.arange(50))
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
VEC = np.concatenate((X.flatten().reshape((X.size,1)),Y.flatten().reshape((Y.size,1))),axis=1)
z=Xcorr_fit_fun(VEC,1,1,0,10,5)
#ax.scatter(X.flatten(),Y.flatten(),z,s=1)
x=X.flatten()#X[0,:]
y=Y.flatten()#Y[0,:]
z=Xcorr_fit_fun1d(np.sqrt((x-25)**2+(y-25)**2),10,0,1)
ax.scatter(x,y,z,s=1,c=z,cmap='jet')

# %%
I = np.arange(FP_Xcorr_df_lst[0].shape[0])
J = np.arange(FP_Xcorr_df_lst[0].shape[1])
inds = np.arange(len(FP_Xcorr_df_lst))
ax=[]
vec_t = []
zfit_t = []
plt.close('all')
for ind in inds:
    fig=plt.figure()
    fig.canvas.mpl_connect('key_press_event', LEplots.press)
#    if ind==inds[0]:
    ax.append(fig.add_subplot(111,projection='3d'))
#    else:
#        ax.append(fig.add_subplot(111,sharex=ax[0],sharey=ax[0],projection='3d'))
    for i in I:
        for j in J:
            mjd_diff = FP_Xcorr_argmat_lst[ind][i,j,3].copy()
            if mjd_diff>10 and mjd_diff<500:
                x = FP_df_lst2[ind].iloc[i]['FP_Ac_bin_x'].copy()
                y = np.ones(x.shape[0])*mjd_diff
                z = FP_Xcorr_df_lst[ind].iloc[i,j]
                t=-5#0.25
                tv=z
                ax[ind].scatter(x[tv>t],y[tv>t],z[tv>t],s=1,c=z[tv>t],vmin=-1,vmax=1,cmap='jet')
                xfit=x[tv>t]
                yfit=y[tv>t]
                zfit=z[tv>t]
                vec = np.concatenate((xfit.reshape((xfit.size,1)),yfit.reshape((yfit.size,1))),axis=1)
                par_init = [500,0,50]
                bnd = (np.array([0,-100,0]),
                       np.array([1000,100,100]))
                if zfit.size>0:
                    zfit_t.append(zfit)
                    vec_t.append(vec)
#                    popt,pcov = curve_fit(Xcorr_fit_fun1d,xfit,zfit,bounds=bnd,p0=par_init)
#                    zzz = Xcorr_fit_fun1d(xfit,*popt)
##                    print(popt)
#                    if not(all(popt==np.array(par_init))) and popt[-1]>0:
#                        ax[ind].scatter(xfit,yfit,zzz,s=1,c='m')
#                        ax[ind].scatter(popt[1],yfit[0],1.2,s=3,c='m')
#                        print(popt)
#                x=FP_Xcorr_argmat_lst[ind][i,j,2]
#                y=y[0]
#                z=np.log10(FP_Xcorr_argmat_lst[ind][i,j,1])
#                ax[ind].scatter(x,y,z,s=2,c='m')
#    vec=np.concatenate(np.array(vec_t),axis=0)
#    zfit=np.concatenate(np.array(zfit_t),axis=0)
#    par_init = [50,-20,0,1,0]
#    bnd = (np.array([0,-100,-np.inf,0,-np.inf]),
#           np.array([1000,0,np.inf,100,np.inf]))
#    popt,pcov = curve_fit(Xcorr_fit_fun,vec,zfit,bounds=bnd)#,p0=par_init)
#    zzz = Xcorr_fit_fun(vec,*popt)
##    zzz = Xcorr_fit_fun(vec,50,-20,0,1,0)
#    ax[ind].scatter(vec[:,0],vec[:,1],zzz,s=1,c='m',vmin=-1,vmax=1)
    
#%%
plt.figure()
ind=3
FP_Xcorr_df = FP_Xcorr_df_lst[ind]
FP_Xcorr_argmat = FP_Xcorr_argmat_lst[ind]
for i in np.arange(FP_Xcorr_df.shape[0]):
    for k in np.arange(FP_Xcorr_df.shape[1]):
#        b[i,k]=np.any(np.isnan(a[i,k]))
        mjd_diff = FP_Xcorr_argmat[i,k,3]
        if i<k and np.abs(mjd_diff)>2:
            plt.plot(FP_Xcorr_df.iloc[i,k])
# %%
moshe_t=['tyc4419.VR.20120118.216797_20140827.48448_2.diff',
         'tyc4419.VR.20120626.24973_20140827.48448_2.diff',
         'tyc4419.VR.20130109t.1692969_20140827.48448_2.diff',
         'tyc4419.VR.20130109t.1692994_20140827.48448_2.diff',
         'tyc4419.VR.20130609.32244_20140827.48448_2.diff',
         'tyc4419.VR.20130826.33218_20140827.48448_2.diff',
         'tyc4419.r.20131202.135756_20140827.48448_2.diff',
         'tyc4419.r.20131202.135757_20140827.48448_2.diff',
         'tyc4419.r.20140226.030626_20140827.48448_2.diff',
         'tyc4419.VR.20140531.105244_20140827.48448_2.diff',
         'tyc4419.VR.20140621.103815_20140827.48448_2.diff',
         'tyc4419.VR.20140923.149398_20140827.48448_2.diff',
         'tyc4419.VR.20141222.54623_20140827.48448_2.diff',
         'tyc4419.VR.20141222.54624_20140827.48448_2.diff',
         'tyc4419.VR.20141223.54713_20140827.48448_2.diff',
         'tyc4419.VR.20150211.157395_20140827.48448_2.diff',
         'tyc4419.VR.20150211.157399_20140827.48448_2.diff',
         'tyc4419.VR.20150212.257541_20140827.48448_2.diff',
         'tyc4419.VR.20150814.64482_20140827.48448_2.diff',
         'tyc4419.VR.20150911.165399_20140827.48448_2.diff']

moshe_t=['tyc2116.061020.033_091118.045_2.diff',
         'tyc2116.061216.032_091118.045_2.diff',
         'tyc2116.071012.067_091118.045_2.diff',
         'tyc2116.080903.110_091118.045_2.diff',
         'tyc2116.080903.111_091118.045_2.diff',
         'tyc2116.080903.117_091118.045_2.diff',
         'tyc2116.080903.118_091118.045_2.diff',
         'tyc2116.090914.1006_091118.045_2.diff',
         'tyc2116.VR.20101110.102775_091118.045_2.diff',
         'tyc2116.VR.20110922.11191_091118.045_2.diff',
         'tyc2116.VR.20111025.14282_091118.045_2.diff',
         'tyc2116.VR.20111025.14283_091118.045_2.diff',
         'tyc2116.VR.20111223.116359_091118.045_2.diff',
         'tyc2116.VR.20111223.116373_091118.045_2.diff',
         'tyc2116.VR.20120117.116577_091118.045_2.diff',
         'tyc2116.VR.20120117.116578_091118.045_2.diff',
         'tyc2116.VR.20120620.124130_091118.045_2.diff',
         'tyc2116.VR.20120622.24466_091118.045_2.diff',
         'tyc2116.VR.20130109t.1692338_091118.045_2.diff',
         'tyc2116.VR.20130605.31570_091118.045_2.diff',
         'tyc2116.VR.20130826.33189_091118.045_2.diff',
         'tyc2116.VR.20130827.33338_091118.045_2.diff',
         'tyc2116.VR.20140530.101653_091118.045_2.diff',
         'tyc2116.VR.20140530.101933_091118.045_2.diff',
         'tyc2116.VR.20140620.081116_091118.045_2.diff',
         'tyc2116.VR.20140621.090708_091118.045_2.diff',
         'tyc2116.VR.20140621.091139_091118.045_2.diff',
         'tyc2116.VR.20140827.48387_091118.045_2.diff',
         'tyc2116.VR.20140923.149324_091118.045_2.diff',
         'tyc2116.VR.20140923.149329_091118.045_2.diff',
         'tyc2116.VR.20141222.54590_091118.045_2.diff',
         'tyc2116.VR.20150814.64419_091118.045_2.diff']
inds=[0,
     1,
     2,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     21,
     22,
     23,
     24,
     28,
     31,
     33,
     34,
     35,
     36,
     37,
     38,
     39,
     40,
     41,
     42,
     43,
     44,
     45,
     46,
     47,
     48,
     49,
     51,
     52,
     54]
DIFF_df=DIFF_df_BU_2116_new_1.iloc[inds].copy()
inds=[(np.argwhere((DIFF_df_BU_2116_new_1['filename']==moshe_t[i]).to_numpy())[0,0]) for i in np.arange(len(moshe_t))]
DIFF_df_4mean = DIFF_df.iloc[[0,1,4,5,6,8,9,10,14]]#[1,5,7,10,11,15]]
DIFF_df_4mean = DIFF_df_BU_2116_new_1.iloc[[0,6,24,34,38,41,49,52]]
# %%
inds=[1,2,3]#[0,1,2,3,4,5,6]
colors = ['r','g','b','m','c','k','y']
J,I=np.meshgrid(np.arange(len(FP_Xcorr_df_lst[0])),np.arange(len(FP_Xcorr_df_lst[0])))
boolinds = I<=J
fig=plt.figure()
x=np.zeros(1)
x[0]=np.nan
y=np.zeros(1)
y[0]=np.nan
for ind in inds:
    
#    fig.canvas.mpl_connect('key_press_event', LEplots.press)
#    ax = fig.add_subplot(111)#, projection='3d')
#    x=np.concatenate((x,FP_Xcorr_argmat_lst[ind][:,:,3][boolinds].flatten()))
    x=FP_Xcorr_argmat_lst[ind][:,:,3][boolinds].flatten()
#    y=np.concatenate((y,FP_Xcorr_argmat_lst[ind][:,:,2][boolinds].flatten()))
    y=FP_Xcorr_argmat_lst[ind][:,:,2][boolinds].flatten()
    x=np.delete(x,[np.argwhere(np.isnan(y))[:,0]])
    y=np.delete(y,[np.argwhere(np.isnan(y))[:,0]])
#    plt.scatter(x,y,s=1)#np.log10(FP_Xcorr_argmat_lst[ind][:,:,1][boolinds].flatten()),s=1)
#    ax.title.set_text(ind)
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#    statistic1, bin_edges, binnumber = stats.binned_statistic(x,y,bins=20)
#    statistic2, bin_edges, binnumber = stats.binned_statistic(x,y,bins=20,statistic='std')
#    plt.errorbar(bin_edges[0:-1]+np.diff(bin_edges).mean()/2,statistic1,yerr=statistic2,c='r')
#    print('slope: '+str(slope))
#    print('intercept: '+str(intercept))
#    plt.plot(x,slope*x+intercept)
    plt.scatter(x,y,s=1,c=colors[ind])
#def R_prc40(x): return np.percentile(x,50)
#statistic1, bin_edges, binnumber = stats.binned_statistic(x,y,bins=20,statistic=R_prc40)
##statistic2, bin_edges, binnumber = stats.binned_statistic(x,y,bins=20,statistic='std')
#plt.plot(bin_edges[0:-1]+np.diff(bin_edges).mean()/2,statistic1)#,yerr=statistic2,c='r')
##slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#plt.plot(x,slope*x+intercept,c='m')
# %%
maxFWHM = Angle(list(DIFF_df['FWHM_ang'])).max()
#mean_image = DIFF_df_4mean.iloc[0]['Diff_HDU'].data*1.0/float(DIFF_df_4mean.iloc[0]['Diff_HDU'].header['KSUM00'])#*LEtb.get_zpt_lin(DIFF_df_4mean,0)#
mean_image = LEtb.matchFWHM(DIFF_df_4mean,0,maxFWHM+over_fwhm)*1.0/float(DIFF_df_4mean.iloc[0]['Diff_HDU'].header['KSUM00'])
#mean_image = DIFF_df_4mean.iloc[i]['Diff_HDU'].data*1.0/float(DIFF_df_4mean.iloc[i]['Diff_HDU'].header['KSUM00'])
mean_image = mean_image.reshape((mean_image.shape[0],mean_image.shape[1],1))
MaskMAT = DIFF_df_4mean.iloc[0]['Mask_mat']
#mean_image[~LEtb.mask2bool(MaskMAT)] = np.nan
for i in np.arange(len(DIFF_df_4mean)-1)+1:
#    mean_image = mean_image+DIFF_df_4mean.iloc[i]['Diff_HDU'].data*1.0/float(DIFF_df_4mean.iloc[i]['Diff_HDU'].header['KSUM00'])#*LEtb.get_zpt_lin(DIFF_df_4mean,i)#
    mean_image = np.concatenate((mean_image,(LEtb.matchFWHM(DIFF_df_4mean,i,maxFWHM+over_fwhm)*1.0/float(DIFF_df_4mean.iloc[i]['Diff_HDU'].header['KSUM00'])).reshape((mean_image.shape[0],mean_image.shape[1],1))),axis=2)
#    mean_image = np.concatenate((mean_image,(DIFF_df_4mean.iloc[i]['Diff_HDU'].data*1.0/float(DIFF_df_4mean.iloc[i]['Diff_HDU'].header['KSUM00'])).reshape((mean_image.shape[0],mean_image.shape[1],1))),axis=2)
    MaskMAT = DIFF_df_4mean.iloc[i]['Mask_mat']
#    mean_image[~LEtb.mask2bool(MaskMAT)] = np.nan
#mean_image=mean_image/(len(DIFF_df_4mean))
mean_image = np.mean(mean_image,axis=2)

# %%
colors = ['r','g','b','m','c','k']
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
fig=plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')
for i in np.arange(len(DIFF_df)):
    for xyz in DIFF_df.iloc[i]['xyz_list']:
        ax.scatter(xyz[0][0],xyz[0][1],xyz[0][2],s=1,c=colors[np.mod(i,len(colors))])
plt.xlabel('x')
plt.ylabel('y')
set_axes_equal(ax)
#plt.zlabel('z')
# %%
from mpl_toolkits.mplot3d import Axes3D
#plt.close('all')
x=DIFF_df_BU['FWHM'].to_numpy().astype(float)

y=DIFF_df_BU['M5SIGMA'].to_numpy().astype(float)

z=DIFF_df_BU['Idate'].to_numpy().astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors=np.zeros((x.shape[0],3))
boolinds = np.logical_and(x<4.5,y>23)
#colors[boolinds,1] = 1.0
colors[~boolinds,0] = 1.0
ax.scatter(x,y,z,s=1,c=colors)

# %%
boolinds_3824_1 = np.zeros(len(DIFF_df_BU_TEMP_3824_1))==1
for i in np.arange(len(boolinds_3824_1)):
    k=0
    while ~boolinds_3824_1[i]&(k<len(boolinds_3824_0)):
        if boolinds_3824_0[k]:
            tmp_str = DIFF_df_BU_TEMP_3824_1.iloc[i,0]
            comp_str = DIFF_df_BU_TEMP_3824_0.iloc[k,0]
            boolinds_3824_1[i] = tmp_str[0:tmp_str.find('_')]==comp_str[0:comp_str.find('_')]
            boolinds_3824_1[i] = boolinds_3824_1[i] & (tmp_str[tmp_str.find('_')+1:tmp_str.find('_',tmp_str.find('_')+1)]=='20140827.48417')
        else:
            pass
        k=k+1
# %%
boolinds2 = np.zeros(len(DIFF_df_BU2))==1
for i in np.arange(len(boolinds2)):
    k=0
    while ~boolinds2[i]&(k<len(boolinds)):
        if boolinds[k]:
            tmp_str = DIFF_df_BU2.iloc[i,0]
            comp_str = DIFF_df_BU.iloc[k,0]
            boolinds2[i] = tmp_str[0:tmp_str.find('_')]==comp_str[0:comp_str.find('_')]
        else:
            pass
        k=k+1

boolinds3 = np.zeros(len(DIFF_df_BU3))==1
for i in np.arange(len(boolinds3)):
    k=0
    while ~boolinds3[i]&(k<len(boolinds)):
        if boolinds[k]:
            tmp_str = DIFF_df_BU3.iloc[i,0]
            comp_str = DIFF_df_BU.iloc[k,0]
            boolinds3[i] = tmp_str[0:tmp_str.find('_')]==comp_str[0:comp_str.find('_')]
        else:
            pass
        k=k+1

boolinds4 = np.zeros(len(DIFF_df_BU4))==1
for i in np.arange(len(boolinds4)):
    k=0
    while ~boolinds4[i]&(k<len(boolinds)):
        if boolinds[k]:
            tmp_str = DIFF_df_BU4.iloc[i,0]
            comp_str = DIFF_df_BU.iloc[k,0]
            boolinds4[i] = tmp_str[0:tmp_str.find('_')]==comp_str[0:comp_str.find('_')]
        else:
            pass
        k=k+1

boolinds5 = np.zeros(len(DIFF_df_BU5))==1
for i in np.arange(len(boolinds5)):
    k=0
    while ~boolinds5[i]&(k<len(boolinds)):
        if boolinds[k]:
            tmp_str = DIFF_df_BU5.iloc[i,0]
            comp_str = DIFF_df_BU.iloc[k,0]
            boolinds5[i] = tmp_str[0:tmp_str.find('_')]==comp_str[0:comp_str.find('_')]
        else:
            pass
        k=k+1

# %%
        for fig in [plt.gcf()]:#figures:
            for ax in fig.get_axes():
                for im in ax.get_images():
#                    cvmin, cvmax = im.get_clim()
#                    inc = float(10)
#                    if pchr=='a':
#                        inc = -inc
#                    elif pchr=='d':
#                        inc = inc
                    im.set_clim(1430, 1550)
            fig.canvas.draw()
# %%
# === ESTIMATE DUST WIDTH ========
maxFWHM_phs = Angle(list(DIFF_df['FWHM_ang'])).max().arcsec*slope
ref_fwhm_phs = over_fwhm.arcsec*slope + maxFWHM_phs#np.sqrt((over_fwhm.arcsec*slope)**2 + maxFWHM_phs**2)

#LC_fit_func = DFit.GenConvLC(LCtable, dm15=1)#, ref_FWHM_phs=ref_fwhm_phs)#, phs_shft_DOF=True)

LC_fit_params = { 'LCtable': LCtable, 'dm15': 0.83, 'ref_FWHM_phs': ref_fwhm_phs, 'phs_shft_DOF': True }

plt.figure()
for i in np.arange(len(DIFF_df)):
    x = -(np.array(FP_df.iloc[i]['ProjAng'].arcsec)-FP_df_PSFeq.iloc[i]['Peak_ProjAng_arcsec'])*slope
#    x = np.array(FP_df.iloc[i]['ProjAng'].arcsec)
    flux_scale = DFit.Mag2ADU(DIFF_df.iloc[i]['ZPTMAG'])*1e12
    y = flux_scale*np.array(FP_df.iloc[i]['FluxProfile'])
    z = flux_scale*np.array(FP_df.iloc[i]['NoiseProfile'])
    
    x = x[y>-z]
    ytmp = y[y>-z]
    z = z[y>-z]
    y = ytmp
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    z = z.reshape((y.size,1))
    
    xy = np.concatenate((x,y,z),axis=1)
    xy = xy[xy[:,0].argsort()]
#    plt.scatter(xy[:,0],xy[:,1],label=str(i),s=1)
#    f_x = LC_fit_func(xy[:,0],phs_wid=20,PeakFlux=500)#,phs_shft=0)#,dm15_param=1.5)
#    f_x = LC_fit_func(-slope*(xy[:,0]-FP_df_PSFeq.iloc[i]['Peak_ProjAng_arcsec']),phs_wid=50,PeakFlux=75)
#    plt.plot(xy[:,0],f_x,label=str(i))
    plt.errorbar(xy[:,0],xy[:,1],yerr=xy[:,2],capsize=2,marker='.',markersize=0.5,label=str(i))
    
    
plt.gca().legend()


FP_df = DFit.FitFluxProfile(DIFF_df,FP_df,FP_df_PSFeq,LC_fit_params,slope,intercept)
# %%

for ind in np.arange(len(DIFF_df)):
    plt.figure()
    i = ind
    x = -(np.array(FP_df.iloc[i]['ProjAng'].arcsec)-FP_df_PSFeq.iloc[i]['Peak_ProjAng_arcsec'])*slope
    flux_scale = DFit.Mag2ADU(DIFF_df.iloc[i]['ZPTMAG'])*1e12
    y = flux_scale*np.array(FP_df.iloc[i]['FluxProfile'])
    z = flux_scale*np.array(FP_df.iloc[i]['NoiseProfile'])
    
    x = x[y>-z]
    ytmp = y[y>-z]
    z = z[y>-z]
    y = ytmp
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    z = z.reshape((y.size,1))
    
    xy = np.concatenate((x,y,z),axis=1)
    xy = xy[xy[:,0].argsort()]
    LC_fit_func = FP_df.loc[i,'fit_LC_fit_func']
    f_x = LC_fit_func(xy[:,0],phs_wid=FP_df.loc[i,'fitTotWid_phs'],PeakFlux=FP_df.loc[i,'fitPeakFlux_phs'],phs_shft=FP_df.loc[i,'fit_phs_shft'])#,dm15_param=FP_df.loc[i,'fitDM15'])
    plt.plot(xy[:,0],f_x,label=str(i))
#    plt.plot(xy[:,0],xy[:,1],label=str(i))
    plt.errorbar(xy[:,0],xy[:,1],yerr=xy[:,2],capsize=2,marker='.',markersize=0.5,label=str(i))

plt.gca().legend()

# %%
moshe0=[0,
 1,
 4,
 5,
 6,
 7,
 9,
 10,
 11,
 12,
 20,
 21,
 22,
 23,
 27,
 28,
 29,
 30,
 33,
 34,
 35,
 36,
 37,
 38,
 39,
 40,
 41,
 45,
 46,
 47,
 48,
 51]