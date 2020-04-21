#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:34:39 2019

@author: roeepartoush
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import File2Data as F2D
import DustFit as DF
import LeTools_Module as LeT
from scipy import stats
#from scipy.optimize import least_squares


# %%
plt.close('all')
LChome_dir = '/Users/roeepartoush/Documents/Astropy/Data/light curves'
LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
ttt = F2D.LightCurves(LCs_file)

ax_img=[]
fig1=plt.figure()
fig2=plt.figure()
fig1.canvas.mpl_connect('key_press_event',LeT.press)
ax_img.append(fig1.add_subplot(111))
fig2.canvas.mpl_connect('key_press_event',LeT.press)
ax_img.append(fig2.add_subplot(111,sharex=ax_img[0],sharey=ax_img[0]))
for ind in np.arange(len(ttt)):
#    print(ttt['dm15'][ind])
    xtmp = ttt['phases'][ind]
    x = np.linspace(xtmp[0]-10,xtmp[-1]+9,1e4)
    y = DF.Mag2ADU(ttt['mags'][ind])
    f = ttt['func_M'][ind]
    f2 = ttt['func_L'][ind]
    plt.figure(fig1.number)
    plt.scatter(x,DF.Mag2ADU(f(x)),s=1)
    plt.figure(fig2.number)
    plt.scatter(x,(f2(x)),s=1)
#    yfilt=gaussian_filter(f(x-5.5),sigma=20/(2*np.sqrt(2*np.log(2))))
#    plt.scatter(x-5.5,DF.Mag2ADU(yfilt),s=1)
    
#    yfilt=gaussian_filter(DF.Mag2ADU(f(x)),sigma=10/(2*np.sqrt(2*np.log(2))))
#    plt.scatter(x,(yfilt),s=1)
    
#    plt.figure(fig2.number)
#    plt.scatter(ind+1,ttt['dm15'][ind],s=1)

# %% experiment to check dependance of peak location on gaussian width

f=ttt['func'][np.nonzero(ttt['dm15']==1.5)[0][0]]
ff = DF.GenConvLC(f,0.1)
widts = np.linspace(4.0615,4.0630,31)
fig17 = plt.figure('wid dep exp')
x = np.linspace(1.5,2.0,3001)
peaks_lst = []
for wid in widts:
    y = ff(x,wid,100)
    plt.plot(x,y)
    peaks_lst.append(np.nanargmax(y))
# %%
fig17 = plt.figure('wid dep exp peaks33')
plt.scatter(widts,x[np.array(peaks_lst)],s=1)
    
# %%
t = np.linspace(-lnlen.arcsec/2,lnlen.arcsec/2,s_mat.shape[1])
pk_lst = list()
ind_lst=list()
colors=['r','g','b','c','m','r','g','b','c']
lnst=['dashed','dashed','dashed','dashed','dashed','solid','solid','solid','solid']
for ind in np.arange(s_mat.shape[0]):
#    x = gaussian_filter(s_mat[ind,:], sigma=3/(2*np.sqrt(2*np.log(2))))
    x = s_mat[ind,:]
    x[x<0]=0
    err = err_mat[ind,:]
#    x_medabs = np.abs(x-np.nanmedian(x))
    p_thres = np.nanmean(x)+np.nanmax(err)
#    promin = np.nanmedian(x_medabs)
    
#    peaks_abs, _ = find_peaks(x_abs, height=p_thres)
#    peaks_n, _   = find_peaks(-x, height=p_thres)
    peaks, _   = find_peaks(x, distance=10, height = p_thres)
#    peaks = np.nanargmax(x)
#    index = (x[peaks]-np.nanmedian(x))/np.nanmean(np.abs(x-np.nanmedian(x)))
#    print(index)

#    print(n
#    print((np.nanmax(x)-np.nanmin(x))/np.nanmean(np.abs(x-np.nanmean(x))))
#    print(np.nanmean(np.abs(x)))
    plt.figure('stam')
    plt.plot(t,x,color=colors[ind],linewidth=1,linestyle=lnst[ind])
    if len(peaks)==1:
        pk_lst.append([t[peaks][0], LeT.flnm2time([files[ind]])[0]])
        ind_lst.append(ind)
    plt.plot(t[peaks], x[peaks], marker='x',mec=colors[ind], linestyle='none')
    plt.plot(t, np.ones(t.shape)*p_thres,color=colors[ind],linewidth=1,linestyle=lnst[ind])
##    plt.plot(np.zeros_like(x), "--", color="gray")
# %%    plt.show()
pk_lst = np.array(pk_lst)
pksarcs = pk_lst[:,0]
mjds = pk_lst[:,1]

slope, intercept, r_value, p_value, std_err = stats.linregress(pksarcs,mjds)

def arc2day(arc):
    day = intercept + slope*arc
    return day

def day2arc(day):
    arc = (day - intercept)/slope
    return arc
# %%
plt.figure('stam2')
#plt.plot(pksarcs,mjds)
plt.scatter(pksarcs,mjds,s=20)
x_arcs = np.linspace(np.min(pksarcs),np.max(pksarcs),2)
plt.plot(x_arcs, arc2day(x_arcs))
# %%

ind_lst=[2]
plt.figure('stam3')
for ind in np.arange(len(ind_lst)):
    yy=s_mat[ind_lst[ind],:].copy()
    err=err_mat[ind_lst[ind],:].copy()
#    yy[np.greater(0,yy)]=np.nan
#    x=mjds[ind_lst[ind]-5]-arc2day(t)
#    x=t-pksarcs[ind_lst[ind]-5]
    x=t-0.28#+1.08# #SPECIAL CASE
#    x=t-day2arc(mjds[ind_lst[ind]])
#    plt.plot(x, yy)
    plt.errorbar(x,yy,yerr=err,capsize=2)

f=ttt['func'][np.nonzero(ttt['dm15']==1.5)[0][0]]

#def f(x):
#    x=np.array([x])
#    x=np.reshape(x,np.max(x.shape))
#    y = np.ones(x.shape)*(np.Inf)
#    y[~np.greater(np.abs(x),0.1)]=0#-np.Inf
#    return y

ff = DF.GenConvLC(f,0.1)

def fff(x_arcs, phs_wid, PeakFlux):
#    x_arcsM = np.linspace(x_arcs[0],x_arcs[-1],10000)
    x_phsM = -slope*x_arcs
    x_phs = -slope*x_arcs
#    x_phs = mjds[ind_lst[ind]-5]-arc2day(x_arcs+day2arc(mjds[ind_lst[ind]-5]))
#    x_phs = (intercept + slope*(pksarcs[ind_lst[ind]-5]))-(intercept + slope*(x_arcs+pksarcs[ind_lst[ind]-5]))
#    x_phs = arc2day(pksarcs[ind_lst[ind]-5])-arc2day(x_arcs+pksarcs[ind_lst[ind]-5])
#    x_phs = mjds[ind_lst[ind]-5]-arc2day(x_arcs+pksarcs[ind_lst[ind]-5])
    y_tmp = ff(x_phsM, phs_wid, PeakFlux)
    shft_ind = np.argmax(y_tmp)
#    y = ff(x_phs, phs_wid, PeakFlux)
    y = ff(x_phs+x_phsM[shft_ind], phs_wid, PeakFlux)
    return y

#y = ff(mjds[ind_lst[ind]-5]-arc2day(x+pksarcs[ind_lst[ind]-5]),0,130)
par_init = [50 ,121]
#x = np.linspace(x[0],x[-1],10000)
y = fff(x,*par_init)
plt.plot(x,y)
plt.plot(x,fff(x,*par_init))
plt.xlabel('arcsecs')
# %%
inds = np.logical_and(~np.isnan(yy),yy>-np.abs(err))
nan_yy = yy[inds]
x = x[inds]
nan_err = err[inds]
bnd = (np.array([-np.inf, 120]), np.array([np.inf, 122]))

popt, pcov = curve_fit(fff, x, nan_yy, p0 = par_init, bounds=bnd)
# %%
plt.figure('FIT')
plt.errorbar(x,nan_yy,yerr=nan_err,capsize=2)
#plt.plot(x,fff(x, *par_init),'b')
plt.plot(x,fff(x, *popt),'r')
plt.xlabel('arcsecs')
# %%
plt.figure('stam')

kk = (2*np.sqrt(2*np.log(2)))

b_p = {'amplitude': (50,500), 'mean': (-50,50), 'stddev':(1/kk,7/kk)}
b_n = {'amplitude': (50,500), 'mean': (-60,60), 'stddev':(1/kk,7/kk)}
p_init = models.Gaussian1D(amplitude=100, mean=20, stddev=3/kk, bounds=b_p)
n_init = models.Gaussian1D(amplitude=100, mean=20, stddev=4/kk, bounds=b_n)


#y = np.power(y,2)
#y=y/np.max(np.abs(y))
#y = gaussian_filter(y, sigma=5/(2*np.sqrt(2*np.log(2))))

#x = np.linspace(-lnlen.arcsec/2,lnlen.arcsec/2,s_mat.shape[1])
xtmp = FP_df.iloc[0]['ProjAng'].arcsec#MAT2[:,0]
x = xtmp.copy()#[np.abs(xtmp)<10].copy()
ytmp = FP_df.iloc[0]['FluxProfile']#MAT2[:,1]
y = ytmp.copy()#[np.abs(xtmp)<10].copy()
#y=y[0:170]
#x=x[0:170]
#mmm = np.zeros(y.shape)
#mmm[np.isnan(y)] = 1
#y = np.ma.array(y, mask = mmm)
#nmmm = np.logical_not(mmm)
#y[nmmm] = y[nmmm]-np.median(y[nmmm])
#y[nmmm] = y[nmmm]-np.mean(y[nmmm])
#y[170:223]=np.nan
## Generate fake data
#np.random.seed(42)
#g1 = models.Gaussian1D(1, -0.5, 0.2)
#g2 = models.Gaussian1D(-2.5, 0.5, 0.1)
#x = np.linspace(-1, 1, 200)
#y = g1(x) + g2(x) + np.random.normal(0., 0.2, x.shape)

pn_init = p_init# + n_init
#pn_init = models.Polynomial1D(degree=7)
fitter = fitting.LevMarLSQFitter()
#fitter = fitting.LevMarLSQFitter()
pn_fit = fitter(pn_init, x, y)#,verblevel=2)#, acc=1e-20, maxiter=100)#,   maxiter=500)#, weights=(y/np.max(y)))

## Now to fit the data create a new superposition with initial
## guesses for the parameters:
#gg_init = g1 + g2
#fitter = fitting.SLSQPLSQFitter()
#gg_fit = fitter(gg_init, x, y)


plt.scatter(x,y,s=3,c='k')
plt.plot(x,y)
plt.plot(x,pn_init(x),'b')
plt.plot(x,pn_fit(x),'r')

## Plot the data with the best-fit model
##plt.figure(figsize=(8,5))
#plt.plot(x, y, 'ko')
#plt.plot(x, gg_fit(x))
#plt.plot(x, gg_init(x),'r')
#plt.xlabel('Position')
#plt.ylabel('Flux')

# %%

for ind in np.arange(len(hdu_list)):
    try:
        magz = hdu_list[ind][0].header['MAGZERO']
        print('hdu no. '+str(ind)+': '+str(np.power(10,-magz/2.5)))
    except:
        print('?')