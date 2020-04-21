#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:30:55 2019

@author: roeeyairpartoush
"""
# %%
import numpy as np
from scipy import signal as sp
from scipy import interpolate as intp
# Set up matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from astropy.io import fits
from astropy import wcs

from astropy import units as u
from astropy.coordinates import Angle

from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import SkyCoord
from astropy.time import Time
#from astropy.utils import data
#from astropy.utils.data import download_file
#from astropy.utils.data import get_readable_fileobj
#from astropy.utils.data import get_file_contents
import LeTools_Module as LeT
from scipy.ndimage import gaussian_filter

import pandas as pd
import re
import os

#%%
plt.close('all')
Xmat,Ymat = np.meshgrid(np.arange(900),np.arange(1800))
cntr = np.array([788,1403])
Rmat = np.sqrt((Xmat-cntr[0])**2 + (Ymat-cntr[1])**2)
inds = np.abs(Rmat-500)<0.5
Xvec = (Xmat-cntr[0])[inds].flatten()
Yvec = (Ymat-cntr[1])[inds].flatten()
theta = np.arctan2(Yvec,Xvec)*180/np.pi
FPs = np.zeros((len(image_data3),np.count_nonzero(inds)))

for i in np.arange(len(image_data3)):
    if i!=2:
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', LeT.press)
        if i==0:
            ax1 = fig.add_subplot(211)
        else:
            fig.add_subplot(211,sharex=ax1,sharey=ax1)
        mat = image_data3[i]
#        mat = gaussian_filter(mat,sigma=2/(2*np.sqrt(2*np.log(2))))
        plt.imshow(mat,vmin=-5,vmax=5,cmap='gray')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.scatter(Xvec+cntr[0],Yvec+cntr[1],s=0.5,c='r')
        FPs[i,:] = mat[inds].flatten()
        if i==0:
            ax2 = fig.add_subplot(212)
        else:
            fig.add_subplot(212,sharex=ax2,sharey=ax2)
        plt.scatter(theta,FPs[i,:],s=0.5)
        plt.ylim([-5,5])
        
# %% MJD-OBS
lst=[]
dates=[]
for i in np.arange(len(hdu_list)):
    try:
        print(str(hdu_list[i][0].header['DATE-OBS'])+', MAGZ no. '+str(i)+': '+str(hdu_list[i][0].header['MAGZERO']))
    except:
        print(str(i)+'?')
    dates.append(hdu_list[i][0].header['MJD-OBS'])
    lst.append([image_files[i],dates[i]])
#ddd = {'flnm': image_files,
#       'date': dates}
ddd = pd.DataFrame(data=lst,columns=['flnms','dates'])
image_files = list(ddd.sort_values('dates')['flnms'])

# %%
#for i in np.arange(62):# aDF.set_value(i,'N_amp',len(pd.unique(data.loc[data[0]==aDF.iloc[i]['field'],1])))
#    Nmax=0
#    Nmin=np.Inf
#    field_inds=(data[0]==aDF.iloc[i]['field'])
#    data_field=data.loc[field_inds]
#    for amp in pd.unique(data_field[1]):
#        Nmax=max(Nmax,np.sum(data_field[1]==amp))
#        Nmin=min(Nmin,np.sum(data_field[1]==amp))
#    aDF.set_value(i,'max_count',Nmax)
#    aDF.set_value(i,'min_count',Nmin)

from astropy.table import Table
from astropy.io.votable import parse
votable = parse('/Users/roeepartoush/Downloads/rows_as_votable_1581715408_8228.vot')
votDF = votable.resources[0].tables[0].to_table().to_pandas()

for i in np.arange(len(votDF)):
    for k in np.arange(votDF.shape[1]):
        val = votDF.loc[i,votDF.columns[k]]
        if type(val)==bytes:
            votDF.at[i,votDF.columns[k]]=val.decode('ascii')
instcalDF = votDF.loc[votDF['proctype']=='InstCal']

invDF=pd.read_csv('/Users/roeepartoush/Desktop/DECAM_QC-INV/20200128.inv',sep='[ ]{1,}')
# %% https://stsci-transients.stsci.edu/atlas/cand/ROEE/dec+46/071+46.58400.wp.fits
plt.close('all')
global image_files
image_files = []
mjds = []#np.array([],ndmin=3).astype(int)
decs = []#np.array([],ndmin=2).astype(int)
ras = []#np.array([],ndmin=2).astype(int)
home_dir = 'https://stsci-transients.stsci.edu/atlas/cand/ROEE/'#'/Users/roeepartoush/Downloads/TYC2116/TEMP/'#'https://stsci-transients.stsci.edu/atlas/cand/ROEE/TEMP2/'#'/Users/roeepartoush/Downloads/ZZZ/zzz/'#'/Users/roeepartoush/Downloads/STScI/STUFF/tmp_download/'

tmpDF = lstDF.iloc[np.logical_and(np.array(lstDF['DEC'])==60,np.array(lstDF['RA'])==352)] 
#tmpDF = lstDF.iloc[np.array(lstDF['DEC'])==-36]
for i in np.arange(len(tmpDF)):
    image_files.append(tmpDF.iloc[i]['flnms'][0:22])#+'.fits')
i=0
#for root, dirs, files in os.walk(home_dir):
#    for file in files:
#        if file[len(file)-9::1]=='diff.fits':#'.wp.fits':
#            image_files.append(file[0:(len(file)-5)])
#            i=i+1
##            dec = int(file[4:6])
##            ra = int(file[0:3])
##            mjd = int(file[7:12])
##            if dec==89:# or dec==68:
##                image_files.append(file[0:(len(file)-5)])
##                mjds.append(int(file[7:12]))
##                decs.append(int(file[4:6]))
##                ras.append(int(file[0:3]))
##                i=i+1

#import urllib
#image_files=urllib.request.urlopen(home_dir+'fileslist.txt').read().decode("utf-8").split('\n')
##image_files=open(home_dir+'fileslist.txt').read().split('\n')
#image_files=image_files[0:-1]

#DF = pd.DataFrame(data=np.concatenate((np.array(ras).reshape((i,1)),np.array(decs).reshape((i,1)),np.array(mjds).reshape((i,1))),axis=1),columns=['RA','DEC','MJD'])
#DF.sort_values(['DEC','RA','MJD'],inplace=True)

DF = pd.DataFrame(data=image_files, columns=['flnm'])
DF.sort_values(['flnm'],inplace=True)
 
inds = list(DF.index)
image_files_tmp = image_files.copy()
for i in np.arange(len(inds)):
    image_files[i] = image_files_tmp[inds[i]]


files = image_files#[37:42]#10:15]
#for i in np.arange(len(files)): files[i]=home_dir+image_files[i]
#DIFF_df_BU3 = F2D.FitsDiff(files)
[image_data, hdu_list_list, aximg2, wcs_list2] = LeT.load_difimg_atlas(home_dir,files, FS=0,gss_FWHM=None)
#hdu_DF=pd.DataFrame(hdu_list_list2).transpose()
#moshe=[manager.canvas.figure.number for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
#moshe0=moshe.copy()
#for i in np.arange(len(moshe0)): moshe0[i]=moshe0[i]-1
#moshe0=np.arange(len(hdu_DF))
#hdu_list_moshe=hdu_DF.iloc[moshe0[0:10]].transpose().values.tolist()
#[image_data2, aximg2, wcs_list2] = LeT.load_difimg_atlas_HDU_LOADED(hdu_list_moshe, files, FS=0,gss_FWHM=None)
#hdu_list = hdu_list_moshe[0].copy()
# %%
with open('/Users/roeepartoush/Downloads/kakabepita.txt') as f:
    a=f.read()
import re
b=[m.start() for m in re.finditer('KRILINITZ!!!', a)]
flds=[]
for i in np.arange(len(b)):
    flds.append(a[(b[i]+12):(b[i]+12+7)])
u_flds = list(set(flds))

strsss='{'
for item in u_flds:
    strsss=strsss+item+','
#    strsss=strsss+'tmpl.'+item+'/*/*.fits,'
strsss=strsss[0:-1]+'}'
# %%
#del Axes3D, proj3d
from mpl_toolkits.mplot3d import Axes3D, proj3d
#def orthogonal_proj(zfront, zback):
#    a = (zfront+zback)/(zfront-zback)
#    b = -2*(zfront*zback)/(zfront-zback)
#    return np.array([[1,0,0,0],
#                        [0,1,0,0],
#                        [0,0,a,b],
#                        [0,0,0,zback]])
#proj3d.persp_transformation = orthogonal_proj

#plt.close('all')
Xm,Ym=np.meshgrid(np.arange(image_data2[0].shape[1]),np.arange(image_data2[0].shape[0]))

fig3=plt.figure(100)
ax3=fig3.add_subplot(111,projection='3d')
for ind in np.arange(len(image_data2)):
    xmat=Xm[~np.isnan(image_data2[ind])]
    ymat=Ym[~np.isnan(image_data2[ind])]
    a=Time(hdu_list[ind][0].header['DATE-OBS'])
    ax3.scatter(xmat,ymat,a.mjd,s=0.01,c='b')

#ax3.view_init(elev=90., azim=0)
ax3.set_zlim(Time(hdu_list[0][0].header['DATE-OBS']).mjd-0.5e3,Time(hdu_list[-1][0].header['DATE-OBS']).mjd+0.5e3)
ax3.set_xlim(3000,4199)
ax3.set_ylim(1100,2199)
# %%
float_formatter = "{:.6f}".format
regDF=pd.read_csv('/Users/roeepartoush/Downloads/ZZZ/tycZZZZ.region',sep="\s+|;|,|\(|\)",header=None)
regDF.insert(regDF.shape[1],'command',None)
for i in np.arange(len(regDF)):
    row = regDF.iloc[i]
    corners=[]
    for k in np.arange(4):
        corners.append(np.array([row[2+2*k],row[2+2*k+1]]))
    ux=corners[1]-corners[0]
    uy=corners[2]-corners[1]
    command=''
    for k in np.arange(2):
        for m in np.arange(4):
            command=command+' \nfk5;polygon('
            origin = corners[0] +(m/4)*uy +(k/2)*ux
            corners8=[origin +0*uy/4 +0*ux/2, origin +0*uy/4 +1*ux/2, origin +1*uy/4 +1*ux/2, origin +1*uy/4 +0*ux/2]
            for l in np.arange(4):
                command=command+float_formatter(corners8[l][0])+','+float_formatter(corners8[l][1])+','
            command=command+float_formatter(corners8[0][0])+','+float_formatter(corners8[0][1])+')'
            command=command+';text('+float_formatter(np.array(corners8).mean(axis=0)[0])+','+float_formatter(np.array(corners8).mean(axis=0)[1])+')'
            command=command+' # color=green text={'+str(8-k*4-m)+'} '
    regDF.at[i,'command'] = command

# %%
amp_fieldDF=pd.read_csv('/Users/roeepartoush/Downloads/amp_fieldDF-4.txt',sep='\t')
amp_fieldDF.insert(amp_fieldDF.shape[1],'command',None)
regex=re.compile(r"\s+|\(|\)|\[|\]|array")
for i in np.arange(len(amp_fieldDF)):
    command = 'fk5;polygon('+regex.sub('',amp_fieldDF.iloc[i]['corners'])+');'
    command = command+'text('+regex.sub('',amp_fieldDF.iloc[i]['center'])+') # color=yellow'
    if amp_fieldDF.iloc[i]['amp']==1:
        text = amp_fieldDF.iloc[i]['field_amp']
    else:
        text = amp_fieldDF.iloc[i]['field_amp'][-1]
    command = command+' text={'+text+'}'
    amp_fieldDF.at[i,'command'] = command
# %%
peletDF=pd.read_csv('/Users/roeepartoush/Downloads/PELET-3.txt',sep='[ ]{1,}')
peletDF.insert(peletDF.shape[1],'field',None)
peletDF.insert(peletDF.shape[1],'field_amp',None)
peletDF.insert(peletDF.shape[1],'FOM',1.0)
MT_C_DELTA_M5SIGMA=1.0
for i in np.arange(len(peletDF)):
    peletDF.at[i,'field'] = peletDF.iloc[i]['Filename'][0:7]
    peletDF.at[i,'field_amp'] = peletDF.iloc[i]['Filename'][0:7]+'_'+peletDF.iloc[i]['Filename'][len(peletDF.iloc[i]['Filename'])-9]
#    peletDF.at[i,'FOM'] = peletDF.iloc[i]['FWHM'] - MT_C_DELTA_M5SIGMA*(peletDF.iloc[i]['M5SIGMA']-medM5SIGMA)

ampDF = pd.DataFrame(columns=['field_amp'],data=pd.unique(peletDF['field_amp']))
for colnm in ['Filename','Filename_date','field','amp','FWHM','M5SIGMA','FOM']:
    ampDF.insert(ampDF.shape[1],colnm,None)

for i in np.arange(len(ampDF)):
    medM5SIGMA= np.median(peletDF.loc[peletDF['field_amp']==ampDF.iloc[i]['field_amp']]['M5SIGMA'])
    for k in peletDF.loc[peletDF['field_amp']==ampDF.iloc[i]['field_amp']].index:
        peletDF.at[k,'FOM'] = peletDF.iloc[k]['FWHM'] - MT_C_DELTA_M5SIGMA*(peletDF.iloc[k]['M5SIGMA']-medM5SIGMA)
    ind = peletDF.loc[peletDF['field_amp']==ampDF.iloc[i]['field_amp']]['FOM'].idxmin()
    for colnm in ['FWHM','M5SIGMA','FOM']:
        ampDF.at[i,colnm] = peletDF.iloc[ind][colnm]
    flnm = peletDF.iloc[ind]['Filename']
    ampDF.at[i,'Filename'] = flnm
    ampDF.at[i,'Filename_date'] = flnm[0:(len(flnm)-9)]
#    ampDF.at[i,'Filename_date'] = ampDF.iloc[i]['Filename'][11:19]
    ampDF.at[i,'field'] = ampDF.iloc[i]['field_amp'][0:7]
    ampDF.at[i,'amp'] = ampDF.iloc[i]['field_amp'][8:9]

fieldDF = pd.DataFrame(columns=['field'],data=pd.unique(peletDF['field']))
fieldDF.insert(fieldDF.shape[1],'Filename',None)
fieldDF.insert(fieldDF.shape[1],'command',None)
fieldDF.insert(fieldDF.shape[1],'command2',None)
for i in np.arange(len(fieldDF)):
    field = fieldDF.iloc[i]['field']
    uniqFilnames = pd.unique(ampDF.loc[ampDF['field']==field]['Filename_date'])
    fieldDF.at[i,'Filename'] = uniqFilnames
    tmpl_dir = 'tmpl.'+field
    amps = '{'
    for k in pd.unique(ampDF.loc[ampDF['field']==field]['amp']):
        amps = amps+str(k)+','
    amps = amps[0:(len(amps)-1)] + '}'
    command = ''#'mkdir '+tmpl_dir+'; mkdir '+tmpl_dir+'/'+amps+';'
    command2 = 'pipeloop.pl -diff '+field+' '+tmpl_dir+' '+amps[1:(len(amps)-1)]+' -stage MATCHTEMPL+ -condor'
#    command2 = 'pipeloop.pl -red '+tmpl_dir+' '+amps[1:(len(amps)-1)]+' -condor'
    for k in uniqFilnames:
        if len(amps)==3:
            command = command+' for i in '+amps[1]+'; do cp '+field+'/$i/'+k+'$i.fits '+tmpl_dir+'/$i/; done;'
        else:
            command = command+' for i in '+amps+'; do cp '+field+'/$i/'+k+'$i.fits '+tmpl_dir+'/$i/; done;'
#        command = command+' for i in '+amps+'; do cp '+field+'/$i/'+field+'.VR.'+k+'*stch*$i*.fits '+tmpl_dir+'/$i/; done;'
    fieldDF.at[i,'command'] = command[0:(len(command)-1)]
    fieldDF.at[i,'command2'] = command2

# %%

home_dir = '/Users/roeepartoush/Downloads/Roee/2/'
prefix = 'tyc4419.VR.'
midfix = '_'#'_R.'
sufix = '_2.diff'
tmplt_img = '20140923.149398'#'20150211.157400'

#image_files = ['20150814.64482'     ,
#               '20150817.564744'    ,
#               '20150817.564745'    ,
#               '20150911.165398'    ,
#               '20150911.165399'     ]

image_files = ['20120118.216797'    ,
               '20120626.24973'     ,
               '20130109t.1692969'  ,
               '20130826.33218'     ,
               '20140621.103815'    ,
               '20141223.54713'     ,
               '20150211.157395'    ,
               '20150814.64482'     ,
               '20150911.165399'     ]

# % ================!!!!!!!===============

flnm_lst=[]
for ind in np.arange(len(image_files)):
    flnm_lst.append(home_dir+prefix+image_files[ind]+midfix+tmplt_img+sufix)

# %%
home_dir = '/Users/roeepartoush/Downloads/STScI/STUFF/OUTPUTdir/dec+12/'
image_files = [home_dir+'082+12.58400.wp',
               home_dir+'082+12.58400.wp']

# %%
plt.close('all')
files = image_files[0:9]
[image_data, mask_img, noise_img, hdu_list, aximg, wcs_list] = LeT.load_difimg(home_dir, prefix, midfix, sufix, tmplt_img, files)
#for i in np.arange(len(aximg)):
#    plt.sca(aximg[i])
#    plt.scatter(FP_df.iloc[i]['WrldVec'][:,0], FP_df.iloc[i]['WrldVec'][:,1], s=2,color='r', transform = aximg[i].get_transform('world'))
# %%
for i in np.arange(4):
    plt.sca(aximg[i]) 
    plt.scatter(FP_df.iloc[i]['WrldVec'][:,0], FP_df.iloc[i]['WrldVec'][:,1], s=2,color='r', transform = aximg[i].get_transform('world'))
# %%
#LE = list()
#LE = [Angle('0h49m18s'),Angle('58°42′30"')]
#LE = [Angle('0h49m16s'),Angle('58°43′00"')]
#LE = [Angle('0h49m3.55s'),Angle('58°42′17"')] # dip for cal
#LE = [Angle('0h48m52.75s'),Angle('58°41′55.7"')] # dip for cal
#LE = [Angle('0h49m21.3s'),Angle('58°43′7"')]
#LE = [Angle('0h48m49.4s'),Angle('58°42′47"')]
#LE = [Angle('0h48m37.6s'),Angle('58°41′8"')]
#LE = [Angle('0h48m55.6s'),Angle('58°42′6"')]
#LE = [Angle('0h48m55.5s'),Angle('58°42′6"')]
#LE = [Angle('0h49m16s'),Angle('58°41′0"')]
#LE = [Angle('0h48m53s'),Angle('58°41′57"')]
#LE = [Angle('0h48m49s'),Angle('58°42′44"')] # peak in 20130109t
#LE = [Angle('0h49m6.5s'),Angle('58°42′40"')]
#LE = [Angle('0h49m16.5s'),Angle('58°42′50"')]
#LE = [Angle('0h49m16s'),Angle('58°42′45"')]
#LE = [Angle('0h49m20s'),Angle('58°43′00"')]
#LE = [Angle('0h49m21.6s'),Angle('58°43′3.5"')]
    
#LE = [Angle('0h49m21s'),Angle('58°43′1"')]
#LE = [Angle('0h49m18.8s'),Angle('58°42′53"')]
#LE = [Angle('0h49m14.4s'),Angle('58°42′27"')]
#SN = [Angle('0h25.3m'),Angle('64°09′')]

LE = [Angle('0h49m14.4s'),Angle('58°42′27"')]
#SN = [Angle('0h49m00s'),Angle('58°42′00"')]
SN = [Angle('0h47m24s'),Angle('58°34′12"')]
#SN = [Angle('0h50m9.4s'),Angle('58°34′12"')]

#LE = [Angle(12.84,u.deg),Angle(58.76,u.deg)]
#SN = [Angle(12.74,u.deg),Angle(58.72,u.deg)]

#LElst=[]
#LElst.append([Angle('0h49m18s'),Angle('58°42′30"')])
#LElst.append([Angle('0h49m16s'),Angle('58°43′00"')])
#LElst.append([Angle('0h49m3.55s'),Angle('58°42′17"')]) # dip for cal
#LElst.append([Angle('0h48m52.75s'),Angle('58°41′55.7"')]) # dip for cal
#LElst.append([Angle('0h49m21.3s'),Angle('58°43′7"')])
#LElst.append([Angle('0h48m49.4s'),Angle('58°42′47"')])
#LElst.append([Angle('0h48m37.6s'),Angle('58°41′8"')])
#LElst.append([Angle('0h48m55.6s'),Angle('58°42′6"')])
#LElst.append([Angle('0h48m55.5s'),Angle('58°42′6"')])
#LElst.append([Angle('0h49m16s'),Angle('58°41′0"')])
#LElst.append([Angle('0h48m53s'),Angle('58°41′57"')])
#LElst.append([Angle('0h48m49s'),Angle('58°42′44"')]) # peak in 20130109t
#LElst.append([Angle('0h49m6.5s'),Angle('58°42′40"')])
#LElst.append([Angle('0h49m16.5s'),Angle('58°42′50"')])
#LElst.append([Angle('0h49m16s'),Angle('58°42′45"')])
#LElst.append([Angle('0h49m20s'),Angle('58°43′00"')])
#LElst.append([Angle('0h49m21.6s'),Angle('58°43′3.5"')])
#LElst.append([Angle('0h49m18.8s'),Angle('58°42′53"')])
#LElst.append([Angle('0h49m14.4s'),Angle('58°42′27"')])

#SN = [Angle('0h49m12s'),Angle('58°42′')]
for ind in np.arange(len(files)):
#    for LE in LElst:
    [arcltPA, angSeper] = LeT.plot_arclt(SN,LE,Angle('0d1m0s'),aximg[ind],wcs_list[ind])

# %%

#point = [Angle('0h49m16s'),Angle('58°42′37.5"')]
point = LE
lnlen = Angle('0d1m0s')
#ang = Angle(-90,u.degree)
ang = Angle(arcltPA.degree+90,u.degree)
[s_mat, err_mat, smp_arcsec_mat, DSM, DEM, DSAM] = LeT.plot_cut(image_data, mask_img, noise_img, hdu_list, np.arange(len(image_data)),point,ang,lnlen,aximg,10)
#s_mat[np.absolute(s_mat)<20] = s_mat[np.absolute(s_mat)<20]*0.2

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

[Xm, Ym] = np.meshgrid(np.arange(s_mat.shape[1]),LeT.flnm2time(files))
X=Xm.flatten()
Y=Ym.flatten()
Z=s_mat.flatten()

ax = Axes3D(fig)
ax.plot(X,Y,Z,color='gray',linewidth=1)
ax.scatter(X,Y,Z,s=1,c=Z,cmap='jet')

# %%
from matplotlib import cm

#def on_move(event):
#    if event.inaxes == ax:
#        ax2.view_init(elev=ax.elev, azim=ax.azim)
#    elif event.inaxes == ax2:
#        ax.view_init(elev=ax2.elev, azim=ax2.azim)
#    else:
#        return
#    fig.canvas.draw_idle()

#fig = plt.figure()
#ax = Axes3D(fig)
W2P = wcs_list[0].wcs_world2pix
pointWrld = np.array([LE[0].degree, LE[1].degree],ndmin=2)
pointPix = np.round(W2P(pointWrld,0)).astype(int)
wid = 50
N = 2*wid+1
vec = np.round(np.linspace(-wid,wid,N)).astype(int)
Xvec = pointPix.flatten()[0]+vec
Yvec = pointPix.flatten()[1]+vec
[Xm, Ym] = np.meshgrid(Xvec,Yvec)
step = 1
Xm = Xm[0:N:step,0:N:step]
Ym = Ym[0:N:step,0:N:step]
for ind in np.arange(len(files)):
    Zm = -((image_data[ind][Yvec[0]:(Yvec[-1]+1):step,Xvec[0]:(Xvec[-1]+1):step]))
#    Zm[Zm<50] = np.nan
#    norm = plt.Normalize(Zm.min(), Zm.max())
#    colors = cm.viridis(norm(Zm))
#    rcount, ccount, _ = colors.shape
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', LeT.press)
#    fig.canvas.mpl_connect('motion_notify_event', on_move)
    ax = Axes3D(fig)
    ax.scatter(Xm.flatten(),Ym.flatten(),Zm.flatten(),s=1,c=Zm.flatten(),vmin=-10, vmax=10,cmap='jet')
#    ax.plot_wireframe(Xm,Ym,Zm,cmap='jet',linewidth=0.1,rstride=1, cstride=1)
#    ax.set_zlim(0,200.5)
#    surf = ax.plot_surface(Xm,Ym,Zm,cmap='jet',linewidth=0.1, antialiased=True,vmin=-100, vmax=200, facecolors = colors, shade=False,rcount=rcount, ccount=ccount)    
#    surf.set_facecolor((0,0,0,0))
#fig.colorbar(surf, shrink=0.5, aspect=5)
#ax.legend()



# %%
ind = 2
LEdeg = np.array([LE[0].degree, LE[1].degree],ndmin=2)
LEpix = wcs_list[0].wcs_world2pix(LEdeg,0)
mat = np.ones(image_data[0].shape)*(-100)
[Xm, Ym] = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
NOT = np.logical_not
px = np.round(LEpix.flatten()[0])
py = np.round(LEpix.flatten()[1])
#mat[(Xm==np.round(LEpix.flatten()[0])) & (Ym==np.round(LEpix.flatten()[1]))] =100
#mat[Xm>2000]=100
#mat[Xm<=2000]=-100
#mat[(np.abs(Xm-px)<=10) &  (np.abs(Ym-py)<=10)]=100
mat = image_data[8]
mat = gaussian_filter(mat, sigma=2)
#image_data[ind] = mat
figures=[manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]
plt.figure(figures[ind].number)
plt.imshow(mat,vmin=-100, vmax = 100, cmap='gray')

# %%
#plt.close('all')
IMmat = np.asarray(image_data)

fig = plt.figure()
[Xm, Ym] = np.meshgrid(np.arange(2200),LeT.flnm2time(files))
X=Xm.flatten()
Y=Ym.flatten()
Z=np.mean(IMmat[:,:,0:2199],2).flatten()

ax = Axes3D(fig)
ax.plot(X,Y,Z,color='gray',linewidth=1)
ax.scatter(X,Y,Z,s=1,c=Z,cmap='jet')

#meanImg = np.mean(IMmat,0)
#[image_data, aximg] = LeT.plot_dif(meanImg, image_data)

# %%

minX = 800
maxX = 940
minY = 1820
maxY = 2040

plt.figure()
ax = plt.axes(projection='3d')

xx = X[minY:maxY,minX:maxX].flatten()
yy = Y[minY:maxY,minX:maxX].flatten()
zz = image_data[6][minY:maxY,minX:maxX].flatten()
ax.scatter(xx, yy, zz,s=1,c=zz,cmap='jet')

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

[Xm, Ym] = np.meshgrid(np.arange(s_mat.shape[1]),LeT.flnm2time(image_files))
X=Xm.flatten()
Y=Ym.flatten()
Z=s_mat.flatten()
#plt.scatter(Xm.flatten(),Ym.flatten(),s=1,c=Z.flatten(),cmap='jet')

ax = Axes3D(fig)
#ax.plot(X,Y,Z,linestyle='none',marker='.',markersize=1)
ax.scatter(X,Y,Z,s=1,c=Z,cmap='jet')
#ax.plot_wireframe(Xm,Ym,s_mat)
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

# %%
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
#from matplotlib import cm
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X, Y, Z = axes3d.get_test_data(0.05)
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
#
#ax.set_xlabel('X')
#ax.set_xlim(-40, 40)
#ax.set_ylabel('Y')
#ax.set_ylim(-40, 40)
#ax.set_zlabel('Z')
#ax.set_zlim(-100, 100)
#
#plt.show()

# %%
ind=8

clim = 50
plt.figure(ind+1)
plt.imshow(image_data[ind], cmap='gray', vmin=-clim, vmax=clim)
#plt.colorbar()
plt.figure(ind+10)

#(z1,z2,z3)=plt.hist(image_data[ind].flatten(),bins=np.unique(image_data))

# %%

plt.figure()

inds=[0,1]

image_sum = np.zeros(image_data[0].shape)
for i in inds:
    image_sum = image_sum + image_data[i]

image_sum = image_sum/(i+1)
plt.figure(500)
plt.imshow(image_sum, cmap='gray', vmin=-clim, vmax=clim)    
# %%
point = np.array([[1520.0],[1760.0]])

angle = np.radians(70);
unitvec = np.array([[np.cos(angle)],[np.sin(angle)]])
uninorm = np.array([[np.sin(angle)],[-np.cos(angle)]])

lnlen = 200;

#vec = np.round(np.linspace(point-unitvec*lnlen/2,point+unitvec*lnlen/2,np.floor(lnlen)+1))
vec = (np.squeeze(np.linspace(point-unitvec*lnlen/2,point+unitvec*lnlen/2,np.floor(lnlen)+1),2))
vec2 = np.concatenate((np.reshape(np.floor(vec[:,0]),(vec.shape[0],1)),np.reshape(vec[:,1],(vec.shape[0],1))),1)
#vec = np.round(np.squeeze(np.linspace(point-uninorm*lnlen/2,point+uninorm*lnlen/2,np.floor(lnlen)+1),2))
#x=np.round(vec)
#plt.sca(ax_img[8])
#plt.scatter(vec[:,0],vec[:,1])

# %%
Btch = 10
stp = 1.0

point = np.array([[1780.0],[1560.0]])

#ind = 0
#imimim = image_data[ind]
#smp_sum = np.zeros(img_smp.shape)

for i in np.arange(Btch):
    point_t = point+uninorm*stp*(i+5)
    vec = np.round(np.squeeze( np.linspace(point_t-unitvec*lnlen/2, point_t+unitvec*lnlen/2, np.floor(lnlen)+1), 2) )
    plt.figure(ind+1)
    plt.scatter(vec[:,0],vec[:,1],s=2)
    
    img_dt_ind = imimim.flatten();
    inds = np.ravel_multi_index([vec[:,1].astype(int),vec[:,0].astype(int)], (2200,4200))
    img_smp = img_dt_ind[inds]
    smp_sum = smp_sum + img_smp
    
    hh = plt.figure(300)
    ax1 = hh.add_subplot(211)
    plt.plot(np.arange(inds.size),img_smp)
    
#plt.figure(301)
ax2 = hh.add_subplot(212, sharex=ax1, sharey=ax1)
plt.plot(np.arange(inds.size),smp_sum/(i+1))
# %%
imimim = image_sum
#imimim = image_data[2]
#imimim = img_flt
img_dt_ind = imimim.flatten();
inds = np.ravel_multi_index([vec[:,1].astype(int),vec[:,0].astype(int)], (2200,4200))
#img_dt_ind[inds] = 10;
img_smp = img_dt_ind[inds]

#f = intp.interp2d(np.arange(4200),np.arange(2200), imimim, kind='linear')
#img_smp = f(vec[:,0],vec[:,1])

#N = 1000
#S = 4500000
#plt.scatter(np.arange(N),img_dt_ind[S:(S+N)])
#img_dt_ind = np.reshape(img_dt_ind,(2200,4200))
#plt.imshow(img_dt_ind, cmap='gray')
hh = plt.figure(300)
#plt.plot(np.arange(inds.size),img_smp.flatten()[np.arange(0,201*201-1+202,202)])
plt.plot(np.arange(inds.size),img_smp)

# %%
img_fft2 = np.fft.fft2(image_data_tmp)
#plt.imshow(np.fft.fftshift(np.log10(np.abs(img_fft2))),extent=[-0.5,0.5,-0.5,0.5], cmap='gray')
plt.imshow(np.fft.ifft2(img_fft2))
plt.colorbar()

# %%
fx = np.linspace(-0.5,0.5,4200)
fy = np.linspace(-0.5,0.5,2200)

fxm, fym = np.meshgrid(fx,fy)
ind_bool = np.fft.ifftshift(np.sqrt(np.power(fym,2)+np.power(fxm,2))>0.05);
#plt.imshow(ind_bool, cmap='gray', extent=[-0.5,0.5,-0.5,0.5])
img_fft2_flt = img_fft2
img_fft2_flt[ind_bool]=0.001
#plt.imshow(np.fft.fftshift(np.log10(np.abs(img_fft2_flt))), cmap='gray', extent=[-0.5,0.5,-0.5,0.5])
#plt.colorbar()
# %%
img_dt_ind = image_data_tmp.flatten();
img_dt_ind[np.ravel_multi_index([vec[:,1].astype(int),vec[:,0].astype(int)], (2200,4200))] = 10;
#N = 1000
#S = 4500000
#plt.scatter(np.arange(N),img_dt_ind[S:(S+N)])
img_dt_ind = np.reshape(img_dt_ind,(2200,4200))
plt.imshow(img_dt_ind, cmap='gray')

# %%
img_flt = sp.medfilt(image_data_tmp,3)
f = plt.figure(17)
plt.imshow(img_flt, cmap='gray')
plt.colorbar()

# %%

import sys
import numpy as np
import matplotlib.pyplot as plt


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')
ax.set_title('Press a key')
plt.show()

# %%
import PS_image_download as PSid
from astropy import wcs
def Get_PS1(RA, DEC, Size, Filter = "i"):
    '''
    Size limit seems to be around 1000
    '''
    size = Size # last term is a fudge factor
    fitsurl = PSid.geturl(RA,DEC, size=size, filters=Filter, format="fits")
    print(fitsurl)
    if len(fitsurl) > 0:
        fh = fits.open(fitsurl[0])
        ps = fh[0].data
        ps_wcs = wcs.WCS(fh[0])
        return ps, ps_wcs, fitsurl
    else:
        raise ValueError("No PS1 images at for this coordinate")
        return 

# %%
#ps,ps_wcs,pfh=Get_PS1(82.75,12.25,7200,Filter='r')
pixwid = np.round(182) # second factor is arcsec
ps,ps_wcs,pfuh=Get_PS1(12.15089583,58.64403056,pixwid,Filter='r')
# %%
#ps = image_data[1]
#ps_wcs = wcs_list[1]

#fig=plt.figure(plt.gcf().number+1)
#fig.canvas.mpl_connect('key_press_event', LeT.press)
#overlay = aximg[0].get_coords_overlay(ps_wcs)
#axps = fig.add_subplot(111,projection=ps_wcs)
#axps = fig.add_subplot(111,sharex=aximg[0],sharey=aximg[0],projection=wcs_list[0])#ps_wcs)

Xind,Yind = np.meshgrid(np.arange(pixwid),np.arange(pixwid))

VECdeg = ps_wcs.wcs_pix2world(np.concatenate((np.reshape(Xind,(pixwid**2,1)),np.reshape(Yind,(pixwid**2,1))),axis=1),0)
# %%
RAvecT = VECdeg[:,0]
DECvecT = VECdeg[:,1]
VALvecT = ps.flatten()

low=3
high=5
inds = np.logical_and(np.abs(VALvecT)>low,np.abs(VALvecT)<high)
RAvec = RAvecT[inds]
DECvec = DECvecT[inds]
VALvec = VALvecT[inds]
#RAvec = RAvecT[np.logical_or(VALvecT<low,VALvecT>high)]
#DECvec = DECvecT[np.logical_or(VALvecT<low,VALvecT>high)]
#VALvec = VALvecT[np.logical_or(VALvecT<low,VALvecT>high)]
plt.scatter(RAvec, DECvec, s=1, c=VALvec, transform = plt.gca().get_transform('world'), cmap='jet', vmin=low, vmax=high)
#leftPS_axpix =   wcs_list[0].wcs_world2pix(ps_wcs.wcs_pix2world(np.array([0,0],ndmin=2),0),0)[0,1]
#rightPS_axpix =  wcs_list[0].wcs_world2pix(ps_wcs.wcs_pix2world(np.array([0,4200-1],ndmin=2),0),0)[0,1]
#topPS_axpix =    wcs_list[0].wcs_world2pix(ps_wcs.wcs_pix2world(np.array([0,0],ndmin=2),0),0)[0,0]
#bottomPS_axpix = wcs_list[0].wcs_world2pix(ps_wcs.wcs_pix2world(np.array([2200-1,0],ndmin=2),0),0)[0,0]
##exps = 
#plt.imshow(ps,cmap='gray',vmin=0,vmax=1e4,extent=[leftPS_axpix, rightPS_axpix, topPS_axpix, bottomPS_axpix])
#plt.grid(color='red', ls='solid')
#
#axps.get_shared_x_axes().join(axps,aximg[0])
#axps.get_shared_y_axes().join(axps,aximg[0])



#axps.set_xlim([3600,7200])
#axps.set_xticklabels([])
#axps.set_yticklabels([])
#plt.show()

# %%

lst = ['dec-04/031-04.58400.wp.fits',
'dec-04/031-04.58450.wp.fits',
'dec-04/031-04.58500.wp.fits',
'dec-04/031-04.58650.wp.fits',
'dec-04/076-04.58400.wp.fits',
'dec-04/076-04.58450.wp.fits',
'dec-04/076-04.58500.wp.fits',
'dec-04/076-04.58550.wp.fits',
'dec-04/076-04.58650.wp.fits',
'dec-05/082-05.58400.wp.fits',
'dec-05/082-05.58450.wp.fits',
'dec-05/082-05.58500.wp.fits',
'dec-05/082-05.58550.wp.fits',
'dec-05/082-05.58650.wp.fits',
'dec-05/195-05.58450.wp.fits',
'dec-05/195-05.58500.wp.fits',
'dec-05/195-05.58550.wp.fits',
'dec-05/195-05.58600.wp.fits',
'dec-05/195-05.58650.wp.fits',
'dec-05/211-05.58450.wp.fits',
'dec-05/211-05.58500.wp.fits',
'dec-05/211-05.58550.wp.fits',
'dec-05/211-05.58600.wp.fits',
'dec-05/211-05.58650.wp.fits',
'dec-06/052-06.58400.wp.fits',
'dec-06/052-06.58450.wp.fits',
'dec-06/052-06.58500.wp.fits',
'dec-06/052-06.58550.wp.fits',
'dec-06/052-06.58650.wp.fits',
'dec-06/057-06.58400.wp.fits',
'dec-06/057-06.58450.wp.fits',
'dec-06/057-06.58500.wp.fits',
'dec-06/057-06.58550.wp.fits',
'dec-06/057-06.58650.wp.fits',
'dec-06/187-06.58450.wp.fits',
'dec-06/187-06.58500.wp.fits',
'dec-06/187-06.58550.wp.fits',
'dec-06/187-06.58600.wp.fits',
'dec-06/187-06.58650.wp.fits',
'dec+08/044+08.58400.wp.fits',
'dec+08/044+08.58450.wp.fits',
'dec+08/044+08.58500.wp.fits',
'dec+08/044+08.58550.wp.fits',
'dec+08/044+08.58650.wp.fits',
'dec+08/099+08.58400.wp.fits',
'dec+08/099+08.58450.wp.fits',
'dec+08/099+08.58500.wp.fits',
'dec+08/099+08.58550.wp.fits',
'dec+08/099+08.58600.wp.fits',
'dec+08/153+08.58400.wp.fits',
'dec+08/153+08.58450.wp.fits',
'dec+08/153+08.58500.wp.fits',
'dec+08/153+08.58550.wp.fits',
'dec+08/153+08.58600.wp.fits',
'dec+08/153+08.58650.wp.fits',
'dec+08/241+08.58400.wp.fits',
'dec+08/241+08.58450.wp.fits',
'dec+08/241+08.58500.wp.fits',
'dec+08/241+08.58550.wp.fits',
'dec+08/241+08.58600.wp.fits',
'dec+08/241+08.58650.wp.fits',
'dec+12/082+12.58400.wp.fits',
'dec+12/082+12.58450.wp.fits',
'dec+12/082+12.58500.wp.fits',
'dec+12/082+12.58550.wp.fits',
'dec+12/082+12.58600.wp.fits',
'dec-16/248-16.58400.wp.fits',
'dec-16/248-16.58450.wp.fits',
'dec-16/248-16.58500.wp.fits',
'dec-16/248-16.58550.wp.fits',
'dec-16/248-16.58600.wp.fits',
'dec-16/248-16.58650.wp.fits',
'dec-17/058-17.58400.wp.fits',
'dec-17/058-17.58450.wp.fits',
'dec-17/058-17.58500.wp.fits',
'dec-17/058-17.58550.wp.fits',
'dec-17/058-17.58650.wp.fits',
'dec+23/236+23.58400.wp.fits',
'dec+23/236+23.58450.wp.fits',
'dec+23/236+23.58500.wp.fits',
'dec+23/236+23.58550.wp.fits',
'dec+23/236+23.58600.wp.fits',
'dec+23/236+23.58650.wp.fits',
'dec+24/067+24.58400.wp.fits',
'dec+24/067+24.58450.wp.fits',
'dec+24/067+24.58500.wp.fits',
'dec+24/067+24.58550.wp.fits',
'dec+24/067+24.58650.wp.fits',
'dec+24/068+24.58400.wp.fits',
'dec+24/068+24.58450.wp.fits',
'dec+24/068+24.58500.wp.fits',
'dec+24/068+24.58550.wp.fits',
'dec+24/068+24.58650.wp.fits',
'dec+26/066+26.58400.wp.fits',
'dec+26/066+26.58450.wp.fits',
'dec+26/066+26.58500.wp.fits',
'dec+26/066+26.58550.wp.fits',
'dec+26/066+26.58650.wp.fits',
'dec+26/069+26.58400.wp.fits',
'dec+26/069+26.58450.wp.fits',
'dec+26/069+26.58500.wp.fits',
'dec+26/069+26.58550.wp.fits',
'dec+26/069+26.58650.wp.fits',
'dec+26/084+26.58400.wp.fits',
'dec+26/084+26.58450.wp.fits',
'dec+26/084+26.58500.wp.fits',
'dec+26/084+26.58550.wp.fits',
'dec+26/084+26.58600.wp.fits',
'dec+26/202+26.58450.wp.fits',
'dec+26/202+26.58500.wp.fits',
'dec+26/202+26.58550.wp.fits',
'dec+26/202+26.58600.wp.fits',
'dec+26/202+26.58650.wp.fits',
'dec+26/217+26.58450.wp.fits',
'dec+26/217+26.58500.wp.fits',
'dec+26/217+26.58550.wp.fits',
'dec+26/217+26.58600.wp.fits',
'dec+26/217+26.58650.wp.fits',
'dec-36/082-36.58400.wp.fits',
'dec-36/082-36.58450.wp.fits',
'dec-36/082-36.58500.wp.fits',
'dec-36/082-36.58550.wp.fits',
'dec+46/071+46.58400.wp.fits',
'dec+46/071+46.58450.wp.fits',
'dec+46/071+46.58500.wp.fits',
'dec+46/071+46.58550.wp.fits',
'dec+52/130+52.58400.wp.fits',
'dec+52/130+52.58450.wp.fits',
'dec+52/130+52.58500.wp.fits',
'dec+52/130+52.58550.wp.fits',
'dec+52/130+52.58600.wp.fits',
'dec+52/139+52.58400.wp.fits',
'dec+52/139+52.58450.wp.fits',
'dec+52/139+52.58500.wp.fits',
'dec+52/139+52.58550.wp.fits',
'dec+52/139+52.58600.wp.fits',
'dec+52/345+52.58400.wp.fits',
'dec+52/345+52.58450.wp.fits',
'dec+52/345+52.58500.wp.fits',
'dec+52/345+52.58600.wp.fits',
'dec+52/345+52.58650.wp.fits',
'dec+52/349+52.58400.wp.fits',
'dec+52/349+52.58450.wp.fits',
'dec+52/349+52.58500.wp.fits',
'dec+52/349+52.58600.wp.fits',
'dec+52/349+52.58650.wp.fits',
'dec+52/350+52.58400.wp.fits',
'dec+52/350+52.58450.wp.fits',
'dec+52/350+52.58500.wp.fits',
'dec+52/350+52.58600.wp.fits',
'dec+52/350+52.58650.wp.fits',
'dec+53/348+53.58400.wp.fits',
'dec+53/348+53.58450.wp.fits',
'dec+53/348+53.58500.wp.fits',
'dec+53/348+53.58600.wp.fits',
'dec+53/348+53.58650.wp.fits',
'dec+54/003+54.58400.wp.fits',
'dec+54/003+54.58450.wp.fits',
'dec+54/003+54.58500.wp.fits',
'dec+54/003+54.58600.wp.fits',
'dec+54/003+54.58650.wp.fits',
'dec+55/003+55.58400.wp.fits',
'dec+55/003+55.58450.wp.fits',
'dec+55/003+55.58500.wp.fits',
'dec+55/003+55.58600.wp.fits',
'dec+55/003+55.58650.wp.fits',
'dec+57/004+57.58400.wp.fits',
'dec+57/004+57.58450.wp.fits',
'dec+57/004+57.58500.wp.fits',
'dec+57/004+57.58600.wp.fits',
'dec+57/004+57.58650.wp.fits',
'dec+57/006+57.58400.wp.fits',
'dec+57/006+57.58450.wp.fits',
'dec+57/006+57.58500.wp.fits',
'dec+57/006+57.58600.wp.fits',
'dec+57/006+57.58650.wp.fits',
'dec+57/325+57.58400.wp.fits',
'dec+57/325+57.58450.wp.fits',
'dec+57/325+57.58500.wp.fits',
'dec+57/325+57.58600.wp.fits',
'dec+57/325+57.58650.wp.fits',
'dec+57/326+57.58400.wp.fits',
'dec+57/326+57.58450.wp.fits',
'dec+57/326+57.58500.wp.fits',
'dec+57/326+57.58600.wp.fits',
'dec+57/326+57.58650.wp.fits',
'dec+57/339+57.58400.wp.fits',
'dec+57/339+57.58450.wp.fits',
'dec+57/339+57.58500.wp.fits',
'dec+57/339+57.58600.wp.fits',
'dec+57/339+57.58650.wp.fits',
'dec+58/012+58.58400.wp.fits',
'dec+58/012+58.58450.wp.fits',
'dec+58/012+58.58500.wp.fits',
'dec+58/012+58.58600.wp.fits',
'dec+58/012+58.58650.wp.fits',
'dec+58/022+58.58400.wp.fits',
'dec+58/022+58.58450.wp.fits',
'dec+58/022+58.58500.wp.fits',
'dec+58/022+58.58550.wp.fits',
'dec+58/022+58.58650.wp.fits',
'dec+58/323+58.58400.wp.fits',
'dec+58/323+58.58450.wp.fits',
'dec+58/323+58.58500.wp.fits',
'dec+58/323+58.58600.wp.fits',
'dec+58/323+58.58650.wp.fits',
'dec+58/325+58.58400.wp.fits',
'dec+58/325+58.58450.wp.fits',
'dec+58/325+58.58500.wp.fits',
'dec+58/325+58.58600.wp.fits',
'dec+58/325+58.58650.wp.fits',
'dec+59/005+59.58400.wp.fits',
'dec+59/005+59.58450.wp.fits',
'dec+59/005+59.58500.wp.fits',
'dec+59/005+59.58600.wp.fits',
'dec+59/005+59.58650.wp.fits',
'dec+59/013+59.58400.wp.fits',
'dec+59/013+59.58450.wp.fits',
'dec+59/013+59.58500.wp.fits',
'dec+59/013+59.58550.wp.fits',
'dec+59/013+59.58600.wp.fits',
'dec+59/013+59.58650.wp.fits',
'dec+59/014+59.58400.wp.fits',
'dec+59/014+59.58450.wp.fits',
'dec+59/014+59.58500.wp.fits',
'dec+59/014+59.58550.wp.fits',
'dec+59/014+59.58600.wp.fits',
'dec+59/014+59.58650.wp.fits',
'dec+59/015+59.58400.wp.fits',
'dec+59/015+59.58450.wp.fits',
'dec+59/015+59.58500.wp.fits',
'dec+59/015+59.58550.wp.fits',
'dec+59/015+59.58600.wp.fits',
'dec+59/015+59.58650.wp.fits',
'dec+59/016+59.58400.wp.fits',
'dec+59/016+59.58450.wp.fits',
'dec+59/016+59.58500.wp.fits',
'dec+59/016+59.58550.wp.fits',
'dec+59/016+59.58600.wp.fits',
'dec+59/016+59.58650.wp.fits',
'dec+59/017+59.58400.wp.fits',
'dec+59/017+59.58450.wp.fits',
'dec+59/017+59.58500.wp.fits',
'dec+59/017+59.58550.wp.fits',
'dec+59/017+59.58600.wp.fits',
'dec+59/017+59.58650.wp.fits',
'dec+59/320+59.58400.wp.fits',
'dec+59/320+59.58450.wp.fits',
'dec+59/320+59.58600.wp.fits',
'dec+59/320+59.58650.wp.fits',
'dec+59/353+59.58400.wp.fits',
'dec+59/353+59.58450.wp.fits',
'dec+59/353+59.58500.wp.fits',
'dec+59/353+59.58600.wp.fits',
'dec+59/353+59.58650.wp.fits',
'dec+59/354+59.58400.wp.fits',
'dec+59/354+59.58450.wp.fits',
'dec+59/354+59.58500.wp.fits',
'dec+59/354+59.58600.wp.fits',
'dec+59/354+59.58650.wp.fits',
'dec+60/350+60.58400.wp.fits',
'dec+60/350+60.58450.wp.fits',
'dec+60/350+60.58500.wp.fits',
'dec+60/350+60.58600.wp.fits',
'dec+60/350+60.58650.wp.fits',
'dec+60/352+60.58400.wp.fits',
'dec+60/352+60.58450.wp.fits',
'dec+60/352+60.58500.wp.fits',
'dec+60/352+60.58600.wp.fits',
'dec+60/352+60.58650.wp.fits',
'dec+63/016+63.58400.wp.fits',
'dec+63/016+63.58450.wp.fits',
'dec+63/016+63.58500.wp.fits',
'dec+63/016+63.58550.wp.fits',
'dec+63/016+63.58600.wp.fits',
'dec+63/016+63.58650.wp.fits',
'dec+63/020+63.58400.wp.fits',
'dec+63/020+63.58450.wp.fits',
'dec+63/020+63.58500.wp.fits',
'dec+63/020+63.58550.wp.fits',
'dec+63/020+63.58600.wp.fits',
'dec+63/020+63.58650.wp.fits',
'dec+63/036+63.58400.wp.fits',
'dec+63/036+63.58450.wp.fits',
'dec+63/036+63.58500.wp.fits',
'dec+63/036+63.58550.wp.fits',
'dec+63/036+63.58650.wp.fits',
'dec+64/028+64.58400.wp.fits',
'dec+64/028+64.58450.wp.fits',
'dec+64/028+64.58500.wp.fits',
'dec+64/028+64.58550.wp.fits',
'dec+64/028+64.58650.wp.fits',
'dec+64/346+64.58400.wp.fits',
'dec+64/346+64.58450.wp.fits',
'dec+64/346+64.58500.wp.fits',
'dec+64/346+64.58600.wp.fits',
'dec+64/346+64.58650.wp.fits',
'dec+64/348+64.58400.wp.fits',
'dec+64/348+64.58450.wp.fits',
'dec+64/348+64.58500.wp.fits',
'dec+64/348+64.58600.wp.fits',
'dec+64/348+64.58650.wp.fits',
'dec+65/012+65.58400.wp.fits',
'dec+65/012+65.58450.wp.fits',
'dec+65/012+65.58500.wp.fits',
'dec+65/012+65.58600.wp.fits',
'dec+65/012+65.58650.wp.fits',
'dec+66/002+66.58400.wp.fits',
'dec+66/002+66.58450.wp.fits',
'dec+66/002+66.58500.wp.fits',
'dec+66/002+66.58600.wp.fits',
'dec+66/002+66.58650.wp.fits',
'dec+67/310+67.58400.wp.fits',
'dec+67/310+67.58450.wp.fits',
'dec+67/310+67.58550.wp.fits',
'dec+67/310+67.58600.wp.fits',
'dec+67/310+67.58650.wp.fits',
'dec+67/314+67.58400.wp.fits',
'dec+67/314+67.58450.wp.fits',
'dec+67/314+67.58550.wp.fits',
'dec+67/314+67.58600.wp.fits',
'dec+67/314+67.58650.wp.fits',
'dec+68/318+68.58400.wp.fits',
'dec+68/318+68.58450.wp.fits',
'dec+68/318+68.58550.wp.fits',
'dec+68/318+68.58600.wp.fits',
'dec+68/318+68.58650.wp.fits',
'dec+73/207+73.58400.wp.fits',
'dec+73/207+73.58450.wp.fits',
'dec+73/207+73.58500.wp.fits',
'dec+73/207+73.58550.wp.fits',
'dec+73/207+73.58600.wp.fits',
'dec+73/207+73.58650.wp.fits',
'dec+89/000+89.58400.wp.fits',
'dec+89/000+89.58450.wp.fits',
'dec+89/000+89.58500.wp.fits',
'dec+89/000+89.58550.wp.fits',
'dec+89/000+89.58600.wp.fits',
'dec+89/000+89.58650.wp.fits']

lstDF = pd.DataFrame(data=lst,columns=['flnms'])#,'DEC','RA','MJD'])
for i in np.arange(len(lst)):
    lstDF.loc[i,'DEC'] = int(lst[i][3:6])
    lstDF.loc[i,'RA']  = int(lst[i][7:10])
    lstDF.loc[i,'MJD'] = int(lst[i][14:19])

# %%

angS_lst = np.array([0.0000001,
18.865090538494,
21.8159593667273,
20.2287341130355,
2.9588831516873,
3.28214036590159,
3.3790010280403,
3.20326207644291,
8.78039544730363,
7.67471345995702,
7.85868516165707,
7.7986094794739,
0.0000001,
4.64661201871526,
6.47420227771659,
6.44521883872748,
13.107019336176,
13.3949460168263,
7.37072027440096,
7.42398880633607,
7.42034166772961,
5.06797554702954,
4.73516374905138,
4.56995367709484,
5.80048046321593,
6.17866249373751,
6.2517129523191,
6.69066764176919,
6.70004036870449,
6.86611480278704,
6.70926194657549,
6.84025775789838,
6.75326723280613,
21.8847969420692,
7.68276466910284,
7.55169257292501,
7.70632104845552,
6.22571011138716,
6.2504080556956,
0.0000001,
20.7104388164822,
19.9633822318894,
20.0028949818296,
19.7718146120123,
6.2916362937626,
6.33972732985148,
6.19825710436395,
20.235414625287,
19.9784441166155,
14.4193431190924,
8.89625938078421,
9.39859238093779,
13.948843822053,
13.9571181520441,
13.9720393108846,
14.0160653654812,
14.0570825751024,
15.6650112018707,
0.0000001,
55.6434543028561,
0.0000001,
53.8698562232955,
55.0597466793822,
61.4907566742975,
0.0000001,
0.0000001,
55.5394945965802,
56.5128697871309,
0.0000001,
0.0000001,
0.0000001,
83.6343517223929,
0.0000001,
96.6316271652384,
70.3323581616512,
84.6349376371537,
88.2534776320181,
0.0000001,
0.0000001,
0.0000001,
0.0000001,
0.0000001,
116.09474182017,
0.0000001,
122.491096206875,
20.8866133640991])
z=np.zeros(angS_lst.shape)
rho=np.zeros(angS_lst.shape)
t = 447#447 #300
D = 8900#8900 #11090
for i in np.arange(len(angS_lst)):
    z[i], rho[i] = LEprcs.z_ly_atD(angS_lst[i],t,D,verbose=False)