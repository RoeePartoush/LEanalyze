#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:12:25 2020

@author: roeepartoush
"""

#!/usr/bin/env python
import argparse, io, os
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter

from astropy import wcs
from astropy.io import fits
from astropy.visualization import ZScaleInterval


def define_options(parser=None, usage=None, conflict_handler='resolve'):
    parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)    
    parser.add_argument("inDir", type=str, help=("input directory (contains *diff.fits files)"))
    parser.add_argument("-outDir", type=str, default=None, help=("output directory for the gif file. if not specified, set to inDir"))
    parser.add_argument("-filesList", type=str, default=None, help=("Path of file containing list of *diff.fits files. If set to OUT or IN, list of *diff.fits files is assumed to be {outDir OR inDir}/fileslist.txt, respectively. If not specified, all *diff.fits files in inDir are used to make the gif file."))
    parser.add_argument("-Nmedfilt", type=str, default="3", help=("used for median filter size and downsampling ratio. default is 3"))
    parser.add_argument("-delays", type=str, default="[100,500]", help=("list of desired delay(s) between each frame in the gif, in miliseconds. default is [100,500]. format for input: -delays=[100,500,...]"))
    parser.add_argument("-maskColor", type=str, default=None, help=("color in 0,0,0<=[R,G,B]<=1,1,1 for masked areas in the image. if not specified, no masking is done. for example, -maskColor=[1,1,1] will produce white masking. NOTE: this assumes inDir contains a mask file with filname corresponding to each *diff.fits file: *diff.mask.fits"))
    parser.add_argument('-T', action='store_true', help=("Time normalization flag. If given, delay(s) are normalized to maximum difference between consecutive diff images."))
    return(parser)

def filesfile2fileslist(fls_lst_file,in_dir):
    fls_lst = []
    for f in open(fls_lst_file).read().split():
        if f[-5:]=='.diff':
            f = f+'.fits'
        f_legit = (os.path.isfile(os.path.join(in_dir,f)) and f[-5:]=='.fits')
        if f_legit:
            fls_lst.append(f)
    return fls_lst

class args2vars:
    def __init__(self,args):
        self.fatal_error = False
        if os.path.isdir(args.inDir):
            self.in_dir = args.inDir
        else:
            print('couldn\'t parse inDir. raising fatal error.')
            self.fatal_error = True
            return
        
        if args.outDir is None:
            self.out_dir = args.inDir
        elif os.path.isdir(args.outDir):
            self.out_dir = args.outDir
        else:
            print('couldn\'t parse outDir. raising fatal error.')
            self.fatal_error = True
            return
        
        if args.filesList is not None:
            if args.filesList == 'IN':
                fls_lst_file = os.path.join(args.inDir, 'fileslist.txt')
            elif args.filesList == 'OUT':
                fls_lst_file = os.path.join(args.outDir, 'fileslist.txt')
            else:
                if not os.path.isfile(args.filesList):
                    args.filesList = os.path.join(args.filesList, 'fileslist.txt')
                fls_lst_file = args.filesList
            if (os.path.isfile(fls_lst_file) and os.path.splitext(fls_lst_file)[1]=='.txt'):
                self.fls_lst = filesfile2fileslist(fls_lst_file,self.in_dir)
            else:
                self.fls_lst = []
        else:
            mypath = self.in_dir
            self.fls_lst = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and f[-10:]=='.diff.fits')]
        if len(self.fls_lst)==0:
            print('No *.diff.fits images found! raising fatal error.')
            self.fatal_error = True
            return
        
        self.N_medfilt = int(args.Nmedfilt)
        self.gif_delays = list(map(int, args.delays.strip('[]').split(',')))
        if args.maskColor is not None:
            self.maskColor = list(map(float, args.maskColor.strip('[]').split(',')))
        else:
            self.maskColor = None
        
        self.Tnormalize = args.T
        self.sorted_fls_lst, self.mjd_dates = sort_diffs(self.fls_lst, self.in_dir, return_dates=True)
        if self.Tnormalize:
            mjd_diff = np.diff(self.mjd_dates)
            norm_diff = np.concatenate((mjd_diff/np.max(mjd_diff),np.array([1])))
            for i in np.arange(len(self.gif_delays)):
                delays = (norm_diff*self.gif_delays[i]).astype(int)
                delays[delays==0] = 1
                self.gif_delays[i] = list(delays)


def WCS_downsample(w,N,shape,inplace=False):
    if inplace:
        w2 = w
    else:
        w2 = w.deepcopy()
    w2.wcs.crpix = w.wcs.crpix/N
    w2.wcs.cd = w.wcs.cd*N
    w2.array_shape = shape
    if not inplace:
        return w2
    else:
        return

def sort_diffs(fls_lst,in_dir,return_dates=False):
    dates=[]
    for file in fls_lst:
        dates.append(fits.getheader(os.path.join(in_dir,file))['MJD-OBS'])
    sorted_fls_lst = [x for _, x in sorted(zip(dates,fls_lst.copy()), key=lambda pair: pair[0])]
    sorted_dates   = [y for y, _ in sorted(zip(dates,fls_lst.copy()), key=lambda pair: pair[0])]
    if return_dates:
        return sorted_fls_lst, np.array(sorted_dates)
    else:
        return sorted_fls_lst

def mask2bool(mask,mode='suspicious'): 
    if mode=='safe':
        boolmat = mask==0 # only "safe" pixels
    elif mode=='suspicious':
        boolmat = np.logical_or(mask==0,mask<0x8000) # permit "suspicious" pixels also
    elif mode=='suspicious-only':
        boolmat = np.logical_and(mask!=0,mask<0x8000) # permit only "suspicious" pixels, without "safe" pixels
    # boolmat=True represents GOOD pixels (boolmat is inteded to be used for image matrix indexing)
    return boolmat

def makeGifs(varis):
    images=[]
    Zscale = ZScaleInterval()
    plt.ioff()
    N = varis.N_medfilt
    for file in varis.sorted_fls_lst:
        fits_flnm = os.path.join(varis.in_dir,file)
        print(fits_flnm)
        mat = fits.getdata(fits_flnm).astype(float)
        if varis.maskColor is not None:
            mask_fits_flnm = fits_flnm[0:-5]+'.mask.fits'
            if os.path.isfile(mask_fits_flnm):
                # setting masked pixels to NaN makes them transparent for imshow
                mat[~mask2bool(fits.getdata(mask_fits_flnm))] = np.nan
        w = wcs.WCS(fits.getheader(fits_flnm))
        if N>1:
            mat = median_filter(mat,size=N)
            mat = mat[::N,::N]
            WCS_downsample(w, N, mat.shape, inplace=True)
        
        clim = Zscale.get_limits(mat)
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111,projection=w)
        if varis.maskColor is not None:
            # transparent (masked) pixels will show the background color specified by set_facecolor.
            # for example: ax.set_facecolor((1,1,1)) or ax.set_facecolor([1,1,1]) means white background
            ax.set_facecolor(varis.maskColor)
        plt.imshow(mat, vmin=clim[0], vmax=clim[1], origin='lower', cmap='gray', interpolation='none')
        plt.title(file)
        plt.grid(color='white', ls='solid')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf).copy()
        buf.close()
        images.append(im)
    for delay in varis.gif_delays: # delay between each frame in the gif [miliseconds] (single value for constant frame rate or list for varying frame rate)
        if varis.Tnormalize:
            flnm_delay = 'Tnorm_'+str(delay[-1])
        else:
            flnm_delay = delay
        write_path = os.path.join(varis.out_dir,'diff_'+str(flnm_delay)+'_ms_animation.gif')
        images[0].save(write_path, save_all=True, append_images=images[1:], duration=delay, loop=0)
        print('File '+write_path+' was written succesfully.')
    return


if __name__ == "__main__":
    args = define_options().parse_args()
    varis = args2vars(args)
    if not varis.fatal_error:
        makeGifs(varis)
    else:
        print('something went wrong.. bye.')