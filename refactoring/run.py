#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:58:34 2020

@author: roeepartoush
"""

import argparse, io, os

# local modules
import File2Data as F2D
import LEutils as LEu
import FluxProf


class default_params:
    # Default setup to be used when no arguments are provided by the user.
    # 
    # == NOTE: This is temporary and will be replaced by a better mechanism for 
    # == default setup later on (example files).
    # 
    def __init__(self):
        ## Light Curves table
        LChome_dir = '/Users/roeepartoush/Documents/Research/Data/light_curves'
        LCs_file = LChome_dir + '/SNIa_model_mlcs2k2_v007_early_smix_z0_av0_desr_ab.txt'
        self.LCs_file = LCs_file
        
        ## Image List
        files=[]
        root='/Users/roeepartoush/Documents/Research/Data/swarp_test/tycA1/'
        files1=['KECK/tyc4419_1_DEIMOS_5_6_coadd',
                '20131202/tyc4419_tyc4519_20131202_coadd',
                'KECK/tyc4419_1_R_LRIS_131228_coadd',
                '20140226/tyc4419_tyc4519_20140226_coadd']
        for file in files1:
            files.append(root+file)
        self.imlist = files
        
        ## Flux Profile Parameters
        self.prof_params = LEu.ProfileParams(ra=12.39251174, dec=58.76486208, pa=180, length=18, width=7.5)


def define_options(parser=None, usage=None, conflict_handler='resolve'):
    parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)    
    parser.add_argument("-LCtable_file", type=str, default=None, help=("light curves file"))
    parser.add_argument("-imlist", type=str, default=None, help=("file containing a list of images"))
    parser.add_argument("-profparams", type=str, default=None, help=("file containing profile parameters"))
    # parser.add_argument("inDir", type=str, help=("input directory (contains *diff.fits files)"))
    # parser.add_argument("-outDir", type=str, default=None, help=("output directory for the gif file. if not specified, set to inDir"))
    # parser.add_argument("-filesList", type=str, default=None, help=("Path of file containing list of *diff.fits files. If set to OUT or IN, list of *diff.fits files is assumed to be {outDir OR inDir}/fileslist.txt, respectively. If not specified, all *diff.fits files in inDir are used to make the gif file."))
    # parser.add_argument("-Nmedfilt", type=str, default="3", help=("used for median filter size and downsampling ratio. default is 3"))
    # parser.add_argument("-delays", type=str, default="[100,500]", help=("list of desired delay(s) between each frame in the gif, in miliseconds. default is [100,500]. format for input: -delays=[100,500,...]"))
    # parser.add_argument("-maskColor", type=str, default=None, help=("color in 0,0,0<=[R,G,B]<=1,1,1 for masked areas in the image. if not specified, no masking is done. for example, -maskColor=[1,1,1] will produce white masking. NOTE: this assumes inDir contains a mask file with filname corresponding to each *diff.fits file: *diff.mask.fits"))
    # parser.add_argument('-T', action='store_true', help=("Time normalization flag. If given, delay(s) are normalized to maximum difference between consecutive diff images."))
    return(parser)


class get_vars:
    def __init__(self,args):
        self.fatal_error = False
        self.default_params = default_params()
        
        self.getLCtable(self,args)
        self.getImages(self,args)
        self.getProfileParams(self,args)
    
    
    def getLCtable(self,args):
        # try:
        if args.LCtable_file is None:
            LCs_file = self.default_params.LCs_file
        # == PLACEHOLDER ==
        # else if {LCtable_file is a path to a file...} then:
        #    {try to parse it as light curve data...}
        # == PLACEHOLDER ==
        # 
        #  catch: self.fatal_error = True
        self.LCtable = F2D.LightCurves(LCs_file)
    
    def getImages(self,args):
        # try:
        if args.imlist is None:
            imlist = self.default_params.imlist
        # == PLACEHOLDER ==
        # else if {imlist is a path to a file...} then:
        #    {try to parse it as a list of images...}
        # == PLACEHOLDER ==
        # 
        #  catch: self.fatal_error = True
        self.IM_df = F2D.IM_df_init(imlist)
    
    def getProfileParams(self,args):
        # try:
        if args.profparams is None:
            prof_params = self.default_params.prof_params
        # == PLACEHOLDER ==
        # else if {profparams is a path to a file or something...} then:
        #    {try to parse it as profile parameters...}
        # == PLACEHOLDER ==
        # 
        #  catch: self.fatal_error = True
        self.prof_params = prof_params





if __name__ == "__main__":
    # set up environment variables, read input data, configuration, etc.
    args = define_options().parse_args()
    env_vars = get_vars(args)
    
    # check if there's anything wrong with input or data
    if env_vars.fatal_error:
        raise SystemExit(0)
    else:
        pass
    
    # get flux profiles from images
    
    
    # perform apparent motion estimation
    
    
    # perform temporal smearing estimation (caused by dust width)
    
    
    # write output
    

