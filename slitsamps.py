#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:13:19 2020

@author: roeepartoush
"""
import os, argparse


def define_options(parser=None, usage=None, conflict_handler='resolve'):
    parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)
    
    parser.add_argument("RA", type=float, help=("RA [deg]"))
    parser.add_argument("Dec", type=float, help=("Dec [deg]"))
    parser.add_argument("PA", type=float, help=("Position Angle [deg]"))
    parser.add_argument("Slit_Length", type=float, help=("Slit Length [arcsec]"))
    parser.add_argument("Samp_Step", type=float, help=("Step size [arcsec] for sampling along the slit"))
    parser.add_argument("Color", type=float, help=("Color {ds9 region file}"))
#    parser.add_argument("-i","--inputlistfilename", default=None,  help=("If specified, then the list of stsci-transients.stsci.edu/atlas/dec???/index.html#?????? in the file is parsed for RA/Dec"))
#    parser.add_argument("--RAmax", type=int, default=None, help=("if RAmax is specified, then the RA is looped from RA to RAmax in steps of one"))
#    parser.add_argument("-d","--inputbasedir", default=inputbasedir, help=("input base directury that contains the input files (default=%(default)s)"))
#    parser.add_argument("-o","--outputbasedir", default=outputbasedir, help=("output base directory (default=%(default)s)"))
#    parser.add_argument("-binsize","-b", type=int, default=3, help=("binsize for smoothing"))
#    parser.add_argument('--skipifexists',action="store_true",default=False,help=['if the output files already exists, skip redoing them!'])
#    parser.add_argument('-s','--saveimages',action="store_true",default=False,help='Save all images into the saveim directory, for further investigating the candidates in these images')
#    parser.add_argument('--verbose', '-v', action='count',default=0)
#    parser.add_argument('--debug',action="store_true",default=False)
    
    return(parser)
    

if __name__ == "__main__":
    print('HELLO!!!')
    args = define_options().parse_args()
#    for att in dir(args):
#        print(att+': '+str(getattr(args,att)))
    sg