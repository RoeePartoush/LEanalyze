#!/usr/bin/env python
import sys, os, re, math, argparse, types, copy, shutil, glob, datetime, socket

import astropy.io.fits as fits
import scipy
from astropy import stats
import numpy.ma as ma
import numpy as np
import pylab
from   matplotlib.ticker import FormatStrFormatter,MultipleLocator

def rebin(a, f):
    M, N = a.shape
    print("Shape: %d %d" % (M, N))
    #m, n = new_shape
    m=int(M/f)
    n=int(N/f)
    print('\nM='+str(M)+', N='+str(N)+', f='+str(f)+', m='+str(m)+', n='+str(n)+'.\n')
    if m<M:
        return a.reshape((m,f,n,f)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)


def makepath(path,raiseError=1):
    if path == '':
        return(0)
    if not os.path.isdir(path):
        os.makedirs(path)
        if not os.path.isdir(path):
            if raiseError == 1:
                raise RuntimeError('ERROR: Cannot create directory %s' % path)
            else:
                return(1)
    return(0)

def makepath4file(filename,raiseError=1):
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        return(makepath(path,raiseError=raiseError))
    else:
        return(0)

def rmfile(filename,raiseError=1,gzip=False):
    " if file exists, remove it "
    if os.path.lexists(filename):
        os.remove(filename)
        if os.path.isfile(filename):
            if raiseError == 1:
                raise RuntimeError('ERROR: Cannot remove %s' % filename)
            else:
                return(1)
    if gzip and os.path.lexists(filename+'.gz'):
        os.remove(filename+'.gz')
        if os.path.isfile(filename+'.gz'):
            if raiseError == 1:
                raise RuntimeError('ERROR: Cannot remove %s' % filename+'.gz')
            else:
                return(2)
    return(0)

def rmfiles(filenames,raiseError=1,gzip=False):
    if not (type(filenames) is list):
        raise RuntimeError("List type expected as input to rmfiles")
    errorflag = 0
    for filename in filenames:
        errorflag |= rmfile(filename,raiseError=raiseError,gzip=gzip)
    return(errorflag)

class figATLASstacksclass:
    def __init__(self):
        self.verbose = 0
        self.debug = False

        self.RA = None
        self.Dec = None
        self.figsuffix = 'jpg'

        # list that will save all the errors
        self.errors=[]
        self.errorflag = False

    def adderror(self,error,skipscreen=False):
        self.errors.append('### '+str(datetime.datetime.now()))
        if isinstance(error, str):
        #if (type(error) is types.StringType) or (type(error) is types.UnicodeType):
            if not skipscreen:
                print(error)
            self.errors.append(error)
        elif isinstance(error, list):
        #elif type(error) is types.ListType:
            if not skipscreen:
                for s in error: print(s)
            self.errors.extend(error)
        else:
            print (error)
            raise RuntimeError("Cannot add ERROR!!!")
        self.errorflag = True
            
    def showerrors(self):
        if len(self.errors)==0:
            if self.verbose>1:
                print('NO ERRORS!')
        else:
            print('#### There were errors! ###')
            for error in self.errors:
                print(error)
        
    def saveerrors(self):
        errorfilename = self.getoutputfilename(None,'error')

        if len(self.errors)==0:
            if self.verbose>1:
                print('NO ERRORS!')
        else:
            print(errorfilename)
            makepath4file(errorfilename)
            f = open(errorfilename,'w')
            for l in self.errors: f.writelines(l+'\n')
            f.close()


    def define_options(self, parser=None, usage=None, conflict_handler='resolve'):
        if parser is None:
            parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

        inputbasedir = '.'
        outputbasedir = '.'
        hostname = socket.gethostname()
        if re.search('^arminmac2',hostname):
            inputbasedir = '/Users/arest/le/ATLAS/allstacks'
            outputbasedir = '/Users/arest/le/ATLAS/allstacks/test'
        elif re.search('^atlas',hostname):
            inputbasedir = '/atlas/stack/1'
            outputbasedir= '/atlas/stack/1/lestacks'
        elif re.search('^plhstproc',hostname):
            inputbasedir = '/ifs/cs/projects/armin1/data/atlas/stacks2'
            outputbasedir = '/grp/websites/stsci-transients.stsci.edu/atlas'
        else:
            print('***\n*** WARNING! hostname={}, therefore could not automatically determine the input and output basedirs! Setting it to \'.\'\n***'.format(hostname))
        
        parser.add_argument("RA", type=int, help=("RA"))
        parser.add_argument("Dec", type=int, help=("Dec"))
        parser.add_argument("-i","--inputlistfilename", default=None,  help=("If specified, then the list of stsci-transients.stsci.edu/atlas/dec???/index.html#?????? in the file is parsed for RA/Dec"))
        parser.add_argument("--RAmax", type=int, default=None, help=("if RAmax is specified, then the RA is looped from RA to RAmax in steps of one"))
        parser.add_argument("-d","--inputbasedir", default=inputbasedir, help=("input base directury that contains the input files (default=%(default)s)"))
        parser.add_argument("-o","--outputbasedir", default=outputbasedir, help=("output base directory (default=%(default)s)"))
        parser.add_argument("-binsize","-b", type=int, default=3, help=("binsize for smoothing"))
        parser.add_argument('--skipifexists',action="store_true",default=False,help=['if the output files already exists, skip redoing them!'])
        parser.add_argument('-s','--saveimages',action="store_true",default=False,help='Save all images into the saveim directory, for further investigating the candidates in these images')

        parser.add_argument('--verbose', '-v', action='count',default=0)
        parser.add_argument('--debug',action="store_true",default=False)
        
        return(parser)

    def getMJD(self,filename):
        m=re.search('\d+[\+|\-]\d+\.(\d+)\.wp',os.path.basename(filename))
        if m!=None:
            return(int(m.groups()[0]))
        else:
            raise RuntimeError('Could not get MJD from filename %s' % filename)
        
    def getinputfilenames(self,RA,Dec,inputbasedir):
        self.RA = RA
        self.Dec = Dec
        self.radecID = '%03d%+03d' % (RA,Dec)

        #print(self.radecID)
        #sys.exit(0)

        self.inputdir = '%s/dec%+03d' % (os.path.abspath(inputbasedir),self.Dec)
        searchpattern = '%s/%s.*.wp' % (self.inputdir,self.radecID)
        if self.verbose:
            print('Searching for files in %s, using search patterns %s' % (self.inputdir,searchpattern))

        # search for the images
        imlist = glob.glob(searchpattern)
        imlist.sort()

        
        # combine all filters to one list, sorted by MJD
        tmplist = []
        for filename in imlist:
            tmplist.append((self.getMJD(filename),filename))
        tmplist.sort()
        print(tmplist)
        alllist=[]
        for (MJD,filename) in tmplist:
            alllist.append(filename)

        return(alllist)

    def getoutbasename(self,infilename):
        (path,basename)=os.path.split(infilename)
        (path,subdir)=os.path.split(path)
        
        outbasename = '%s/%s/%s' % (self.outputbasedir,subdir,basename)
        print('ROEE in outbasename - outputbasedir='+str(self.outputbasedir)+' ***\n')
        return(outbasename)
        
    def mkimageplot(self,image,zmin,zmax,outputfig,trimx=None,trimy=None,xfigsize=8.0,yfigsize=8.0,
                    outputfigTN=None,xfigsizeTN=1.0,yfigsizeTN=1.0,
                    infostring=None,infostringfontsize=14):
        pylab.clf()
        fig = pylab.figure()
        fig.set_size_inches(xfigsize,yfigsize)
        pylab.gray()
        pylab.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98)
        xmin = ymin = 0
        xmax=image.shape[1]
        ymax=image.shape[0]
        #xmax = hdr['NAXIS1']
        #ymax = hdr['NAXIS2']        
        if trimx!=None:
            xmin = int(0.5*trimx)
            xmax = image.shape[1]-int(0.5*trimx)
        if trimy!=None:
            ymin = int(0.5*trimy)
            ymax = image.shape[0]-int(0.5*trimy)
        trimimage = image[ymin:ymax,xmin:xmax]
        
        sp = pylab.subplot(111)
        sp.axis('off')
        sp.set_xticklabels([])
        sp.set_yticklabels([])
        sp.set_xticks([])
        sp.set_yticks([])
       
        #im = pylab.imshow(image,vmin=zmin,vmax=zmax,aspect='equal',origin='lower')
        im = sp.imshow(trimimage,vmin=zmin,vmax=zmax,origin='lower')
 
        ymin, ymax = sp.get_ylim()
        xmin, xmax = sp.get_xlim()
        ymax = trimimage.shape[0]*1.02
        if infostring!=None:
            sp.text(1,trimimage.shape[0]*0.995,infostring,color="black",backgroundcolor="white",fontsize=infostringfontsize,horizontalalignment="left",verticalalignment="bottom")
        sp.set_xlim((xmin, xmax))
        sp.set_ylim((ymin, ymax))

        print('Saving ', outputfig)
        pylab.savefig(outputfig)
        if outputfigTN != None:
            #figsize = fig.get_size_inches()
            #figsize *= 1.0/TNscale
            #fig.set_size_inches(figsize[0], figsize[1] )
            fig.set_size_inches(xfigsizeTN, yfigsizeTN )
            if self.verbose>1:print('Saving',outputfigTN)
            fig.savefig(outputfigTN)
        pylab.clf()
        #pylab.show()
        pylab.close()

    
    def mkfigs4files(self,filenames,saveimages=False,
                     binsize=4,sigmanoisecut4mask=1.5,z_Nsigma=1.5,z_Nsigma_minus=1.5,trimx=None,trimy=None,skipifexists=False):
        for filename in filenames:
            print('### file: %s binsize: %d' % (filename,binsize))
            print('\nROEE: filename='+str(filename)+'\n')
            outbasename = self.getoutbasename(filename)
            print('\nROEE: outbasename='+str(outbasename)+'\n')
            print('outbasename: %s' % outbasename)
            figfilename = '%s.%s' % (outbasename,self.figsuffix)
            makepath4file(figfilename)
            binfigfilename  = '%s.bin.%s' % (outbasename,self.figsuffix)
            
            if skipifexists:
                if os.path.isfile(figfilename) and os.path.isfile(binfigfilename):
                    print('figures %s and %s already exists, skipping recreating them since skipifexists option!' % (figfilename,binfigfilename))
                    continue
                              
            fitsim = fits.open(filename)
            print(fitsim.info())
            #results = stats.sigma_clipped_stats(fitsim[1].data[1,:,:],sigma=3)
            median_noise = np.sqrt(np.median(fitsim[1].data[1,:,:]))
            print('Median noise: %.3f' % median_noise)
            mask = scipy.where(fitsim[1].data[1,:,:]>=sigmanoisecut4mask*sigmanoisecut4mask*median_noise*median_noise,0.0,1.0)# used to be: sigmanoisecut4mask*median_noise,0.0,1.0) 12/13/2019
            fitsim_masked = 1.0*fitsim[1].data[0,:,:]*mask
            #rebinim = rebin(fitsim[1].data[0,:,:],binsize)
            rebinim = rebin(fitsim_masked,binsize)

            if saveimages:
                fitsfilename = '%s.fits' % (outbasename)
                noisefitsfilename = '%s.noise.fits' % (outbasename)
                fitsbinfilename = '%s.bin.fits' % (outbasename)
                fitsim[1].header['RADESYS']='FK5'
                fitsim[1].header['CTYPE1']='RA---LIN'
                fitsim[1].header['CTYPE2']='DEC--LIN'
                fitsim[1].header['CUNIT1']='deg'
                fitsim[1].header['CUNIT2']='deg'
                print('Saving fits file %s' % (fitsfilename))
                fits.writeto(fitsfilename,fitsim[1].data[0,:,:],fitsim[1].header,overwrite=True)
                print('Saving noise fits file %s' % (noisefitsfilename))
                fits.writeto(noisefitsfilename,fitsim[1].data[1,:,:],fitsim[1].header,overwrite=True)
                print('Saving binned fits file %s' % (fitsbinfilename))
                fits.writeto(fitsbinfilename,rebinim,fitsim[1].header,overwrite=True)
                
            zmin = -z_Nsigma_minus*median_noise
            zmax = z_Nsigma*median_noise

            self.mkimageplot(fitsim_masked,
                             zmin,zmax,figfilename,
                             trimx=trimx,trimy=trimy,
                             xfigsize=16.0,yfigsize=10.0,infostring=os.path.basename(filename))

            self.mkimageplot(rebinim,
                             zmin/binsize*2.0,zmax/binsize*2.0,binfigfilename,
                             None,None,
                             xfigsize=16.0,yfigsize=10.0,infostring='%s, binned with %d' % (os.path.basename(filename),binsize))
           
if __name__ == "__main__":
    print('HELLO!!!')
    tstart = datetime.datetime.now()
    
    figATLASstacks = figATLASstacksclass()
    parser = figATLASstacks.define_options()
    args = parser.parse_args()
    
    figATLASstacks.verbose = args.verbose
    figATLASstacks.debug = args.debug
    if args.saveimages:
        figATLASstacks.outputbasedir = args.outputbasedir#'/grp/websites/stsci-transients.stsci.edu/atlas/cand'
    else:
        figATLASstacks.outputbasedir = args.outputbasedir


    if not(args.inputlistfilename is None):
        RAlist=[]
        DEClist=[]
        lines = open(args.inputlistfilename, 'r').readlines()
        for line in lines:
            if figATLASstacks.verbose>1:print(line)
            m = re.search('index\.html\#(\S\S\S)(\S\S\S)',line)
            if m is None:
                raise RuntimeError("Could not parse %s!" % line)
            (RA,Dec)= m.groups()
            if figATLASstacks.verbose>1:print('%s %s' % (RA,Dec))
            RAlist.append(int(RA))
            DEClist.append(int(Dec))

        for i in xrange(len(RAlist)):
            RA=RAlist[i]
            Dec=DEClist[i]
            print("\n#####\n##### RA={} Dec={}\n#####\n".format(RA,Dec))
            inputfilenames = figATLASstacks.getinputfilenames(RA,Dec,args.inputbasedir)
            if len(inputfilenames)==0:
                print('NO IMAGES!!! Skipping....')
                continue
            figATLASstacks.mkfigs4files(inputfilenames,binsize=args.binsize,skipifexists=args.skipifexists,saveimages=args.saveimages)
            
        sys.exit(0)
        
    if args.RAmax is None:
        RArange = [args.RA]
    else:
        RArange = range(args.RA,args.RAmax+1)

    print('RArange',RArange)

    for RA in RArange:
        print("\n#####\n##### RA={} Dec={}\n#####\n".format(RA,args.Dec))
        inputfilenames = figATLASstacks.getinputfilenames(RA,args.Dec,args.inputbasedir)
        print('\n'+args.inputbasedir+'\n'+args.outputbasedir+'\n'+str(inputfilenames))
        if len(inputfilenames)==0:
            print('NO IMAGES!!! Skipping....')
            continue
        figATLASstacks.mkfigs4files(inputfilenames,binsize=args.binsize,skipifexists=args.skipifexists,saveimages=args.saveimages)

