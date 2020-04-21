#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:45:14 2020

@author: roeepartoush
"""

import LEplots
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import ZScaleInterval

class IndexTracker(object):
    def __init__(self, ax, X, titles, w_lst):
        self.Zscale = ZScaleInterval()
        self.titles = titles
        self.w_lst = w_lst
        self.ind = 0
        self.clim = self.Zscale.get_limits(X[:,:,self.ind])
        if ax is not None:
            self.ax = ax
        else:
            self.ax = plt.subplot(111,projection=self.w_lst[self.ind])
        self.ax.figure.canvas.mpl_connect('key_press_event', self.onpress)
        self.X = X
        rows, cols, self.slices = X.shape
        self.im = self.ax.imshow(self.X[:, :, self.ind],vmin=self.clim[0],vmax=self.clim[1],cmap='gray')
        self.ax.grid(color='white', ls='solid')
        self.update()

    def onscroll(self, event):
#        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
    
    def onpress(self, event):
#        print("%s %s" % (event.button, event.step))
        if event.key == 'right':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
#        self.clim = self.Zscale.get_limits(X[:,:,self.ind])
#        self.ax = plt.subplot(111,projection=self.w_lst[self.ind])
#        self.im = ax.imshow(self.X[:, :, self.ind],vmin=self.clim[0],vmax=self.clim[1],cmap='gray')
#        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title(self.titles[self.ind])
        self.ax.set_transform(self.w_lst[self.ind])
#        self.ax.grid(color='white', ls='solid')
        self.im.axes.figure.canvas.draw()


# %%
fig, ax = plt.subplots(1, 1)

#X = np.random.rand(20, 20, 40)
X=[]
for file in files2:
    mat = fits.getdata(file + '.fits')
    X.append(mat)
X = np.stack(X,axis=2)


tracker = IndexTracker(ax, X)


#fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('key_press_event', tracker.onpress)

plt.show()