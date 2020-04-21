#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:42:59 2020

@author: roeepartoush
"""

import numpy as np
import matplotlib.pyplot as plt

# %%
plt.close('all')
start = 0
dur = 1
Fs = 3e3
N = int(dur*Fs)+1-1
t = np.linspace(start,start+dur,N,endpoint=False)

y = np.fft.fftshift(np.sinc(0.1*1e3*(t-0.49)))
plt.subplot(311)
plt.scatter(t,y,s=1)
plt.xlabel('[sec]')

Y = np.fft.fftshift(np.fft.fft(y))
ax1=plt.subplot(312)
f = np.linspace(-0.5,0.5,N,endpoint=False)*Fs
plt.scatter(f,np.abs(Y),s=1)
plt.xlabel('[Hz]')

plt.subplot(313,sharex=ax1)
f = np.linspace(-0.5,0.5,N,endpoint=False)*Fs
plt.plot(f,np.angle(Y)/np.pi)
plt.xlabel('[Hz]')
mng=plt.get_current_fig_manager()
mng.window.setGeometry(1680, -208, 1600, 1178)