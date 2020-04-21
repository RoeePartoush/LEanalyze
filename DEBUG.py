#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:30:32 2019

@author: roeepartoush
"""

point=np.array([[970.0],[2050.0]])#([[1600.0],[1850.0]])
#[s_mat, zzz] = LeT.plot_cut(image_data,np.array([7,8]),np.array([[1600.0],[1850.0]]),70,600,aximg,15)
[s_mat, zzz] = LeT.plot_cut(image_data, mask_img, noise_img, np.arange(len(image_files)),point,70,600,aximg,1)