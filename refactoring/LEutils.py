#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:40:24 2020

@author: roeepartoush
"""


from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u


class ProfileParams:
    def __init__(self, **kwargs):
        origin_ra_deg, origin_dec_deg   = kwargs["ra"], kwargs["dec"]
        PA_deg                          = kwargs["pa"]
        Len_arcsec, Wid_arcsec          = kwargs["length"], kwargs["width"]
        
        self.origin = SkyCoord([[origin_ra_deg, origin_dec_deg]],frame='fk5',unit=(u.deg, u.deg))[0]
        self.position_angle = Angle(PA_deg,'deg')
        self.length = Angle(Len_arcsec,u.arcsec)
        self.width = Angle(Wid_arcsec,u.arcsec)