#!/usr/bin/env python
import re,sys,string,math,os

pc_cm    = 3.086E18          # pc in cm
c_cm_sec = 2.9979E10         # speed of light in cm/sec
c_pc_sec = c_cm_sec / pc_cm  # speed of light in pc/sec
sidereal_year = 3.156E7      # sidereal year in sec
pc2ly = 3.26
deg2rad=0.0174532925199      # math.pi / 180.
rad2deg = 57.2957795131      # 180.0 / math.pi

def z_ly_rhot(rho_ly,delta_t_years):
    z =rho_ly * rho_ly / (2*delta_t_years)  - 0.5 * delta_t_years
    return z

def z_ly_atD(angularsep_degree,delta_t_years,D_ly,precision=0.000001,verbose=False):
    angularsep_radians = angularsep_degree * deg2rad
    z = 0
    iteration=0
    dz = 0.0
    goodflag = False
    while iteration==0 or (not goodflag):
        D_prime_ly = D_ly - z
        if D_prime_ly<0:
            while D_prime_ly<0:
                D_prime_ly= (D_ly + D_prime_ly)*0.3
        rho_ly = D_prime_ly * math.tan(angularsep_radians)
        znew =rho_ly * rho_ly / (2*delta_t_years)  - 0.5 * delta_t_years
        dz = znew-z
        goodflag = abs(dz)<abs(z)*precision
        if verbose:
            print('++++\niteration: ',iteration)
            print('D_prime_ly',D_prime_ly)
            print('rho_ly',rho_ly)
            print('znew',znew)
            print('dz',dz,z*precision,abs(dz))
        #z = (znew + z) * 0.5
        z = z + (znew - z) * 0.2
        iteration+=1
        if iteration>400:
            print('ERROR: too many iterations!')
            sys.exit(0)

    return z, rho_ly

def rho_ly_aDz(angularsep_degree,D_ly,z_ly):
    angularsep_radians = angularsep_degree * deg2rad
    D_prime_ly = D_ly - z_ly
    rho_ly = D_prime_ly * math.tan(angularsep_radians)
    return(rho_ly)

def rho_ly_zt(z_ly,delta_t_years):
    tmp = (z_ly + delta_t_years*0.5) * 2*delta_t_years
    if tmp >=0.0:
        rho_ly = math.sqrt(tmp)
    else:
        rho_ly = None
    return(rho_ly)

def angularsep_deg_ztD(z_ly,delta_t_years,D_ly):
    rho_ly = rho_ly_zt(z_ly,delta_t_years)
    if rho_ly == None:
        return(None)
    D_prime_ly = D_ly - z_ly
    #D_prime_ly = D_ly
    #print 'XXX',delta_t_years,D_ly,z_ly,rho_ly,D_prime_ly
    angularsep_radians = math.atan(rho_ly/D_prime_ly)
    #print 'angularsep_radians',angularsep_radians
    angularsep_degree = angularsep_radians * rad2deg
    return angularsep_degree

# the apparent motion is in units of the speed of light
# this assumes dz/dt=0 (constant z)
def apparentmotion_const_z(z_ly,delta_t_years):
    appmotion = (z_ly + delta_t_years)/math.sqrt(2*z_ly*delta_t_years+delta_t_years*delta_t_years)
    return(appmotion)


if __name__=='__main__':
    delta_t_years = 334.0   # Cas A
    z_ly = 1000
    while (z_ly>-delta_t_years*0.5):
        print('t: %.2f years   z:%8.2f ly   dh/dt:%.4f c' % (delta_t_years,z_ly,apparentmotion_const_z(z_ly,delta_t_years)))
        z_ly-=20
