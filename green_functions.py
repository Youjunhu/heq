# Green's functions for poloidal magnetic flux (psi) and magnetic field (Br, Bz)
import numpy as np
from numpy import sqrt, sin, cos, pi, clip
from scipy.special import ellipk, ellipe

mu0 = 4*pi*1.0e-7 #permeability in SI unit
twopi = 2*pi

nphi = 50
dphi = pi/(nphi-1)
phi = np.linspace(0, pi, nphi)
phi = phi + dphi*0.5
phi = phi[:-1]
cos_phi = cos(phi)
sin_phi = sin(phi)

def greenPsi0(rp, zp, r, z):
   #dist =sqrt((z-zp)**2+(r-rp*cos_phi)**2+(rp*sin_phi)**2)
   dist = sqrt((z-zp)**2+r**2+rp**2 - 2*r*rp*cos_phi)
   integral = np.sum(rp*cos_phi/dist)*dphi
   return 2*mu0*r/(4*pi)*integral

def greenPsi(rp, zp, r, z):
   k = sqrt(4*r*rp/((r+rp)**2+(z-zp)**2))
   k = clip(k, 1e-10, 1.0 - 1e-10)
   return mu0/twopi*sqrt(r*rp)/k*((2-k**2)*ellipk(k**2)-2*ellipe(k**2))


def greenBrz(rp, zp, r, z):
   #dist3 = sqrt((z-zp)**2+(r-rp*cos_phi)**2+(rp*sin_phi)**2)**3
   dist3 = sqrt((z-zp)**2+r**2+rp**2 - 2*r*rp*cos_phi)**3
   #dist3 = clip(dist3, 1e-8, 1000)
   integral =  np.sum(rp*(z-zp)*cos_phi/dist3)*dphi
   integral2 = np.sum((rp**2-r*rp*cos_phi)/dist3)*dphi
   return 2*mu0/(4*pi)*integral, 2*mu0/(4*pi)*integral2 #R,Z components

def greenBr(rp, zp, r, z):
   dist3 = sqrt((z-zp)**2+r**2+rp**2 - 2*r*rp*cos_phi)**3
   integral =  np.sum(rp*(z-zp)*cos_phi/dist3)*dphi
   return 2*mu0/(4*pi)*integral


def greenBz(rp, zp, r, z):
   dist3 = sqrt((z-zp)**2+r**2+rp**2 - 2*r*rp*cos_phi)**3
   integral2 = np.sum((rp**2-r*rp*cos_phi)/dist3)*dphi
   return 2*mu0/(4*pi)*integral2 

def greenBrz0(Rc, Zc, R, Z, eps=1e-3):
    """
    Calculate magnetic field at (R,Z)
    due to unit current at (Rc, Zc)
    Br = -(1/R) d psi/dZ
    Bz = (1/R) d psi/dR
    """
    return (greenPsi(Rc, Zc, R, Z - eps) - greenPsi(Rc, Zc, R, Z + eps)) / (2.0 * eps * R), (greenPsi(Rc, Zc, R + eps, Z) - greenPsi(Rc, Zc, R - eps, Z)) / (2.0 * eps * R)
