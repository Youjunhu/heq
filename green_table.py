# Prepare numerical tables of Green's functions on gridpoints
# Machine dependent parameters are set in this file, e.g., RZ grid, PF coils location, magnetic proble(MP), flux loop (FL) locations.

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
cos = np.cos
sin = np.sin
sqrt = np.sqrt
pi = np.pi
mu0 = 4*pi*1.0e-7 #permeability in SI unit
twopi = 2*np.pi

nphi = 100
phi = np.linspace(0,twopi,nphi)
phi = phi[:-1]
dphi = phi[1] - phi[0]
cos_phi=cos(phi)
sin_phi=sin(phi)

def greenPsi0(rp, zp, r, z):
   #dist = sqrt((z-zp)**2+(r-rp*cos_phi)**2+(rp*sin_phi)**2)
   dist = sqrt((z-zp)**2+r**2+rp**2 - 2*r*rp*cos_phi)
   integral = np.sum(rp*cos_phi/dist)*dphi
   return mu0*r/(4*pi)*integral

def greenPsi(rp, zp, r, z):
   k = sqrt(4*r*rp/((r+rp)**2+(z-zp)**2))
   return mu0/twopi*sqrt(r*rp)/k*((2-k**2)*scipy.special.ellipk(k**2)-2*scipy.special.ellipe(k**2))

def greenBrz(rp, zp, r, z):
   dist = sqrt((z-zp)**2+(r-rp*cos_phi)**2+(rp*sin_phi)**2)
   integral =  np.sum(rp*(z-zp)*cos_phi/dist**3)*dphi
   integral2 = np.sum((rp**2-r*rp*cos_phi)/dist**3)*dphi
   return mu0/(4*pi)*integral, mu0/(4*pi)*integral2 #R,Z components

#Computational rectangle in (R,Z)
r = np.genfromtxt('data/r.txt')
z = np.genfromtxt('data/z.txt') 
(nr, nz) = (r.size, z.size)
print('r.size, z.size=', nr, nz)
print('r.min(), r.max()=',r.min(), r.max())
print('z.min(), z.max()=',z.min(), z.max())


dr =  r[1] - r[0]
dz =  z[1] - z[0]
rm = r + dr/2
zm = z + dz/2
rm = rm[:-1]
zm = zm[:-1]
nrm, nzm = nr-1, nz-1

data = np.genfromtxt('data/coil.txt')
r_coil = data[:,0]
z_coil = data[:,1]
Nc = r_coil.size

pla_pla = np.zeros((nrm, nzm, nr, nz))
for im in range(nrm):
    for jm in range(nzm):
        for i in range(nr):
            for j in range(nz):
                pla_pla[im,jm,i,j] = greenPsi(rm[im], zm[jm], r[i], z[j])


coil_pla = np.zeros((Nc,nr,nz))
for ic in range(Nc):
     for i in range(nr):
         for j in range(nz):
            coil_pla[ic, i, j] = greenPsi(r_coil[ic], z_coil[ic], r[i],z[j])

#----Flux loops (FL) measuring poloidal magnetic flux----
data = np.genfromtxt('data/FL2.txt')
n_FL = data.shape[0]
r_FL = data[:,2].astype('float')
z_FL = data[:,3].astype('float')
print('n_FL=',n_FL)

pla_FL = np.zeros( (nrm, nzm, n_FL) )
for i in range(nrm):
    for j in range(nzm):
            for k in range(n_FL):
                pla_FL[i,j,k] = greenPsi(rm[i], zm[j], r_FL[k], z_FL[k])
                              
coil_FL = np.zeros( (Nc, n_FL) )
for ic in range(Nc):
     for k in range(n_FL):
           coil_FL[ic,k] = greenPsi(r_coil[ic], z_coil[ic], r_FL[k], z_FL[k])

#--Magnetic probe (MP) measuring poloidal magnetic field-----
data = np.genfromtxt('data/probe4.txt',dtype='str')
n_MP = data.shape[0]
print('n_MP=', n_MP)
MP_name = data[:,0]
r_MP = data[:,2].astype('float')
z_MP = data[:,3].astype('float')
angle_MP = data[:,4].astype('float')/180*pi

# fig, ax = plt.subplots()
# ax.quiver(r_MP, z_MP, cos(angle_MP), sin(angle_MP), color='b')
# ax.set_aspect('equal','box')
# plt.show()
pla_MP = np.zeros( (nrm, nzm, n_MP) )
for i in range(nrm):
    for j in range(nzm):
            for k in range(n_MP):
                br, bz = greenBrz(rm[i], zm[j], r_MP[k], z_MP[k])
                pla_MP[i,j,k] = br*cos(angle_MP[k]) + bz*sin(angle_MP[k])
                              

coil_MP = np.zeros( (Nc, n_MP) )                
for ic in range(Nc):
         for k in range(n_MP):
                br, bz = greenBrz(r_coil[ic], z_coil[ic], r_MP[k], z_MP[k])
                coil_MP[ic,k] = br*cos(angle_MP[k]) + bz*sin(angle_MP[k])


ind = np.unravel_index(np.argmin(pla_pla), pla_pla.shape)
print(ind)
print(pla_pla[ind], pla_pla.min())

ind = np.unravel_index(np.argmax(pla_pla), pla_pla.shape)
print(ind)
print(pla_pla[ind], pla_pla.max())



fig, ax = plt.subplots()
R,Z = np.meshgrid(r,z,indexing='ij')
ax.plot(R,Z, 'b.')
ax.plot(r_MP,z_MP, 'r.')
ax.plot(r_coil,z_coil, 'k.')
ax.set_aspect('equal', 'box')
plt.show()


import scipy.io
scipy.io.savemat('data/pla_pla.mat', mdict={'out': pla_pla}, oned_as='row')
scipy.io.savemat('data/coil_pla.mat', mdict={'out': coil_pla}, oned_as='row')
scipy.io.savemat('data/pla_FL.mat', mdict={'out': pla_FL}, oned_as='row')
scipy.io.savemat('data/coil_FL.mat', mdict={'out': coil_FL}, oned_as='row')
scipy.io.savemat('data/pla_MP.mat', mdict={'out': pla_MP}, oned_as='row')
scipy.io.savemat('data/coil_MP.mat', mdict={'out': coil_MP}, oned_as='row')

print('nr,nz=',nr,nz)
print('nrm,nzm=',nrm,nzm)
print('pla_pla.shape=',pla_pla.shape)
print('coil_pla.shape=',coil_pla.shape)
print('pla_FL.shape=',pla_FL.shape)
print('coil_FL.shape=',coil_FL.shape)
print('pla_MP.shape=',pla_MP.shape)
print('coil_MP.shape=',coil_MP.shape)
