# Prepare numerical Green's function tables on gridpoints
# require data files : PF coils location, magnetic proble(MP), flux loop (FL) locations.
import numpy as np
from numpy import sqrt, sin, cos, pi
import matplotlib.pyplot as plt
from green_functions import greenPsi, greenBrz

# Grid for psi
nr, nz = 33, 33
resolution = f'{nr}x{nz}'
print(resolution)

# choose computational box
#rmin, rmax = 1.2, 2.8
#zmin, zmax = -1.6, 1.6
rmin, rmax= 1.2000000476837158, 2.5999999046325684
zmin, zmax= -1.2000000476837158, 1.2000000476837158

R_VDE, Z_VDE = (rmax+rmin)/2, zmax+0.01
r = np.linspace(rmin, rmax, nr)
z = np.linspace(zmin, zmax, nz)
np.savetxt(f'input/{resolution}/r.txt', np.c_[r])
np.savetxt(f'input/{resolution}/z.txt', np.c_[z])
np.savetxt(f'input/{resolution}/VDE_coil.txt', [R_VDE, Z_VDE])

print('r.min(), r.max()=',r.min(), r.max())
print('z.min(), z.max()=',z.min(), z.max())
dr =  r[1] - r[0]
dz =  z[1] - z[0]

rm = r + dr/2
zm = z + dz/2
rm = rm[:-1]
zm = zm[:-1]
nrm, nzm = nr-1, nz-1

print('computing green function table....')
pla_pla = np.zeros((nrm, nzm, nr, nz))
for im in range(nrm):
    for jm in range(nzm):
        for i in range(nr):
            for j in range(nz):
                pla_pla[im,jm,i,j] = greenPsi(rm[im], zm[jm], r[i], z[j])


data = np.genfromtxt('input/coils.txt')
rcenter = data[:,0]
zcenter = data[:,1]
width = data[:,2]
hight = data[:,3]
nx  = data[:,4].astype(np.int64)
ny  = data[:,5].astype(np.int64)
turns = data[:,-1]
Nc = rcenter.size

r_coil = [np.empty(nx[i]) for i in range(Nc) ]
z_coil = [np.empty(ny[i]) for i in range(Nc) ]
for ic in range(Nc):
   for ix in range(nx[ic]):
      r_coil[ic][ix] =  rcenter[ic] - width[ic]/2 + width[ic]/(nx[ic]-1)*ix
   for iy in range(ny[ic]):
      z_coil[ic][iy] =  zcenter[ic] - hight[ic]/2 + hight[ic]/(ny[ic]-1)*iy


coil_pla = np.zeros((Nc,nr,nz))
for i in range(nr):
    for j in range(nz):
        for ic in range(Nc):
            for ix in range(nx[ic]):
                for iy in range(ny[ic]):
                    coil_pla[ic, i, j] +=  greenPsi(r_coil[ic][ix], z_coil[ic][iy], r[i], z[j])

for ic in range(Nc):
   coil_pla[ic,:,:] *= turns[ic]/(nx[ic]*ny[ic])


virtualCoil_pla = np.zeros((nr,nz))
for i in range(nr):
    for j in range(nz):
             virtualCoil_pla[i, j] =  ( greenPsi(R_VDE, Z_VDE, r[i], z[j])
                                   - greenPsi(R_VDE, -Z_VDE, r[i], z[j]))
   

#----Flux loops (FL) measuring poloidal magnetic flux----
data = np.genfromtxt('input/FL.txt')
n_FL = data.shape[0]
r_FL = data[:,-2].astype('float')
z_FL = data[:,-1].astype('float')
print('n_FL=',n_FL)

pla_FL = np.zeros( (nrm, nzm, n_FL) )
for i in range(nrm):
    for j in range(nzm):
            for k in range(n_FL):
                pla_FL[i,j,k] = greenPsi(rm[i], zm[j], r_FL[k], z_FL[k])
                              
coil_FL = np.zeros( (Nc, n_FL) )
for ic in range(Nc):
     for k in range(n_FL):
        for ix in range(nx[ic]):
           for iy in range(ny[ic]):
              coil_FL[ic,k] += greenPsi(r_coil[ic][ix], z_coil[ic][iy], r_FL[k], z_FL[k])

for ic in range(Nc):
   coil_FL[ic,:] *= turns[ic]/(nx[ic]*ny[ic])

#--Magnetic probe (MP) measuring poloidal magnetic field-----
data = np.genfromtxt('input/probeT.txt',dtype='str')
n_MP = data.shape[0]
print('n_MP=', n_MP)
r_MP = data[:,-3].astype('float')
z_MP = data[:,-2].astype('float')
angle_MP = data[:,-1].astype('float')/180*pi

# fig, ax = plt.subplots()
# ax.quiver(r_MP, z_MP, cos(angle_MP), sin(angle_MP), color='b')
# ax.set_aspect('equal','box')
# plt.show()
pla_MP = np.zeros( (nrm, nzm, n_MP) )
for i in range(nrm):
    for j in range(nzm):
            for k in range(n_MP):
                br, bz = greenBrz(rm[i], zm[j], r_MP[k], z_MP[k])
                pla_MP [i,j,k] = br*cos(angle_MP[k]) + bz*sin(angle_MP[k])
                              
coil_MP = np.zeros( (Nc, n_MP) )                
for ic in range(Nc):
         for k in range(n_MP):
             for ix in range(nx[ic]):
                 for iy in range(ny[ic]):
                    br, bz = greenBrz(r_coil[ic][ix], z_coil[ic][iy], r_MP[k], z_MP[k])
                    coil_MP[ic,k] += br*cos(angle_MP[k]) + bz*sin(angle_MP[k])

for ic in range(Nc):
   coil_MP[ic,:] *= turns[ic]/(nx[ic]*ny[ic])

data = np.genfromtxt('input/probeN.txt',dtype='str')
n_MPN = data.shape[0]
print('n_MPN=', n_MPN)
r_MPN = data[:,-3].astype('float')
z_MPN = data[:,-2].astype('float')
angle_MPN = data[:,-1].astype('float')/180*pi
                
pla_MPN = np.zeros( (nrm, nzm, n_MPN) )
for i in range(nrm):
    for j in range(nzm):
            for k in range(n_MPN):
                br, bz = greenBrz(rm[i], zm[j], r_MPN[k], z_MPN[k])
                pla_MPN[i,j,k] = br*cos(angle_MPN[k]) + bz*sin(angle_MPN[k])

coil_MPN = np.zeros( (Nc, n_MPN) )
for ic in range(Nc):
         for k in range(n_MPN):
             for ix in range(nx[ic]):
                 for iy in range(ny[ic]):
                     br, bz = greenBrz(r_coil[ic][ix], z_coil[ic][iy], r_MPN[k], z_MPN[k])
                     coil_MPN[ic,k] += br*cos(angle_MPN[k]) + bz*sin(angle_MPN[k])

for ic in range(Nc):
   coil_MPN[ic,:] *= turns[ic]/(nx[ic]*ny[ic])


ind = np.unravel_index(np.argmin(pla_pla), pla_pla.shape)
print(pla_pla[ind], pla_pla.min())

ind = np.unravel_index(np.argmax(pla_pla), pla_pla.shape)
print(pla_pla[ind], pla_pla.max())

import scipy.io
scipy.io.savemat(f'input/{resolution}/pla_pla.mat', mdict={'out': pla_pla}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/coil_pla.mat', mdict={'out': coil_pla}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/pla_FL.mat', mdict={'out': pla_FL}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/coil_FL.mat', mdict={'out': coil_FL}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/pla_MP.mat', mdict={'out': pla_MP}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/coil_MP.mat', mdict={'out': coil_MP}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/pla_MPN.mat', mdict={'out': pla_MPN}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/coil_MPN.mat', mdict={'out': coil_MPN}, oned_as='row')
scipy.io.savemat(f'input/{resolution}/virtualCoil_pla.mat', mdict={'out': virtualCoil_pla}, oned_as='row')
print('nr,nz=',nr,nz)
print('nrm,nzm=',nrm,nzm)
print('pla_pla.shape=',pla_pla.shape)
print('coil_pla.shape=',coil_pla.shape)
print('pla_FL.shape=',pla_FL.shape)
print('coil_FL.shape=',coil_FL.shape)
print('pla_MP.shape=',pla_MP.shape)
print('coil_MP.shape=',coil_MP.shape)
print('pla_MPN.shape=',pla_MPN.shape)
print('coil_MPN.shape=',coil_MPN.shape)
print('finished')

fig, ax = plt.subplots()
R,Z = np.meshgrid(r,z,indexing='ij')
ax.plot(R,Z, 'b.')
ax.plot(r_MP,z_MP, 'r.')

for ic in range(Nc):
    rc_grid, zc_grid = np.meshgrid(r_coil[ic], z_coil[ic])
    ax.plot(rc_grid, zc_grid, 'k.')
ax.set_aspect('equal', 'box')
plt.show()

