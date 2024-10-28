import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import lstsq
from scipy.sparse.linalg import spsolve
from aux import determine_lcfs_axis, magnetic_surfaces, laplace_cylindrical2d, get_laplace_matrix
from aux import inductance, betap #, find_critical
from green_functions import greenBr
import geqdsk

mu0 = 4*pi*1.0e-7 #permeability in SI unit
twopi = 2*pi

nr, nz = 33, 33 # Choose resolution
resolution = f'{nr}x{nz}'

# Grid for psi
r = np.genfromtxt(f'input/{resolution}/r.txt')
z = np.genfromtxt(f'input/{resolution}/z.txt') 
R_VDE, Z_VDE = np.loadtxt(f'input/{resolution}/VDE_coil.txt') # upper Virtual coil location
R, Z = np.meshgrid(r, z, indexing='ij')
dr = r[1]-r[0]
dz = z[1]-z[0]
if not r.size == nr: sys.exit("grid resolution does not match") 
ds = dr*dz

rin = r[1:-1] # R grid in the inner region (excluding the boundary points)

# Grid for jphi
rm = r + dr/2
zm = z + dz/2
rm = rm[:-1]
zm = zm[:-1]
nrm, nzm = rm.size, zm.size
Rm, Zm = np.meshgrid(rm, zm, indexing='ij')

# Construct 2D Laplacian operator matrix (in cylindrical coordinate)
laplace_matrix = get_laplace_matrix(r,z)

# read Green function numerical table
print(resolution)
matdata = loadmat(f'input/{resolution}/pla_pla.mat')
pla_pla = matdata['out']
matdata = loadmat(f'input/{resolution}/coil_pla.mat')
coil_pla = matdata['out']
matdata = loadmat(f'input/{resolution}/virtualCoil_pla.mat')
virtualCoil_pla = matdata['out']



print(pla_pla.shape)
print(coil_pla.shape)
Nc = coil_pla.shape[0]
print('Nc=', Nc)

nP = 1 # Number of coeffients in p'
nF = 1 # Number of coeffients in FF'
nb = nP + nF
# Jphi expansion coefficients
coef = np.array([9.482305004450900014e+05,
                 -1.468170461510982516e+00,])

# coil current per turn (Ampere)
Icoil = np.array([2.653381103696751779e+03,
                  2.604317138679315576e+03,
                  6.887657715092106628e+03,
                  5.985154296901499947e+03,
                  4.213260742432499683e+03,
                  5.764477051064446641e+03,
                  7.343001953306179530e+03,
                  7.297209472847495817e+03,
                  7.294779785952942802e+03,
                  7.288896484909391802e+03,
                  -6.929062987588693431e+03,
                  -7.292153320461962721e+03,
                  5.275637821274214048e+02,
                  5.083233032447752748e+02,
                  -2.816467284112563902e+02,
                  2.593994136454391253e+01,])

Ip = 3.962260312500000000e+05

def Pprime_basis(j,x): # P' basis functions
    return x**j - x**nP

def FFprime_basis(j,x): # FF' basis functions
    return x**j - x**nF

def Jphi_basis(j,x,R):
    if j<nP:
        return Pprime_basis(j,x)*R
    else:
        return FFprime_basis(j-nP,x)/(mu0*R)
first_wall = np.genfromtxt('input/limiter.txt') # mm to meter

psi = 0.5 - ((R-1.8)**2/0.5**2 + Z**2/1**2)*0.05 # Initial guess of psi(R,Z)


#psi = np.loadtxt('/home/yj/tmp/output/psi.txt')
f_error = open('output/error_evolution.txt','w')
nPicard = 100
for kk in range(nPicard): # Picard iteration
    psi_old = np.ndarray.copy(psi)
    psi_mid = np.zeros((nrm,nzm))
    for i in range(nrm):
        for j in range(nzm):
            psi_mid[i,j] = (psi[i,j]+psi[i+1,j]+psi[i,j+1]+psi[i+1,j+1])/4

    r_lcfs, z_lcfs, psi_lcfs, psi_axis,raxis, zaxis, elongation = determine_lcfs_axis(r,z,first_wall, psi)
    X = (psi_mid - psi_axis)/(psi_lcfs - psi_axis)
    
    in_lcfs = np.ones((nrm, nzm))
    bdry = mpltPath.Path(np.asarray([ [x,y] for x, y in zip(r_lcfs,z_lcfs) ]))
    for i in range(nrm):
        for j in range(nzm):
           p = (rm[i],zm[j])
           if not bdry.contains_point(p): in_lcfs[i,j] = 0.0

    jphi = np.zeros((nrm, nzm))
    for k in range(nb):
        jphi +=  coef[k]*Jphi_basis(k,X,Rm)  * in_lcfs

    coef_Ip = Ip/(np.sum(jphi)*ds)
    coef = coef*coef_Ip #sacle coefficient to make plasma current equal Ip
        
    psi_coils = np.zeros((nr,nz))    
    # Update the poloidal flux using Green's function method:
    for i in range(nr):
        for j in range(nz):
            #if (i==0 or i==nr-1 or j==0 or j==nz-1):
                psi[i,j]  = np.sum(pla_pla[:,:,i,j]*jphi)*ds # plasma contribution
                psi_coils[i,j] = np.sum(coil_pla[:,i,j]*Icoil[:]) # coil contribution
                
    psi += psi_coils # coil contribution
    
    # control vertical displacement instability
    Rcur = np.sum(jphi*Rm)*ds/Ip
    Zcur = np.sum(jphi*Zm)*ds/Ip
    Br_star = greenBr(R_VDE,Z_VDE, Rcur,Zcur) - greenBr(R_VDE,-Z_VDE, Rcur,Zcur)
    psi_coils_spline = RectBivariateSpline(r,z, psi_coils, kx=3,ky=3)
    gz = 2.5 # set by user, coefficient for Br compensation
    I_VDE = - gz * (-psi_coils_spline.ev(Rcur,Zcur, dx=0, dy=1)/Rcur)/ Br_star
    psi += virtualCoil_pla * I_VDE # VDE_coil contribution
                
    """
    M, N = nrm-1, nzm-1
    jphi_in = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            jphi_in[i,j] = (jphi[i,j]+jphi[i+1,j]+jphi[i,j+1]+jphi[i+1,j+1])/4

    # Update the rhs of the GS equation:
    source = -mu0*rin[:,None]*jphi_in
    # Special treatment near rectangle boundary
    for i in range(M):
        for j in range(N):
            if i==0:    source[i,j] -= (1/dr**2 + 1/(2*rin[i]*dr))*psi[0,j+1]
            if i==M-1:  source[i,j] -= (1/dr**2 - 1/(2*rin[i]*dr))*psi[nr-1,j+1]
            if j==0:    source[i,j] -= 1/dz**2*psi[i+1,0]
            if j==N-1:  source[i,j] -= 1/dz**2*psi[i+1,nz-1]
                
    # Update psi within the rectangle by inverting the Laplace operator
    psi[1:-1, 1:-1] = spsolve(laplace_matrix, source.reshape((M*N,), order='F')).reshape((M,N), order='F')
    """

    error = np.abs(psi_old - psi).max() / (psi.max() - psi.min())
    print(f'difference between two iterations ={error:12.4E}')
    print(error, I_VDE, file=f_error)
    if(error<0.001): break
    #print(f'Plasma current (kA) =', np.sum(jphi)*ds/1000)

f_error.close()
print('number of Picard iteration=', kk+1)

# post processing
#opoint, xpoint = find_critical(R, Z, psi)


Ip = np.sum(jphi)*ds
data = np.loadtxt("output/error_evolution.txt")
fig, ax = plt.subplots(nrows=1)
ax.plot(data[:,0], 'o-b')
ax.set_yscale('log')
ax.set_ylabel(f'max|$\psi_1$ - $\psi_2$|/ ($\psi_1$.max - $\psi_1$.min)',fontsize=20)
ax.tick_params(axis='both',which='both', labelsize=20)
plt.show()

r_lcfs, z_lcfs, psi_lcfs, psi_axis, raxis, zaxis, ellipticity = determine_lcfs_axis(r,z,first_wall,psi)
np.savetxt('output/jphi1.txt', jphi)
np.savetxt('output/jphi2.txt', laplace_cylindrical2d(r,z, psi))

np.savetxt('output/psi.txt', psi) 
np.savetxt('output/Ic.txt', Icoil) 

psi0 = -np.loadtxt('input/psi_efit.txt').T # benchmark
levels0 = np.linspace(psi.min(), psi.max(), 30)
fig, ax = plt.subplots()
ax.contour(R,Z, psi0, levels=levels0, colors='black')
ax.contour(R,Z, psi, levels=levels0, colors='red')
ax.plot(first_wall[:,0], first_wall[:,1])
ax.set_aspect('equal', 'box')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(R,Z,psi,  color='red')
surf = ax.plot_surface(R,Z, psi0,  color='black')
plt.show()

np.savetxt('output/lcfs.txt', np.c_[r_lcfs, z_lcfs])


from aux import magnetic_surfaces
dpsi = (psi_lcfs - psi_axis)/(nr-1)
psival = np.array([psi_axis + dpsi*i for i in range(nr)])

rs, zs = magnetic_surfaces(r,z,psi, psival[1:-1])
rs.insert(0, raxis)
zs.insert(0, zaxis)
rs.append(r_lcfs)
zs.append(z_lcfs)
npsi = nr

psi_spline = RectBivariateSpline(r,z, psi, kx=3,ky=3)
psi_axis = psi_spline.ev(raxis, zaxis, dx=0, dy=0)
pfn = (psival - psi_axis)/(psival[-1] - psi_axis) # Normalized poloidal flux

Pprime = np.zeros((npsi,))
for j in range(nP):
    Pprime[:] += Pprime_basis(j, pfn)*coef[j]

FFprime = np.zeros((npsi,))
for j in range(nF):
    FFprime[:] += FFprime_basis(j, pfn)*coef[j+nP]

p = np.empty((npsi,)) # Pressure, unit: Pascal
p[-1] = 0 # Pressure at lcfs
for i in range(npsi-2, -1, -1): # from magnetic edge to axis
      p[i] = p[i+1] + Pprime[i+1]*(-dpsi)
np.savetxt('output/pressure.txt', np.c_[pfn, p])


Is = 90.66*1000 # current per turn in a TF coil (Ampere)
I_tf = Is*2*12 # 12 TF coils, 2 turns per coil
g0 = -I_tf*mu0/twopi # vacuum value of g=Bphi*R

Bt0 = g0/raxis # Vacuum toroidal magnetic field at raxis
print('Bt0=',Bt0, 'g0=',g0)
print('Raxis =', raxis)

g = np.empty( (npsi,) )
g[0] = g0
for i in range(npsi-1): # from magnetic axis to edge
      g[i+1] = g[i] + (FFprime[i]/g[i])*dpsi

q = np.zeros(npsi,) # Safety factor
for i in range(1,npsi):
    for j in range(rs[i].size-1): # poloidal integration
         Br = -psi_spline.ev(rs[i][j], zs[i][j], dx=0, dy=1)/rs[i][j]
         Bz =  psi_spline.ev(rs[i][j], zs[i][j], dx=1, dy=0)/rs[i][j]
         Bp = sqrt(Br**2 + Bz**2)
         dlp = sqrt((rs[i][j+1]-rs[i][j])**2 + (zs[i][j+1]-zs[i][j])**2)
         q[i] += g[i]/rs[i][j]**2/Bp*dlp/twopi
q[0] = 2*q[1]-q[2] # Linear interpolation at magnetic axis
np.savetxt('output/profiles.txt', np.c_[pfn, q, g, p, Pprime, FFprime],
           fmt='%8.6e', header="pfn, safety_factor, Bphi*R, Pressure(Pa), P', FF'")

psi_r = psi_spline(r,z,dx=1,dy=0)
psi_z = psi_spline(r,z,dx=0,dy=1)
Br = (-1/r[:,None])*psi_z
Bz =  (1/r[:,None])*psi_r
Bp_sq = Br**2 + Bz**2
li = inductance(r, z, Bp_sq, r_lcfs, z_lcfs, Ip)

print('normalized inductance=',li)

psiN_2d = (psi - psi_axis)/(psival[-1] - psi_axis) # Normalized poloidal flux
betap0 = betap(pfn, p, r, z,  psiN_2d, r_lcfs, z_lcfs, Ip)

print('betap0=',betap0)
print(f'Plasma current (kA) =', Ip/1000)

fig, ax = plt.subplots()
ax.plot(pfn, np.abs(q), label='safety factor')
ax.set_xlabel("$\Psi_N$")
ax.legend()

fig, ax = plt.subplots()
ax.plot(pfn**0.5, g, label='Bphi*R (T*m)')
ax.set_xlabel("$\sqrt{\Psi_N}$")
ax.legend()

fig, ax = plt.subplots()
ax.plot(pfn**0.5, p/1000, label='pressure (kPa)')
ax.set_xlabel("$\sqrt{\Psi_N}$")
ax.legend()

# fig, ax = plt.subplots()
# ax.set_aspect('equal', 'box')
# for i in range(0, len(rs), 2):
#     ax.plot(rs[i], zs[i],'-b',)

plt.show()

rmajor = 1.0 #meter
data = {}
data['nx'] = nr
data['ny'] = nz
data['rdim'] = r[-1] - r[0]
data['zdim'] = z[-1] - z[0]
data['rcentr'] = rmajor
data['rgrid1'] = r[0]
data['zmid'] = (z[0]+z[-1])/2
data['rmagx'] = raxis
data['zmagx'] = zaxis
data['simagx'] = psi_axis
data['sibdry'] = psi_lcfs
data['bcentr'] = g0/rmajor
data['cpasma'] = Ip
data['sibdry'] = psi_lcfs
data['fpol'] = g
data['pressure'] = p
data['FFprime'] = FFprime
data['Pprime'] = Pprime
data['psi'] = psi.flatten(order='F')
data['qpsi'] = q
data['np_lcfs'] = len(r_lcfs)
data['nlim'] = first_wall[:,0].size
data['lcfs'] = np.array([ [x,y] for x,y in zip(r_lcfs, z_lcfs)]).flatten()
data['limiter'] = first_wall.flatten()

geqdsk.write('output/gfile.txt',data)
