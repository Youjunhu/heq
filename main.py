import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import scipy.interpolate
import scipy.io

pi = np.pi
mu0 = 4*pi*1.0e-7 #permeability in SI unit
twopi = 2*pi
sqrt = np.sqrt

#grid for psi
r = np.genfromtxt('data/r.txt')
z = np.genfromtxt('data/z.txt') 
R, Z = np.meshgrid(r,z, indexing='ij')
dr = r[1]-r[0]
dz = z[1]-z[0]
nr, nz = r.size, z.size

#grid for jphi
rm = r + dr/2
zm = z + dz/2
rm = rm[:-1]
zm = zm[:-1]
nrm, nzm = rm.size, zm.size
Rm, Zm = np.meshgrid(rm, zm, indexing='ij')
ds = dr*dz

matdata = scipy.io.loadmat('data/pla_pla.mat')
pla_pla = matdata['out']
matdata = scipy.io.loadmat('data/coil_pla.mat')
coil_pla = matdata['out']
matdata = scipy.io.loadmat('data/pla_FL.mat')
pla_FL = matdata['out']
matdata = scipy.io.loadmat('data/coil_FL.mat')
coil_FL = matdata['out']
matdata = scipy.io.loadmat('data/pla_MP.mat')
pla_MP = matdata['out']
matdata = scipy.io.loadmat('data/coil_MP.mat')
coil_MP = matdata['out']

print(pla_pla.shape)
print(coil_pla.shape)
print(pla_FL.shape)
print(coil_FL.shape)
print(pla_MP.shape)
print(coil_MP.shape)

mFL = pla_FL.shape[-1]
mMP = pla_MP.shape[-1]
Nc = coil_FL.shape[0]
m = mFL + mMP + 2 + Nc #number of measurements and constraints
n = 4 + Nc #number of free parameters
print('mFL=',mFL, 'mMP=', mMP, 'Nc=', Nc, 'm,n=', m,n)

def b0(r,x):
    return r*(1-x**3)

def b1(r,x):
    return r*(x-x**3)

def b2(r,x):
    return r*(x**2-x**3)

def b3(r,x):
    return (1-x)/(mu0*r)

b = {0 : b0, 1 : b1, 2 : b2, 3 : b3} #function map

psi = np.loadtxt('data/psi.txt')*1.2 #initial guess of Psi
psi = psi.T # then psi[i,j] with i=>r, j=>z
psi = - psi #my definition, psi=Aphi*R, has a sign difference from EAST's
print('psi.shape=',psi.shape)
print('psi.min(), psi.max()=', psi.min(), psi.max())

fig, ax = plt.subplots()
def determine_lcfs_axis(psi):
    nl = 440
    levels0 = np.linspace(psi.min(),psi.max(),nl)
    cs = plt.contour(r,z,psi.T, levels=levels0)
    polygon = [ [r[1],z[1]], [r[-2],z[1]], [r[-2],z[-2]], [r[1],z[-2]] ]
    polygon = np.asarray(polygon)
    bdry = mpltPath.Path(polygon)
    xt = []
    yt = []
    for i, conts in enumerate(cs.collections):
        xt.append([])
        yt.append([])
        for j, p in enumerate(conts.get_paths()):
            v = p.vertices
            inside = np.all(bdry.contains_points(v))
            if inside:
                xt[i].append(v[:,0])
                yt[i].append(v[:,1])     
    tmp = 0
    for i, c in  enumerate(xt):
       for j, path in  enumerate(c):
         if np.max(path)> tmp:
             tmp = np.max(path)
             I, J = i, j
    tmp=10**9        
    for i, c in  enumerate(xt):
      for j, path in  enumerate(c):
        t = np.max(path) -np.min(path)
        if abs(t)<tmp:
            tmp = np.abs(t)
            I2, J2= i, j
    raxis = (xt[I2][J2].max()+xt[I2][J2].min())/2
    zaxis = (yt[I2][J2].max()+yt[I2][J2].min())/2
    return np.asarray(xt[I][J]), np.asarray(yt[I][J]), levels0[I], levels0[I2], raxis, zaxis


rhs = np.loadtxt('data/mm.txt') # Values of magnetic measurements
rhs[:mFL] = rhs[:mFL]/twopi
g0 = 2.0*1.9 # Bphi*R at the magnetic axis
qaxis = 1.60845699796 # safety factor constraint
rhs = np.concatenate( (rhs, np.asarray([qaxis,])) )
psi_old = np.zeros((nr,nz))
psi_mid = np.zeros((nrm,nzm))

for kk in range(5): # Picard iteration
    psi_old[:,:] = psi[:,:]
    _, _, psi_lcfs, psi_axis,raxis,zaxis = determine_lcfs_axis(psi)
    for i in range(nrm):
        for j in range(nzm):
            psi_mid[i,j] = (psi[i,j]+psi[i+1,j]+psi[i,j+1]+psi[i+1,j+1])/4
    X = (psi_mid-psi_axis)/(psi_lcfs - psi_axis)
    basis = [0]*4
    for j in range(4):
       basis[j] = b[j](Rm,X)
       
    psi_spline = scipy.interpolate.RectBivariateSpline(r,z, psi, kx=3,ky=3)
    psi_rr = psi_spline(raxis,zaxis,dx=2,dy=0)[0,0]
    psi_zz = psi_spline(raxis,zaxis,dx=0,dy=2)[0,0]
    ellipticity = psi_rr/psi_zz
    
    Gamma = np.zeros((m,n)) # Response matrix
    for i in range(mFL):
        for j in range(4):
            Gamma[i,j] = np.sum(pla_FL[:,:,i]*basis[j])*ds
        for j in range(4,n):
            Gamma[i,j] = coil_FL[j-4,i]

    for i in range(mMP):
        for j in range(4):
            Gamma[i+mFL,j] = np.sum(pla_MP[:,:,i]*basis[j])*ds
        for j in range(4,n):
            Gamma[i+mFL,j] = coil_MP[j-4,i]

    i = mFL + mMP
    for j in range(4):
        Gamma[i,j] = np.sum(basis[j])*ds
            
    shift = mFL + mMP +1
    for i in range(Nc):
        Gamma[i+shift,i+4] = 1
        
    i = mFL + mMP + Nc +1
    Gamma[i,0] = sqrt(ellipticity)*raxis**2*mu0/((ellipticity+1)*g0)*raxis
    Gamma[i,3] = sqrt(ellipticity)*raxis**2    /((ellipticity+1)*g0)/raxis

    # for i in range(m):
    #     Gamma[i,:] = Gamma[i,:]/rhs[i]
    # rhs[:] = 1
    
    # Solve the least square problem to determine [c0,...,c4,I[0],...I[Nc-1]]
    u, _, _, _ = np.linalg.lstsq(Gamma,rhs, rcond=None)
    c = u[0:4]
    I = u[4:n] # Update the coil currents

    # Update the current density using latest coefficients c[:]
    jphi = np.zeros((nrm, nzm))
    for k in range(4):
        jphi = jphi + c[k]*basis[k]
    # Update the poloidal flux using the lastest plasma and coil currents
    for i in range(nr):
        for j in range(nz):
            psi[i,j] = np.sum(pla_pla[:,:,i,j]*jphi[:,:])*ds
    for k in range(Nc):
        psi[:,:] = psi[:,:] + coil_pla[k,:,:]*I[k]

    error = np.sum(np.abs(psi_old - psi))/np.sum(np.abs(psi_old))
    print('difference between two iterations =', error)
    #print('u=',u)

plt.clf()
nl = 100
levels0 = np.linspace(psi.min(),psi.max(),nl)
cs = plt.contour(r,z, psi.T, levels=levels0)
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(R,Z,psi.T,  linewidth=1, antialiased=False)
plt.show()
