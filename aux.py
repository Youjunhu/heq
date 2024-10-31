import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from numpy import pi
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import inv
mu0 = 4*pi*1.0e-7 #permeability in SI unit

from scipy.sparse import eye
def get_laplace_matrix(r, z):
    """
    Construct 2D Laplacian operator matrix (in cylindrical coordinate)
    """
    nr, nz = r.size, z.size
    rin = r[1:-1] #R grid in the inner region (excluding the boundary points)
    m = nr -2 #inner grid point number
    n = nz -2
    mn = m*n
    dr = r[1]-r[0]
    dz = z[1]-z[0]
    # Create a sparse matrix and set values of its elements.
    A = eye(mn, format="lil") # using linked list format for storage efficiency
    for II in range(mn):
        j = II//m #recover the original index in Z
        i = II - j*m #recover the original index in R
        A[II,II] = -2/dr**2 - 2/dz**2
        if i>0:   A[II, II-1] = 1/dr**2 + 1/(2*rin[i]*dr)
        if i<m-1: A[II, II+1] = 1/dr**2 - 1/(2*rin[i]*dr)

        if j >0: A[II, II-m] = 1/dz**2
        if j < n-1: A[II, II+m] =  1/dz**2
    return A.tocsr() # convert to csr format for efficient operation


def find_critical(R, Z, psi, discard_xpoints=True):
    """
    Find critical points

    Inputs
    ------
    R:
        R(nr, nz) 2D array of major radii
    Z:
        Z(nr, nz) 2D array of heights
    psi:
        psi(nr, nz) 2D array of psi values

    Returns
    -------
    opoint: list
        List of O-points consisting of ``(R, Z, psi)`` tuples
    xpoint: list
        List of X-points consisting of ``(R, Z, psi)`` tuples

    """

    # Get a spline interpolation function
    f = RectBivariateSpline(R[:, 0], Z[0, :], psi)

    # Find candidate locations, based on minimising Bp^2
    Bp2 = (f(R, Z, dx=1, grid=False) ** 2 + f(R, Z, dy=1, grid=False) ** 2) / R ** 2

    # Get grid resolution, which determines a reasonable tolerance
    # for the Newton iteration search area
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    radius_sq = 9 * (dR ** 2 + dZ ** 2)

    # Find local minima

    J = np.zeros([2, 2])

    xpoint = []
    opoint = []

    nx, ny = Bp2.shape
    for i in range(2, nx - 2):
        for j in range(2, ny - 2):
            if (
                (Bp2[i, j] < Bp2[i + 1, j + 1])
                and (Bp2[i, j] < Bp2[i + 1, j])
                and (Bp2[i, j] < Bp2[i + 1, j - 1])
                and (Bp2[i, j] < Bp2[i - 1, j + 1])
                and (Bp2[i, j] < Bp2[i - 1, j])
                and (Bp2[i, j] < Bp2[i - 1, j - 1])
                and (Bp2[i, j] < Bp2[i, j + 1])
                and (Bp2[i, j] < Bp2[i, j - 1])
            ):

                # Found local minimum

                R0 = R[i, j]
                Z0 = Z[i, j]

                # Use Newton iterations to find where
                # both Br and Bz vanish
                R1 = R0
                Z1 = Z0

                count = 0
                while True:

                    Br = -f(R1, Z1, dy=1, grid=False) / R1
                    Bz = f(R1, Z1, dx=1, grid=False) / R1

                    if Br ** 2 + Bz ** 2 < 1e-6:
                        # Found a minimum. Classify as either
                        # O-point or X-point

                        dR = R[1, 0] - R[0, 0]
                        dZ = Z[0, 1] - Z[0, 0]
                        d2dr2 = (psi[i + 2, j] - 2.0 * psi[i, j] + psi[i - 2, j]) / (
                            2.0 * dR
                        ) ** 2
                        d2dz2 = (psi[i, j + 2] - 2.0 * psi[i, j] + psi[i, j - 2]) / (
                            2.0 * dZ
                        ) ** 2
                        d2drdz = (
                            (psi[i + 2, j + 2] - psi[i + 2, j - 2]) / (4.0 * dZ)
                            - (psi[i - 2, j + 2] - psi[i - 2, j - 2]) / (4.0 * dZ)
                        ) / (4.0 * dR)
                        D = d2dr2 * d2dz2 - d2drdz ** 2

                        if D < 0.0:
                            # Found X-point
                            xpoint.append((R1, Z1, f(R1, Z1)[0][0]))
                        else:
                            # Found O-point
                            opoint.append((R1, Z1, f(R1, Z1)[0][0]))
                        break

                    # Jacobian matrix
                    # J = ( dBr/dR, dBr/dZ )
                    #     ( dBz/dR, dBz/dZ )

                    J[0, 0] = -Br / R1 - f(R1, Z1, dy=1, dx=1)[0][0] / R1
                    J[0, 1] = -f(R1, Z1, dy=2)[0][0] / R1
                    J[1, 0] = -Bz / R1 + f(R1, Z1, dx=2) / R1
                    J[1, 1] = f(R1, Z1, dx=1, dy=1)[0][0] / R1

                    d = np.dot(inv(J), [Br, Bz])

                    R1 = R1 - d[0]
                    Z1 = Z1 - d[1]

                    count += 1
                    # If (R1,Z1) is too far from (R0,Z0) then discard
                    # or if we've taken too many iterations
                    if ((R1 - R0) ** 2 + (Z1 - Z0) ** 2 > radius_sq) or (count > 100):
                        # Discard this point
                        break

    # Remove duplicates
    def remove_dup(points):
        result = []
        for n, p in enumerate(points):
            dup = False
            for p2 in result:
                if (p[0] - p2[0]) ** 2 + (p[1] - p2[1]) ** 2 < 1e-5:
                    dup = True  # Duplicate
                    break
            if not dup:
                result.append(p)  # Add to the list
        return result

    xpoint = remove_dup(xpoint)
    opoint = remove_dup(opoint)

    if len(opoint) == 0:
        # Can't order primary O-point, X-point so return
        print("Warning: No O points found")
        return opoint, xpoint

    # Find primary O-point by sorting by distance from middle of domain
    Rmid = 0.5 * (R[-1, 0] + R[0, 0])
    Zmid = 0.5 * (Z[0, -1] + Z[0, 0])
    opoint.sort(key=lambda x: (x[0] - Rmid) ** 2 + (x[1] - Zmid) ** 2)

    # Draw a line from the O-point to each X-point. Psi should be
    # monotonic; discard those which are not

    if discard_xpoints:
        Ro, Zo, Po = opoint[0]  # The primary O-point
        xpt_keep = []
        for xpt in xpoint:
            Rx, Zx, Px = xpt

            rline = np.linspace(Ro, Rx, num=50)
            zline = np.linspace(Zo, Zx, num=50)

            pline = f(rline, zline, grid=False)

            if Px < Po:
                pline *= -1.0  # Reverse, so pline is maximum at X-point

            # Now check that pline is monotonic
            # Tried finding maximum (argmax) and testing
            # how far that is from the X-point. This can go
            # wrong because psi can be quite flat near the X-point
            # Instead here look for the difference in psi
            # rather than the distance in space

            maxp = np.amax(pline)
            if (maxp - pline[-1]) / (maxp - pline[0]) > 0.001:
                # More than 0.1% drop in psi from maximum to X-point
                # -> Discard
                continue

            ind = np.argmin(pline)  # Should be at O-point
            if (rline[ind] - Ro) ** 2 + (zline[ind] - Zo) ** 2 > 1e-4:
                # Too far, discard
                continue
            xpt_keep.append(xpt)
        xpoint = xpt_keep

    # Sort X-points by distance to primary O-point in psi space
    psi_axis = opoint[0][2]
    xpoint.sort(key=lambda x: (x[2] - psi_axis) ** 2)

    return opoint, xpoint


def determine_lcfs_axis(r,z, first_wall, psi, nl = 400):
    R, Z = np.meshgrid(r, z, indexing='ij')
    psival = np.linspace(psi.min(), psi.max(), nl)
    cs = plt.contour(R, Z, psi, levels = psival)
    # Select out all the magnetic surfaces that do not touch the first wall
    bdry = mpltPath.Path(first_wall)
    xt = []
    yt = []
    for i, conts in enumerate(cs.allsegs):
        xt.append([])
        yt.append([])
        for path in conts:
            if path.shape[0] == 0: break
            v = path
            #print('path.shape=',path.shape)
            inside = np.all(bdry.contains_points(v))
            if inside:
                xt[i].append(v[:,0])
                yt[i].append(v[:,1])

    # Select out the outmost contour (LCFS)
    tmp = 0
    for i, c in  enumerate(xt):
       for j, path in  enumerate(c):
         t = np.max(path)
         if t > tmp:
             tmp = t
             I, J = i, j

    # Select out the innermost contour (the magnetic axis)
    tmp=10**9        
    for i, c in  enumerate(xt):
      for j, path in  enumerate(c):
        t = abs(np.max(path) -np.min(path))
        if t<tmp:
            tmp = t
            I2, J2= i, j
    r_max = xt[I2][J2].max()
    r_min = xt[I2][J2].min()
    z_max = yt[I2][J2].max()
    z_min = yt[I2][J2].min()
    raxis = (r_max + r_min)/2
    zaxis = (z_max + z_min)/2
    elongation = (z_max - z_min)/(r_max - r_min)
    plt.close()
    return np.asarray(xt[I][J]), np.asarray(yt[I][J]), psival[I], psival[I2], raxis, zaxis, elongation

def magnetic_surfaces(r,z,psi, levels0):
    if levels0[1]<levels0[0]:
        levels0 = np.flip(levels0)
        flag = 1
    else:
        flag = 0
    R, Z = np.meshgrid(r,z, indexing='ij')
    cs = plt.contour(R,Z, psi, levels=levels0)
    #print(len(cs.collections))
    polygon = [ [r[1],z[1]], [r[-2],z[1]], [r[-2],z[-2]], [r[1],z[-2]] ]
    bdry = mpltPath.Path(np.asarray(polygon))
    xt = []
    yt = []
    psival = []
    # get all contours that do not touch the boundary (i.e., they are closed)
    for i, conts in enumerate(cs.allsegs):
        for path in conts:
            v = path
            if np.all(bdry.contains_points(v)):
               xt.append(v[:,0])
               yt.append(v[:,1])
               psival.append(levels0[i])
    if flag==1:
        xt.reverse()
        yt.reverse()
        psival.reverse()
    plt.close()
    return xt, yt #, np.asarray(psival)


def inductance(r, z, Bp_sq, r_lcfs, z_lcfs, Ip):
    dr = r[1]-r[0]
    dz = z[1]-z[0]
    polygon = [ [r_lcfs[k], z_lcfs[k]] for k in range(r_lcfs.size) ]
    bdry = mpltPath.Path(np.asarray(polygon))

    s = 0.0
    V = 0.0
    area = 0.0
    for i, rt in enumerate(r):
        for j, zt in enumerate(z):
            point = [[rt, zt]]
            if bdry.contains_points(point):
                dV = 2*pi*rt*dr*dz
                s += Bp_sq[i,j]*dV
                V +=  dV
                area += dr*dz

    bp_sq_vol_av = s/V
    li = bp_sq_vol_av*4*pi*area/(mu0**2*Ip**2)
    return li


def betap(psiN, p, r, z,  psiN_2d, r_lcfs, z_lcfs, Ip):
    dr = r[1]-r[0]
    dz = z[1]-z[0]
    dS = dr*dz
    polygon = [ [r_lcfs[k], z_lcfs[k]] for k in range(r_lcfs.size) ]
    bdry = mpltPath.Path(np.asarray(polygon))
    pfn_spline = RectBivariateSpline(r,z, psiN_2d, kx=3,ky=3)

    p_av = 0.0
    S = 0.0
    V = 0.0
    for rt in r:
        for zt in z:
            point = [[rt, zt]]
            if bdry.contains_points(point):
                pfn = pfn_spline.ev(rt, zt, dx=0, dy=0)
                pval = np.interp(pfn, psiN, p)
                dV = 2*pi*rt*dS
                S += dS
                V +=  dV
                p_av += pval*dS

    p_av /= S
    L =0
    for j in range(r_lcfs.size-1): # poloidal perimeter of lcfs
       L += np.sqrt((r_lcfs[j+1]-r_lcfs[j])**2+(z_lcfs[j+1]-z_lcfs[j])**2)
    betap0 = p_av*2*L**2/(mu0*Ip**2)
    return betap0



def partial_derivatives_2d(x,z,b):
    nx,nz=x.size, z.size
    b_x = np.zeros((nx,nz))
    b_z = np.zeros((nx,nz))
    for i in  range(nx):
        for j in range(nz):
              i2=i+1
              i1=i-1
              j2=j+1
              j1=j-1
              if i==0: i1=i
              if j==0: j1=j
              if i==nx-1: i2=i
              if j==nz-1: j2=j
              b_x[i,j]= (b[i2,j]-b[i1,j])/(x[i2]-x[i1])
              b_z[i,j]= (b[i,j2]-b[i,j1])/(z[j2]-z[j1])
    return b_x, b_z              

def laplace_cylindrical2d(x,z,psi):
    nx,nz =x.size, z.size
    psi_x, psi_z   = partial_derivatives_2d(x,z, psi)
    psi_xx, psi_xz = partial_derivatives_2d(x,z, psi_x)
    psi_zx, psi_zz = partial_derivatives_2d(x,z, psi_z)
    jphi = np.zeros((nx,nz))
    for i in range(nx):
        for j in range(nz):
          jphi[i,j] = psi_zz[i,j] + psi_xx[i,j]  - 1/x[i]*psi_x[i,j]
          jphi[i,j] = -jphi[i,j]/(mu0*x[i]) #to toroidal current density
    return jphi[:-1, :-1]




