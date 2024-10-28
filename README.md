
# heq
tokamak equilibrium reconstruction

How to run heq? 
1. First run green_table.py, which generate numerical table of the Geen's functions. 
The tokamak machine geometry data (coils, flux loops, magneti probles locations) are read in from the files in input fold.
2. Then run fitting.py, which constructs an equilibrium magnetic field using the magnetic measurements (mm.txt in input fold)
   The example is for the EAST tokamak.

   HEQ use an artifical coil to stabilize the VDE instability encountered in the Picard iteration.
   Details are given in Young Mu Jeon's paper: Development of a free-boundary tokamak equilibrium solver for advanced study of tokamak equilibria.
   Journal of the Korean Physical Society, Vol. 67, No. 5 September 2015, pp. 843-853.