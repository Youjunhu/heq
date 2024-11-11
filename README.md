
# HEQ
Tokamak equilibrium reconstruction.

How to run HEQ? 
1. First run green_table.py, which generates numerical table of Green's functions. 
The tokamak machine geometry data (coils, flux loops, magneti probles locations) are read in from the files in input fold.
2. Then run fitting.py, which constructs an equilibrium magnetic field using the magnetic measurements (mm.txt in input fold)
   The example is for the EAST tokamak.

The details of algorithms used in HEQ can be found <a href="https://youjunhu.github.io/research_notes/tokamak_equilibrium_htlatex/tokamak_equilibrium.html">here</a>. 
