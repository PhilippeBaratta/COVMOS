# COVMOS in a nutshell

COVMOS is a python code designed to simulate in a fast way catalogues of particles in a box of volume V and of redshift z.
The main advantages of this approximated method are:

- the underlying density field can be non-Gaussian, with a probability distribution function (PDF) and a power spectrum set as inputs by the user
- the nature of the simulated objects (dark matter particles, neutrinos, galaxies, haloes, etc) is only related to the type of the previous statistical targets set by the user
- the nature of the simulated cosmological model (lcdm, massive neutrinos, dynamical dark energy, modified gravity models, etc) is also only related to the type of the previous statistical targets set by the user
- the simulation of peculiar velocities is caracterised by a velocity dispersion related to the local density, thus allowing a reliable reproduction of redshift-space distortions in linear and non-linear regimes
- can be used to quickly produce covariance matrix of two-point statistics

The main pipeline lays on the local, non-linear mapping of an initial Gaussian field ν(x) such that δ(x)=L(ν(x)), where the resulting δ-field must represent the cosmological, non-Gaussian contrast density field.
COVMOS first find out the right power spectrum for ν in such a way that once the Gaussian field is transformed under the local transformation L, the δ variable is both following the targeted power spectrum and PDF set by the user.
In a second step, the density field is discretised into particles following a local Poisson sampling.
Then a velocity field following a Gaussian prescription and a power spectrum set by the user is generated. It is finally discretised to assign peculiar velocities to particles.


# Structure of the method

COVMOS is splitted into two main codes :

**COVMOS_ini.py**

it generates the initilisation files that will be used for the simulation code COVMOS_sim.py. To be run, COVMOS_ini.py needs as argument a setting.ini file specifying the catalogue settings, requiered inputs and outputs (see .ini_files/setting_example.ini).
It can be run from the terminal for example calling `python COVMOS_ini.py setting_example.ini` where the .ini file must be saved in the ini_files repertory

**COVMOS_sim.py**

it is the main code simulating boxes of particles (and associated velocities). It required the files produced by the previous code to be run. The command is similar:
`python COVMOS_ini.py setting_example.ini`

Several statistical inputs are requiered by these two codes. They must be set by the user in a .ini file. If the user wants COVMOS to clone the user own data, several codes are provided to compute these statistical targets (see the next section).
  

# User inputs

The aim of this approximated universe simulation method is to target a small sample of statistical quantities:

*The density power spectrum*

an ascii file provided by the user. This power spectrum can follow a linear or non-linear prescription, and be associated to arbitrary cosmology and redshift. Moreover the refered objects associated to this power spectrum (dark matter particles, galaxies, etc) are also arbitrary. Since COVMOS needs to alias this power spectrum, it must be provided by the user deconvolved and unaliased.
If the user does not provide this file, COVMOS can use [classy](https://github.com/lesgourg/class_public) to compute it. In this case the user must provide the cosmological parameters values associated to classy. Also compute_monop_using_NBK.py helps the user computing the monopole of the power spectrum from his own data, using the [NBodyKit](https://github.com/bccp/nbodykit) module.

*The theta-theta power spectrum*

an ascii file provided by the user. Two options are proposed here. Either the user provides it or classy (combined to the [Bel et al.](https://www.aanda.org/articles/aa/full_html/2019/02/aa34513-18/aa34513-18.html) fitting functions) computes it.

*The probability distribution function of the contrast density field*

an ascii file provided by the user containing the normalised density probability distribution function. The user can also ask COVMOS to estimate it directly from data provided by the user using compute_delta_PDF.py. Note that this PDF must be estimated on the same grid precision as the one of the simulated COVMOS box (same smoothing defined by the quantity L/N_sample, see setting_example.ini).

*The one-point velocity variance*

This target statistical quantity must be set by the user. The code compute_velocity_rms.py helps the user to estimate it from his own data

*The alpha parameter*

alpha is directly linked to the way COVMOS assignes peculiar velocities to particles. The relation velocity variance as a function of the local density field can be approximated by a power law, i.e. Σ^2(ρ) = βρ^α. Either alpha is provided by the user, or COVMOS can estimate it from data provided by the user (using compute_alpha.py).


# COVMOS outputs

from COVMOS_ini.py 

This initialisation code provides numerous required files for COVMOS_sim.py to be run. This includes the power spectra for the Gaussian field (density and velocity), prior to their non-linear transformations into non-Gaussian fields (only for the density field).
On request of the user, it can also provide in ascii files the predictions for the output power spectra and correlation functions. Indeed at small scales, mainly due to the grid precision and Poisson sampling, the input and outputs two-points statistics cannot be equal.

from COVMOS_sim.py

The outputs or this code are the simulated boxes (particle positions and associated velocities). The catalogues are stored in binary files and can easily be loaded using
```
from COVMOS_func import *
x,y,z,vx,vy,vz = loadcatalogue(filename,RSD=True)
```
Moreover, the multipoles (monopole, quadrupole, hexadecapole) of the power spectrum, both in comoving and in redshift spaces, can be asked by the user. In this case COVMOS will call [NBodyKit](https://github.com/bccp/nbodykit) for the estimation. 


# Parallel computation

When running on a single node COVMOS_ini.py and COVMOS_sim.py, a multiprocessing can be exploited thanks to the [numba](https://numba.pydata.org/numba-doc/latest/index.html) library by setting the environment variable OMP_NUM_THREADS to the wanted number of processes.
Moreover to make the execution faster, the codes can also be run through MPI to share the jobs though nodes.
To do so you need to provide a machinefile: a text file that stores the IP addresses of all the nodes in the cluster network. .machinefiles/machinefile_example1 gives an example of its structure.
The command is the following:
`mpiexec -f ./machinefiles/machinefile_example1 -n 10 python COVMOS_ini.py setting.ini` here only acts on the aliasing part of the code, which is highly cpu and memory demanding.
COVMOS_sim.py can also be parallelised using MPI. If more than one catalogue is asked, each node will independently generate catalogues.
Note that if several processes are asked on the same machine (see ./machinefiles/machinefile_example2), some arrays are shared through the differents processes, so that the memory is optimised.

# References

If you are using COVMOS in a publication, please refer the code by citing the following paper:
(baratta et al. 2021, in prep.)
Also if you used the [classy](https://github.com/lesgourg/class_public) or the [NBodyKit](https://github.com/bccp/nbodykit) modules, you should cite the original works.

