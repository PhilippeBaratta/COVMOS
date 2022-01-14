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

COVMOS is resumed in one main python code:

**COVMOS.py**

To be run, COVMOS.py needs two key arguments:

- the COVMOS mode that takes the values 'ini', 'sim' or 'both'
- the path to a .ini file specifying the project settings, requiered inputs and outputs (see ./ini_files/setting_example.ini)

In the 'ini' mode, COVMOS generates the initilisation files that will be used for the simulating mode 'sim'. It can be run calling `python COVMOS.py ini path/to/the/inifile.ini` or in parallel (see section 'Parallel computation').

Once done, the 'sim' mode simulates boxes of particles and associated velocities. It can also be run calling `python COVMOS.py sim path/to/the/inifile.ini` or in parallel (see section 'Parallel computation').

You can run the whole pipeline using the 'both' mode: `python COVMOS.py both path/to/the/inifile.ini`

Several statistical inputs are requiered when using COVMOS. They must be set by the user in a .ini file. If the user wants COVMOS to clone its own data, several codes are provided to compute these statistical targets in the ./helping_codes folder (see the next section).
  

# User inputs

The aim of this approximated universe simulation method is to target a small sample of statistical quantities:

*The density power spectrum*

an ascii file provided by the user. This power spectrum can follow a linear or non-linear prescription, and be associated to arbitrary cosmology and redshift. Moreover the refered objects associated to this power spectrum (dark matter particles, galaxies, etc) are also arbitrary. Since COVMOS needs to alias this power spectrum, it must be provided by the user deconvolved and unaliased.
If the user does not provide this file, COVMOS can use [classy](https://github.com/lesgourg/class_public) to compute it. In this case the user must provide the cosmological parameters values associated to classy. Also ./helping_codes/compute_shell_average_monopole.py helps the user computing the monopole of the power spectrum from his own data, using the [NBodyKit](https://github.com/bccp/nbodykit) module. Finally, the user can provide a 3D target power spectrum in a .npy format, in this case convolved and aliased, see ./helping_codes/compute_3D_aliased_Pk.py to estimate it from data (this option offers the best results)

*The theta-theta power spectrum*

an ascii file provided by the user. Two options are proposed here. Either the user provides it or classy (combined to the [Bel et al.](https://www.aanda.org/articles/aa/full_html/2019/02/aa34513-18/aa34513-18.html) fitting functions) computes it.

*The probability distribution function of the contrast density field*

an ascii file provided by the user containing the normalised density probability distribution function. The user can also ask COVMOS to estimate it directly from data provided by the user using ./helping_codes/compute_delta_PDF.py. Note that this PDF must be estimated on the same grid precision as the one of the simulated COVMOS box (same smoothing defined by the quantity L/N_sample, see ./ini_files/setting_example.ini).

*The one-point velocity variance*

This target statistical quantity must be set by the user. The code ./helping_codes/compute_velocity_rms.py helps the user to estimate it from his own data

*The alpha parameter*

alpha is directly linked to the way COVMOS assignes peculiar velocities to particles. The relation velocity variance as a function of the local density field can be approximated by a power law, i.e. Σ^2(ρ) = βρ^α. Either alpha is provided by the user, or COVMOS can estimate it from data provided by the user (using ./helping_codes/compute_alpha.py).


# COVMOS outputs


*The two-point statistics prediction*

COVMOS is a method that needs several filterings. In this way the output power spectra of the produced catalogues are not exactly matching the targeted ones at small scales (depending on the grid precision). However the impact of these filterings on the output two-point statistics can be analyticaly computed at a better than the percent accuracy. The user can ask for the prediction of the output two-point correlation functions and power spectra

*The catalogues*

They are the simulated boxes (particle positions and associated velocities). The catalogues are stored in binary files and can easily be loaded using
```
from tools.COVMOS_func import loadcatalogue
x,y,z,vx,vy,vz = loadcatalogue(filename,velocity=True)
```

*The estimated power spectra*

The multipoles (monopole, quadrupole, hexadecapole) of the power spectrum, both in real and redshift spaces, can be asked by the user. In this case COVMOS will call [NBodyKit](https://github.com/bccp/nbodykit) for the estimation.

*The unbiased covariance*

The COVMOS covariance of the multipoles of the power spectrum are slightly biased at small scales (k ~ 0.2h/Mpc). This bias can be removed applying the method presented in Baratta et al. 22 (in prep) and asked by the user in the .ini file

# Installation

When cloning COVMOS to your directory, make sure to pass the option `--recurse-submodules` to the git `clone command`, it will automatically initialize and update one external submodule used by COVMOS, called [fast_interp](https://github.com/dbstein/fast_interp):
`git clone --recurse-submodules https://github.com/PhilippeBaratta/COVMOS.git`

Moreover, make sure that the [numba](https://numba.pydata.org/numba-doc/latest/index.html) library is already installed on you machine (mandatory), as well as [NBodyKit](https://github.com/bccp/nbodykit) and [classy](https://github.com/lesgourg/class_public) if you want COVMOS to use them (optionnal).

# Parallel computation

When running on a single node COVMOS.py, a multiprocessing can be exploited thanks to the [numba](https://numba.pydata.org/numba-doc/latest/index.html) library by setting the environment variable OMP_NUM_THREADS to the wanted number of processes.
Moreover to make the execution faster, the codes can also be run through MPI to share the jobs though nodes.
To do so you need to provide a machinefile: a text file that stores the IP addresses of all the nodes in the cluster network. ./machinefiles/machinefile_example1 gives an example of its structure.
The command is the following:
`mpiexec -f ./machinefiles/machinefile_example -n 32 python COVMOS.py both setting.ini`

# References

If you are using COVMOS in a publication, please refer the code by citing the following paper:
(baratta et al. 22, in prep.)
Also if you used the [classy](https://github.com/lesgourg/class_public) or the [NBodyKit](https://github.com/bccp/nbodykit) modules, you should cite the original works.

