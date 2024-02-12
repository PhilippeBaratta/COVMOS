# COVMOS in a nutshell

COVMOS is a Python code designed to simulate, in a fast way, catalogues of particles in a box of volume V and of redshift z. The main advantages of this approximated method are:

- The underlying density field can be non-Gaussian, with a probability distribution function (PDF) and a power spectrum set as inputs by the user.
- The nature of the simulated objects (dark matter particles, neutrinos, galaxies, haloes, etc.) is only related to the type of the statistical targets previously set by the user.
- The nature of the simulated cosmological model (ΛCDM, massive neutrinos, dynamical dark energy, modified gravity models, etc.) is also only related to the type of the statistical targets previously set by the user.
- The simulation of peculiar velocities is characterized by a velocity dispersion related to the local density, thus allowing a reliable reproduction of redshift-space distortions in linear and non-linear regimes.
- It can be used to quickly produce a covariance matrix of two-point statistics.

The main pipeline relies on the local, non-linear mapping of an initial Gaussian field ν(x) such that δ(x) = L(ν(x)), where the resulting δ-field must represent the cosmological, non-Gaussian contrast density field. COVMOS first finds the right power spectrum for ν in such a way that once the Gaussian field is transformed under the local transformation L, the δ variable follows both the targeted power spectrum and PDF set by the user. In a second step, the density field is discretized into particles following a local Poisson sampling. Then, a velocity field following a Gaussian prescription and a power spectrum set by the user is generated. It is finally discretized to assign peculiar velocities to particles.


# Structure of the method

COVMOS is resumed in one main python code:

**COVMOS.py**

To run COVMOS.py, two essential arguments are needed:

- The COVMOS mode, which can be 'ini', 'sim', or 'both'.
- The path to an .ini file that specifies the project settings, including required inputs and outputs (refer to ./ini_files/setting_example.ini for an example).

In 'ini' mode, COVMOS generates the initialization files needed for the simulation mode 'sim'. This mode can be executed by calling `python COVMOS.py ini path/to/the/inifile.ini`, either as a standalone process or in parallel (for details, see the 'Parallel computation' section).

After preparing the initialization files, the 'sim' mode is used to simulate boxes of particles along with their velocities. This mode can also be executed by calling `python COVMOS.py sim path/to/the/inifile.ini` (refer to the 'Parallel computation' section for more information).

To execute the entire process in one go, use the 'both' mode with the command: `python COVMOS.py both path/to/the/inifile.ini`.

COVMOS requires various statistical inputs, which users must define in an .ini file. For those looking to generate data similar to their own (clones), COVMOS provides several utilities in the ./helping_codes folder to help compute these statistical targets (further details are provided in the following section).
  

# User inputs

The aim of this approximated universe simulation method is to target a small sample of statistical quantities:

*The density power spectrum*

an ascii file provided by the user. This power spectrum can follow a linear or non-linear prescription, and be associated to arbitrary cosmology and redshift. Moreover the referred objects associated with this power spectrum (dark matter particles, galaxies, etc.) are also arbitrary. Since COVMOS needs to alias this power spectrum, it must be provided by the user deconvolved and unaliased.
If the user does not provide this file, COVMOS can use [classy](https://github.com/lesgourg/class_public) to compute it. In this case the user must provide the cosmological parameter values associated to classy. Also ./helping_codes/compute_shell_average_monopole.py helps the user computing the monopole of the power spectrum of his own data, using the [NBodyKit](https://github.com/bccp/nbodykit) module. Finally, the user can provide a 3D target power spectrum in a .npy format, in this case convolved and aliased, see ./helping_codes/compute_3D_aliased_Pk.py to estimate it from data (this option offers the best results)

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

COVMOS is a method that needs several mode-filterings. In this way the output power spectra of the produced catalogues are not exactly matching the targeted ones at small scales (depending on the grid precision). However the impact of these filterings on the output two-point statistics can be analytically computed at a better than the percent accuracy. The user can ask for the prediction of the output two-point correlation functions and power spectra

*The catalogues*

They are the simulated boxes (particle positions and associated velocities). The catalogues are stored in binary files and can easily be loaded using
```
from tools.COVMOS_func import loadcatalogue
x,y,z,vx,vy,vz = loadcatalogue(filename,velocity=True)
```

*The estimated power spectra*

The multipoles (monopole, quadrupole, hexadecapole) of the power spectrum, both in real and redshift spaces, can be asked by the user. In this case COVMOS will call [NBodyKit](https://github.com/bccp/nbodykit) for the estimation.

*The unbiased covariance*

The COVMOS covariance of the multipoles of the power spectrum is slightly biased at small scales (k ~ 0.2h/Mpc). This bias can be removed applying the method presented in Baratta et al. 22 (in prep) and asked by the user in the .ini file

# Installation

First, clone the COVMOS repository along with its submodules (class_public and fast_interp) by running the following command:
`git clone --recurse-submodules https://github.com/PhilippeBaratta/COVMOS.git`
After cloning the repository, navigate into the COVMOS directory and create a new conda environment using the COVMOS-env.yml file provided in the repository:
`cd COVMOS
conda env create -f COVMOS-env.yml`
This command creates a new conda environment with all the dependencies specified in the COVMOS-env.yml file (including the nbodykit code). Once the environment is created, activate it using:
`conda activate COVMOS-env`
Now, navigate to the CLASS submodule directory to compile the class_public library:
`cd tools/class_public
make`

# Parallel computation

When running on a single node COVMOS.py, a multiprocessing can be exploited thanks to the [numba](https://numba.pydata.org/numba-doc/latest/index.html) library by setting the environment variable OMP_NUM_THREADS to the wanted number of processes.
Moreover to make the execution faster, the codes can also be run through MPI to share the jobs though nodes.
To do so, you need to provide a machinefile: a text file that stores the IP addresses of all the nodes in the cluster network. ./machinefiles/machinefile_example1 gives an example of its structure.
The command is the following:
`mpiexec -f ./machinefiles/machinefile_example -n 32 python COVMOS.py both setting.ini`

# References

If you are using COVMOS in a publication, please refer the code by citing the following papers:

@article{Baratta:2019bta,
    author = "Baratta, Philippe and Bel, Julien and Plaszczynski, Stephane and Ealet, Anne",
    title = "{High-precision Monte-Carlo modelling of galaxy distribution}",
    eprint = "1906.09042",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1051/0004-6361/201936163",
    journal = "Astron. Astrophys.",
    volume = "633",
    pages = "A26",
    year = "2020"
}

@article{Baratta:2022gqd,
    author = "Baratta, Philippe and Bel, Julien and Gouyou Beauchamps, Sylvain and Carbone, Carmelita",
    title = "{COVMOS: a new Monte Carlo approach for galaxy clustering analysis}",
    eprint = "2211.13590",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1051/0004-6361/202245683",
    journal = "Astron. Astrophys.",
    volume = "673",
    pages = "A1",
    year = "2023"
}

Also if you used the [classy](https://github.com/lesgourg/class_public) or the [NBodyKit](https://github.com/bccp/nbodykit) modules, you should cite the original works.

