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

The density power spectrum is provided by the user as an ASCII file. This power spectrum may adhere to either a linear or non-linear prescription and be associated with arbitrary cosmology and redshift. Furthermore, the objects referred to by this power spectrum (e.g., dark matter particles, galaxies, etc.) are also arbitrary. Since COVMOS requires aliasing this power spectrum, it must be supplied by the user in a deconvolved and unaliased form (the standard form). 

If the user does not provide this file, COVMOS can utilize classy to compute it. In this scenario, the user must provide the cosmological parameter values for [classy](https://github.com/lesgourg/class_public). Additionally, ./helping_codes/compute_shell_average_monopole.py assists the user in computing the monopole of their own data's power spectrum using the [NBodyKit](https://github.com/bccp/nbodykit) module. Alternatively, the user can supply a 3D target power spectrum in .npy format, which in this case should be convolved and aliased. For estimating this from data, see ./helping_codes/compute_3D_aliased_Pk.py (this option offers the best results).

*The theta-theta power spectrum*

The user must provide an ASCII file containing the theta-theta velocity power spectrum. There are two options available:

1. The user directly supplies the theta-theta power spectrum file.
2. COVMOS, leveraging classy and the fitting functions from [Bel et al.](https://www.aanda.org/articles/aa/full_html/2019/02/aa34513-18/aa34513-18.html), computes the power spectrum.

*The probability distribution function of the contrast density field*

The user needs to provide an ASCII file containing the normalized density probability distribution function (PDF) in a linear binning. Alternatively, COVMOS can estimate this PDF directly from user-provided data using the script located at ./helping_codes/compute_delta_PDF.py. It is crucial that this PDF is estimated using the same grid precision as the simulated COVMOS box. This means the smoothing should be defined by the quantity L/Nsample, as outlined in the example settings file located at ./ini_files/setting_example.ini.

*The one-point velocity variance*

This statistical target quantity must be defined by the user. To facilitate estimation from user's own data, the script ./helping_codes/compute_velocity_rms.py is provided. It assists in calculating the one-point velocity variance, ensuring users can accurately set this parameter based on their dataset.

*The alpha parameter*

The alpha parameter plays a crucial role in how COVMOS assigns peculiar velocities to particles. The relationship between velocity variance and the local density field can be approximated by a power law, expressed as Σ^2(ρ) = βρ^α. Users have two options:

1. Provide alpha based on their own theoretical calculation or prior knowledge.
2. Utilize the provided script ./helping_codes/compute_alpha.py to estimate alpha directly from their data.

This approach allows COVMOS to adapt its velocity assignment process to closely match the user's data characteristics or theoretical model.

# COVMOS outputs

*The two-point statistics prediction*

COVMOS employs multiple mode-filterings, affecting the output power spectra's match to targeted ones at small scales, dependent on grid precision. However, the impact of these filterings on two-point statistics can be analytically computed with better than one percent accuracy.
Users can request predictions for the output two-point correlation functions and power spectra.

*The catalogues*

The core output includes simulated boxes, detailing particle positions and associated velocities. These catalogues are stored in binary format and can be easily loaded as follows:
```
from tools.COVMOS_func import loadcatalogue
x,y,z,vx,vy,vz = loadcatalogue(filename,velocity=True)
```

*The estimated power spectra*

COVMOS allows users to request the multipoles of the power spectrum (monopole, quadrupole, hexadecapole) in both real and redshift spaces. For estimation, it utilizes [NBodyKit](https://github.com/bccp/nbodykit).

*The unbiased covariance*

At small scales (k ~ 0.2h/Mpc), the COVMOS covariance of the multipoles of the power spectrum exhibits slight bias. This bias can be corrected using the method presented in [Baratta et al. 22](https://www.aanda.org/articles/aa/full_html/2023/05/aa45683-22/aa45683-22.html), which users can request in the .ini file.



## Installation

**Prerequisites:** Linux/macOS with `wget`, `unzip`, `make`, `gcc` (or clang), and MPI if you plan to run distributed jobs.

1. **Clone the repository (with submodules):**

   ```bash
   git clone --recurse-submodules https://github.com/PhilippeBaratta/COVMOS.git
   cd COVMOS
   ```

2. **Download the prebuilt environment from Zenodo (1.24 GB):**

   ```bash
   wget "https://zenodo.org/record/17551945/files/COVMOS-env.zip?download=1" -O COVMOS-env.zip
   ```

3. **Unzip and remove the archive:**

   ```bash
   unzip COVMOS-env.zip
   rm COVMOS-env.zip
   ```

   This extracts a self-contained environment folder `COVMOS-env/` that already includes all required packages and pinned versions.

4. **Build CLASS (submodule):**

   ```bash
   cd tools/class_public
   make
   cd ../../
   ```

### How to run COVMOS with the bundled Python

You do **not** need to “activate” anything: simply call the Python interpreter from the bundled environment when running COVMOS.

**Single process example:**

```bash
./COVMOS-env/bin/python3 COVMOS.py both ./ini_files/setting_example.ini
```

**MPI example (4 ranks):**

```bash
mpiexec -n 4 ./COVMOS-env/bin/python3 COVMOS.py both ./ini_files/setting_example.ini
```

If your environment was extracted elsewhere, provide the absolute path to its Python:

```bash
mpiexec -n 4 '/path/to/COVMOS-env/bin/python3' COVMOS.py both initialisation.ini
```

> Tip: for convenience, you can define an environment variable in your shell:
>
> ```bash
> export COVMOS_PY="$(pwd)/COVMOS-env/bin/python3"
> $COVMOS_PY COVMOS.py both ./ini_files/setting_example.ini
> mpiexec -n 4 $COVMOS_PY COVMOS.py both ./ini_files/setting_example.ini
> ```

# References

If you are using COVMOS in a publication, please refer the code by citing the following papers:

```
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
```
Also if you used the [classy](https://github.com/lesgourg/class_public) or the [NBodyKit](https://github.com/bccp/nbodykit) modules, you should cite the original works.

Any question? Please contact me at philippe.baratta@univ-amu.fr
