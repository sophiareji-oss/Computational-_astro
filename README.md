# Daneel

A practical example to detect and characterize exoplanets.

The full documentation is at https://tiziano1590.github.io/comp_astro_25/index.html

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install .
```

### Development installation

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install -e .
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> [options]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-d DETECT, --detect DETECT`: Initialize detection algorithms for exoplanets where DETECT can be rf(Random Forest)/dt(Decision tree)/cnn(convolutional neural network) 
- `-a ATMOSPHERE, --atmosphere ATMOSPHERE`: Atmospheric characterization from input Parameters when ATMOSPHERE is model and retrive the parameters from spectrum when ATMOSPHERE is retrieve
- `-t, --transit`: Plotting the light curve of the transit

### Examples

```bash
# Run exoplanet detection
daneel -i parameters.yaml -d rf

# Run atmospheric characterization
daneel -i parameters.yaml -a model

# Run both detection and atmospheric analysis
daneel -i parameters.yaml -d rf -a model

# Run the light curve plotting for the transit
daneel -i parameters.yaml -t

```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

The YAML file for transit should be in following format.

transit:
  XO-2N b:               # name of the planet
    t0: 0.               # time of inferior conjunction
    per: 2.615838        # orbital period in days
    rp: 0.103            # planet radius (in units of stellar radii)
    a: 7.993             # semi-major axis (in units of stellar radii)
    inc: 88.01           # orbital inclination (in degrees)
    ecc: 0.028           # eccentricity
    w : 261              # longitude of periastron (in degrees)
    u : [0.4984,0.0785]  # limb darkening coefficients [u1, u2]
    limb_dark: quadratic # limb darkening model

Example input parameters for atmosphere file is in the example folder.

## License

This project is licensed under the MIT License.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
