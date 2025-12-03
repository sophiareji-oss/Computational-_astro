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
- `-d MODEL, --detect MODEL`: Initialize detection algorithms for exoplanets where MODEL can be rf(Random Forest)/dt(Decision tree)/cnn(convolutional neural network) 
- `-a, --atmosphere`: Atmospheric characterization from input transmission spectrum
- `-t, --transit`: Plotting the light curve of the transit

### Examples

```bash
# Run exoplanet detection
daneel -i parameters.yaml -d rf

# Run atmospheric characterization
daneel -i parameters.yaml -a

# Run both detection and atmospheric analysis
daneel -i parameters.yaml -d rf -a

# Run the light curve plotting for the transit
daneel -i parameters.yaml -t

```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

The YAML file should be in following format.

transit:
  XO-2N b:
    t0: 0.               # time of inferior conjunction
    per: 2.615838        # orbital period in days
    rp: 0.103            # planet radius (in units of stellar radii)
    a: 7.993             # semi-major axis (in units of stellar radii)
    inc: 88.01           # orbital inclination (in degrees)
    ecc: 0.028           # eccentricity
    w : 261              # longitude of periastron (in degrees)
    u : [0.4984,0.0785]  # limb darkening coefficients [u1, u2]
    limb_dark: quadratic # limb darkening model

## License

This project is licensed under the MIT License.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
