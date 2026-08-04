# Kepler-221 Exoplanet System: Transit Timing Variations Pipeline

## About ASPIRE (2026)

ASPIRE (Astrophysics Summer Program for International Research Experience) at the University of Amsterdam is an 8-week (for me from 18th June until 7th August 2026) research internship program designed to help young astronomers — particularly master's students — accelerate their research and technical skills.

## The Kepler-221 Exoplanet System

Kepler-221 is located 385 pc from Earth and hosts four known planets: Kepler-221 b, c, d, and e. Notably, planet d is *not* in orbital resonance, despite sitting in the middle of the system — an unusual configuration given the resonant chain formed by the other planets.

Yi et al. (2025) proposed that planet d's broken resonance is the result of a past collision within the system. Our team — Dr. Silvia Toonen, Dr. Tjarda C. N. Boekholt, and myself — proposes an alternative scenario: the presence of an additional, undiscovered planet (**planet f**) in the outer region of the system, whose gravitational influence could account for the observed configuration without invoking a collisional history.

To test this hypothesis, we use **Transit Timing Variations (TTVs)** to search for the gravitational signature of planet f on the rest of the system. Since no existing package directly converts N-body simulation output into a synthetic light curve, we built a custom pipeline to bridge that gap.

## Our Pipeline

1. **N-Body Simulation** — An [AMUSE](https://www.amusecode.org/) (Astrophysical Multipurpose Software Environment) pipeline drives the N-body integration, using an adaptive timestep to ensure the resulting light curve is smooth and free of numerical artifacts.
2. **Geometric Conversion** — Geometric transformations project the 3D N-body simulation output onto the sky plane, converting positions into a 1D transit light curve.
3. **Limb Darkening** — The Mandel & Agol (2002) analytic transit model is used to incorporate limb darkening, producing physically realistic transit light curves.
4. **Analysis** — The resulting light curves are analyzed for timing perturbations (TTVs) in the transiting planets, which would indicate the gravitational influence of the hypothetical planet f.

## Results

Preliminary results show non-zero TTV residuals for planet e, suggesting a real possibility that an unseen planet f is dynamically perturbing its orbit.

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/AkmalHusain2003/ASPIRE.git
cd ASPIRE
```

### 2. Install Dependencies (AMUSE via pip)

The simplest way to install AMUSE and the Huayno N-body integrator used in this project is via `pip`:

```bash
# Core AMUSE framework
pip install amuse-framework

# Huayno N-body integrator module
pip install amuse-huayno
```

You will also need the standard scientific Python stack:

```bash
pip install numpy scipy matplotlib astropy tqdm h5py shapely
```

### 3. Alternative: Install AMUSE from Source

If you prefer (or need) to build AMUSE from source, download a release archive directly:

```bash
# Download the source code
curl -L -O "https://github.com/amusecode/amuse/archive/refs/tags/v2025.9.0.tar.gz"

# Unpack the archive
tar xf v2025.9.0.tar.gz

# Enter the directory and run setup
cd amuse-2025.9.0
./setup
```

### 4. Running the Pipeline

Once AMUSE and the dependencies are installed, run the main simulation and light curve generator:

```bash
python run_sim.py input_sim.txt
```

But if you are using the precomputed one, it will be:
```bash
python run_sim.py --precompute input_sim_precompute.txt
```

## Acknowledgements

This project was carried out as part of the ASPIRE program at the University of Amsterdam, under the supervision of Dr. Silvia Toonen and Dr. Tjarda C. N. Boekholt.

## References

- Mandel, K., & Agol, E. (2002). Analytic light curves for planetary transit searches. The Astrophysical Journal Letters, 580(2), L171–L175.
- Yi, T., Ormel, C. W., Huang, S., & Petit, A. C. (2025). The dynamical history of the Kepler-221 planet system. Astronomy & Astrophysics, 695, A191.
