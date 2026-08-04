# About ASPIRE
ASPIRE (Astrophysics Summer Program for International Research Experience) by University of Amsterdam is 8-weeks research internship program for young astronomers especially masters students to escalate their research skills and technical skills.

# Kepler-221 Exoplanet System
Kepler-221 located 385 pc away from earth that contains 4 planets planet b, c, d, and e. But planet d is not in orbital resonance even though its located in the middle of the system. Yi et al, 2025 proposed their scenario about this. Planet d was a result of the collided system in the past then his lost orbital resonance after that.

But we (Me, Dr. Silvia Toonen, and Dr. Tjarda. C. N. Boekholt) propose our scenario to put another planet (planet f) in outer region of the system. We use TTVs (Transit Timing Variations) to see the effect of the planet f to the system. Since no package that could convert N-Body simulation data into light curve, we built our own pipelines to do that.

# Pipline
## 1. We built AMUSE (Astrophysical Multipurpose Software Environment) pipeline to do the N-Body simulation and apply adaptive timestep to make the light curve smoother.
## 2. We do some geomatrical thing to convert N-Body Simulation data into light curve.
## 3. We also add limb darkening effect from Mandel & Agol, 2002 model to make it more realistic.
## 4. Analyze the light curve from transitting planets to see the effect of planet f

# Results
## There is possibility the existance of planet f since the residuls of planet e was not 0.
