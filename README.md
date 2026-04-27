# Introduction
Runaway electrons can be formed when a disruption occurs in a tokamak reactor. An important aspect of these runaways is the amount of energy that they carry. This study aims to find the average energy of runaways, and does so by first simplifying the fully kinetic description given by the Focker-Planck equation to a fluid model, then by taking the energy moment of different terms. These terms are then discretised using a finite-volume scheme (FVM) in space and a backwards Euler method in time, and the total energy is calculated in time and across flux surfaces for the ITER tokamak.
The dataset represents a disruption mitigation scenario in which a healthy plasma is injected with a shattered pellet made of 99% neon and the rest of deuterium, and looks at the current quench phase after the injection (and the passage of the thermal quench). For more information refer to article, "Runaway electron generation in ITER mitigated disruptions with improved physics models" (L. Votta et al., 2026).
# Setup
First clone the repo then
```
cd plasma-numerics/src
```
Dependencies:
```
pip install numpy, matplotlib
```

## Run a test by
```
python -m tests.testname
``

## Run the model by

To run the model with parameters from a specific dream data model file (that is not in this repo), put the file at
src/dream_outputs.h5 and then run the model by
```
python ./main.py
```
