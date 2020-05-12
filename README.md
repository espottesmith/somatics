# SOMaTICS: Swarm Optimization of Minima and Transition-states In Chemical Systems

SOMaTICS is a code for identifying important Potential Energy Surface (PES) features
(minima, transition states) using modified Particle Swarm Optimization (PSO) algorithms.

Our algorithm:
1. Explores the PES to identify minima using a convergent niching PSO method
2. Uses Delaunay triangulation to predict which minima are likely to be linked
by transition states (TS)
3. Approximates the Minimum Energy Pathway (MEP) between neighboring minima as
a means to approximate the TS


SOMaTICS is meant to interface with quantum chemistry codes. Currently, an interface
with the eXtended Tight-Binding (xTB) code of Stephan Grimme's group is in operation.
There are plans to add interfaces with the Vienna Ab initio Simulation Program (VASP)
and Q-Chem.


## Installation

To install SOMaTICS, clone from this Github repo:

```
git clone https://github.com/espottesmith/somatics
```

SOMaTICS requires OpenMP, MPI, QHull, and xTB. These codes are all installed on the Cori
supercompter of the National Energy Research Supercomputing Center; for now, we strongly
recommend installing on Cori:

```
cd somatics
make
```

Once somatics has been installed, just run
```
.somatics -h
```
to see how to run the program.

Note that use of molecular PES is currently not advised, though it should be stable.
