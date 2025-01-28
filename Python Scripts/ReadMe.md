# Python script overview 

**CahnHilliard_ModelValidation**: PF simulation used for model validation. Solves Cahn-Hilliard equation for periodic boundary conditions.

**STrIPS_PF**: PF simulation of bijel formation via STrIPS with one open boundary (top) for solvent diffusion. 

**STrIPS_PF_2Sided**: PF simulation of bijel formation via STrIPS with two open boundaries (top and bottom) for solvent diffusion. 

**PoreSizeAnalysis_SigmaC**: Script for the pore size analysis of bijel morphologies with different values of the critical IFT. [Associated Dataset](../Data/Simulation/SigmaC) 

**PoreSizeAnalysis_ExperimentalVsSimulation**: Script for the pore size analysis of both experimental (confocal) and simulated bijel morphologies. [Associated Dataset (Experiment) ](../Data/Experiment) & [Associated Dataset (Simulation)](../Data/Simulation/ExperimentVsSimulation)

Apart from standard libraries (NumPy, SciPy etc), the scripts make use of the SciencePlots library (see https://pypi.org/project/SciencePlots/ for installation instructions).
