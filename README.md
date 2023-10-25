# Linear Model Based Gradient Estimation (LMBGE)
This Git repository contains the Julia code for running the numerical experiment in Subsection 4.6.2 of Brian Irwin's 
PhD thesis. Brian Irwin's PhD thesis is available at [http://hdl.handle.net/2429/86077](http://hdl.handle.net/2429/86077).


# Running The Code
To run the quartic function experiment from Subsection 4.6.2 of Brian Irwin's PhD thesis, first open a Julia REPL in the
directory "Quartic Experiment". Then execute the following commands in the Julia REPL:
```
include("aggregate_gd_lmbge_ge_optimize_noisy_quartic.jl")
include("aggregate_gd_lmbge_fge_optimize_noisy_quartic.jl")
include("aggregate_bfgs_lmbge_ge_optimize_noisy_quartic.jl")
include("aggregate_bfgs_lmbge_fge_optimize_noisy_quartic.jl")
include("aggregate_spbfgs_lmbge_ge_optimize_noisy_quartic.jl")
include("aggregate_spbfgs_lmbge_fge_optimize_noisy_quartic.jl")
```
The 6 commands above produce Figures 4.5, 4.6, and 4.7, as well as the data for columns 4 - 7 in Table 4.1.

The numerical experiments code contained in this Git repository was originally tested using Julia 1.5.4 on a computer 
with the Ubuntu 20.04 LTS operating system installed.


# Citation
If you use this code, please cite the PhD thesis:

Irwin, Brian. 2023. ``How To Descend A Rocky Slope: Numerical Techniques For The Solution Of Noisy Optimization Problems.''
Electronic Theses and Dissertations (ETDs) 2008+. T, University of British Columbia. doi:http://dx.doi.org/10.14288/1.0436950

BibTeX:
```
@phdthesis{irwin-phd-thesis-2023,
    Author = {Brian Irwin},
    Title = {How To Descend A Rocky Slope: Numerical Techniques For The Solution Of Noisy Optimization Problems},
    School = {University of British Columbia},
    Year = {2023},
    URL = {http://hdl.handle.net/2429/86077},
    DOI = {10.14288/1.0436950}
}
```
