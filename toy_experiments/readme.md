# Experiments on Toy MOO problems

### Implementation of problems
The `problems` module contains the toy MOO problems, for which the Pareto front in the objective space is available.
- `toy_biobjective.py`: two objectives
- `toy_triobjective.py`: three objectives

Apart from these two, `simulation.py` implements a many objective toy problem.

### Implementation of solvers
The `solvers` module contains four different solvers:
1. Linear Sclarization: `linscalar.py`
2. MGDA based MOO: `moo_mtl.py` and `min_norm_solvers_numpy.py`
3. Pareto MTL: 
	- cpu: `pmtl.py` and `min_norm_solvers_numpy.py`
	- gpu: `pmtl_gpu.py` and `min_norm_solvers.py`
4. EPO Search: `epo_search.py` and `epo_lp.py`
5. FERERO: `pmol.py`


## Experiments in the Main Paper

### Comparison of four Solvers
`compare_solvers.py` compares the spread of Pareto optimal solutions, and its precision for different preference vectors.



