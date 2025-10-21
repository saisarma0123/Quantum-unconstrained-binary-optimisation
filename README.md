# Quantum-unconstrained-binary-optimisation
This code demonstrates the working and functionality of QUBO model, that uses "QUANTUM ANALYSIS" to solve complex puzzles and this algorithm can be directly applied to other problems too.

# Sudoku QUBO Benchmark

Compare a classical backtracking Sudoku solver vs. a QUBO-based solver (Simulated Annealing) and benchmark success rates, timing and energy distributions.

## Overview
This repo contains:
- `sudokuQUBO.py` — QUBO builder, SA sampler wrapper, classic backtracking solver.
- `tester.py` — batch benchmark runner that loads puzzles, runs both solvers, saves CSV results and plots.
- `sudoku.csv` — dataset of Sudoku puzzles (not included by default).
- `results/` — generated outputs (plots, CSVs) — ignored from Git.

This project is aimed at experimentation and research-style benchmarking, not production-level annealing.

## When using make sure to have the required libraries:
numpy
pandas
matplotlib
dimod
neal            # SA sampler if you use neal.SimulatedAnnealingSampler
dwave-ocean-sdk # optional (if you use other D-Wave Ocean tooling)

# Make sure to download a sudoku.csv of 1 million puzzles, and keep every file in same folder. Access it through thif link :
https://www.kaggle.com/datasets/bryanpark/sudoku
