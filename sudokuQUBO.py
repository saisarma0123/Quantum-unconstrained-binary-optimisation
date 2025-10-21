import numpy as np
import dimod
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib.pyplot as plt
from itertools import combinations
import time
# --- Helper functions ---
def var_index(i, j, k):
    return i*81 + j*9 + k

# --- Build QUBO ---
def build_sparse_qubo(clues_matrix, A=10, B=10, C=10, D=10, penalty_strength=15, clue_strength=20):
    n = 9
    Q = {}

    # Convert clues matrix to list of (i,j,val)
    clues = [(i,j,int(clues_matrix[i,j])) for i in range(n) for j in range(n) if clues_matrix[i,j]>0]

    # 1️⃣ Cell uniqueness
    for i in range(n):
        for j in range(n):
            for k1,k2 in combinations(range(n),2):
                Q[(var_index(i,j,k1), var_index(i,j,k2))] = 2*A
            for k in range(n):
                Q[(var_index(i,j,k), var_index(i,j,k))] = -2*A

    # 2️⃣ Row uniqueness
    for i in range(n):
        for k in range(n):
            for j1,j2 in combinations(range(n),2):
                Q[(var_index(i,j1,k), var_index(i,j2,k))] = 2*B

    # 3️⃣ Column uniqueness
    for j in range(n):
        for k in range(n):
            for i1,i2 in combinations(range(n),2):
                Q[(var_index(i1,j,k), var_index(i2,j,k))] = 2*C

    # 4️⃣ Block uniqueness
    for br in range(3):
        for bc in range(3):
            cells = [(i,j) for i in range(br*3,(br+1)*3) for j in range(bc*3,(bc+1)*3)]
            for k in range(n):
                for (i1,j1),(i2,j2) in combinations(cells,2):
                    Q[(var_index(i1,j1,k), var_index(i2,j2,k))] = 2*D

    # 5️⃣ Clue biases and penalties
    for i, j, val in clues:
        correct_idx = var_index(i, j, val-1)
        # reward correct choice
        Q[(correct_idx, correct_idx)] = Q.get((correct_idx, correct_idx), 0) - penalty_strength
        # penalty for wrong numbers
        for k in range(n):
            if k != val-1:
                wrong_idx = var_index(i,j,k)
                Q[(wrong_idx, wrong_idx)] = Q.get((wrong_idx, wrong_idx), 0) + 2*penalty_strength
        # strong clue bias
        Q[(correct_idx, correct_idx)] -= clue_strength
        for k in range(n):
            if k != val-1:
                idx2 = var_index(i,j,k)
                Q[(idx2, idx2)] = Q.get((idx2, idx2),0) + clue_strength

    # 6️⃣ Encourage picking at least one number per cell
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = var_index(i,j,k)
                Q[(idx, idx)] = Q.get((idx, idx),0) - penalty_strength

    return Q, clues

# --- Solve Sudoku ---
def solve_sudoku_sparse(matrix, num_reads=100, seed=42):
    start_time = time.time()  # ← START TIMER HERE

    Q_dict, clues = build_sparse_qubo(matrix)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)
    best = sampleset.first.sample

    grid = np.zeros((9,9), dtype=int)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                idx = var_index(i,j,k)
                if best.get(idx,0) == 1:
                    grid[i,j] = k+1

    total_time = time.time() - start_time   # ← NOW total_time works
    time_per_read = total_time / num_reads

    return grid, clues, sampleset, total_time, time_per_read

# --- Validity check ---
def check_validity(grid):
    def valid(block): return sorted(block)==list(range(1,10))
    for i in range(9):
        if not valid(grid[i,:]) or not valid(grid[:,i]): return False
    for br in range(3):
        for bc in range(3):
            block = grid[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
            if not valid(block): return False
    return True


# --- Quantum-style outputs ---
def quantum_style_outputs(sampleset, clues, clue_strength):
    best = sampleset.first
    print("✅ Best solution energy:", best.energy)
    print("Number of reads:", sampleset.info.get('num_reads', 'N/A'))

    print("\nClues applied (with bias strength):")
    for i, j, val in clues:
        print(f" - Cell ({i+1},{j+1}) = {val}, bias = {clue_strength}")

    print("\nSample energies, occurrences, and probabilities (first 10 samples):")
    data = list(sampleset.data(['sample','energy','num_occurrences']))
    total_occurrences = sum([count for _, _, count in data])
    for idx, (sample, energy, count) in enumerate(data[:10]):
        prob = count / total_occurrences
        print(f"Energy: {energy}, Occurrences: {count}, Probability: {prob:.4f}")

    # Energy histogram
    energies = [energy for _, energy, _ in data]
    plt.hist(energies, bins=30)
    plt.title("Energy distribution of samples (quantum-style)")
    plt.xlabel("Energy")
    plt.ylabel("Occurrences")
    plt.show()
# --- Classic backtracking solver ---
def classic_sudoku_solver(grid):
    start_time = time.time()
    def is_safe(r, c, n):
        # check row/column
        if n in grid[r,:] or n in grid[:,c]:
            return False
        # check block
        br, bc = 3*(r//3), 3*(c//3)
        if n in grid[br:br+3, bc:bc+3]:
            return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if grid[i,j] == 0:
                    for n in range(1,10):
                        if is_safe(i,j,n):
                            grid[i,j] = n
                            if solve():
                                return True
                            grid[i,j] = 0
                    return False
        return True

    solved = solve()
    elapsed = time.time() - start_time
    return grid, solved, elapsed


# --- Example: AI Escargot ---
ai_escargot = np.array( [
    [2, 0, 0, 0, 0, 0, 0, 0, 5],
    [0, 0, 0, 8, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3]
])

# --- Solve & print ---
solved_grid, clues, sampleset, total_time, time_per_read = solve_sudoku_sparse(ai_escargot)
print("\n✅ Solved Grid:")
for row in solved_grid:
    print(" ".join(map(str,row)))

print("\n✔️ Valid Sudoku:", check_validity(solved_grid))

print("---------------------------------------------------")
print("---------------------------------------------------")

print(f"⏱ Average time taken per read: {time_per_read:.4f} s")
print(f"⏱ total time taken: {total_time:.4f} s")
# --- Solve Classic ---
classic_grid, solved_flag, classic_time = classic_sudoku_solver(ai_escargot.copy())
print("\n✅ Classic Solved Grid:")
for row in classic_grid:
    print(" ".join(map(str,row)))
print("✔️ Valid Sudoku:", solved_flag)
print(f"⏱ Classic solver time: {classic_time:.4f} s")


# --- Quantum-style analysis ---
quantum_style_outputs(sampleset, clues, clue_strength=20)

