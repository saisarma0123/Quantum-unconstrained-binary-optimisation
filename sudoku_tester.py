import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sudokuQUBO import solve_sudoku_sparse, classic_sudoku_solver

# --- Setup ---
os.makedirs("results", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Experiment parameters ---
N_SAMPLES = 100          # how many puzzles to test
NUM_READS = 10            # number of annealing reads per puzzle
SAVE_ENERGIES = True     # store energy distribution per puzzle

# --- Load and sample random puzzles ---
try:
    df = pd.read_csv("sudoku.csv")  # or pd.read_csv("sudoku.csv")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

df_sampled = df.sample(n=N_SAMPLES, random_state=76).reset_index(drop=True)

records = []

# --- Solve puzzles ---
for idx, row in df_sampled.iterrows():
    puzzle_str = str(row['quizzes']).strip()  # adjust column name to your Excel
    if len(puzzle_str) != 81:
        print(f"Skipping invalid puzzle {idx}")
        continue

    grid = np.array([[int(puzzle_str[i*9 + j]) for j in range(9)] for i in range(9)])
    num_clues = int(np.count_nonzero(grid))
    print(f"\n▶ Solving Puzzle {idx} ({num_clues} clues)...")

    # --- Classical solver ---
    classic_valid = False
    try:
        classic_grid, solved_flag, classic_time = classic_sudoku_solver(grid.copy())
        classic_valid = np.count_nonzero(classic_grid) == 81

    except Exception as e:
        print(f"Classic solver failed: {e}")
        classic_time= np.nan

    # --- QUBO solver ---
    try:
        solved_grid, clues, sampleset, total_time, time_per_read = solve_sudoku_sparse(grid, num_reads=NUM_READS)
        qubo_valid = np.count_nonzero(solved_grid) == 81
        best_energy = sampleset.first.energy
        energies = [rec.energy for rec in sampleset.record]
        occurrences = [rec.num_occurrences for rec in sampleset.record]
        avg_energy = np.mean(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)
    except Exception as e:
        print(f"QUBO solver failed: {e}")
        total_time = time_per_read = best_energy = avg_energy = min_energy = max_energy = np.nan
        qubo_valid = False
        energies = occurrences = []

    # --- Store record ---
    records.append({
        "puzzle_index": idx,
        "num_clues": num_clues,
        "classic_time": classic_time,
        "classic_valid": classic_valid,  # renamed from classic_valid
        "qubo_time_total": total_time,
        "qubo_time_avg": time_per_read,
        "best_energy": best_energy,
        "avg_energy": avg_energy,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "qubo_valid": qubo_valid,
        "energies": energies if SAVE_ENERGIES else None,
        "occurrences": occurrences if SAVE_ENERGIES else None
    })

# --- Save raw results ---
results_df = pd.DataFrame(records)
results_csv = f"results/sudoku_benchmark_{timestamp}.csv"
results_df.to_csv(results_csv, index=False)
print(f"\n✅ Saved results → {results_csv}")

# --- Filter successful runs ---
success_rows = results_df[
    (results_df['classic_valid'] == True) & (results_df['qubo_valid'] == True)
].copy()
success_rows['speed_ratio'] = success_rows['qubo_time_avg'] / success_rows['classic_time']

# --- Timing vs clues plot ---
plt.figure(figsize=(10,6))
plt.scatter(success_rows['num_clues'], success_rows['classic_time'], color='blue', alpha=0.7, label='Classic')
plt.scatter(success_rows['num_clues'], success_rows['qubo_time_avg'], color='red', alpha=0.7, label='QUBO (avg/read)')
plt.xlabel("Number of Clues")
plt.ylabel("Average Solve Time (s)")
plt.title("Classical vs QUBO Solve Time vs Number of Clues (Successful Runs)")
plt.legend()
plt.grid(alpha=0.4)
timing_plot = f"results/crossover_plot_{timestamp}.png"
plt.savefig(timing_plot)
plt.show()

# --- Crossover ratio plot ---
plt.figure(figsize=(10,5))
plt.plot(success_rows['num_clues'], success_rows['speed_ratio'], 'o-', color='purple')
plt.axhline(y=1.0, color='gray', linestyle='--', label='Crossover = 1.0')
plt.xlabel("Number of Clues")
plt.ylabel("QUBO Time / Classic Time")
plt.title("Performance Crossover Ratio (Successful Runs)")
plt.legend()
plt.grid(alpha=0.4)
crossover_ratio_plot = f"results/crossover_ratio_{timestamp}.png"
plt.savefig(crossover_ratio_plot)
plt.show()

# --- Energy distributions (first few puzzles) ---
if SAVE_ENERGIES:
    for i, row in enumerate(results_df.head(5).to_dict('records')):
        if not row["energies"]:
            continue
        plt.figure(figsize=(8,4))
        plt.bar(range(len(row["energies"])), row["occurrences"], color='orange', alpha=0.7)
        plt.title(f"Energy Distribution — Puzzle {row['puzzle_index']} ({row['num_clues']} clues)")
        plt.xlabel("Sample Index")
        plt.ylabel("Occurrences")
        plt.grid(alpha=0.4)
        energy_plot_path = f"results/energy_distribution_{row['puzzle_index']}_{timestamp}.png"
        plt.savefig(energy_plot_path)
        plt.show()

# --- Success/Failure summary by clue count ---
summary_data = []
for clues, group in results_df.groupby('num_clues'):
    total = len(group)
    classic_success = np.sum(group['classic_valid'])
    qubo_success = np.sum(group['qubo_valid'])
    summary_data.append({
        "num_clues": clues,
        "total_puzzles": total,
        "classic_success_rate": classic_success / total * 100,
        "qubo_success_rate": qubo_success / total * 100,
        "classic_fail_rate": 100 - (classic_success / total * 100),
        "qubo_fail_rate": 100 - (qubo_success / total * 100)
    })

summary_df = pd.DataFrame(summary_data)
summary_csv = f"results/success_failure_summary_{timestamp}.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"✅ Saved success/failure summary → {summary_csv}")

# --- Success rate plot ---
plt.figure(figsize=(10,6))
plt.plot(summary_df['num_clues'], summary_df['classic_success_rate'], 'o-', color='blue', label='Classic Success %')
plt.plot(summary_df['num_clues'], summary_df['qubo_success_rate'], 'o-', color='red', label='QUBO Success %')
plt.xlabel("Number of Clues")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate vs Number of Clues (All Runs)")
plt.legend()
plt.grid(alpha=0.4)
success_plot = f"results/success_rate_{timestamp}.png"
plt.savefig(success_plot)
plt.show()

# --- Failure rate plot ---
plt.figure(figsize=(10,6))
plt.plot(summary_df['num_clues'], summary_df['classic_fail_rate'], 'o-', color='blue', linestyle='--', label='Classic Fail %')
plt.plot(summary_df['num_clues'], summary_df['qubo_fail_rate'], 'o-', color='red', linestyle='--', label='QUBO Fail %')
plt.xlabel("Number of Clues")
plt.ylabel("Failure Rate (%)")
plt.title("Failure Rate vs Number of Clues (All Runs)")
plt.legend()
plt.grid(alpha=0.4)
failure_plot = f"results/failure_rate_{timestamp}.png"
plt.savefig(failure_plot)
plt.show()

print("\n📊 All analyses complete! Plots and summaries saved in 'results' folder.")
