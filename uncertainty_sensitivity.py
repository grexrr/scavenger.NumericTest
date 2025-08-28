# uncertainty_sensitivity.py
import random
import csv
import matplotlib.pyplot as plt
from elo_calculator_demo import testPlayer, testLandmark, testEloCalculator

# ===============================
# SETTINGS
# ===============================
testround_landmark_pool = [
    "Glucksman Gallery",
    "Honan Collegiate Chapel",
    "Boole Library",
    "The Quad / Aula Maxima",
    "Brookfield Health Sciences Complex",
    "Student Centre",
    "Geography Building"
]
test_players = ["Novice", "Average", "Expert"]

EPOCHS = 200           
TIME_LIMIT = 10        
UNCERTAINTY_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
GLOBAL_SEED = 20250819 
PLOT_STYLE = {
    "player_linewidth": 2.0,
    "landmark_alpha": 0.7,
    "landmark_linestyle": "--"
}

# ===============================
# Unified Schedule Generation (Fixed Random Seed)
# ===============================
def simulate_performance_profile(player_name):
    """Player profile consistent with baseline: accuracy + mean time used"""
    if player_name == "Novice":
        correct_prob = 0.4
        mean_time = 0.9 * TIME_LIMIT
    elif player_name == "Average":
        correct_prob = 0.65
        mean_time = 0.7 * TIME_LIMIT
    else:  # Expert
        correct_prob = 0.9
        mean_time = 0.5 * TIME_LIMIT
    return correct_prob, mean_time

def sample_minutes(mean_time, std_ratio=0.1):
    # Gaussian distribution sampling and avoid negative values
    return max(0.1, random.gauss(mean_time, std_ratio * TIME_LIMIT))

def build_schedule():
    """Pre-generate N rounds: [ (player_name, landmark_name, minutes_used, correct), ... ]"""
    random.seed(GLOBAL_SEED)
    schedule = []
    for _ in range(EPOCHS):
        p = random.choice(test_players)
        lm = random.choice(testround_landmark_pool)
        correct_prob, mean_time = simulate_performance_profile(p)
        minutes_used = sample_minutes(mean_time, std_ratio=0.1)
        correct = random.random() < correct_prob
        schedule.append((p, lm, minutes_used, correct))
    return schedule

# ===============================
# Single U Experiment
# ===============================
def run_single_experiment(init_uncertainty, schedule, csv_path):
    """
    Use the same schedule, only change the initial U, keep test_mode=True (do not dynamically update U over time).
    Output the ratings of all players and landmarks at the end of each round to CSV.
    """
    # Initialization
    players = {name: testPlayer(name, rating=0.0, lastPlay=None, uncertainty=init_uncertainty)
               for name in test_players}
    landmarks = {lm: testLandmark(lm, rating=0.0, lastAnswered=None, uncertainty=init_uncertainty)
                 for lm in testround_landmark_pool}
    calc = testEloCalculator(players, landmarks)

    history = []
    round_idx = 0
    for (p_name, lm_name, minutes_used, correct) in schedule:
        round_idx += 1
        p = players[p_name]
        lm = landmarks[lm_name]

        # Core: test_mode=True => Do not modify uncertainty, only compare the effect of "initial U" on convergence speed
        calc.calculateElo(
            p, lm,
            minutes_used=minutes_used,
            time_limit_minutes=TIME_LIMIT,
            correct=correct,
            test_mode=True
        )

        row = {"round": round_idx}
        for pn, pobj in players.items():
            row[f"player_{pn}"] = pobj.rating
        for lmn, lobj in landmarks.items():
            row[f"landmark_{lmn}"] = lobj.rating
        history.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    return history  # For plotting/comparison

# ===============================
# Main Process: Run Multiple U and Plot
# ===============================
if __name__ == "__main__":
    # 1) Unified Schedule
    schedule = build_schedule()

    # 2) Run all U, collect results
    results_by_U = {}  # U -> history(list of dict)
    for U0 in UNCERTAINTY_LEVELS:
        csv_name = f"data/uncertainty_U={U0}.csv"
        print(f"[•] Running U={U0} -> {csv_name}")
        history = run_single_experiment(U0, schedule, csv_name)
        results_by_U[U0] = history
    print("[✓] All CSV files written.")

    # 3) Plot (Player Perspective): One subplot per U, three curves for three players
    fig_rows = len(UNCERTAINTY_LEVELS)
    fig, axs = plt.subplots(fig_rows, 1, figsize=(12, 3.5 * fig_rows), sharex=True)
    if fig_rows == 1:
        axs = [axs]  # Compatible with single row case

    for idx, U0 in enumerate(UNCERTAINTY_LEVELS):
        ax = axs[idx]
        hist = results_by_U[U0]
        rounds = [row["round"] for row in hist]

        for pn in test_players:
            ratings = [row[f"player_{pn}"] for row in hist]
            ax.plot(rounds, ratings, label=f"{pn}", linewidth=PLOT_STYLE["player_linewidth"])

        ax.set_title(f"Players — U={U0}")
        ax.set_ylabel("Rating")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    axs[-1].set_xlabel("Round")
    plt.suptitle(f"Uncertainty Sensitivity (Players) — EPOCHS={EPOCHS}, TimeLimit={TIME_LIMIT}min", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figure/players_by_U.png", dpi=300)
    plt.show()
    print("[✓] Plot saved to players_by_U.png")

    # 4) Plot (Landmark Perspective): One subplot per U, all landmarks in dashed semi-transparent lines
    fig, axs = plt.subplots(fig_rows, 1, figsize=(12, 3.5 * fig_rows), sharex=True)
    if fig_rows == 1:
        axs = [axs]

    for idx, U0 in enumerate(UNCERTAINTY_LEVELS):
        ax = axs[idx]
        hist = results_by_U[U0]
        rounds = [row["round"] for row in hist]

        for lm in testround_landmark_pool:
            ratings = [row[f"landmark_{lm}"] for row in hist]
            ax.plot(
                rounds, ratings,
                linestyle=PLOT_STYLE["landmark_linestyle"],
                alpha=PLOT_STYLE["landmark_alpha"],
                label=f"{lm}"
            )

        ax.set_title(f"Landmarks — U={U0}")
        ax.set_ylabel("Rating")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, ncol=2, loc="best")

    axs[-1].set_xlabel("Round")
    plt.suptitle(f"Uncertainty Sensitivity (Landmarks) — EPOCHS={EPOCHS}, TimeLimit={TIME_LIMIT}min", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figure/landmarks_by_U.png", dpi=300)
    plt.show()
    print("[✓] Plot saved to landmarks_by_U.png")
