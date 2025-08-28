# time_limit_array.py  (absolute-time model + timeout stats)
import random, csv
import matplotlib.pyplot as plt
from elo_calculator_demo import testPlayer, testLandmark, testEloCalculator

# ----- Settings -----
TIME_LIMITS = [5, 10, 15, 20, 25, 30]   # minutes
EPOCHS = 200
GLOBAL_SEED = 20250826

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
PROFILE = {
    "Novice":  {"p_correct": 0.40, "mean_min": 12.0, "std_min": 3.0},
    "Average": {"p_correct": 0.65, "mean_min": 9.0,  "std_min": 2.0},
    "Expert":  {"p_correct": 0.90, "mean_min": 6.0,  "std_min": 1.5},
}


def build_schedule():
    random.seed(GLOBAL_SEED)
    sched = []
    for _ in range(EPOCHS):
        p = random.choice(test_players)
        lm = random.choice(testround_landmark_pool)
        z = random.gauss(0, 1)         
        u = random.random()            
        sched.append((p, lm, z, u))
    return sched

def draw_perf_absolute(player, tl, z, u):
    prof = PROFILE[player]
    minutes = max(0.1, prof["mean_min"] + z * prof["std_min"])  
    if minutes > tl:
        correct = False        
        tag = "timeout"
    else:
        correct = (u < prof["p_correct"])
        tag = "in_time"
    return minutes, correct, tag

# ----- Single Time Limit -----
def run_time_limit(tl, schedule, csv_path):
    players = {n: testPlayer(n, 0.0, None, 0.5) for n in test_players}
    landmarks = {lm: testLandmark(lm, 0.0, None, 0.5) for lm in testround_landmark_pool}
    calc = testEloCalculator(players, landmarks)
    hist = []

    # Statistics: Count the number of timeouts for each player under this TL
    timeout_counter = {n: 0 for n in test_players}
    visit_counter   = {n: 0 for n in test_players}

    for r, (p, lm, z, u) in enumerate(schedule, 1):
        m, c, tag = draw_perf_absolute(p, tl, z, u)
        calc.calculateElo(players[p], landmarks[lm], m, tl, c, test_mode=True)

        visit_counter[p] += 1
        if tag == "timeout":
            timeout_counter[p] += 1

        row = {"round": r, "time_limit": tl, "player": p, "landmark": lm,
               "minutes_used": m, "timing_tag": tag, "is_correct": int(c)}
        for pn, pobj in players.items():
            row[f"player_{pn}"] = pobj.rating
        for lmn, lobj in landmarks.items():
            row[f"landmark_{lmn}"] = lobj.rating
        hist.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hist[0].keys())
        writer.writeheader(); writer.writerows(hist)

    # Calculate timeout rate
    timeout_rate = {
        pn: (timeout_counter[pn] / max(1, visit_counter[pn]))
        for pn in test_players
    }
    return hist, timeout_rate

# ----- Main -----
if __name__ == "__main__":
    schedule = build_schedule()
    results = {}          # tl -> history
    timeout_stats = {}    # tl -> {player: rate}

    for tl in TIME_LIMITS:
        csv_file = f"time_limit_{tl}min.csv"
        print(f"[•] Running time_limit={tl} min -> {csv_file}")
        hist, tr = run_time_limit(tl, schedule, csv_file)
        results[tl] = hist
        timeout_stats[tl] = tr
    print("[✓] CSV written.")

    # ===== Players plot =====
    fig, axs = plt.subplots(len(TIME_LIMITS), 1, figsize=(12, 3.5*len(TIME_LIMITS)), sharex=True)
    if len(TIME_LIMITS) == 1: axs = [axs]
    for idx, tl in enumerate(TIME_LIMITS):
        ax = axs[idx]; hist = results[tl]; rounds = [r["round"] for r in hist]
        for pn in test_players:
            ax.plot(rounds, [r[f"player_{pn}"] for r in hist], label=pn, lw=2)
        ax.set_title(f"Players — Time Limit {tl} min"); ax.grid(True, ls="--", alpha=0.5); ax.legend()
    axs[-1].set_xlabel("Round")
    plt.suptitle("Time-Limit Strategy (Players, absolute-time model, U=0.5)", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("players_by_T.png", dpi=300); plt.show()

    # ===== Landmarks plot =====
    fig, axs = plt.subplots(len(TIME_LIMITS), 1, figsize=(12, 3.5*len(TIME_LIMITS)), sharex=True)
    if len(TIME_LIMITS) == 1: axs = [axs]
    for idx, tl in enumerate(TIME_LIMITS):
        ax = axs[idx]; hist = results[tl]; rounds = [r["round"] for r in hist]
        for lm in testround_landmark_pool:
            ax.plot(rounds, [r[f"landmark_{lm}"] for r in hist], ls="--", alpha=0.7, label=lm)
        ax.set_title(f"Landmarks — Time Limit {tl} min"); ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=8, ncol=2)
    axs[-1].set_xlabel("Round")
    plt.suptitle("Time-Limit Strategy (Landmarks, absolute-time model, U=0.5)", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("landmarks_by_T.png", dpi=300); plt.show()

    # ===== Timeout rate summary =====
    with open("time_limit_stats.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["time_limit"] + [f"timeout_rate_{pn}" for pn in test_players]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tl in TIME_LIMITS:
            row = {"time_limit": tl}
            for pn in test_players:
                row[f"timeout_rate_{pn}"] = timeout_stats[tl][pn]
            writer.writerow(row)
    print("[✓] Summary saved to time_limit_stats.csv")

    import numpy as np
    x = np.arange(len(TIME_LIMITS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pn in enumerate(test_players):
        ax.bar(x + (i-1)*width, [timeout_stats[tl][pn] for tl in TIME_LIMITS], width, label=pn)
    ax.set_xticks(x); ax.set_xticklabels([f"{tl}m" for tl in TIME_LIMITS])
    ax.set_ylabel("Timeout Rate")
    ax.set_title("Player Timeout Rate vs. Time Limit (absolute-time model)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout(); plt.savefig("players_timeout_by_T.png", dpi=300); plt.show()
