# single_player_timing_test.py
import random
import csv
import math
import matplotlib.pyplot as plt
from elo_calculator_demo import testPlayer, testLandmark, testEloCalculator

# ---------------- Settings ----------------
TIME_LIMIT = 10        # minutes
EPOCHS = 200
GLOBAL_SEED = 20250826

# Nominal correctness rate (only effective if not timed out)
P_CORRECT = 0.65

# Time distribution: given mean percentage and fluctuation, ensure some will exceed the time limit
MEAN_PCT = 0.75        # Average time used is about 0.75 * TIME_LIMIT
STD_RATIO = 0.20       # Standard deviation ratio (the larger, the more likely to exceed 10 minutes)

testround_landmark_pool = [
    "Glucksman Gallery",
    "Honan Collegiate Chapel",
    "Boole Library",
    "The Quad / Aula Maxima",
    "Brookfield Health Sciences Complex",
    "Student Centre",
    "Geography Building"
]

# ---------------- Helpers ----------------
def sample_minutes(time_limit, mean_pct=MEAN_PCT, std_ratio=STD_RATIO):
    mean = mean_pct * time_limit
    std  = std_ratio * time_limit

    m = max(0.1, random.gauss(mean, std))
    return m

def decide_correct(minutes_used, time_limit, p_correct=P_CORRECT):
    if minutes_used > time_limit:
        return False, "timeout"
    return (random.random() < p_correct), "in_time"

# ---------------- Run ----------------
if __name__ == "__main__":
    random.seed(GLOBAL_SEED)

    
    player = testPlayer("TestPlayer", rating=0.0, lastPlay=None, uncertainty=0.5)
    landmarks = {
        lm: testLandmark(lm, rating=0.0, lastAnswered=None, uncertainty=0.5)
        for lm in testround_landmark_pool
    }
    calc = testEloCalculator({"P": player}, landmarks)

    history = []
    player_ratings = []
    correctness_marks = []  # ('correct' or 'wrong'), also tag 'timeout'/'in_time'

    for r in range(1, EPOCHS + 1):
        lm_name = random.choice(testround_landmark_pool)
        lm = landmarks[lm_name]

        minutes_used = sample_minutes(TIME_LIMIT)
        is_correct, timing_tag = decide_correct(minutes_used, TIME_LIMIT)

        calc.calculateElo(
            player, lm,
            minutes_used=minutes_used,
            time_limit_minutes=TIME_LIMIT,
            correct=is_correct,
            test_mode=True  # Fix U, does not change over time
        )

        player_ratings.append(player.rating)
        correctness_marks.append(("correct" if is_correct else "wrong", timing_tag))

        row = {
            "round": r,
            "minutes_used": minutes_used,
            "timing_tag": timing_tag,     # "in_time"/"timeout"
            "is_correct": int(is_correct),
            "player_rating": player.rating,
            "landmark": lm_name
        }
        
        for name, lobj in landmarks.items():
            row[f"landmark_{name}"] = lobj.rating
        history.append(row)

    # ---------------- CSV ----------------
    with open("single_player_timing.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print("[✓] Results saved to single_player_timing.csv")

    # ---------------- Plot ----------------
    rounds = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, player_ratings, linewidth=2, label="Player Rating")

   
    for i, (flag, timing) in enumerate(correctness_marks):
        x = rounds[i]; y = player_ratings[i]
        if flag == "correct":
            plt.scatter([x], [y], marker="o", color="green")
        else:
            if timing == "timeout":
                plt.scatter([x], [y], marker="+", color="red")
            else:
                plt.scatter([x], [y], marker="x", color="orange")

    
    plt.scatter([], [], marker="o", color="green", label="Correct")
    plt.scatter([], [], marker="x", color="orange", label="Wrong (in time)")
    plt.scatter([], [], marker="+", color="red", label="Wrong (timeout)")

    plt.xlabel("Round")
    plt.ylabel("Player Rating")
    plt.title("Single-Player Timing Test (p=0.65, U=0.5, TL=10min)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("single_player_timing.png", dpi=300)
    plt.show()

