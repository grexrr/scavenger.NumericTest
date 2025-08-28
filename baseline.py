import random
import csv
import matplotlib.pyplot as plt
from elo_calculator_demo import testPlayer, testLandmark, testEloCalculator

# ===============================
# Setting
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
OUTPUT_CSV = "data/baseline.csv"
OUTPUT_FIG = "figure/baseline.png"

# ===============================
# INIT PLAYERs & LMs
# ===============================
players = {name: testPlayer(name, rating=0.0, lastPlay=None, uncertainty=0.5) for name in test_players}
landmarks = {lm: testLandmark(lm, rating=0.0, lastAnswered=None, uncertainty=0.5) for lm in testround_landmark_pool}
calc = testEloCalculator(players, landmarks)

# ===============================
# SIMULATION
# ===============================
def simulate_performance(player_name):
    if player_name == "Novice":
        correct_prob = 0.4
        mean_time = 0.9 * TIME_LIMIT
    elif player_name == "Average":
        correct_prob = 0.65
        mean_time = 0.7 * TIME_LIMIT
    else:  # Expert
        correct_prob = 0.9
        mean_time = 0.5 * TIME_LIMIT

    correctness = random.random() < correct_prob

    # Time（Gaussian Distribution）
    minutes_used = max(0.1, random.gauss(mean_time, 0.1 * TIME_LIMIT))
    return minutes_used, correctness

# ===============================
# MAIN
# ===============================
history = []

for round_idx in range(1, EPOCHS + 1):
    
    player_name = random.choice(test_players)
    landmark_name = random.choice(testround_landmark_pool)

    player = players[player_name]
    landmark = landmarks[landmark_name]

    minutes_used, correct = simulate_performance(player_name)

    calc.calculateElo(
        player, landmark,
        minutes_used=minutes_used,
        time_limit_minutes=TIME_LIMIT,
        correct=correct,
        test_mode=True
    )

    row = {"round": round_idx}
    for pn, pobj in players.items():
        row[f"player_{pn}"] = pobj.rating
    for lm, lobj in landmarks.items():
        row[f"landmark_{lm}"] = lobj.rating
    history.append(row)

# ===============================
# output
# ===============================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

print(f"[✓] Results saved to {OUTPUT_CSV}")

# ===============================
# VISUALIZATION
# ===============================
rounds = [row["round"] for row in history]

plt.figure(figsize=(12, 6))
# player
for pn in test_players:
    ratings = [row[f"player_{pn}"] for row in history]
    plt.plot(rounds, ratings, label=f"Player: {pn}", linewidth=2)
# landmarks
for lm in testround_landmark_pool:
    ratings = [row[f"landmark_{lm}"] for row in history]
    plt.plot(rounds, ratings, linestyle="--", alpha=0.7, label=f"Landmark: {lm}")

plt.xlabel("Round")
plt.ylabel("Rating")
plt.title(f"Baseline Simulation (EPOCHS={EPOCHS}, K=0.0075, U=0.5, Time Limit=10min)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"[✓] Plot saved to {OUTPUT_FIG}")
