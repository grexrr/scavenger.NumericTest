import math

class testPlayer():
    def __init__(self, playerId, rating, lastPlay, uncertainty=0.5) -> None:
        self.playerId = playerId
        self.rating = rating # placeholder, not sure
        self.uncertainty = uncertainty
        self.lastPlayed = lastPlay

class testLandmark():
    def __init__(self, landmarkId, rating, lastAnswered, uncertainty=0.5) -> None:
        self.landmarkId = landmarkId
        self.rating = rating # placeholder, not sure
        self.uncertainty = uncertainty
        self.lastAnswered = lastAnswered


class testEloCalculator():
    def __init__(self, players, landmarks) -> None:
        self.players = players
        self.landmarks = landmarks

    def calculateElo(self, player:testPlayer, landmark:testLandmark, minutes_used, time_limit_minutes=20, correct=True, test_mode=True):
        """summary_line
        
        q^_player = q_player + K_player(S_hshs - E(S_hshs))
        q^_riddle = q_riddle + K_riddle(E(S_hshs) - S_hshs)
        
        - If player performs better than expected (S_hshs > E(S_hshs)), player rating increases, landmark rating decreases
        - If player performs worse than expected (S_hshs < E(S_hshs)), player rating decreases, landmark rating increases
        - K factors control how much ratings change: higher uncertainty = bigger rating swings
        - This creates a zero-sum game where total rating points are conserved between player and landmark

        Keyword arguments:
        argument: description
        Return: return_description
        """
        
    
        K_player, K_riddle = self._dynamicK(player, landmark, test_mode)
        hshs, expectation = self._hshs(minutes_used *  60, player.rating, landmark.rating, time_limit_minutes * 60, correct)

        player.rating += K_player * (hshs - expectation)
        landmark.rating -= K_riddle * (hshs - expectation)

        return

    def _dynamicK(self, player: testPlayer, landmark:testLandmark, test_mode=True):
        """
        Glickman's K factor in CAP system calculates the rating uncertainty U of the player and the item 

        K_player = K(1 + K_+ * U_player - K_- * U_riddle)
        K_riddle = K(1 + K_+ * U_riddle - K_- * U_player)
        
        - K factors determine how much ratings change after each game
        - Higher uncertainty (U) means we're less confident about current ratings, so we allow bigger changes
        - K_+ amplifies uncertainty effects: more uncertain players/landmarks get bigger rating swings
        - K_- dampens uncertainty effects: when opponent is uncertain, we're more conservative with our own changes
        - This creates adaptive learning rates that slow down as confidence increases
        """
        
        # K, K_plus, K_minus are constants used in the dynamic K-factor calculation
        # K is the base learning rate, set to 0.0075, which determines the overall sensitivity of rating changes.
        # K_plus is set to 4, amplifying the effect of uncertainty on rating changes.
        # K_minus is set to 0.5, dampening the effect of the opponent's uncertainty on rating changes.
        K, K_plus, K_minus = 0.0075, 4, 0.5

        player.uncertainty = self._updateUncertainty(player.uncertainty, test_mode)
        landmark.uncertainty = self._updateUncertainty(landmark.uncertainty, test_mode)

        K_player = K * (1 + K_plus * player.uncertainty - K_minus * landmark.uncertainty)
        K_riddle = K * (1 + K_plus * landmark.uncertainty - K_minus * player.uncertainty)
        
        return K_player, K_riddle
    
    def _updateUncertainty(self, current_U, day_since_last_play=30, test_mode=False):
        """
        
        Update uncertainty based on last interaction time and total sessions
        
        U = U - 1/40 + 1/30 * D
        
        - Uncertainty naturally decreases over time (U - 1/40) as we become more confident
        - But long periods of inactivity (D days) increase uncertainty (1/30 * D)
        - Uncertainty is clamped between 0 (completely certain) and 1 (completely uncertain)
            
        Keyword arguments:
            current_U (float): Current uncertainty lev [0, 1]
            day_since_last_play (int): inactivity time 

        Return: 
            update uncertainty (float)
        """
        if test_mode:
            return current_U
        else:
            U_new = current_U - (1/40.0) + (1/30.0) * day_since_last_play
            return max(0, min(1, U_new))
        

    def _hshs(self, seconds_used, player_rating, landmark_rating, time_limit, correct):
        """
        Comupute High-Speed High-Stakes(HSHS) score and Expectation .

        HSHS: s_ij = (2 * x_ij - 1) * (a_i * d_i - a_i * t_ij)
        
        Intuitive understanding of HSHS:
        - Score rewards both correctness AND speed
        - (2 * x_ij - 1) converts correct=1/incorrect=0 to +1/-1
        - (a_i * d_i - a_i * t_ij) measures time efficiency: faster = higher score
        - Discrimination factor a_i makes harder puzzles (shorter time limits) worth more points
        - This creates a "speed-accuracy tradeoff" where players must balance both factors

        E(HSHS): a_i * d_i * (
            (e^ (2 * a_i * d_i) * (theta_j - beta_i) + 1)/
            (e^ (2 * a_i * d_i) * (theta_j - beta_i) - 1)
            ) - (1 / (theta_j - beta_i))
        
        Intuitive understanding of Expectation:
        - Expected score based on skill difference (theta_j - beta_i)
        - When player skill >> landmark difficulty, expectation approaches maximum possible score
        - When landmark difficulty >> player skill, expectation approaches minimum possible score
        - The exponential term creates a smooth transition between these extremes
        - This expectation serves as the "baseline" for determining if performance was above/below average
        
        Keyword arguments:
            time_limit(float): Max allowed seconds for the riddle(d_i)
            time_used(float): Actual response seconds(t_ij)
            correct(bool): Whether the answer is correct (x_ij) = 1 if correct else 0
            player_rating (float): Player's skill level (theta_j)
            landmark_rating (float): Landmark's difficulty level (beta_i)

        Return: 
            hshs (float) 
        """

        if seconds_used > time_limit:
            correct = False
            seconds_used = time_limit 

        ### HSHS
        # discrimination a_i = 1 / d_i (d_i is time_limit)
        discrimination = self._discrimination(time_limit)
        
        # calculate (a_i * d_i - a_i * t_ij) = a_i * (d_i - t_ij)
        time_component = discrimination * (time_limit - seconds_used)
        
        # apply(2 * x_ij - 1) factor
        hshs = time_component if correct else -time_component

        ### Expectation
        delta = player_rating - landmark_rating
        if abs(delta) < 1e-6:
            delta = 1e-6

        weight_ability_difference = 2 * discrimination * time_limit * (delta)
        exp_term = math.exp(weight_ability_difference)
        expectation = discrimination * time_limit * ((exp_term + 1) / (exp_term - 1)) - (1 / delta)

        return hshs, expectation
    
    def _discrimination(self, time_limit, mode="default"):   
        
        """
        Allowing mode="default" and fallback 1/10, expandable for future work like "different judgement rate for different riddle types"
        
        a_i = 1 / d_i (default mode)
        
        Intuitive understanding:
        - Discrimination factor determines how much "weight" a puzzle carries in rating calculations
        - Shorter time limits (harder puzzles) have higher discrimination = more rating impact
        - Longer time limits (easier puzzles) have lower discrimination = less rating impact
        - This makes sense because solving a hard puzzle quickly is more impressive than solving an easy one slowly
        - The inverse relationship (1/time_limit) creates a natural difficulty scaling
        """
        
        return 1.0/time_limit if mode == "default" else 1/10
        

# class testSession():
#     def __init__(self, player, landmark, minutes_used, minutes_limit = 20, is_correct=True) -> None:
#         self.player = player
#         self.landmark = landmark
#         self.time_limit_seconds = minutes_limit * 60
#         self.seconds_used = minutes_used * 60
#         self.correct = is_correct


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    ### TEST 1 BENCHMARK
    ITERATIONS = 1000
    TIME_LIMIT_MINUTES = 30 # minutes   
    BENCHMARK_UNCERTAINTY = 0.5 # Glicko / CAP model uncertainty benchmark values
    TIME_USED_CASES    = [0.0, 5, 10, 15, 20, 25, 30]
    CORRECT = True
    RATING_CAP = 3

    results = []

    # ---- TESTING ----
    for minutes_used in TIME_USED_CASES:
        player   = testPlayer("P1", rating=-RATING_CAP, lastPlay=None, uncertainty=BENCHMARK_UNCERTAINTY)
        landmark = testLandmark("L1", rating= RATING_CAP, lastAnswered=None, uncertainty=BENCHMARK_UNCERTAINTY)
        calc     = testEloCalculator([player], [landmark])

        p_hist, l_hist = [], []
        for _ in range(ITERATIONS):
            calc.calculateElo(player, landmark,
                            minutes_used=minutes_used,
                            time_limit_minutes=TIME_LIMIT_MINUTES,
                            correct=CORRECT,
                            test_mode=True)
            p_hist.append(player.rating)
            l_hist.append(landmark.rating)
        results.append((p_hist, l_hist, minutes_used))

    # ---- PLOT ----
    fig, axs = plt.subplots(len(TIME_USED_CASES), 1, figsize=(10, 24), sharex=True)

    for idx, (p_hist, l_hist, minutes_used) in enumerate(results):
        x_data = list(range(len(p_hist)))
        axs[idx].plot(x_data, p_hist, label="Player", color="royalblue", linewidth=2)
        axs[idx].plot(x_data, l_hist, label="Puzzle", color="darkorange", linewidth=2)
        axs[idx].set_title(f"minutes_used = {minutes_used} min", fontsize=13)
        axs[idx].set_ylabel("Internal rating (±2)")
        axs[idx].grid(True, linestyle='--', alpha=0.5)
        axs[idx].legend(loc="lower right")
        axs[idx].tick_params(axis='x', which='both', labelbottom=True)

    axs[-1].set_xlabel("Iteration")
    plt.suptitle("Convergence under Different Time Used", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig("Convergence under Different Time Used.png", dpi=300, bbox_inches='tight')
    plt.show()

    ## TEST 2 UNCERTAINTY

    ITERATIONS = 1000
    TIME_LIMIT_MINUTES = 30 # minutes   
    BENCHMARK_UNCERTAINTY = [0.1, 0.3, 0.5, 0.7, 0.9] # Glicko / CAP model uncertainty benchmark values
    TIME_USED_CASES = 10
    CORRECT = True
    RATING_CAP = 3

    results = []

    for u in BENCHMARK_UNCERTAINTY:
        player   = testPlayer("P1", rating=-RATING_CAP, lastPlay=None, uncertainty=u)
        landmark = testLandmark("L1", rating= RATING_CAP, lastAnswered=None, uncertainty=u)
        calc     = testEloCalculator([player], [landmark])  

        p_hist, l_hist = [], []
        for _ in range(ITERATIONS):
            calc.calculateElo(player, landmark,
                            minutes_used=TIME_USED_CASES,
                            time_limit_minutes=TIME_LIMIT_MINUTES,
                            correct=CORRECT,
                            test_mode=False)  # update uncertainty
            p_hist.append(player.rating)
            l_hist.append(landmark.rating)
        results.append((p_hist, l_hist, TIME_USED_CASES))

    # ---- PLOT ----
    plt.figure(figsize=(10, 6))

    for idx, (p_hist, _, _) in enumerate(results):
        u_value = BENCHMARK_UNCERTAINTY[idx]
        plt.plot(p_hist, label=f"U = {u_value}", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Player Rating (internal scale ±3)")
    plt.title(f"Effect of Uncertainty on Rating Convergence ({TIME_USED_CASES} min, last_play=30days)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    plt.savefig(f"Effect of Uncertainty on Rating Convergence ({TIME_USED_CASES}) min.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    