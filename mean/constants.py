LOG_STD_MIN, LOG_STD_MAX = -5, 2  # std in [exp(-5)≈0.0067, exp(2)≈7.4]

# REWARD CONSTS
REWARD_SCALE = 5 # shaped_reward = reward_scale * improvement - 0.01 where improvemnt = RMSD change
STEP_COST = 0.01  # Small time penalty to encourage faster convergence
TERMINAL_CORRECT_DOCK_REWARD = 2