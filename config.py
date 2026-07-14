"""Central configuration for the RL-guided simplex solver.

Every knob for a training/evaluation run lives here as a module-level constant
(game mode, LP size, PPO hyperparameters, the observation/reward feature flags,
the pivot-rule action maps, and the per-rule step-penalty weight tables). Other
modules read these directly via ``from config import ...``, so changing a run's
configuration means editing this file. A few values are derived at import time
(e.g. ``USE_COMPACT_OBS`` is forced True in Leduc mode, and
``STEP_PENALTY_WEIGHTS`` / ``MODEL_RUN_TAG`` are computed from the flags above).
"""

# Game mode: "matrix" or "leduc"
GAME_MODE = "leduc"
TIMESTEPS = 30_000_000

# Checkpoint settings: save model every CHECKPOINT_FREQ steps after CHECKPOINT_START
CHECKPOINT_START = 1_000_000
CHECKPOINT_FREQ = 1_000_000
LOAD_MODEL = False


# Leduc poker settings (only used when GAME_MODE = "leduc")
LEDUC_GAME = "leduc_poker(suit_isomorphism=true)"  # OpenSpiel game string
LEDUC_ALPHA = 100.0      # Dirichlet concentration: high alpha -> small perturbation around uniform
LEDUC_NUM_RANKS = 3     # Number of card ranks (J, Q, K)

# Normal form games  settings (used when GAME_MODE = "matrix")
# Training parameters
M = 40
N = 40

# Matrix-mode settings (uniform random ints in [MIN_VAL, MAX_VAL])
MIN_VAL = -1
MAX_VAL = 1

# EPSILON: perturbation magnitude applied to the base matrix each episode
EPSILON = 0.001

# Perturbation magnitude used by experiment.py when generating in-distribution
# test matrices. Defaults to EPSILON so the test set matches the training
# distribution. Set to a different value to probe how the agent generalizes
# to larger or smaller perturbations than it was trained on (the saved model's
# filename / training epsilon is unaffected).
TEST_EPSILON = 0.001

# Training settings
N_ENVS = 4

# (MODEL_NAME_TEMPLATE is defined further down, after USE_COMPACT_OBS and
# USE_WEIGHTED_STEP_PENALTY are set, so it can include obs/penalty tags
# automatically.)

# Action -> pivot-rule map for TRAINING: the agent's action space is exactly
# these 3 rules. PIVOT_MAP_TEST below adds random_edge and blands_rule, which
# are evaluation-only baselines the agent cannot select.
PIVOT_MAP = {
    0: 'largest_coefficient',
    1: 'largest_increase',
    2: 'steepest_edge',
    # 3: 'random_edge'
    # 3: 'blands_rule'
}

PIVOT_MAP_TEST = {
    0: 'largest_coefficient',
    1: 'largest_increase',
    2: 'steepest_edge',
    3: 'random_edge',
    4: 'blands_rule'
}

NUM_PIVOT_STRATEGIES = len(PIVOT_MAP)
NUM_PIVOT_STRATEGIES_TEST = len(PIVOT_MAP_TEST)
PIVOT_STRATEGY_NAMES = list(PIVOT_MAP_TEST.values())

# History tracking feature
USE_COMPACT_OBS = False  # Wrap env with CompactObsWrapper (31 size-independent features)
# Leduc tableau is ~(483, 965); a Dict-obs policy would have ~120M params just
# in the input layer and a 500MB+ saved model. Force compact obs for Leduc.
if GAME_MODE == "leduc":
    USE_COMPACT_OBS = True
# Sanity-check baseline: replace the observation with a single constant feature
# (information-free). The policy is forced to be state-independent and can only
# learn one fixed pivot rule. If this matches the compact-obs agent, the agent
# wasn't using the tableau. Takes precedence over USE_COMPACT_OBS when True.
USE_EMPTY_OBS = False
ENT_COEF = 0.05  # PPO entropy coefficient (higher = more exploration / action diversity)
# PPO discount factor. Effective horizon ~ 1/(1-GAMMA). Leduc phase-2 episodes
# run ~280-480 pivots, so 0.999 (horizon ~1000) is used so the terminal success
# bonus / total-pivot-count signal propagates across the whole episode (an
# earlier 0.995 setting had horizon ~200, shorter than the episode).
GAMMA = 0.999

# Two-phase simplex: when True, build Phase 1 tableau and solve it (Bland's rule),
# then transition to Phase 2 for the RL agent. When False, construct
# the Phase 2 tableau directly (skipping Phase 1).
# NOTE: the two modes produce different tableau shapes, so models are NOT interchangeable.
USE_TWO_PHASE = False

# Weighted step penalty: scale the per-step penalty by the empirical cost of
# the chosen pivot rule (calibrate via benchmark_pivot_cost.py /
# benchmark_pivot_cost_leduc.py). When False, every step costs -1 regardless
# of strategy.
USE_WEIGHTED_STEP_PENALTY = True

# Per-rule wallclock weights, normalized so largest_coefficient = 1.0.
# Re-run the matching benchmark if you change PIVOT_MAP or the simplex
# internals.
#
# Calibrated for TOTAL per-pivot wallclock on a 40x40 matrix-mode phase-2
# tableau (~42x83): col-select + row-select + apply-pivot. At this size
# apply-pivot is only ~10us (the tableau fits in cache), so col-select is the
# dominant cost — the old col-only ratios (1 / 1.5 / 2.5 / 0.8 / 0.35) were
# close but slightly understated SE and LI, and Bland's was way too cheap
# (the old 0.35 ignored the fixed row+apply costs every rule pays).
STEP_PENALTY_WEIGHTS_MATRIX = {
    'largest_coefficient': 1.00,
    'steepest_edge':       1.89,
    'largest_increase':    3.17,
    'random_edge':         0.90,
    'blands_rule':         0.66,
}
# Calibrated for TOTAL per-pivot wallclock on the Leduc phase-2 tableau
# (~483x965): col-select + row-select + apply-pivot. Apply-pivot is a
# ~234us memory-bandwidth-bound op that every rule pays equally, so the
# realistic wallclock ratios across rules are far gentler than the
# col-select-only ratios (the previous calibration: 1 / 9.5 / 20 / 0.8 / 0.4).
# Re-measure if the simplex inner loop or the LP size changes meaningfully.
STEP_PENALTY_WEIGHTS_LEDUC = {
    'largest_coefficient': 1.00,
    'steepest_edge':       1.85,
    'largest_increase':    2.95,
    'random_edge':         0.98,
    'blands_rule':         0.95,
}
# Active dict — auto-picked from GAME_MODE so envs.py / experiment scripts
# don't need to know the mode.
STEP_PENALTY_WEIGHTS = (STEP_PENALTY_WEIGHTS_LEDUC if GAME_MODE == "leduc"
                       else STEP_PENALTY_WEIGHTS_MATRIX)

# Run tag derived from observation-space and step-penalty flags so each
# training configuration writes to a unique model filename. Examples:
#   USE_COMPACT_OBS=False, USE_WEIGHTED_STEP_PENALTY=True  -> "dict_weighted"
#   USE_COMPACT_OBS=True,  USE_WEIGHTED_STEP_PENALTY=False -> "compact_unweighted"
_OBS_TAG = "empty" if USE_EMPTY_OBS else ("compact" if USE_COMPACT_OBS else "dict")
_PEN_TAG = "weighted" if USE_WEIGHTED_STEP_PENALTY else "unweighted"
MODEL_RUN_TAG = f"{_OBS_TAG}_{_PEN_TAG}"

# Model save path template — uses MODEL_RUN_TAG so different obs/penalty
# combinations don't overwrite each other's saved models.
MODEL_NAME_TEMPLATE = (
    f"models/ppo_simplex_random_{{steps}}_matrix{{m}}x{{n}}"
    f"_min{{min}}_max{{max}}_epsilon{{eps}}_{MODEL_RUN_TAG}.zip"
)
