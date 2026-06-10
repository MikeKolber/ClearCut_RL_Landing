"""All tunable constants for the 1D rocket-landing project, in one place.

Grouped into: SIMULATION, EPISODE, REWARD, TRAINING, FILES.
Change a value here and every script picks it up.
"""

"""
take off and land
take from sharon height, distance, time, ect... and state vector
try to duplicate sharons astos trajectory
"""

# ============================== SIMULATION (physics) ==============================
DRY_MASS = 798.1          # kg, vehicle mass with no propellant
FUEL_MASS = 168.1         # kg, propellant available at the start of the landing
MAX_THRUST = 13_000.0     # N, thrust at full throttle 
# MAX_THRUST = 12_470.0
ISP = 208.1               # s, specific impulse (exhaust velocity = ISP * G0)
GRAVITY = 9.81            # m/s^2, constant
AIR_DENSITY = 1.225       # kg/m^3, constant
DRAG_AREA = 2.0           # m^2, drag coefficient * reference area, lumped together
INIT_ALTITUDE = 122.2     # m, starting height (apogee we begin the landing from)
INIT_VELOCITY = 0.0       # m/s, starting vertical velocity (0 = at apogee)
DT = 0.1                  # s, physics + decision timestep
MIN_THROTTLE = 0.6        # minimum throttle the engine can produce (action space lower bound)
INIT_THROTTLE = 0.6745    # forced throttle at dt=0 (ASTOS: 8768 N / 13000 N)
MAX_THROTTLE_RATE = 0.1  # max throttle change per step (0.05 = 5% per dt)
G0 = 9.80665              # m/s^2, standard gravity, only for Isp -> exhaust velocity

# ============================== EPISODE ==============================
MAX_STEPS = 600           # max timesteps before timeout (= 60 s at dt = 0.1)

# ============================== REWARD ==============================
# Terminal (paid once at episode end). Touchdown reward decays with impact speed
# and is never negative -- no crash "cliff" -- so reaching the ground always beats
# floating. The only real failures are timeout / out of fuel.
SAFE_LANDING_SPEED = 2.0      # m/s; touchdown at/below this counts as a "landing"
LANDING_BONUS = 200.0         # touchdown reward = LANDING_BONUS * exp(-impact / TAU)
SOFTNESS_TAU = 1.0            # m/s; decay rate of touchdown reward vs impact speed
TIMEOUT_PENALTY = -100.0      # still airborne at the time limit
OUT_OF_FUEL_PENALTY = -100.0  # ran out of fuel before reaching the ground

# Per-step (paid every step).
SHAPING_W_ALTITUDE = 0.05     # weight on |altitude| in the shaping potential
SHAPING_W_VELOCITY = 1.0      # weight on |velocity| in the shaping potential
FUEL_PENALTY = 0.0            # reward lost per kg of propellant burned
TIME_PENALTY = 0.1            # living cost per airborne step, so "land now" beats hovering

# Observation normalization.
V_SCALE = 100.0               # reference speed for normalizing velocity (~terminal velocity)

# ============================== TRAINING ==============================
PPO_TIMESTEPS = 1_000_000     # PPO budget (on-policy: needs many steps)
SAC_TIMESTEPS = 150_000       # SAC budget (off-policy + sample-efficient; updates every step)
GAMMA = 0.999                # discount factor; horizon ~ 1 / (1 - GAMMA) steps
LEARNING_RATE = 3e-4          # initial LR, linearly decayed to 0 over training
ENT_COEF = 0.01               # entropy bonus: keeps exploration alive
SEED = 0
EVAL_FREQ = 20_000            # evaluate + maybe save the best model every N steps
PLOT_FREQ = 100_000           # save a progress rollout plot every N steps
EVAL_EPISODES = 1             # eval episodes to average (raise for a stochastic env)

# ============================== FILES ==============================
# Outputs are prefixed per algorithm in train_common.py, so PPO and SAC stay
# separate: e.g. ppo_lander.zip / sac_lander.zip, ppo_progress/ ...
