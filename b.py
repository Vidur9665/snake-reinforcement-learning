import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer

print("Running b.py...")

# -------------------- Hyperparameters --------------------- #
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
LR = 0.001
BATCH_SIZE = 128
MEMORY_SIZE = 10000
MAX_TICKS = 100
TARGET_SYNC_EVERY = 10

# -------------------- Bot Base ---------------------------- #
class SnakeBot:
    DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
    def next_move(self, state): return random.choice(self.DIRECTIONS)

# ---------------- Trainable Bot Class --------------------- #
class TrainableSnakeBot(SnakeBot):
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_score = 0

    def _build_model(self):
        model = Sequential([
            InputLayer(input_shape=(11,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(4, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='mse')
        return model

    def _state_vector(self, state):
        snake = state["snake"]
        food = state["food"]
        direction = state["direction"]
        head_x, head_y = snake[0]

        def move(pos, dir):
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[dir]
            return (pos[0] + dx, pos[1] + dy)

        def in_bounds(pos):
            x, y = pos
            return 0 <= x < state["board_width"] and 0 <= y < state["board_height"]

        left  = self._turn_left(direction)
        right = self._turn_right(direction)
        point_f = move((head_x, head_y), direction)
        point_l = move((head_x, head_y), left)
        point_r = move((head_x, head_y), right)

        danger_f = int(not in_bounds(point_f) or point_f in snake)
        danger_l = int(not in_bounds(point_l) or point_l in snake)
        danger_r = int(not in_bounds(point_r) or point_r in snake)

        dir_flags = [int(direction == d) for d in self.DIRECTIONS]

        food_left = int(food[0] < head_x)
        food_right = int(food[0] > head_x)
        food_up = int(food[1] < head_y)
        food_down = int(food[1] > head_y)

        return np.array([
            danger_f, danger_l, danger_r,
            *dir_flags,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)

    def _turn_left(self, d): return {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}[d]
    def _turn_right(self, d): return {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}[d]

    def next_move(self, state):
        state_vec = self._state_vector(state)
        reward = self._calc_reward(state)
        done = not (0 <= state["snake"][0][0] < state["board_width"] and 0 <= state["snake"][0][1] < state["board_height"])

        if self.prev_state is not None:
            self.memory.append((self.prev_state, self.prev_action, reward, state_vec, done))

        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            q_vals = self.model.predict(state_vec.reshape(1, -1), verbose=0)[0]
            action = np.argmax(q_vals)

        self.prev_state = state_vec
        self.prev_action = action
        self.prev_score = state["score"]

        self.steps += 1
        if self.steps % TARGET_SYNC_EVERY == 0:
            self.target_model.set_weights(self.model.get_weights())

        return self.DIRECTIONS[action]

    def _calc_reward(self, state):
        if self.prev_score is None:
            return 0
        if state["score"] > self.prev_score:
            return 50  # more emphasis on eating
        if not (0 <= state["snake"][0][0] < state["board_width"] and 0 <= state["snake"][0][1] < state["board_height"]):
            return -50
        return -0.2

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, targets = [], []

        for s, a, r, s2, done in batch:
            q_vals = self.model.predict(s.reshape(1, -1), verbose=0)[0]
            if done:
                q_vals[a] = r
            else:
                future_q = self.target_model.predict(s2.reshape(1, -1), verbose=0)[0]
                q_vals[a] = r + GAMMA * np.max(future_q)
            states.append(s)
            targets.append(q_vals)

        self.model.train_on_batch(np.array(states), np.array(targets))

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# -------------------- Training Loop ------------------------ #
N_EPISODES = 1000
BLANK_LINES_BEFORE_DEMO = 10


def train_and_demo():
    print("Starting training...")
    import s
    bot = TrainableSnakeBot()
    scores, best_score = [], 0

    print(f"Training {N_EPISODES} episodes with max_ticks={MAX_TICKS}")
    for ep in range(1, N_EPISODES + 1):
        bot.prev_state = bot.prev_action = bot.prev_score = None

        res = s.run_episode(bot, render=False, max_ticks=MAX_TICKS)
        scores.append(res["score"])
        best_score = max(best_score, res["score"])
        bot.train()
        bot.decay_epsilon()

        if ep % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"Ep {ep:4d} | Best: {best_score:2d} | Avg(10): {avg:.2f} | ε={bot.epsilon:.3f}")

    print("\nTraining done.\n")
    print("\n" * BLANK_LINES_BEFORE_DEMO)
    s.run_episode(bot, render=True)

# -------------------- Entry Point ------------------------ #
if __name__ == "__main__":
    train_and_demo()