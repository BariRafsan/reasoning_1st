import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="Windy Gridworld RL", layout="wide")

st.title("🌪️ Windy Gridworld — Reinforcement Learning")


# ==========================================================
# ENVIRONMENT
# ==========================================================


class WindyGridworld:

    def __init__(self, kings_moves=False, stochastic=False):

        self.width = 10
        self.height = 7

        self.start = (0, 3)
        self.goal = (7, 3)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.kings_moves = kings_moves
        self.stochastic = stochastic

        if kings_moves:

            self.actions = [
                (1, 0),
                (0, -1),
                (-1, 0),
                (0, 1),
                (1, -1),
                (-1, -1),
                (1, 1),
                (-1, 1),
            ]

        else:

            self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

        self.num_actions = len(self.actions)

        self.reset()

    def reset(self):

        self.state = self.start

        return self.state

    def step(self, action):

        x, y = self.state

        dx, dy = self.actions[action]

        x += dx
        y += dy

        wind_strength = self.wind[x] if 0 <= x < self.width else 0

        if self.stochastic:

            wind_strength += random.choice([-1, 0, 1])

        y -= wind_strength

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        self.state = (x, y)

        reward = -1

        done = self.state == self.goal

        return self.state, reward, done


# ==========================================================
# SARSA
# ==========================================================


class SARSAAgent:

    def __init__(self, env, alpha, epsilon, gamma):

        self.env = env

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(env.num_actions))

        self.steps_per_episode = []

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:

            return np.random.randint(self.env.num_actions)

        return np.argmax(self.Q[state])

    def train(self, episodes):

        for ep in range(episodes):

            state = self.env.reset()

            action = self.choose_action(state)

            done = False

            steps = 0

            while not done and steps < 500:

                next_state, reward, done = self.env.step(action)

                next_action = self.choose_action(next_state)

                td_target = reward

                if not done:

                    td_target += self.gamma * self.Q[next_state][next_action]

                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                action = next_action

                steps += 1

            self.steps_per_episode.append(steps)

    def get_policy(self):

        policy = {}

        for state in self.Q:

            policy[state] = np.argmax(self.Q[state])

        return policy


# ==========================================================
# MONTE CARLO
# ==========================================================


class MonteCarloAgent:

    def __init__(self, env, epsilon, gamma):

        self.env = env

        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(env.num_actions))

        self.returns = defaultdict(list)

        self.steps_per_episode = []

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:

            return np.random.randint(self.env.num_actions)

        return np.argmax(self.Q[state])

    def train(self, episodes):

        for ep in range(episodes):

            episode = []

            state = self.env.reset()

            done = False

            steps = 0

            while not done and steps < 500:

                action = self.choose_action(state)

                next_state, reward, done = self.env.step(action)

                episode.append((state, action, reward))

                state = next_state

                steps += 1

            self.steps_per_episode.append(steps)

            G = 0

            visited = set()

            for t in reversed(range(len(episode))):

                state, action, reward = episode[t]

                G = self.gamma * G + reward

                if (state, action) not in visited:

                    visited.add((state, action))

                    self.returns[(state, action)].append(G)

                    self.Q[state][action] = np.mean(self.returns[(state, action)])

    def get_policy(self):

        policy = {}

        for state in self.Q:

            policy[state] = np.argmax(self.Q[state])

        return policy


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Configuration")

algorithm = st.sidebar.selectbox("Algorithm", ["SARSA", "Monte Carlo"])

episodes = st.sidebar.slider("Training Episodes", 100, 10000, 5000)

alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.5)

epsilon = st.sidebar.slider("Exploration (epsilon)", 0.01, 1.0, 0.1)

gamma = st.sidebar.slider("Discount Factor (gamma)", 0.1, 1.0, 1.0)

kings_moves = st.sidebar.checkbox("King's Moves", value=True)

stochastic = st.sidebar.checkbox("Stochastic Wind", value=False)


# ==========================================================
# TRAIN BUTTON
# ==========================================================

train_button = st.sidebar.button("🚀 Train Agent")


# ==========================================================
# TRAIN
# ==========================================================

if train_button:

    env = WindyGridworld(kings_moves=kings_moves, stochastic=stochastic)

    if algorithm == "SARSA":

        agent = SARSAAgent(env, alpha, epsilon, gamma)

    else:

        agent = MonteCarloAgent(env, epsilon, gamma)

    with st.spinner("Training Agent..."):

        agent.train(episodes)

    policy = agent.get_policy()

    st.success("Training Complete!")

    # ======================================================
    # POLICY VISUALIZATION
    # ======================================================

    arrows = {0: "→", 1: "↑", 2: "←", 3: "↓", 4: "↗", 5: "↖", 6: "↘", 7: "↙"}

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)

    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))

    ax.grid()

    for col in range(env.width):

        ax.text(col, -0.3, f"W={env.wind[col]}", ha="center", color="red")

    for state, action in policy.items():

        x, y = state

        ax.text(x, y, arrows[action], fontsize=20, ha="center", va="center")

    sx, sy = env.start
    gx, gy = env.goal

    ax.text(sx, sy, "S", fontsize=22, color="orange", ha="center")

    ax.text(gx, gy, "G", fontsize=22, color="green", ha="center")

    ax.set_title(f"{algorithm} Learned Policy")

    st.pyplot(fig)

    # ======================================================
    # LEARNING CURVE
    # ======================================================

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    moving_avg = np.convolve(agent.steps_per_episode, np.ones(100) / 100, mode="valid")

    ax2.plot(moving_avg)

    ax2.set_title("Learning Curve")

    ax2.set_xlabel("Episode")

    ax2.set_ylabel("Steps to Goal")

    ax2.grid()

    st.pyplot(fig2)

    # ======================================================
    # STATISTICS
    # ======================================================

    st.subheader("📊 Statistics")

    st.write("Average Final Steps:", np.mean(agent.steps_per_episode[-100:]))

    st.write("Total States Learned:", len(policy))

    # ======================================================
    # EXPLANATION
    # ======================================================

    st.subheader("🧠 Explanation")

    if kings_moves:

        st.write("King's Moves allow diagonal movement, creating shorter paths.")

    if stochastic:

        st.write("Stochastic wind creates more cautious policies.")

    if algorithm == "SARSA":

        st.write("SARSA learns faster using Temporal Difference learning.")

    else:

        st.write("Monte Carlo waits until episode end before updating.")
