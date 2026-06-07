import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="Windy Gridworld RL", layout="wide")

st.title("🌪️ Windy Gridworld — SARSA & Monte Carlo")

# ==========================================================
# SESSION STATE
# ==========================================================

if "trained" not in st.session_state:
    st.session_state.trained = False

# ==========================================================
# ENVIRONMENT
# ==========================================================


class WindyGridworld:

    def __init__(self, kings_moves=False, stochastic=False, stay_action=False):

        self.width = 10
        self.height = 7

        self.start = (0, 3)
        self.goal = (7, 3)

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.kings_moves = kings_moves
        self.stochastic = stochastic
        self.stay_action = stay_action

        # ==================================================
        # ACTIONS
        # ==================================================

        # STANDARD 4 ACTIONS
        if not kings_moves:

            self.actions = [
                (1, 0),  # RIGHT
                (0, -1),  # UP
                (-1, 0),  # LEFT
                (0, 1),  # DOWN
            ]

        # KING'S MOVES (8 ACTIONS)
        elif kings_moves and not stay_action:

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

        # KING'S + STAY (9 ACTIONS)
        elif kings_moves and stay_action:

            self.actions = [
                (1, 0),
                (0, -1),
                (-1, 0),
                (0, 1),
                (1, -1),
                (-1, -1),
                (1, 1),
                (-1, 1),
                (0, 0),  # STAY STILL
            ]

        self.num_actions = len(self.actions)

        self.reset()

    def reset(self):

        self.state = self.start

        return self.state

    def step(self, action):

        x, y = self.state

        dx, dy = self.actions[action]

        # MOVE
        x += dx
        y += dy

        # WIND
        wind_strength = self.wind[x] if 0 <= x < self.width else 0

        # STOCHASTIC WIND
        if self.stochastic:

            wind_strength += random.choice([-1, 0, 1])

        # APPLY WIND
        y -= wind_strength

        # BOUNDARY
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        self.state = (x, y)

        reward = -1

        done = self.state == self.goal

        return self.state, reward, done


# ==========================================================
# SARSA AGENT
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

                # SARSA UPDATE
                td_target = reward

                if not done:

                    td_target += self.gamma * self.Q[next_state][next_action]

                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state
                action = next_action

                steps += 1

            self.steps_per_episode.append(steps)


# ==========================================================
# MONTE CARLO AGENT
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

            # FIRST VISIT MONTE CARLO
            G = 0

            visited = set()

            for t in reversed(range(len(episode))):

                state, action, reward = episode[t]

                G = self.gamma * G + reward

                if (state, action) not in visited:

                    visited.add((state, action))

                    self.returns[(state, action)].append(G)

                    self.Q[state][action] = np.mean(self.returns[(state, action)])


# ==========================================================
# GENERATE PATH
# ==========================================================


def generate_path(env, agent):

    state = env.reset()

    path = [state]

    done = False

    steps = 0

    visited = set()

    while not done and steps < 100:

        if state in visited:
            break

        visited.add(state)

        action = np.argmax(agent.Q[state])

        next_state, reward, done = env.step(action)

        path.append(next_state)

        state = next_state

        steps += 1

    return path


# ==========================================================
# DRAW POLICY
# ==========================================================


def draw_policy(env, agent, blink_goal=False, current_position=None):

    arrows = {0: "→", 1: "↑", 2: "←", 3: "↓", 4: "↗", 5: "↖", 6: "↘", 7: "↙", 8: "•"}

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)

    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))

    ax.grid()

    # WIND LABELS
    for col in range(env.width):

        ax.text(col, -0.4, f"W={env.wind[col]}", color="red", ha="center")

    # POLICY ARROWS
    for state in agent.Q:

        x, y = state

        action = np.argmax(agent.Q[state])

        ax.text(x, y, arrows[action], fontsize=20, ha="center", va="center")

    # START
    sx, sy = env.start

    ax.text(sx, sy, "S", fontsize=24, color="orange", ha="center")

    # GOAL
    gx, gy = env.goal

    goal_color = "green"

    if blink_goal and current_position == env.goal:

        if int(time.time() * 3) % 2 == 0:

            goal_color = "yellow"

    ax.text(gx, gy, "G", fontsize=24, color=goal_color, ha="center")

    # CURRENT AGENT POSITION
    if current_position is not None:

        ax.plot(current_position[0], current_position[1], "bo", markersize=18)

    ax.set_title("Learned Policy")

    return fig


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Configuration")

algorithm = st.sidebar.selectbox("Algorithm", ["SARSA", "Monte Carlo"])

episodes = st.sidebar.slider("Training Episodes", 100, 10000, 5000)

alpha = st.sidebar.slider("Learning Rate α", 0.01, 1.0, 0.5)

epsilon = st.sidebar.slider("Exploration ε", 0.01, 1.0, 0.1)

gamma = st.sidebar.slider("Discount Factor γ", 0.1, 1.0, 1.0)

kings_moves = st.sidebar.checkbox("King's Moves (8 Actions)", value=True)

stay_action = st.sidebar.checkbox("Stay Action (9th Action)", value=False)

stochastic = st.sidebar.checkbox("Stochastic Wind", value=False)

blink_goal = st.sidebar.checkbox("Blink Goal", value=True)

train_button = st.sidebar.button("🚀 Train Agent")

play_button = st.sidebar.button("▶ Play Learned Path")

# ==========================================================
# TRAINING
# ==========================================================

if train_button:

    env = WindyGridworld(
        kings_moves=kings_moves, stochastic=stochastic, stay_action=stay_action
    )

    # SARSA
    if algorithm == "SARSA":

        agent = SARSAAgent(env, alpha, epsilon, gamma)

    # MONTE CARLO
    else:

        agent = MonteCarloAgent(env, epsilon, gamma)

    with st.spinner("Training Agent..."):

        agent.train(episodes)

    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.path = generate_path(env, agent)

    st.session_state.trained = True

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

if st.session_state.trained:

    env = st.session_state.env
    agent = st.session_state.agent
    path = st.session_state.path

    # ======================================================
    # STEP SLIDER
    # ======================================================

    step_slider = st.slider("Step Viewer", 0, max(len(path) - 1, 0), 0)

    current_position = path[step_slider]

    # ======================================================
    # POLICY DISPLAY
    # ======================================================

    fig = draw_policy(
        env, agent, blink_goal=blink_goal, current_position=current_position
    )

    st.pyplot(fig)

    # ======================================================
    # PLAY ANIMATION
    # ======================================================

    if play_button:

        animation_placeholder = st.empty()

        for pos in path:

            plt.close("all")

            fig = draw_policy(env, agent, blink_goal=blink_goal, current_position=pos)

            animation_placeholder.pyplot(fig, clear_figure=True)

        time.sleep(0.55)

    # ======================================================
    # LEARNING CURVE
    # ======================================================

    st.subheader("📈 Learning Curve")

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    moving_avg = np.convolve(agent.steps_per_episode, np.ones(100) / 100, mode="valid")

    ax2.plot(moving_avg)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps to Goal")

    ax2.grid()

    st.pyplot(fig2)

    # ======================================================
    # PERFORMANCE
    # ======================================================

    avg_steps = np.mean(agent.steps_per_episode[-100:])

    st.subheader("📊 Performance")

    st.write("Average Final Steps:", round(avg_steps, 2))

    st.write("Total Learned States:", len(agent.Q))

    # ======================================================
    # OBSERVATIONS
    # ======================================================

    st.subheader("🧠 Observations")

    if not kings_moves:

        st.write("Standard Windy Gridworld produces zig-zag trajectories.")

    if kings_moves:

        st.write("King's Moves produce shorter and smoother trajectories.")

    if stay_action:

        st.write("The 9th stay action allows the agent to use wind efficiently.")

    if stochastic:

        st.write("Stochastic wind creates safer and more robust policies.")

    if algorithm == "SARSA":

        st.write("SARSA learns online using Temporal Difference updates.")

    else:

        st.write(
            "Monte Carlo learns slower because updates happen after full episodes."
        )
