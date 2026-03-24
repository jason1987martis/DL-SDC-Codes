"""
Q-learning agent for CartPole.

Run:
    python 07_cartpole_q_learning.py --episodes 2000

Working:
1. CartPole states are continuous, so we discretize them into bins.
2. Q-table stores expected reward for each state-action pair.
3. Agent chooses actions using epsilon-greedy exploration.
4. Table updates using the Bellman equation.
"""

import argparse
import numpy as np
import gymnasium as gym


def discretize_state(state, bins):
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))


def create_bins():
    return [
        np.linspace(-4.8, 4.8, 10),        # cart position
        np.linspace(-4.0, 4.0, 10),        # cart velocity
        np.linspace(-0.418, 0.418, 10),    # pole angle
        np.linspace(-4.0, 4.0, 10)         # pole angular velocity
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    bins = create_bins()

    q_table = np.zeros((11, 11, 11, 11, env.action_space.n))

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        state_disc = discretize_state(state, bins)
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_disc])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_disc = discretize_state(next_state, bins)

            best_next_q = np.max(q_table[next_state_disc])
            td_target = reward + args.gamma * best_next_q * (not done)
            td_error = td_target - q_table[state_disc][action]
            q_table[state_disc][action] += args.alpha * td_error

            state_disc = next_state_disc
            total_reward += reward

        args.epsilon = max(args.epsilon_min, args.epsilon * args.epsilon_decay)

        if episode % 100 == 0:
            print(f"Episode {episode}/{args.episodes} | Reward: {total_reward:.1f} | Epsilon: {args.epsilon:.4f}")

    np.save("cartpole_q_table.npy", q_table)
    print("Q-table saved as cartpole_q_table.npy")
    env.close()


if __name__ == "__main__":
    main()
