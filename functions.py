import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from hiive.mdptoolbox.mdp import QLearning
import pandas as pd


def best_action(T, V, π, action, s, γ, n_actions):
    Σ = np.zeros(n_actions)
    for a in range(n_actions):
        q = 0
        P = np.array(T[s][a])
        x = np.shape(P)[0]

        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            q += p * (r + γ * V[s_])
            Σ[a] = q

    best_a = np.argmax(Σ)
    action[s] = best_a
    π[s][best_a] = 1

    return π


def bellman(T, V, s, γ, n_states, n_actions):
    """update V[s] by taking action which maximizes current value"""
    π = np.zeros((n_states, n_actions))
    Σ = np.zeros(n_actions)
    rewards = []
    for a in range(n_actions):
        q = 0
        P = np.array(T[s][a])
        (x, y) = np.shape(P)

        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            rewards.append(r)
            q += p * (r + γ * V[s_])
        Σ[a] = q

    best_a = np.argmax(Σ)
    π[s][best_a] = 1

    v = 0
    for a in range(n_actions):
        u = 0
        P = np.array(T[s][a])
        x = np.shape(P)[0]
        for i in range(x):

            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            u += p * (r + γ * V[s_])

        v += π[s, a] * u

    V[s] = v
    return V[s]


def get_cur_policy(n_states, n_actions, T, V, γ):
    cur_policy = np.zeros((n_states, n_actions))
    action = np.zeros((n_states))
    for s in range(n_states):
        cur_policy = best_action(T, V, cur_policy, action, s, γ, n_actions)
    return [list(val).index(1) for val in cur_policy]


def value_iteration(T, γ, ϵ, ax_conv, ax_succ, n_states, n_actions, R=[]):
    V = np.zeros(n_states)  # initialize v(0) to arbitory value, my case "zeros"
    δs = []
    success_rate = []
    runtime = 0
    while True:
        start = time.time()
        # initialize convergence marker
        δ = 0
        # iterate for all states
        for s in range(n_states):
            v = V[s]
            V[s] = bellman(
                T, V, s, γ, n_states, n_actions
            )  # update state_value with bellman
            δ = max(
                δ, abs(v - V[s])
            )  # assign the change in value per iteration to delta
        δs.append(δ)
        runtime += time.time() - start
        cur_policy = get_cur_policy(n_states, n_actions, T, V, γ)
        if R == []:
            success_rate.append(test_frozen_lake_policy(cur_policy))
        else:
            success_rate.append(test_forest_policy(T, R, cur_policy))
        if δ < ϵ:
            break

    plot_convergence(
        range(len(δs)),
        δs,
        ax_conv,
        "δ",
        title="Convergence Plot for Value Iteration",
        hlines=[ϵ],
        hline_labels=["ϵ"],
    )
    π = np.zeros((n_states, n_actions))
    action = np.zeros((n_states))
    for s in range(n_states):
        π = best_action(T, V, π, action, s, γ, n_actions)
    if R == []:
        plot_success_rate(success_rate, ax_succ, label="Value Iteration")
    else:
        plot_success_rate(success_rate, ax_succ, label="Value Iteration")
    return V, π, action, runtime


def plot_convergence(
    x,
    y,
    ax,
    ylabel,
    title="Convergence Plot",
    hlines=[],
    hline_labels=[],
):
    ax.plot(x, y, label=ylabel)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, line in enumerate(hlines):
        ax.axhline(line, linestyle="--", label=hline_labels[i], color="orange")

    ax.legend()


def policy_test(T, policy, V, discount_factor, n_states):
    policy_value = np.zeros(n_states)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, _ in T[state][action]:
            policy_value[state] += probablity * (
                reward + (discount_factor * V[next_state])
            )

    return policy_value


def policy_iteration(
    T,
    ϵ,
    ax_conv,
    ax_succ,
    n_states,
    n_actions,
    discount_factor=0.999,
    max_iteration=1000,
    R=[],
):
    V = np.zeros(n_states)
    runtime = 0
    policy = np.random.randint(0, n_actions, n_states)
    policy_prev = np.copy(policy)
    δs = []
    success_rate = []
    for i in range(max_iteration):
        start = time.time()
        V = policy_test(T, policy, V, discount_factor, n_states)
        policy = update(T, policy, V, discount_factor, n_states, n_actions)
        δ = (policy != policy_prev).sum()
        if i % 10 == 0:
            δs.append(δ)
            if δ == 0:
                break
            policy_prev = np.copy(policy)
        runtime += time.time() - start
        if R == []:
            success_rate.append(test_frozen_lake_policy(policy))
        else:
            success_rate.append(test_forest_policy(T, R, policy))
    plot_convergence(
        [val * 10 for val in range(len(δs))],
        δs,
        ax_conv,
        "Num Differences in Policies",
        "Convergence Plot for Policy Iteration",
    )
    if R == []:
        plot_success_rate(success_rate, ax_succ, "Policy Iteration")
    else:
        plot_success_rate(success_rate, ax_succ, "Policy Iteration")
    return V, policy, runtime


def plot_success_rate(success_rate, ax, label):
    ax.plot(success_rate, label=label)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title("Policy Success Rate vs Iteration")


def plot_forest_rewards(rewards, ax, label):
    ax.plot(rewards, label=label)
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Episode")
    ax.set_title("Avg. Rewards per Episode")
    ax.legend()


def update(T, policy, V, discount_factor, n_states, n_actions):
    for state in range(n_states):
        # for a given state compute state-action value.
        action_values = step(T, state, V, n_actions, discount_factor)

        # choose the action which maximizez the state-action value.
        policy[state] = np.argmax(action_values)

    return policy


def step(T, state, V, n_actions, gamma=0.99):
    actions = np.zeros(n_actions)
    for a in range(n_actions):
        for p, s_, r, _ in T[state][a]:
            actions[a] += p * (r + (gamma * V[s_]))
    return actions


def test_frozen_lake_policy(policy, trials=100):
    env = gym.make("FrozenLake-v1")
    success = 0
    for _ in range(trials):
        done = False
        state = env.reset()[0]
        while not done:
            action = policy[state]
            state, _, done, _, _ = env.step(action)
            if state == 15:
                success += 1

    avg_success_rate = success / trials
    return avg_success_rate


def test_forest_policy(T, R, policy, test_count=100, gamma=0.9):
    num_state = len(T)
    total_episode = num_state * test_count
    successes = 0
    for state in range(num_state):
        for _ in range(test_count):
            while True:
                action = policy[state]
                probs = [T[state][action][s][0] for s in range(num_state)]
                candidates = list(range(num_state))
                next_state = np.random.choice(candidates, 1, p=probs)[0]
                if next_state == 499:
                    successes += 1
                    break
                if next_state == 0:
                    break
    return successes / total_episode


def evaluate_forest_policy(P, R, policy, test_count=100, gamma=0.9):
    num_state = P.shape[-1]
    total_runs = 0
    total_reward = 0
    successes = 0
    for state in range(num_state):
        for _ in range(test_count):
            disc_rate = 1
            while True:
                action = policy[state]
                probs = P[action][state]
                candidates = list(range(len(P[action][state])))
                next_state = np.random.choice(candidates, 1, p=probs)[0]
                reward = R[state][action] * disc_rate
                total_reward += reward
                disc_rate *= gamma
                if next_state == 499:
                    successes += 1
                if next_state == 0:
                    break
            total_runs += 1
    return total_reward / total_runs


def train_q_learning_forest(
    P,
    R,
    test_count,
    discount=0.9,
    alpha_dec=[0.99],
    alpha_min=[0.001],
    epsilon=[1.0],
    epsilon_decay=[0.99],
    n_iter=[1000],
):
    q_df = pd.DataFrame(
        columns=[
            "Iterations",
            "Alpha Decay",
            "Alpha Min",
            "Epsilon",
            "Epsilon Decay",
            "Reward",
            "Time",
            "Policy",
            "Value Function",
            "Training Rewards",
        ]
    )

    count = 0
    for i in n_iter:
        for eps in epsilon:
            for eps_dec in epsilon_decay:
                for a_dec in alpha_dec:
                    for a_min in alpha_min:
                        q = QLearning(
                            P,
                            R,
                            discount,
                            alpha_decay=a_dec,
                            alpha_min=a_min,
                            epsilon=eps,
                            epsilon_decay=eps_dec,
                            n_iter=i,
                        )
                        q.run()
                        reward = evaluate_forest_policy(P, R, q.policy, test_count)
                        count += 1
                        #                         clear_output(wait=True)
                        print("{}: {}".format(count, reward))
                        st = q.run_stats
                        rews = [s["Reward"] for s in st]
                        info = [
                            i,
                            a_dec,
                            a_min,
                            eps,
                            eps_dec,
                            reward,
                            q.time,
                            q.policy,
                            q.V,
                            sum(rews),
                        ]

                        df_length = len(q_df)
                        q_df.loc[df_length] = info
    return q_df
