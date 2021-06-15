import numpy as np
import random
import time
import gym
import sys, getopt


def print_frames(frames):
    for i, frame in enumerate(frames):
        if 'stop' in frame:
            time.sleep(4)
        else:
            print(frame['frame'])
            print(f"Episode: {frame['episode']}")
            print(f"Step: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            time.sleep(1)


def train(learning_rate, epsilon, gamma, iterations):
    env = gym.make('Taxi-v3')
    env.reset()

    actions_count = env.action_space.n
    states_count = env.observation_space.n
    q_table = np.zeros([states_count, actions_count])

    max_steps = 200
    for i in range(1, iterations + 1):
        state = env.reset()

        incorrect_moves, total_reward = 0, 0
        done = False

        step_count = 0
        while not done and step_count < max_steps:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)

            prev_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - learning_rate) * prev_value + learning_rate * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward < -1:
                incorrect_moves += 1

            total_reward += reward

            state = next_state
            step_count += 1

        if i % 500 == 0:
            print(f"Episode: {i}, average reward: {total_reward / step_count}, incorrectMoves: {incorrect_moves}, episode length: {step_count} ")

    print("Training finished.\n")
    np.savetxt("q_table.txt", q_table, delimiter=",")


def demo(episodes):
    env = gym.make('Taxi-v3')
    env.reset()

    q_table = np.genfromtxt("q_table.txt", delimiter=',')

    total_epochs, total_incorrect_moves = 0, 0
    frames = []

    for ep in range(episodes):
        state = env.reset()
        incorrect_moves, reward = 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward < -1:
                incorrect_moves += 1

            frames.append({
                'frame': env.render(mode='ansi'),
                'episode': ep,
                'state': state,
                'action': action,
                'reward': reward,
            })

        frames.append({
            'stop': 1
        })
        total_incorrect_moves += incorrect_moves

    print(f"Results after {episodes} episodes:")
    print(f"Average incorrect moves per episode: {total_incorrect_moves / episodes}")

    print_frames(frames)


def main(argv):
    learning_rate, epsilon, gamma, iterations = 0, 0, 0, 0

    try:
        opts, args = getopt.getopt(argv, "hl:e:g:i:", ["learning_rate=", "epsilon=", "gamma=", "iterations="])
    except getopt.GetoptError:
        print('ex8.py -l <learning_rate> -e <epsilon> -g <gamma> -i <iterations>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ex8.py -l <learning_rate> -e <epsilon> -g <gamma> -i <iterations>')
            sys.exit()
        elif opt in ("-l", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-e", "--epsilon"):
            epsilon = float(arg)
        elif opt in ("-g", "--gamma"):
            gamma = float(arg)
        elif opt in ("-i", "--iterations"):
            iterations = int(arg)

    print('EARIN EX 8 - by Oskar Hącel and Marcin Lisowski')
    print(f"Learning rate: {learning_rate}")
    print(f"Epsilon: {epsilon}")
    print(f"Gamma: {gamma}")
    print(f"Iterations: {iterations}")

    train(learning_rate, epsilon, gamma, iterations)
    demo(3)

if __name__ == "__main__":
   main(sys.argv[1:])