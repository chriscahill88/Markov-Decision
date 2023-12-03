#This code was written by ChatGPT 3.5 on December 3rd at 2:30 P.M and was given the prompt "can you give me the code for epsilon greedy Q learning given the same grid as value iterations"
import numpy as np

# Hyperparameters
ALPHA = 0.5  # Learning rate
GAMMA = 0.98  # Discount factor
EPSILON = 0.50  # Exploration-exploitation trade-off
STALL_CHANCE = 0.15  # Chance to stall
BACKWARD_CHANCE = 0.15  # Chance to go backward
DESIRED_MOVE_CHANCE = 0.7  # Chance to perform the desired move
NUM_EPISODES = 1000

# Define all states
all_states = [(i, j) for i in range(6) for j in range(6)]

# Define rewards for all states
rewards = {
    (2, 2): -1700, (3, 1): -1700, (3, 2): -1700, (3, 3): -1700, (4, 2): -1700, (1, 4): 1700,
    #(3, 0): 20, (4, 5): -20, (2, 0): -5, (0, 1): 10
}

# Initialize Q values
Q = {(state, action): 0 for state in all_states for action in ['U', 'D', 'L', 'R']}

# Initialize values
values = {state: 0 for state in all_states}

# Epsilon-Greedy Q-Learning
for episode in range(NUM_EPISODES):
    state = (5, 0)  # Initial state
    total_reward = 0

    while state != (1, 4):  # Continue until reaching the goal state
        # Epsilon-Greedy Exploration
        if np.random.rand() < EPSILON:
            rand_val = np.random.rand()
            if rand_val < STALL_CHANCE:
                action = 'N/A'  # Stall
            elif rand_val < STALL_CHANCE + BACKWARD_CHANCE:
                action = np.random.choice(['U', 'D', 'L', 'R'])
            else:
                action = np.random.choice(['U', 'D', 'L', 'R'])
        else:
            values[state] = max(Q.get((state, a), 0) for a in ['U', 'D', 'L', 'R'])
            values = {state: max(Q.get((state, a), 0) for a in ['U', 'D', 'L', 'R']) for state in all_states}
            action = ['U', 'D', 'L', 'R'][np.argmax(values)]

        # Take action and observe the next state and reward
        if action != 'N/A':
            if action == 'U':
                next_state = (max(0, state[0] - 1), state[1])
            elif action == 'D':
                next_state = (min(5, state[0] + 1), state[1])
            elif action == 'L':
                next_state = (state[0], max(0, state[1] - 1))
            elif action == 'R':
                next_state = (state[0], min(5, state[1] + 1))

            # Check if the next state is a -1700 square
            if next_state in rewards and rewards[next_state] == -1700:
                total_reward = 0
            else:
                reward = rewards.get(next_state, 0)
                total_reward += reward

                # Update Q value using the Bellman equation
                next_max_q = max(Q.get((next_state, a), 0) for a in ['U', 'D', 'L', 'R'])
                Q[(state, action)] += ALPHA * (reward + GAMMA * next_max_q - Q[(state, action)])

            state = next_state  # Move to the next state

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Display the learned policy
policy = {state: ['U', 'D', 'L', 'R'][np.argmax([Q.get((state, a), 0) for a in ['U', 'D', 'L', 'R']])] for state in all_states}
print("\nLearned Policy:")
for i in range(6):
    row = ""
    for j in range(6):
        state = (i, j)
        if state in policy:
            if policy[state] == 'U':
                row += "^  \t"
            elif policy[state] == 'D':
                row += "v  \t"
            elif policy[state] == 'R':
                row += ">  \t"
            elif policy[state] == 'L':
                row += "<  \t"
            elif policy[state] == 'N/A':
                row += "N/A\t"
        else:
            row += "N/A\t"
    print(row)

# Display the results
print("Optimal Value Function:")
for i in range(6):
    row = ""
    for j in range(6):
        state = (i, j)
        if state in values:
            row += f"{values[state]:.0f}\t"
        else:
            row += "N/A\t"
    print(row)