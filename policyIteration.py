import numpy as np

# Initial set up

# Hyperparameters
SMALL_ENOUGH = 0.005
GAMMA = 0.9
NOISE = 0.1

# Define all states
all_states = []
for i in range(6):
    for j in range(6):
        all_states.append((i, j))

# Define rewards for all states
rewards = {}
for i in all_states:
    if i == (2, 2) or i == (3, 1) or i == (3, 2) or i == (3, 3) or i == (4, 2):
        rewards[i] = -1700
    elif i == (1, 4):
        rewards[i] = 1700
   # elif i == (3, 0):
    #    rewards[i] = 20
    #elif i == (4, 5):
    #    rewards[i] = -20
    #elif i == (2, 0):
    #    rewards[i] = -5
    #elif i == (0, 1):
    #    rewards[i] = 10
    else:
        rewards[i] = 0

# Define initial policy and value function
policy = {state: np.random.choice(['U', 'D', 'L', 'R']) for state in all_states}
V = {state: 0 for state in all_states}

# Policy Iteration

max_iterations = 100
iteration = 0
while True:
    # Policy Evaluation
    while True:
        biggest_change = 0
        for state in all_states:
            old_v = V[state]

            action = policy[state]
            if action == 'U':
                next_state = (max(0, state[0] - 1), state[1])
            elif action == 'D':
                next_state = (min(5, state[0] + 1), state[1])
            elif action == 'L':
                next_state = (state[0], max(0, state[1] - 1))
            elif action == 'R':
                next_state = (state[0], min(5, state[1] + 1))

            next_state = tuple(next_state)
            V[state] = rewards[state] + (GAMMA * ((1 - NOISE) * V[next_state] + (NOISE * V[state])))
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))

        if biggest_change < SMALL_ENOUGH:
            break

    # Policy Improvement
    policy_stable = True
    for state in all_states:
        old_action = policy[state]

        # Find the action that maximizes expected value
        values = []
        for action in ['U', 'D', 'L', 'R']:
            if action == 'U':
                next_state = (max(0, state[0] - 1), state[1])
            elif action == 'D':
                next_state = (min(5, state[0] + 1), state[1])
            elif action == 'L':
                next_state = (state[0], max(0, state[1] - 1))
            elif action == 'R':
                next_state = (state[0], min(5, state[1] + 1))

            next_state = tuple(next_state)
            value = rewards[state] + (GAMMA * ((1 - NOISE) * V[next_state] + (NOISE * V[state])))
            values.append(value)

        best_action = ['U', 'D', 'L', 'R'][np.argmax(values)]
        policy[state] = best_action

        if old_action != best_action:
            policy_stable = False

    if policy_stable:
        print(f"Converged after {iteration} iterations")
        break

    iteration += 1
    if iteration == max_iterations:
        print("Maximum number of iterations reached.")

# Print the optimal path with random decisions
print("\nOptimal Path:")
current_state = (5, 0)  # Starting state
while True:
    print(f"Current State: {current_state}, Action: {policy[current_state]}")

    # Make random decisions
    rand_val = np.random.rand()
    if rand_val < 0.15:
        next_state = current_state  # Stays in the same state (stall out)
    elif rand_val < 0.3:
        # Move in reverse
        action = policy[current_state]
        next_state = (max(0, current_state[0] - 1), current_state[1]) if action == 'U' else \
                     (min(5, current_state[0] + 1), current_state[1]) if action == 'D' else \
                     (current_state[0], max(0, current_state[1] - 1)) if action == 'L' else \
                     (current_state[0], min(5, current_state[1] + 1))
    elif rand_val >= 0.3:
        # Move forward
        action = policy[current_state]
        next_state = (max(0, current_state[0] - 1), current_state[1]) if action == 'U' else \
                     (min(5, current_state[0] + 1), current_state[1]) if action == 'D' else \
                     (current_state[0], max(0, current_state[1] - 1)) if action == 'L' else \
                     (current_state[0], min(5, current_state[1] + 1))

    current_state = next_state

    # Check if the path has reached the goal state
    if current_state == (1, 4):
        print(f"Reached the goal state: {current_state}")
        break

# Display the results
print("Optimal Value Function:")
for i in range(6):
    row = ""
    for j in range(6):
        state = (i, j)
        if state in V:
            row += f"{V[state]:.0f}\t"
        else:
            row += "N/A\t"
    print(row)

print("\nOptimal Policy:")
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
        else:
            row += "N/A\t"
    print(row)
