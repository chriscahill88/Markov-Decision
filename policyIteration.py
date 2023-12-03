#This code was written by ChatGPT 3.5 on December 3rd at 2:25 P.M and was given the prompt "can you give me the code for policy iteration given the same grid as value iterations"
import numpy as np

# Initial set up

# Hyperparameters
SMALL_ENOUGH = 0.05
GAMMA = 0.95
NOISE = 0.0

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
    #elif i == (3, 0):
        #rewards[i] = 20
    #elif i == (4, 5):
        #rewards[i] = -20
    #elif i == (2, 0):
        #rewards[i] = -5
    #elif i == (0, 1):
        #rewards[i] = 10
    else:
        rewards[i] = 0

# Define initial state (starting position of the character)
initial_state = (5, 0)

# Dictionary of possible actions
actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('D', 'R', 'L'),
    (0, 2): ('D', 'L', 'R'),
    (0, 3): ('D', 'L', 'R'), 
    (0, 4): ('D', 'L', 'R'),  
    (0, 5): ('D', 'L'),  
    (1, 0): ('D', 'U', 'R'),
    (1, 1): ('D', 'R', 'L', 'U'),
    (1, 2): ('D', 'R', 'L', 'U'),
    (1, 3): ('D', 'R', 'L', 'U'),
    (1, 4): ('D', 'R', 'L', 'U'),
    (1, 5): ('D', 'L', 'U'),
    (2, 0): ('D', 'U', 'R'),
    (2, 1): ('D', 'U', 'L', 'R'),
    (2, 2): ('D', 'U', 'L', 'R'),
    (2, 3): ('D', 'U', 'L', 'R'),
    (2, 4): ('D', 'U', 'R', 'L'),
    (2, 5): ('D', 'U', 'L'),  
    (3, 0): ('D', 'U', 'R'),
    (3, 1): ('D', 'R', 'L', 'U'),
    (3, 2): ('D', 'R', 'L', 'U'),
    (3, 3): ('D', 'U', 'R', 'L'),
    (3, 4): ('D', 'U', 'R', 'L'),
    (3, 5): ('D', 'U', 'L'),
    (4, 0): ('D', 'U', 'R'),
    (4, 1): ('D', 'R', 'L', 'U'),
    (4, 2): ('D', 'R', 'L', 'U'),
    (4, 3): ('D', 'U', 'R', 'L'),
    (4, 4): ('D', 'U', 'R', 'L'),
    (4, 5): ('D', 'U', 'L'),
    (5, 0): ('U', 'R'),
    (5, 1): ('U', 'L', 'R'),
    (5, 2): ('U', 'L', 'R'),  
    (5, 3): ('R', 'L', 'U'),
    (5, 4): ('R', 'L', 'U'),
    (5, 5): ('L', 'U'),
}

# Define an initial policy and value function
policy = {state: np.random.choice(actions[state]) for state in actions.keys()}
V = {state: 0 for state in all_states}

# Set the initial state and value
policy[initial_state] = 'U'
V[initial_state] = 0

def get_next_state(state, action):
    if action == 'U':
        return max(0, state[0] - 1), state[1]
    elif action == 'D':
        return min(5, state[0] + 1), state[1]
    elif action == 'L':
        return state[0], max(0, state[1] - 1)
    elif action == 'R':
        return state[0], min(5, state[1] + 1)

def reverse_action(action):
    if action == 'U':
        return 'D'
    elif action == 'D':
        return 'U'
    elif action == 'L':
        return 'R'
    elif action == 'R':
        return 'L'

# Value Iteration
max_iterations = 100
iteration = 0

while True:
    biggest_change = 0
    for state in all_states:
        if state in policy:
            old_v = V[state]
            new_v = float('-inf')
            best_action = None



            for action in actions[state]:
                next_state = get_next_state(state, action)
                # Calculate the value
                value = rewards[state] + (GAMMA * ((1 - NOISE) * V[next_state] + (NOISE * V[state])))
                if value > new_v:  # Is this the best action so far? If so, keep it
                    new_v = value
                    best_action = action

            # Save the best action for the state
            V[state] = new_v
            policy[state] = best_action
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))


    # See if the loop should stop now
    if biggest_change < SMALL_ENOUGH:
        print(f"Converged after {iteration} iterations due to SMALL_ENOUGH")
        break
    iteration += 1
    if iteration == max_iterations:
        print(f"Maximum number of iterations {iteration} reached.")
        break

# Print the optimal path with random decisions
print("\nOptimal Path:")
current_state = initial_state
while True:
    print(f"Current State: {current_state}, Action: {policy[current_state]}")
    
    # Make random decisions
    rand_val = np.random.rand()
    if rand_val < 0.15:
        next_state = current_state  # Stays in the same state (stall out)
    elif rand_val < 0.3:
        # Move in reverse
        action = policy[current_state]
        next_state = get_next_state(current_state, reverse_action(action))
    elif rand_val >= 0.3:
        # Move forward
        action = policy[current_state]
        next_state = get_next_state(current_state, action)

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

