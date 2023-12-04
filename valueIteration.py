#This code was generated using ChatGPT 3.5 on December 1st 2023 at 4:45-8:00 P.M.  
#The website that was used as a reference to learn about value iteration and the code used as a reference can be found here https://tarikgit.github.io/coding/valueiteration-gridworld.html
#ChatGPT was given the assignemtn description so it could reference what needed to be completed
#Below are the prompts thaat were used to generate this code
#"Can you give me a code for value iteration on a 6x6 grid?"
#"What's a simple implementation of the value iteration algorithm for a grid world?"
#the grid world is a 6x6, the user loses everything if they step in (1,2)(,2,2)(3,2)(2,1),(2,3) and the end goal is at (4,4) which has a value of 1700 dollars. Every other square is empty
#"Can you write a Python script that implements the value iteration algorithm for a 6x6 grid world?"
#"Provide me with a Python code snippet for value iteration on a 6x6 grid using the Bellman equation."
#Can you make the output a grid and only allow 2 decimal points
#what would be a good value to change small enough to
#can you give me the full code with the character starting at 6,0
#The rest of the code was edited and created me

import numpy as np

# Initial set up

# Hyperparameters
SMALL_ENOUGH = 0.1
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
    elif i == (3, 0):
        rewards[i] = 20
    elif i == (4, 5):
        rewards[i] = -20
    elif i == (2, 0):
        rewards[i] = -5
    elif i == (0, 1):
        rewards[i] = 10
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

            # Check if the state has a negative reward
            if rewards[state] < -1000:
                V[state] = 0
                continue  # Skip the rest of the loop for this state

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