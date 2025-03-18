
# # Learning and Decision Making

# ## Laboratory 2: Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (``File >> Download as >> Python (.py)``). Make sure that the resulting script includes all code written in the tasks marked as "**Activity n. N**", together with any replies to specific questions posed. Your file should be named `padi-labKK-groupXXX.py`, where `KK` corresponds to the lab number and the `XXX` corresponds to your group number. Similarly, your homework should consist of a single pdf file named `padi-hwKK-groupXXX.pdf`. You should create a zip file with the lab and homework files and submit it in Fenix **at most 30 minutes after your lab is over**.
# 
# Make sure to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The MDP Model
# 
# We will use a simplified game inspired in the games StopIT and Insey-Winsey-Spider.
# 
# The player has several levels to climb (corresponding to steps in a ladder) and wants to reach the top level.
# 
# At each instant the player can decide to go and they throw a dice. After that the player has the possibility to climb a number of steps. However, the player will only go up if it is a sunny day; if it is a rainny day then the player will go back to the last safe level. At each instant there is also the option to stop. This makes the current level a safe one. Once the last step is reached, the game will reset to the initial state corresponding to the level 0 and safe level 0, independently of the action taken by the player at the last level.

# ##### ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file will contain the transition matrix for the two actions, with P[0] corresponding to the transition of the action stop and P[1] for the action go.
# 
# Create a tuple including
# 
# * An array `X` that contains all the states in the MDP represented as strings.
# * An array `A` that contains all the actions in the MDP, represented as strings. In the domain above, for example, each action is represented as a string `"St"`, and `"Go"`.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `c` with dimension `len(X)` &times; `len(A)` containing the cost function for the MDP. The cost must be 1 in all states except in the states corresponding to the top level (level 9); for top-level states, the cost must be zero.
# 
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# 
# ---

import numpy as np

def load_mdp(fname, gamma):
    """
    Builds an MDP model from the provided file.

    :param fname: Name of the file containing the MDP information
    :type: str
    :param gamma: Discount
    :type: float
    :returns: tuple (tuple, tuple, tuple, nd.array, float)
    """
    tMatrices = np.load(fname)
    X = list(str(i) for i in range(tMatrices.shape[1]))
    A = ["St", "Go"]
    P = np.array([np.array(tMatrices[0]), np.array(tMatrices[1])])
    c = np.ones((len(X), len(A)))
    c[90:, :] = 0

    return (X, A, P, c, gamma)

import numpy.random as rand

M = load_mdp('StopITSpider04.npy', 0.9)

rand.seed(42)

# States
print('= State space (%i states) =' % len(M[0]))
print('\nStates:')
for i in range(min(10, len(M[0]))):
    print(M[0][i])

print('...')

# Random state
x = rand.randint(len(M[0]))
print('\nRandom state: x =', M[0][x])

# Last state
print('Last state: x =', M[0][-1])

# Actions
print('\n= Action space (%i actions) =\n' % len(M[1]))
for i in range(len(M[1])):
    print(M[1][i])

# Random action
a = rand.randint(len(M[1]))
print('\nRandom action: a =', M[1][a])

# Transition probabilities
print('\n= Transition probabilities =')

for i in range(len(M[1])):
    print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[2][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[2][i]), len(M[0])))

print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][x], M[1][a]))
print("x' in", np.array(M[0])[np.where(M[2][a][x, :] > 0)])

# Cost
print('\n= Costs =')
print('\nCost for the state-action pair (%s, %s):' % (M[0][x], M[1][a]))
print(x,a)
print('c(x, a) =', M[3][x, a])


# Discount
print('\n= Discount =')
print('\ngamma =', M[4])

# We provide below an example of application of the function with MDP from the example in **Activity 1**, that you can use as a first "sanity check" for your code. Note that, as emphasized above, your function should work with **any** MDP that is specified as a tuple with the structure of the one from **Activity 1**.
# 
# ```python
# 
# import numpy.random as rand
# 
# M = load_mdp('StopITSpider04.npy', 0.9)
# 
# rand.seed(42)
# 
# # States
# print('= State space (%i states) =' % len(M[0]))
# print('\nStates:')
# for i in range(min(10, len(M[0]))):
#     print(M[0][i])
# 
# print('...')
# 
# # Random state
# x = rand.randint(len(M[0]))
# print('\nRandom state: x =', M[0][x])
# 
# # Last state
# print('Last state: x =', M[0][-1])
# 
# # Actions
# print('\n= Action space (%i actions) =\n' % len(M[1]))
# for i in range(len(M[1])):
#     print(M[1][i])
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('\nRandom action: a =', M[1][a])
# 
# # Transition probabilities
# print('\n= Transition probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[2][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[2][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][x], M[1][a]))
# print("x' in", np.array(M[0])[np.where(M[2][a][x, :] > 0)])
# 
# # Cost
# print('\n= Costs =')
# print('\nCost for the state-action pair (%s, %s):' % (M[0][x], M[1][a]))
# print(x,a)
# print('c(x, a) =', M[3][x, a])
# 
# 
# # Discount
# print('\n= Discount =')
# print('\ngamma =', M[4])
# ```
# 
# Output
# ```
# = State space (100 states) =
# 
# States:
# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# ...
# 
# Random state: x = 51
# Last state: x = 99
# 
# = Action space (2 actions) =
# 
# St
# Go
# 
# Random action: a = St
# 
# = Transition probabilities =
# 
# Transition probability matrix dimensions (action St): (100, 100)
# Dimensions add up for action "St"? True
# 
# Transition probability matrix dimensions (action Go): (100, 100)
# Dimensions add up for action "Go"? True
# 
# State-action pair (51, St) transitions to state(s)
# x' in ['55']
# 
# = Costs =
# 
# Cost for the state-action pair (51, St):
# 51 0
# c(x, a) = 1.0
# 
# = Discount =
# 
# gamma = 0.9
# ```

# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# Write a function `noisy_policy` that builds a noisy policy "around" a provided action. Your function should receive, as input, an MDP described as a tuple like that of **Activity 1**, an integer `a`, corresponding to the _index_ of an action in the MDP, and a real number `eps`. The function should return, as output, a policy for the provided MDP that selects action with index `a` with a probability `1 - eps` and, with probability `eps`, selects another action uniformly at random. The policy should be a `numpy` array with as many rows as states and as many columns as actions, where the element in position `[x, a]` should contain the probability of action `a` in state `x` according to the desired policy.
# 
# **Note:** The examples provided correspond for the MDP in the previous environment. However, your code should be tested with MDPs of different sizes, so **make sure not to hard-code any of the MDP elements into your code**.
# 
# ---

def noisy_policy(mdp, a, eps):
    """
    Builds a noisy policy around action a for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :param a: main action for the policy
    :type: integer
    :param eps: noise level
    :type: float
    :return: nd.array
    """

    num_states = len(mdp[0])  # Number of states
    num_actions = len(mdp[1])  # Number of actions

    pol = np.zeros((num_states, num_actions))

    for x in range(num_states):
        pol[x, a] = 1 - eps
        for other_a in range(num_actions):
            if other_a != a:
                pol[x, other_a] = eps / (num_actions - 1)

    return pol

# -- End: noisy_policy

# Noiseless policy for action "Go" (action index: 1)
pol_noiseless = noisy_policy(M, 1, 0.)

# Arbitrary state
x = 50 # State

# Policy at selected state
print('Arbitrary state (from previous example):', M[0][x])
print('Noiseless policy at selected state (eps = 0):', pol_noiseless[x, :])

# Noisy policy for action "Go" (action index: 1)
pol_noisy = noisy_policy(M, 1, 0.1)

# Policy at selected state
print('Noisy policy at selected state (eps = 0.1):', np.round(pol_noisy[x, :], 2))

# Random policy for action "Go" (action index: 1)
pol_random = noisy_policy(M, 1, 0.75)

# Policy at selected state
print('Random policy at selected state (eps = 0.75):', np.round(pol_random[x, :], 2))

# We provide below an example of application of the function with MDP from the example in **Activity 2**, that you can use as a first "sanity check" for your code. Note that, as emphasized above, your function should work with **any** MDP that is specified as a tuple with the structure of the one from **Activity 2**.
# 
# ```python
# # Noiseless policy for action "Go" (action index: 1)
# pol_noiseless = noisy_policy(M, 1, 0.)
# 
# # Arbitrary state
# x = 50 # State
# 
# # Policy at selected state
# print('Arbitrary state (from previous example):', M[0][x])
# print('Noiseless policy at selected state (eps = 0):', pol_noiseless[x, :])
# 
# # Noisy policy for action "Go" (action index: 1)
# pol_noisy = noisy_policy(M, 1, 0.1)
# 
# # Policy at selected state
# print('Noisy policy at selected state (eps = 0.1):', np.round(pol_noisy[x, :], 2))
# 
# # Random policy for action "Go" (action index: 1)
# pol_random = noisy_policy(M, 1, 0.75)
# 
# # Policy at selected state
# print('Random policy at selected state (eps = 0.75):', np.round(pol_random[x, :], 2))
# ```
# 
# Output:
# 
# ```
# Arbitrary state (from previous example): 50
# Noiseless policy at selected state (eps = 0): [0. 1.]
# Noisy policy at selected state (eps = 0.1): [0.1 0.9]
# Random policy at selected state (eps = 0.75): [0.75 0.25]
# ```

# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# **Note:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. Note also that, as before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**. In your solution, you may find useful the function `np.linalg.inv`, which can be used to invert a matrix.
# 
# ---

def evaluate_pol(mdp, policy):
    """
    Computes the cost-to-go function for a given policy in a given MDP.

    :param mdp: MDP description
    :type: tuple
    :param pol: Policy to be evaluated
    :type: nd.array
    :returns: nd.array
    """
    X, A, P, c, gamma = mdp
    num_states = len(X)
    num_actions = len(A)
    
    P_pi = np.zeros((num_states, num_states))
    for a in range(num_actions):
        P_pi += np.diag(policy[:, a]) @ P[a]

    c_pi = np.sum(policy * c, axis=1, keepdims=True)

    # J^π = (I - γ P^π)^(-1) c^π
    J_pi = np.linalg.inv(np.eye(num_states) - gamma * P_pi) @ c_pi

    return J_pi

# -- End: evaluate

Jact2 = evaluate_pol(M, pol_noisy)

print('Dimensions of cost-to-go:', Jact2.shape)

print('\nExample values of the computed cost-to-go:')

x = 0 # State (0, 0)
print('\nCost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))

x = 50 # State (5, 0)
print('Cost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))

x = 55 # State (5, 5)
print('Cost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))

# Example with random policy

rand.seed(42)

rand_pol = rand.randint(2, size=(len(M[0]), len(M[1]))) + 0.01 # We add 0.01 to avoid all-zero rows
rand_pol = rand_pol / rand_pol.sum(axis = 1, keepdims = True)

Jrand = evaluate_pol(M, rand_pol)

print('\nExample values of the computed cost-to-go:')

x = 0 # State (0, 0)
print('\nCost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))

x = 50 # State (5, 0)
print('Cost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))

x = 80 # State (8, 0)
print('Cost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))

# As an example, you can evaluate the random policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jact2 = evaluate_pol(M, pol_noisy)
# 
# print('Dimensions of cost-to-go:', Jact2.shape)
# 
# print('\nExample values of the computed cost-to-go:')
# 
# x = 0 # State (0, 0)
# print('\nCost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))
# 
# x = 50 # State (5, 0)
# print('Cost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))
# 
# x = 55 # State (5, 5)
# print('Cost-to-go at state %s:' % M[0][x], np.round(Jact2[x], 3))
# 
# # Example with random policy
# 
# rand.seed(42)
# 
# rand_pol = rand.randint(2, size=(len(M[0]), len(M[1]))) + 0.01 # We add 0.01 to avoid all-zero rows
# rand_pol = rand_pol / rand_pol.sum(axis = 1, keepdims = True)
# 
# Jrand = evaluate_pol(M, rand_pol)
# 
# print('\nExample values of the computed cost-to-go:')
# 
# x = 0 # State (0, 0)
# print('\nCost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))
# 
# x = 50 # State (5, 0)
# print('Cost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))
# 
# x = 80 # State (8, 0)
# print('Cost-to-go at state %s:' % M[0][x], np.round(Jrand[x], 3))
# ```
# 
# Output:
# ```
# Dimensions of cost-to-go: (100, 1)
# 
# Example values of the computed cost-to-go:
# 
# Cost-to-go at state 0: [9.521]
# Cost-to-go at state 50: [9.259]
# Cost-to-go at state 55: [9.039]
# 
# Example values of the computed cost-to-go:
# 
# Cost-to-go at state 0: [9.73]
# Cost-to-go at state 50: [9.418]
# Cost-to-go at state 80: [9.208]
# ```

# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$.
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$. To compute the error between iterations, you should use the function `norm` from `numpy.linalg`.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``. You may also find useful the code provided in the theoretical lecture.
# 
# **Note 3:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. As before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**.
# 
# 
# ---

import time

def value_iteration(mdp):
    """
    Computes the optimal cost-to-go function for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :returns: nd.array
    """

    X, A, P, c, gamma = mdp
    num_states = len(X)
    num_actions = len(A)

    J = np.zeros(num_states)

    epsilon = 1e-8 
    iteration = 0
    error = 1.0

    start_time = time.time()

    while error >= epsilon :
        J_prev = J.copy()
        for x in range(num_states):

            J[x] = min([c[x, a] + gamma * np.sum(P[a][x, :] * J_prev) for a in range(num_actions)])

        error = np.linalg.norm(J - J_prev)
        iteration += 1

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"N. iterations: {iteration}")

    return J.reshape(-1, 1)

# -- End: value_iteration

Jopt = value_iteration(M)

print('\nDimensions of cost-to-go:', Jopt.shape)

print('\nExample values of the optimal cost-to-go:')

x = 0 # State (0, 0)
print('\nCost to go at state %s:' % M[0][x], Jopt[x])

x = 50 # State (5, 0)
print('Cost to go at state %s:' % M[0][x], Jopt[x])

x = 55 # State (5, 5)
print('Cost to go at state %s:' % M[0][x], Jopt[x])

print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jact2)))

# For example, using the MDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# Jopt = value_iteration(M)
# 
# print('\nDimensions of cost-to-go:', Jopt.shape)
# 
# print('\nExample values of the optimal cost-to-go:')
# 
# x = 0 # State (0, 0)
# print('\nCost to go at state %s:' % M[0][x], Jopt[x])
# 
# x = 50 # State (5, 0)
# print('Cost to go at state %s:' % M[0][x], Jopt[x])
# 
# x = 55 # State (5, 5)
# print('Cost to go at state %s:' % M[0][x], Jopt[x])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jact2)))
# ```
# 
# Output:
# ```
# Execution time: 0.179 seconds
# N. iterations: 197
# 
# Dimensions of cost-to-go: (100, 1)
# 
# Example values of the optimal cost-to-go:
# 
# Cost to go at state 0: [9.36120013]
# Cost to go at state 50: [9.02014276]
# Cost to go at state 55: [8.91126974]
# 
# Is the policy from Activity 2 optimal? False
# ```

# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Consider the initial policy is the uniformly random policy. Before returning, your function should print:
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$). You may also find useful the code provided in the theoretical lecture.
# 
# ---

def policy_iteration(mdp):
    """
    Computes the optimal policy for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :returns: nd.array
    """
    start_time = time.time()
    
    X, A, P, c, gamma = mdp

    pol = np.ones((len(X), len(A))) / len(A)  
    iterations = 0

    while True:
        iterations += 1
        J_pi = evaluate_pol(mdp, pol)  # Pol eval
        
        Q = np.zeros((len(X), len(A)))
        for a in range(len(A)):
            Q[:, a] = c[:, a] + gamma * (P[a] @ J_pi).flatten()
        
        new_policy = np.zeros((len(X), len(A)))
        best_actions = np.argmin(Q, axis=1)
        new_policy[np.arange(len(X)), best_actions] = 1
        
        if np.all(np.isclose(pol, new_policy, atol=1e-8)):
            break
        pol = new_policy
    
    execution_time = round(time.time() - start_time, 3)
    
    print(f"Execution time: {execution_time} seconds")
    print(f"N. iterations: {iterations}")
    
    return pol



# -- End: policy_iteration

popt = policy_iteration(M)

print('\nDimension of the policy matrix:', popt.shape)

rand.seed(42)

print('\nExamples of actions according to the optimal policy:')

# Select random state, and action using the policy computed
x = 0 # State (0, 0)
a = rand.choice(len(M[1]), p=popt[x, :])
print('Policy at state %s: %s' % (M[0][x], M[1][a]))

# Select random state, and action using the policy computed
x = 50 # State (5, 0)
a = rand.choice(len(M[1]), p=popt[x, :])
print('Policy at state %s: %s' % (M[0][x], M[1][a]))

# Select random state, and action using the policy computed
x = 55 # State (5, 5)
a = rand.choice(len(M[1]), p=popt[x, :])
print('Policy at state %s: %s' % (M[0][x], M[1][a]))

# Verify optimality of the computed policy

print('\nOptimality of the computed policy:')

Jpi = evaluate_pol(M, popt)
print('- Is the new policy optimal?', np.all(np.isclose(Jopt, Jpi)))

# For example, using the MDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# popt = policy_iteration(M)
# 
# print('\nDimension of the policy matrix:', popt.shape)
# 
# rand.seed(42)
# 
# print('\nExamples of actions according to the optimal policy:')
# 
# # Select random state, and action using the policy computed
# x = 0 # State (0, 0)
# a = rand.choice(len(M[1]), p=popt[x, :])
# print('Policy at state %s: %s' % (M[0][x], M[1][a]))
# 
# # Select random state, and action using the policy computed
# x = 50 # State (5, 0)
# a = rand.choice(len(M[1]), p=popt[x, :])
# print('Policy at state %s: %s' % (M[0][x], M[1][a]))
# 
# # Select random state, and action using the policy computed
# x = 55 # State (5, 5)
# a = rand.choice(len(M[1]), p=popt[x, :])
# print('Policy at state %s: %s' % (M[0][x], M[1][a]))
# 
# # Verify optimality of the computed policy
# 
# print('\nOptimality of the computed policy:')
# 
# Jpi = evaluate_pol(M, popt)
# print('- Is the new policy optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.005 seconds
# N. iterations: 4
# 
# Dimension of the policy matrix: (100, 2)
# 
# Examples of actions according to the optimal policy:
# Policy at state 0: Go
# Policy at state 50: St
# Policy at state 55: Go
# 
# Optimality of the computed policy:
# - Is the new policy optimal? True
# ```

# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, `x0`, corresponding to a state index
# * A second integer, `length`
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **`NRUNS`** trajectories of `length` steps each, starting in the provided state and following the provided policy.
# * For each trajectory, compute the accumulated (discounted) cost.
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---


NRUNS = 100 # Do not delete this

def simulate(mdp, pol, x0, length=10000):
    """
    Estimates the cost-to-go for a given MDP, policy and state.

    :param mdp: MDP description
    :type: tuple
    :param pol: policy to be simulated
    :type: nd.array
    :param x0: initial state
    :type: int
    :returns: float
    """
    X, A, P, c, gamma = mdp
    total_costs = []

    for _ in range(NRUNS):
        state = x0
        total_cost = 0
        
        for t in range(length):

            a = np.random.choice(len(A), p=pol[state, :])
            total_cost += (gamma ** t) * c[state, a] 
            state = np.random.choice(len(X), p=P[a][state, :] ) 
        
        total_costs.append(total_cost)

    return np.mean(total_costs)


# -- End: simulate

rand.seed(42)

# Select arbitrary state, and evaluate for the optimal policy
x = 0 # State (0, 0)
print('Cost-to-go for state %s:' % M[0][x])
print('\tTheoretical:', np.round(Jopt[x], 4))
print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))

# Select arbitrary state, and evaluate for the optimal policy
x = 10 # State (1, 0)
print('Cost-to-go for state %s:' % M[0][x])
print('\tTheoretical:', np.round(Jopt[x], 4))
print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))

# Select arbitrary state, and evaluate for the optimal policy
x = 65 # State (6, 5)
print('Cost-to-go for state %s:' % M[0][x])
print('\tTheoretical:', np.round(Jopt[x], 4))
print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))

# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# rand.seed(42)
# 
# # Select arbitrary state, and evaluate for the optimal policy
# x = 0 # State (0, 0)
# print('Cost-to-go for state %s:' % M[0][x])
# print('\tTheoretical:', np.round(Jopt[x], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))
# 
# # Select arbitrary state, and evaluate for the optimal policy
# x = 10 # State (1, 0)
# print('Cost-to-go for state %s:' % M[0][x])
# print('\tTheoretical:', np.round(Jopt[x], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))
# 
# # Select arbitrary state, and evaluate for the optimal policy
# x = 65 # State (6, 5)
# print('Cost-to-go for state %s:' % M[0][x])
# print('\tTheoretical:', np.round(Jopt[x], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, x, 10000), 4))
# ```
# 
# Output:
# ```
# Cost-to-go for state 0:
# 	Theoretical: [9.3612]
# 	Empirical: 9.4104
# Cost-to-go for state 10:
# 	Theoretical: [9.3206]
# 	Empirical: 9.3311
# Cost-to-go for state 65:
# 	Theoretical: [8.8534]
# 	Empirical: 8.9033
# ```


