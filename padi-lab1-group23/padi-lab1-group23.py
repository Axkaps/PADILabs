
# # Learning and Decision Making


# ## Laboratory 1: Markov chains
# 
# In the end of the lab, you should export the notebook to a Python script (``File >> Download as >> Python (.py)``). Make sure that the resulting script includes all code written in the tasks marked as "**Activity n. N**", together with any replies to specific questions posed. Your file should be named `padi-labKK-groupXXX.py`, where `KK` corresponds to the lab number and the `XXX` corresponds to your group number. Similarly, your homework should consist of a single pdf file named `padi-hwKK-groupXXX.pdf`. You should create a zip file with the lab and homework files and submit it in Fenix **at most 30 minutes after your lab is over**.
# 
# Make sure to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The Markov chain model
# We will use a simplified game inspired in the games StopIT and Insey-Winsey-Spider.
# 
# The players have several levels to climb (corresponding to steps in a ladder) and want to reach the top level.
# 
# At each instant the player can decide to go and they throw a dice.
# After that the player has the possibility to climb a number of steps. But they will only go up if it is a sunny day, if it is a rainy day then they go back to the last safe level.
# At each instant there is also the option to stop. This makes the current level a safe one.
# Once the last step is reached the game is won.
# A stop action will stop again but a go action will reset the game to the initial state corresponding to the level 0 and safe level 0.
# 
# In this first activity, you will implement your Markov chain model in Python. You will start by loading the transition probability matrix from a `numpy` binary file, using the `numpy` function `load`. You will then consider the state space to consist of all valid indices for the loaded transition matrix, each represented as a string. For example, if the transition probability matrix is $20\times 20$, the states should include the strings `'0'` to `'19'`.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_chain` that receives, as input, a string corresponding to the name of the file with a transition matrix to be loaded, and a real number $\gamma$ between $0$ and $1$. Assume that:
# 
# * The transition matrices in the file have been built from a representation of the game with P[0] corresponding to the transition of the action stop and P[1] for the action go.
# 
# * For this first lab we do not consider the process of choosing action so we consider that the action are choosen at random with the action go selected with probability $\gamma$ .
# 
# Your function should build the transition probability matrix for the chain by combining the two actions using the value of $\gamma$. Your function should then return, as output, a two-element tuple corresponding to the Markov chain, where:
# 
# * ... the first element is a tuple containing an enumeration of the state-space (i.e., each element of the tuple corresponds to a state of the chain, represented as a string);
# * ... the second element is a `numpy` array corresponding to the transition probability matrix for the chain.
# 
# ---

# %%
import numpy as np
# Add your code here.
def load_chain(filename: str, gamma: float) -> tuple[tuple, np.ndarray]:
    
    markovChain = np.load(filename)
    stateSpace = tuple(str(i) for i in range(markovChain.shape[1]))
    MarkovChain = (1 - gamma) * markovChain[0] + gamma * markovChain[1]
    return (stateSpace, MarkovChain)

# Mgo = load_chain("StopITSpider02.npy", 0.5)
# print("Number of states: ", len(Mgo[0]))
# np.random.seed(42)
# x = np.random.randint(len(Mgo[0]))
# print("Random state: ", Mgo[0][x])
# print("Transition probabilities in random state: ")
# print(Mgo[1][x, :])

# We provide below an example of application of the function that you can use as a first "sanity check" for your code. Note, however, that the fact that you can replicate the examples below is not indicative that your code is correct. Moreover, your code will be tested with networks of different sizes, so **make sure not to hard-code the size of the environments into your code**.
# 
# 
# ```python
# print('- Mgo - always select go -')
# Mgo = load_chain('StopITSpider02.npy', 1)
# print('Number of states:', len(Mgo[0]))
# print('Transition probabilities:')
# print(Mgo[1])
# 
# import numpy.random as rand
# 
# rand.seed(42)
# 
# print('- Mgostop - select go half of the time -')
# Mgostop = load_chain('StopITSpider02.npy', 0.5)
# print('Number of states:', len(Mgostop[0]))
# x = rand.randint(len(Mgostop[0]))
# print('Random state:', Mgostop[0][x])
# print('Transition probabilities in random state:')
# print(Mgostop[1][x, :])
# ```
# 
# Output:
# ```
# - Mgo - always select go -
# Number of states: 100
# Transition probabilities:
# [[0.2 0.  0.  ... 0.  0.  0. ]
#  [0.  0.  0.  ... 0.  0.  0. ]
#  [0.  0.  0.  ... 0.  0.  0. ]
#  ...
#  [1.  0.  0.  ... 0.  0.  0. ]
#  [1.  0.  0.  ... 0.  0.  0. ]
#  [1.  0.  0.  ... 0.  0.  0. ]]
# - Mgostop - select go half of the time -
# Number of states: 100
# Random state: 51
# Transition probabilities in random state:
# [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.1 0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.5 0.  0.  0.  0.  0.  0.1 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.1
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.1 0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.1 0.  0.  0.  0.  0.  0.  0.  0. ]
# ```
# 

# The following functions might be useful to convert the state representation. For instance the state with the index 10 can also be represented as 1(0) meaning that the player is in the level 1 with safe level 0. State index 22 corresponds to being in level 2 with safe level 2.
# 

# Auxiliary function to convert state representation to state index
Lm = 10

def observ2state(observ):
    dimsizes = [Lm, Lm]
    return np.ravel_multi_index(observ, dimsizes)

# Auxiliary function to convert state index to state representation
def state2observ(state):
    dimsizes = [Lm, Lm]
    return np.unravel_index(int(state), dimsizes)

# Auxiliary function to print a sequence of states
def printTraj(seq):
    ss = ""
    for st in seq:
        ss += printState(st) + "\n"

    return ss

# Auxiliary function to print a state
def printState(state):
    if type(state) in [list,tuple]:
        l = state[0]
        s = state[1]
    else:
        l,s = state2observ(state)

    return "%d (%d)" % (l, s)

print(10, state2observ('10'))
print(22, state2observ('22'))

# In the next activity, you will use the Markov chain model to evaluate the likelihood of any given path for the player.
# 
# ---
# 
# #### Activity 2.
# 
# Write a function `prob_trajectory` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a trajectory, corresponding to a sequence of states (i.e., a tuple or list of strings, each string corresponding to a state).
# 
# Your function should return, as output, a floating point number corresponding to the probability of observing the provided trajectory, taking the first state in the trajectory as initial state.  
# 
# ---

# Add your code here.
def prob_trajectory(markov_chain: tuple[tuple, np.ndarray], trajectory):

    states, transition_matrix = markov_chain
    probability = 1.0

    # Iterar os elementos da trajetória
    for i in range(len(trajectory) - 1):
        current_state = trajectory[i]
        next_state = trajectory[i + 1]

        # index atual e proximo index
        current_index = states.index(current_state)
        next_index = states.index(next_state)

        # ir buscar a probabilidade de transição
        probability *= transition_matrix[current_index, next_index]

    return probability

# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# print('- Mgo - always select go -')
# print("Prob. of trajectory 0(0)-2(0)-4(0):", prob_trajectory(Mgo, ('0', '20', '40')))
# print("Prob. of trajectory 0(0)-0(0)-2(0):", prob_trajectory(Mgo, ('0', '20', '40')))
# print("Prob. of trajectory 0(0)-2(0)-2(2):", prob_trajectory(Mgo, ('0', '20', '22')))
# print("Prob. of trajectory 0(0)-2(0)-4(0)-4(4)-6(4):", prob_trajectory(Mgo, ('0', '20', '40', '44','64')))
# print("Prob. of trajectory 6(0)-8(0)-0(0):", prob_trajectory(Mgo, ('60','80','0')))
# 
# print('- Mgostop - select go half of the time -')
# print("Prob. of trajectory 0(0)-2(0)-4(0):", prob_trajectory(Mgostop, ('0', '20', '40')))
# print("Prob. of trajectory 0(0)-0(0)-2(0):", prob_trajectory(Mgostop, ('0', '20', '40')))
# print("Prob. of trajectory 0(0)-2(0)-2(2):", prob_trajectory(Mgostop, ('0', '20', '22')))
# print("Prob. of trajectory 0(0)-2(0)-4(0)-4(4)-6(4):", prob_trajectory(Mgostop, ('0', '20', '40', '44','64')))
# print("Prob. of trajectory 6(0)-8(0)-0(0):", prob_trajectory(Mgostop, ('60','80','0')))
# ```
# 
# Output:
# ```
# - Mgo - always select go -
# Prob. of trajectory 0(0)-2(0)-4(0): 0.04000000000000001
# Prob. of trajectory 0(0)-0(0)-2(0): 0.04000000000000001
# Prob. of trajectory 0(0)-2(0)-2(2): 0.0
# Prob. of trajectory 0(0)-2(0)-4(0)-4(4)-6(4): 0.0
# Prob. of trajectory 6(0)-8(0)-0(0): 0.04000000000000001
# - Mgostop - select go half of the time -
# Prob. of trajectory 0(0)-2(0)-4(0): 0.010000000000000002
# Prob. of trajectory 0(0)-0(0)-2(0): 0.010000000000000002
# Prob. of trajectory 0(0)-2(0)-2(2): 0.05
# Prob. of trajectory 0(0)-2(0)-4(0)-4(4)-6(4): 0.0005000000000000001
# Prob. of trajectory 6(0)-8(0)-0(0): 0.010000000000000002
# ```
# 
# Note that your function should work with **any** Markov chain that is specified as a tuple like the one from Activity 1.

# ### 2. Stability

# The next activities explore the notion of *stationary distribution* for the chain.
# 
# ---
# 
# #### Activity 3
# 
# Write a function `stationary_dist` that receives, as input, a Markov chain in the form of a tuple like the one returned by the function in Activity 1. Your function should return, as output, a `numpy` array corresponding to a row vector containing the stationary distribution for the chain.
# 
# **Note:** The stationary distribution is a *left* eigenvector of the transition probability matrix associated to the eigenvalue 1. As such, you may find useful the numpy function `numpy.linalg.eig`. Also, recall that the stationary distribution is *a distribution*. You may also find useful the function `numpy.real` which returns the real part of a complex number.
# 
# ---

# Add your code here.
def stationary_dist(MarkovChain) -> np.ndarray:
    # Unwrap
    states, transitionMatrix = MarkovChain

    eigenvalues, left_eigenvectors = np.linalg.eig(transitionMatrix.T)

    ids = np.argmin(np.abs(eigenvalues - 1)) 
    stationary_vector = np.real(left_eigenvectors[:, ids])

    stationary_distribution = stationary_vector / np.sum(stationary_vector)
    return stationary_distribution.reshape(1, -1)

# Mgo = load_chain("StopITSpider02.npy", 1)
# uStar = stationary_dist(Mgo)
# print(np.round(uStar, 2))

# print(" ---- Mgo Go - Always select go -----")
# uPrime = uStar.dot(Mgo[1])
# print("Is this a distribution?", np.isclose(np.sum(uStar), 1))
# print("Most likely state: ", Mgo[0][np.argmax(uStar)])
# print("probability of being in the last level: ", np.sum(uStar[0][90:]))

# print("\nIs u* * P = u*?", np.all(np.isclose(uPrime, uStar)))

# print(" ---- Mgo Go - select go half of the time -----")
# MgoStop = load_chain("StopITSpider02.npy", 0.5)
# uStar = stationary_dist(MgoStop)
# print(np.round(uStar, 2))

# uPrime = uStar.dot(MgoStop[1])
# print("Is this a distribution?", np.isclose(np.sum(uStar), 1))
# print("Most likely state: ", MgoStop[0][np.argmax(uStar)])
# print("probability of being in the last level: ", np.sum(uStar[0][90:]))

# print("\nIs u* * P = u*?", np.all(np.isclose(uPrime, uStar)))

# Example of application of the function.
# 
# ```python
# print('- Mgo - always select go -')
# u_star = stationary_dist(Mgo)
# 
# print('Stationary distribution:')
# print(np.round(u_star, 2))
# 
# u_prime = u_star.dot(Mgo[1])
# 
# print('Is this a distribution?', np.isclose(np.sum(u_star), 1))
# 
# print('Most likely state:', Mgo[0][np.argmax(u_star)])
# print('probability of being in the last level:', np.sum(u_star[0][90:]))
# 
# 
# print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))
# 
# print('- Mgostop - select go half of the time -')
# 
# u_star = stationary_dist(Mgostop)
# print(np.round(u_star, 2))
# 
# print('Is this a distribution?', np.isclose(np.sum(u_star), 1))
# 
# print('Most likely state:', Mgostop[0][np.argmax(u_star)])
# print('probability of being in the last level:', np.sum(u_star[0][90:]))
# 
# u_prime = u_star.dot(Mgostop[1])
# print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))
# ```
# 
# Output:
# ```
# - Mgo - always select go -
# Stationary distribution:
# [[ 0.3   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.06 -0.
#    0.    0.    0.    0.    0.    0.    0.    0.    0.07 -0.    0.    0.
#    0.    0.    0.    0.    0.    0.    0.09 -0.    0.    0.    0.    0.
#    0.    0.    0.    0.    0.1  -0.    0.    0.   -0.    0.    0.    0.
#    0.    0.    0.06 -0.    0.    0.   -0.    0.    0.    0.    0.    0.
#    0.07 -0.    0.    0.   -0.    0.    0.    0.    0.    0.    0.06 -0.
#    0.    0.   -0.    0.    0.    0.    0.    0.    0.06 -0.    0.    0.
#   -0.    0.    0.    0.    0.    0.    0.12 -0.    0.    0.   -0.    0.
#    0.    0.    0.    0.  ]]
# Is this a distribution? True
# Most likely state: 0
# probability of being in the last level: 0.12493248575731676
# 
# Is u* * P = u*? True
# - Mgostop - select go half of the time -
# [[ 0.24 -0.   -0.   -0.   -0.   -0.   -0.   -0.   -0.   -0.    0.02  0.04
#   -0.   -0.   -0.   -0.   -0.   -0.   -0.   -0.    0.03  0.    0.04 -0.
#   -0.   -0.   -0.   -0.   -0.   -0.    0.03  0.    0.    0.05 -0.   -0.
#   -0.   -0.   -0.   -0.    0.03  0.    0.    0.01  0.07 -0.   -0.   -0.
#   -0.   -0.    0.01  0.    0.01  0.01  0.01  0.05 -0.   -0.   -0.   -0.
#    0.01  0.    0.01  0.01  0.01  0.    0.05 -0.   -0.   -0.    0.01  0.
#    0.    0.01  0.01  0.01  0.    0.05 -0.   -0.    0.01  0.    0.    0.
#    0.01  0.01  0.01  0.    0.05 -0.    0.01  0.    0.    0.01  0.01  0.01
#    0.01  0.02  0.02 -0.  ]]
# Is this a distribution? True
# Most likely state: 0
# probability of being in the last level: 0.08293059995179561
# 
# Is u* * P = u*? True
# ```

# To complement Activity 3, you will now empirically establish that the chain is ergodic, i.e., no matter where the player starts, its visitation frequency will eventually converge to the stationary distribution.
# 
# ---
# 
# #### Activity 4.
# 
# Write a function `compute_dist` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a numpy array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the chain is expected to take.
# 
# Your function should return, as output, a row vector (a `numpy` array) containing the distribution after $N$ steps of the chain. Use your function to justify that the chain is ergodic.
# 
# ---

# Add your code here.
def compute_dist(markov_chain, initial_distribution, N):

    states, transition_matrix = markov_chain

    # destribution dps de N steps
    distribution = initial_distribution @ np.linalg.matrix_power(transition_matrix, N)

    return distribution

# <font color='cyan'>
# The Markov chain is ergodic because it satisfies the two key properties required for ergodicity: irreducibility and aperiodicity. We observed that the distribution of the Markov chain converges to u * after a large number of steps. This convergence occurs regardless of the initial distribution, confirming that the chain is aperiodic. We also observed that the distribution of the Markov chain converges to the same stationary distribution u* 
# regardless of the initial distribution. This implies that all states communicate with each other, confirming that the chain is irreducible.
# </font>

# Example of application of the function.
# 
# ```python
# import numpy.random as rnd
# 
# rnd.seed(42)
# 
# REPETITIONS = 5
# 
# print('- Mgo - always select go -')
# 
# # Number of states
# nS = len(Mgo[0])
# u_star = stationary_dist(Mgo)
# 
# # Repeat a number of times
# for n in range(REPETITIONS):
# 
#     print('\n- Repetition', n + 1, 'of', REPETITIONS, '-')
# 
#     # Initial random distribution
#     u = rnd.random((1, nS))
#     u = u / np.sum(u)
# 
#     # Distrbution after 10 steps
#     v = compute_dist(Mgo, u, 100)
#     print('Is u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# 
#     # Distrbution after 100 steps
#     v = compute_dist(Mgo, u, 200)
#     print('Is u * P^2000 = u*?', np.all(np.isclose(v, u_star)))
# 
# print('- Mgostop - select go half of the time -')
# 
# # Number of states
# nS = len(Mgostop[0])
# u_star = stationary_dist(Mgostop)
# 
# # Repeat a number of times
# for n in range(REPETITIONS):
# 
#     print('\n- Repetition', n + 1, 'of', REPETITIONS, '-')
# 
#     # Initial random distribution
#     u = rnd.random((1, nS))
#     u = u / np.sum(u)
# 
#     # Distrbution after 100 steps
#     v = compute_dist(Mgostop, u, 100)
#     print('Is u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# 
#     # Distrbution after 2000 steps
#     v = compute_dist(Mgostop, u, 200)
#     print('Is u * P^2000 = u*?', np.all(np.isclose(v, u_star)))
# ```
# 
# Output:
# ````
# - Mgo - always select go -
# 
# - Repetition 1 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 2 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 3 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 4 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 5 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# - Mgostop - select go half of the time -
# 
# - Repetition 1 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 2 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 3 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 4 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 5 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# ```

# ### 3. Simulation
# 
# In this part of the lab, you will *simulate* the actual player, and empirically compute the visitation frequency of each state.

# %% [markdown]
# ---
# 
# #### Activity 5
# 
# Write down a function `simulate` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a `numpy` array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the chain is expected to take.
# 
# Your function should return, as output, a tuple containing a trajectory with $N$ states, where the initial state is sampled according to the initial distribution provided. Each element in the tuple should be a string corresponding to a state index.
# 
# ---
# 
# **Note:** You may find useful to import the numpy module `numpy.random`.

# %%
import numpy.random as rnd
rnd.seed(42)
# Add your code here.
def simulate(markovChain: tuple, initialDistribution: np.ndarray, N: int) -> tuple:
    nStates = len(markovChain[0])
    # Sample the initial state based on the initial distribution
    currentStateID = np.random.choice(nStates, p=initialDistribution.flatten())
    trajectory = [markovChain[0][currentStateID]]  # Store the state name

    # Simulate N steps of the Markov chain
    for _ in range(N - 1):
        currentStateID = np.random.choice(nStates, p=markovChain[1][currentStateID])
        trajectory.append(markovChain[0][currentStateID])

    return tuple(trajectory)


rnd.seed(42)
Mgo = load_chain("StopITSpider02.npy", 1)
Mgostop = load_chain("StopITSpider02.npy", 0.5)

print('- Mgo - always select go -')

# Number of states
nS = len(Mgo[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

# Simulate short trajectory
traj = simulate(Mgo, u, 10)
print('Small trajectory:', traj)

# Simulate a long trajectory
traj = simulate(Mgo, u, 10000)
print('End of large trajectory:', traj[-10:])

print('- Mgostop - select go half of the time -')

# Number of states
nS = len(Mgostop[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

# Simulate short trajectory
traj = simulate(Mgostop, u, 10)
print('Small trajectory:', traj)

# Simulate a long trajectory
traj = simulate(Mgostop, u, 10000)
print('End of large trajectory:', traj[-10:])

# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# import numpy.random as rnd
# 
# rnd.seed(42)
# 
# print('- Mgo - always select go -')
# 
# # Number of states
# nS = len(Mgo[0])
# 
# # Initial, uniform distribution
# u = np.ones((1, nS)) / nS
# 
# # Simulate short trajectory
# traj = simulate(Mgo, u, 10)
# print('Small trajectory:', traj)
# 
# # Simulate a long trajectory
# traj = simulate(Mgo, u, 10000)
# print('End of large trajectory:', traj[-10:])
# 
# print('- Mgostop - select go half of the time -')
# 
# # Number of states
# nS = len(Mgostop[0])
# 
# # Initial, uniform distribution
# u = np.ones((1, nS)) / nS
# 
# # Simulate short trajectory
# traj = simulate(Mgostop, u, 10)
# print('Small trajectory:', traj)
# 
# # Simulate a long trajectory
# traj = simulate(Mgostop, u, 10000)
# print('End of large trajectory:', traj[-10:])
# ```
# 
# Output:
# ```
# - Mgo - always select go -
# Small trajectory: ('37', '77', '97', '0', '0', '0', '0', '40', '70', '90')
# End of large trajectory: ('20', '30', '0', '30', '50', '90', '0', '30', '60', '90')
# - Mgostop - select go half of the time -
# Small trajectory: ('88', '98', '0', '0', '0', '0', '0', '0', '0', '0')
# End of large trajectory: ('22', '22', '22', '22', '22', '52', '72', '92', '0', '0')
# ```
# 
# Note that, even if the seed is fixed, it is possible that your trajectories are slightly different.

# ---
# 
# #### Activity 6
# 
# We will now compare the relative speed of two chains.
# Create two chains, one where we always choose Go and another where we choose Go 3/4 of the time and Stop 1/4 of the time.
# 
# Which one is faster? Verify using one sampling approach, and one analytical approach.
# 
# Is the best way to choose the action the same for the game with 20% rainy days ('StopITSpider02.npy') and the game with 40% rainy days?.
# 
# ---

# Add your code here.
MgoAlways = load_chain("StopITSpider02.npy", 1)
MgoAlmostAlways = load_chain("StopITSpider02.npy", 0.75)

MgoAlways2 = load_chain("StopITSpider04.npy", 1)
MgoAlmostAlways2 = load_chain("StopITSpider04.npy", 0.75)

uStarAlways02 = stationary_dist(MgoAlways)
uStarAlmostAlways02 = stationary_dist(MgoAlmostAlways)
uStarAlways04 = stationary_dist(MgoAlways2)
uStarAlmostAlways04 = stationary_dist(MgoAlmostAlways2)


timeAtGoalStatesAlways02 = np.sum(uStarAlways02[0][90:])
timeAtGoalStatesAlmostAlways02 = np.sum(uStarAlmostAlways02[0][90:])
timeAtGoalStatesAlways04 = np.sum(uStarAlways04[0][90:])
timeAtGoalStatesAlmostAlways04 = np.sum(uStarAlmostAlways04[0][90:])


print("Spider02 - gamma = 1:    ", timeAtGoalStatesAlways02)
print("Spider02 - gamma = 0.75: ", timeAtGoalStatesAlmostAlways02)
print("Spider04 - gamma = 1:    ", timeAtGoalStatesAlways04)
print("Spider04 - gamma = 0.75: ", timeAtGoalStatesAlmostAlways04)


expectedStepsAlways02 = 1 / timeAtGoalStatesAlways02
expectedStepsAlmostAlways02 = 1 / timeAtGoalStatesAlmostAlways02
expectedStepsAlways04 = 1 / timeAtGoalStatesAlways04
expectedStepsAlmostAlways04 = 1 / timeAtGoalStatesAlmostAlways04

print("Expected Steps (Spider02, gamma=1):   ", expectedStepsAlways02)
print("Expected Steps (Spider02, gamma=0.75):", expectedStepsAlmostAlways02)
print("Expected Steps (Spider04, gamma=1):   ", expectedStepsAlways04)
print("Expected Steps (Spider04, gamma=0.75):", expectedStepsAlmostAlways04)

# <font color='cyan'>
# Using the analytical approach, we determined that StopITSpider02 with P['go'] = 1 is the fastest chain. This conclusion is based on the fact that it spends the most time in goal states according to the stationary distribution and, as expected, has the fewest expected steps to reach a goal state.
# From the results, we observe that for StopITSpider02, reducing γγ from 1 to 0.75 decreases the time spent in goal states (from 0.1249 to 0.1105) and increases the expected steps (from 8.00 to 9.05), indicating a slower convergence to the goal.
# On the other hand, we observe the reverse pattern for StopITSpider04: decreasing γ γ decreases the predicted steps (from 16.18 to 14.22) and increases the time spent in goal states (from 0.0618 to 0.0703), indicating that a different action selection technique is more effective in this context.
# Given that a single fixed method does not work optimally in all circumstances, these findings support the idea that the optimal probability for action selection should be modified when the likelihood of wet days varies.
# </font>

MgoAlways = load_chain("StopITSpider02.npy", 1)
MgoAlmostAlways = load_chain("StopITSpider02.npy", 0.75)

MgoAlways2 = load_chain("StopITSpider04.npy", 1)
MgoAlmostAlways2 = load_chain("StopITSpider04.npy", 0.75)

u = np.zeros(len(MgoAlways[0]))
u[0] = 1
print(u)
counterGoalState = 0
# Set the first element to 1

traj = simulate(MgoAlways, u, 10000)
counterGoalState += sum(1 for state in traj if state2observ(state)[0] == 9)
print("Time at goal states:")
print(counterGoalState / 10000)
print("Expected Steps:   ", 1 / (counterGoalState / 10000))
counterGoalState = 0
traj = simulate(MgoAlmostAlways, u, 10000)
counterGoalState += sum(1 for state in traj if state2observ(state)[0] == 9)
print("Time at goal states:")
print(counterGoalState / 10000)
print("Expected Steps:   ", 1 / (counterGoalState / 10000))
counterGoalState = 0
traj = simulate(MgoAlways2, u, 10000)
counterGoalState += sum(1 for state in traj if state2observ(state)[0] == 9)
print("Time at goal states:")
print(counterGoalState / 10000)
print("Expected Steps:   ", 1 / (counterGoalState / 10000))
counterGoalState = 0
traj = simulate(MgoAlmostAlways2, u, 10000)
counterGoalState += sum(1 for state in traj if state2observ(state)[0] == 9)
print("Time at goal states:")
print(counterGoalState / 10000)
print("Expected Steps:   ", 1 / (counterGoalState / 10000))

# <font color='cyan'>
# As we can observe, the sampling approach verified what we concluded on the analytical approach. That is the choice for gamma should not be the same when considering different probabilities of raining.
# Also the StopITSpider02 Markov chain is the fastest with gamma = 1 and the StopITSpider04 markov chain with gamma 0.75 is faster than its counterpart with gamma = 1.
# </font>


