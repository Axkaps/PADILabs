
s = "Hello, world!!"
print(s)

import math

a = 0

# Print a radian --> degree conversion table
while a < 2 * math.pi: 
    print(a, "radians correspond to", a * 180 / math.pi, "degrees.")
    a = a + 0.5

# a = input("Please insert a number:\n>> ")

# for i in range(5):
#     a = math.sqrt(float(a))
#     print("Next square root:", a)

# if a > 1:
#     print(a, "is larger than 1.") 
# else: 
#     print(a, "is smaller than or equal to 1.")

def square(x):
    return x * x

print(square(2))
print("The variable s is accessible here:", s)

import numpy as np

A1 = np.array([[1, 2, 3], [4, 5, 6]])
print("2 x 3 array of numbers:")
print(A1)
print("This array is of dimension", A1.shape)

A2 = np.eye(3)
print("3 x 3 identity:")
print(A2)
print("This array is of dimension", A2.shape)

A3 = np.zeros((2, 3))
print("2 x 3 array of zeros:")
print(A3)
print("This array is of dimension", A3.shape)

A4 = np.ones(4)
print("4 x 0 array of ones (note how there is no second dimension):")
print(A4)
print("This array is of dimension", A4.shape)

# You can now easily perform standard algebraic operations, such as matrix sums or products. You can also perform indexing, slicing, and other operations, as illustrated in the following examples.

# = Matrix creation = #

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("3 x 3 matrix:")
print(A)

A = np.eye(3, dtype='int64')
print(A)

# You can explicitly indicate the array data type with the optional argument "dtype"
print(A.dtype)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float64')
print(A)
print(A.dtype)
print(A.shape)

B = np.arange(1,4, dtype='float64')
print("Vector with all numbers between 1 and 3:")
print(B)

C = np.diag(B)
print("Diagonal matrix built from the vector B:")
print(C)

# = Matrix operations = #

# Sum
D = A + np.eye(3)
print("A + I:")
print(D)

# Vector transpose and regular matrix product
E = np.dot(A, B.T)
print("A * B':")
print(E)

# Matrix inverse
F = np.linalg.inv(D)
print("inv(D):")
print(F)

# = Matrix concatenation = #
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float64')
G = np.concatenate((np.array([[1, 2, 3]]), A), axis=0)
print("Append matrix A to vector [1, 2, 3]:")
print(G)
print(G.shape)

# When the axis to append is specified, the 
# matrices/vectors must have the correct shape

H1 = np.append(A, [[10, 11, 12]], axis = 0)
H2 = np.append(A, [[4], [7], [10]], axis = 1)
print("Append [10, 11, 12] to A:")
print(H1)

print("Append [[4], [7], [10]] to A:")
print(H2)

# = Matrix indexing = #

# Simple indexing
print("A[0]:", A[0])
print("A[1]:", A[1])
print("A[1, 2]:", A[1, 2])  # More efficient
print("A[0][2]:", A[0][2])  # Less efficient

# -- Slicing

# Rows between 1 and 2 (excluding the latter), 
# columns between 0 and 1 (excluding the latter)
print("A[1:2,0:1]:", A[1:2,0:1])

# All rows except the last two,
# every other column
print("A[:-2,::2]:", A[:-2,::2]) 

I = np.arange(10, 1, -1)
print("Vector I, with numbers between 10 and 1:")
print(I)

# -- Matrices as indices

# Indexing with a list
print("I[[3, 3, 1, 8]]:", I[[3, 3, 1, 8]])

# Indexing with an nparray
print("I[np.array([3, 3, -3, 8])]:", I[np.array([3, 3, -3, 8])])

# Indexing with an npmatrix
print("I[np.array([[1, 1], [2, 3]])]:", I[np.array([[1, 1], [2, 3]])])

import numpy.random as rnd
import time

A = rnd.rand(1000,1000)
B = rnd.rand(1000,1000)
C = np.zeros((1000,1000))

t = time.time()

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        C[i, j] = A[i, j] + B[i, j]
    
t1 = time.time() - t

t = time.time()
C = A + B
t2 = time.time() - t

print("Computation time with cycle (in seconds):", t1)
print("Computation time with numpy operation (in seconds):", t2)



# Insert your code here.
A = rnd.rand(100000, 1)
B = rnd.rand(100000, 1)

t = time.time()
for i in range(1 , A.shape[0]):
    C = A[i] + B[i]
t1 = time.time() - t

print("Time with a loop:", t1)

t = time.time()
C = A + B
t1 = time.time() - t
print("Time with vectorization:", t1)


# Insert your code here.
u = 1
p1 = np.zeros(1000)
p1[0] = -np.pi/3 + 0.6
v = np.zeros(1000)

for i in range(1, v.shape[0]):
    v[i] = v[i - 1] - 1/400 * np.cos(3 * p1[i - 1]) + u / 1000
    p1[i] = p1[i - 1] + v[i]




# Insert your code here.
values = [1, 0, -1]
probabilities = [0.7, 0.2, 0.1]  # Probabilities must sum to 1
u = np.zeros(1000)
u[0] = np.random.choice(values, p=probabilities)
p = np.zeros(1000)
p[0] = -np.pi/3 + 0.6

for i in range(1, v.shape[0]):
    u[i] = np.random.choice(values, p=probabilities)
    v[i] = v[i - 1] - 1/400 * np.cos(3 * p[i - 1]) + u[i - 1] / 1000
    p[i] = p[i - 1] + v[i]

    

# %matplotlib inline
import matplotlib.pyplot as plt

# Create data
x = 100 * rnd.rand(100, 1)
y = 2 * x + 10 * rnd.randn(100, 1)

# Estimate linear relation between X and Y

X = np.append(x, np.ones((100,1)), axis = 1)

f_est = np.dot(np.linalg.pinv(X), y)
y_est = np.dot(X, f_est)

# Plot commands

plt.figure()
plt.plot(x, y_est)
plt.plot(x, y, 'x')

plt.xlabel('Input X')
plt.ylabel('Output Y')

plt.title('Linear regression')


# Consider more carefully the piece of code above, where we included line numbers for easier reference.
# 
# * On lines 5 and 6, the vectors *x* and *y* are created, using mostly functionalities that you already encountered in Sections 1 and 2. The novelty is the function `randn` which is similar to the function `rand` except on their underlying distribution: while `rand` generates random numbers uniformly from the interval [0, 1], `randn` generates normally distributed random numbers with mean 0 and a standard deviation of 1.
# 
# * Lines 10-13 estimate a linear relation between *x* and *y* using the data already created. Do not worry about the actual computations, and simply observe the use of matrix concatenation in line 10, and the `pinv` function in line 12. The function `pinv` computes the Moore-Penrose pseudo-inverse of a matrix (you can find more infor, for example, in [Wikipedia](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse))
# 
# * Lines 17 through 24 contain the actual plotting commands. In particular:
# 
#   * The figure command in line 17 creates a new figure.
# 
#   * The plot command in line 18 is responsible for displaying the continuous line in the plot. In here, it is used with its most basic syntax. However, the plot command has a very rich syntax, and you can type `help("mplotlib.pyplot.plot")` to know more about this useful function.
#   
#   * The plot command in line 19 plots the original data. Note how the line specification 'x' indicates that, unlike the plot in line 18, this data should not be plotted continuously but instead marking each data-point with an "&times;".
#   
# * Finally, the commands in lines 21 to 24 are used to include additional information in the plot, such as the labels for both axis and the title for the plot.


# ---
# 
# #### Activity 4. Mountain car problem (cont.)
# 
# Plot in the same axis the position $p$ of the car as a function of $t$ as observed in Activities 2 and 3. What do you observe?
# 
# ---


# Insert your code here.
t = np.arange(1000)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, p, label='p (position)')
plt.plot(t, p1, label='p1 (second position)', linestyle='dashed')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.title('Random Process Evolution')
plt.show()




# <font color='blue'>Yadayaydaaydyadyadyada</font>


# ---
# 
# #### Activity 5. Mountain car problem (conc.)
# 
# Suppose now that the driver selects $u(t)=-1$ until it reaches the position $p(t)=-0.85$ and then selects $u(t)=1$. Repeact Activity 2 and plot the three trajectories you obtained. What can you observe?
# 
# ---


# Insert your code here.
# Insert your code here.
u2 = -1
p2 = np.zeros(1000)
p2[0] = -np.pi/3 + 0.6
v = np.zeros(1000)

for i in range(1, v.shape[0]):
    u2 = -1 if p2[i - 1] > -0.85 else 1 
    v[i] = v[i - 1] - 1/400 * np.cos(3 * p2[i - 1]) + u2 / 1000
    p2[i] = p2[i - 1] + v[i]






# Insert your code here.
t = np.arange(1000)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, p, label='p (position)')
plt.plot(t, p1, label='p1 (second position)', linestyle='dashed')
plt.plot(t, p2, label='p2 (third position)', linestyle='dashed')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.title('Random Process Evolution')
plt.show()


# ---
# 
# #### Activity 6. Saving and exporting
# 
# Export your file as a Python script. In particular:
# 
# * Go to the menu "**File**", select the option "**Download as &#x27A4;**" and then "**Python (.py)**".
# * Name your file `padi-lab0-groupXXX.py`, where you replace `XXX` by your group number.
# 
# Open the resulting Python file. Note that all the markdown in this notebook is converted to comments in the resulting Python file. 
# 
# * Delete **all** comments in the Python file, keeping only the "pure" Python code.
# 
# At some point in the python file, you will find the line:
# 
# `get_ipython().run_line_magic('matplotlib', 'notebook')`
# 
# This is a line used by Jupyter and which will yield an error if the Python script is executed as is. 
# 
# * Remove the aforementioned line.
# * Run the resulting file. To do so, you can open a terminal window and execute the shell command `python padi-lab0-grouXXX.py`. Verify that everything works properly. 
# 
# In future labs, you should always follow these steps to make sure that everything works properly before submitting a lab assignment.
# 
# ---





