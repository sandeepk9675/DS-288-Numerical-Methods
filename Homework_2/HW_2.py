import numpy as np
import os
import matplotlib.pyplot as plt
import sympy as sp

# Question 1 (a) and (b) solution

# Function to calculate Newton's step
def calculate_newton_step(x_old, func, x):
    func_at_xold = func.subs(x, x_old)
    func_at_xold_real = func_at_xold.evalf()
    func_prime = sp.diff(func, x)
    func_prime_at_xold = func_prime.subs(x, x_old)
    func_prime_at_xold_real = func_prime_at_xold.evalf()

    numerator = func_at_xold_real
    denominator = func_prime_at_xold_real
    x_n_plus_1 = x_old - numerator / denominator
    return x_n_plus_1

# Newton's Method
def Newton_method(p_o, func, x):
    x_old = p_o 
    x_new = calculate_newton_step(x_old, func, x)
    n = 1
    root = np.array([], dtype=float)
    root = np.append(root, x_old)
    root = np.append(root, x_new)
    while (abs(x_new - x_old)/abs(x_new)) > 0.000001:
        x_old = x_new
        x_new = calculate_newton_step(x_old, func, x)
        root = np.append(root, x_new)
        n = n + 1
    return x_new, root,n

# Function to calculate Secant step
def Calculate_secant_step(x_n, x_n_minus_1, func, x):
    func_at_x_n = func.subs(x, x_n)
    func_at_x_n_real = func_at_x_n.evalf()

    func_at_x_n_minus_1 = func.subs(x, x_n_minus_1)
    func_at_x_n_minus_1_real = func_at_x_n_minus_1.evalf()

    numerator = func_at_x_n_real * x_n_minus_1 - func_at_x_n_minus_1_real * x_n
    denominator = func_at_x_n_real - func_at_x_n_minus_1_real

    x_n_plus_1 = numerator / denominator
    return x_n_plus_1

# Secant Method
def Secant_method(p_o, p_1, func, x):
    x_n_minus_1 = p_o 
    x_n = p_1
    x_n_plus_1 = Calculate_secant_step(x_n, x_n_minus_1, func, x)   
    n = 1
    root = np.array([], dtype=float)
    root = np.append(root, x_n_minus_1)
    root  = np.append(root, x_n)
    root = np.append(root, x_n_plus_1)
    while (abs(x_n_plus_1 - x_n)/abs(x_n_plus_1)) > 0.000001:
        x_temp = x_n 
        x_n = x_n_plus_1
        x_n_plus_1 = Calculate_secant_step(x_n_plus_1, x_temp, func, x)
        root = np.append(root, x_n_plus_1)
        n = n + 1
    return x_n, root,n

# Function to calculate Modified Newton's step
def Calculate_modified_newton_step(x_old, func, x):
    func_prime = sp.diff(func, x)
    func_double_prime = sp.diff(func_prime, x)
    func_at_xold = func.subs(x, x_old)
    func_at_xold_real = func_at_xold.evalf()
    func_prime_at_xold = func_prime.subs(x, x_old)
    func_prime_at_xold_real = func_prime_at_xold.evalf()
    func_double_prime_at_xold = func_double_prime.subs(x, x_old)
    func_double_prime_at_xold_real = func_double_prime_at_xold.evalf()
    
    numerator = func_at_xold_real * func_prime_at_xold_real
    denominator = func_prime_at_xold_real**2 - func_at_xold_real * func_double_prime_at_xold_real
    x_new = x_old - numerator / denominator
    return x_new

# Modified Newton's Method
def Modified_newton_method(p_o, func, x):
    x_old = p_o
    x_new = Calculate_modified_newton_step(x_old, func, x)
    root = np.array([], dtype=float)
    root = np.append(root, x_old)
    root = np.append(root, x_new)
    n = 1
    while (abs(x_new - x_old)/abs(x_new)) > 0.000001:
        x_old = x_new
        x_new = Calculate_modified_newton_step(x_old, func, x)
        root = np.append(root, x_new)
        n += 1
    return x_new, root, n

# Define the symbolic variable and function
x = sp.symbols('x')
f1 = sp.exp(-x**2) * sp.cos(x) + x
f2 = (sp.exp(-x**2) * sp.cos(x) + x)**2

# Get root propagation for Newton, Secant, and Modified Newton methods
root_newton, root_newton_propagation, iterations_newtion = Newton_method(0, f1, x)
root_sec, root_sec_propagation, iterations_sec = Secant_method(0, 1, f1, x)
root_modified_newton, root_newton_modified_propagation, iterations_modified_newton = Modified_newton_method(0, f1, x)

print("Q1(a) \n")
print("Newton Method: ")
print("root : {}".format(root_newton))
print("Number of iterations: ", iterations_newtion)
print("\n")
print("Secant Method: ")
print("root : {}".format(root_sec))
print("Number of iterations: ", iterations_sec)
print("\n")
print("Modified Newton Method: ")
print("root : {}".format(root_modified_newton))
print("Number of iterations: ", iterations_modified_newton)
print("\n")

# Get root propagation for Newton, Secant, and Modified Newton methods
root_newton, root_newton_propagation,iterations_newtion = Newton_method(0, f2, x)
root_sec, root_sec_propagation, iterations_sec = Secant_method(0, 1, f2, x)
root_modified_newton, root_newton_modified_propagation, iterations_modified_newton = Modified_newton_method(0, f2, x)

print("Q1(b) \n")
print("Newton Method: ")
print("root : {}".format(root_newton))
print("Number of iterations: ", iterations_newtion)
print("\n")
print("Secant Method: ")
print("root : {}".format(root_sec))
print("Number of iterations: ", iterations_sec)
print("\n")
print("Modified Newton Method: ")
print("root : {}".format(root_modified_newton))
print("Number of iterations: ", iterations_modified_newton)
print("\n")


print("--------------------------------------------\n")
print("Question 3 solution")

# Question 2 (b) solution

# Constants
r1 = 10
r2 = 6
r3 = 8
r4 = 4
theta4 = np.deg2rad(220)

# Initial guesses (in radians)
theta2_old = np.deg2rad(30)
theta3_old = np.deg2rad(0)

# Function definitions
def f1(theta2, theta3):
    return r2 * np.cos(theta2) + r3 * np.cos(theta3) + r4 * np.cos(theta4) - r1

def f2(theta2, theta3):
    return r2 * np.sin(theta2) + r3 * np.sin(theta3) + r4 * np.sin(theta4)

# Jacobian matrix
def jacobian(theta2, theta3):
    return np.array([
        [-r2 * np.sin(theta2), -r3 * np.sin(theta3)],
        [r2 * np.cos(theta2),  r3 * np.cos(theta3)]
    ])

F = np.array([f1(theta2_old, theta3_old), f2(theta2_old, theta3_old)])
    
# Compute Jacobian and its inverse
J = jacobian(theta2_old, theta3_old)
J_inv = np.linalg.inv(J)
    
# Update guess using Newton's method
delta = np.dot(J_inv, F)
theta2 = theta2_old - delta[0]
theta3 = theta3_old - delta[1]

# Newton's Method parameters
tolerance = 1e-4
iterations = 1

# Initialize error array

err = np.array([[abs(theta2 - theta2_old)/(abs(theta2)), abs(theta3 - theta3_old)/(abs(theta3))]])


while  np.sqrt((abs(theta3 - theta3_old)/abs(theta3))**2 + (abs(theta2 - theta2_old)/abs(theta2))**2) > tolerance:
    theta2_old = theta2
    theta3_old = theta3
    # Compute function values
    F = np.array([f1(theta2, theta3), f2(theta2, theta3)])
    
    # Compute Jacobian and its inverse
    J = jacobian(theta2, theta3)
    J_inv = np.linalg.inv(J)
    
    # Update guess using Newton's method
    delta = np.dot(J_inv, F)
    theta2 = theta2 - delta[0]
    theta3 = theta3 - delta[1]

    # Update error array
    err = np.vstack([err, [abs(theta2 - theta2_old)/(abs(theta2)), abs(theta3 - theta3_old)/(abs(theta3))]])
    
    
    iterations += 1

# Output final results
theta2_deg = np.rad2deg(theta2)
theta3_deg = np.rad2deg(theta3)
print("Number of iterations: ", iterations)
print("theta2: ", theta2_deg)
print("theta3: ", theta3_deg)

