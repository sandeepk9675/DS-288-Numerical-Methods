import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Question 1 solution
print("Question 1 solution")
x = np.array([0.3, 0.4, 0.5, 0.6])
y = np.array([0.740818, 0.670320, 0.606531, 0.548812])

fun = x - y

def neville(x, y, xbar):
    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y
    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i, j] = ((xbar - x[i - j]) * Q[i, j - 1] - (xbar - x[i]) * Q[i - 1, j - 1]) / (x[i] - x[i - j])
    print(Q)
    return Q[n - 1, n - 1]

def Bisection_method(func, a, b, tol):
    if func(a) * func(b) > 0:
        print("No root found.")
        return None
    while (b - a) / 2 > tol:
        midpoint = a + (b - a) / 2
        if func(midpoint) == 0:
            return midpoint
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return midpoint

func_x = lambda x: x - np.exp(-x)
x_root = Bisection_method(func_x, 0.5, 0.6, 1e-10)
print(x_root)

fun_value = 0
# Example value for interpolation
print(neville(fun, x, fun_value))

print("--------------------------------------------\n")
# Question 2 solution
print("Question 2 solution")

t_i = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
x_i = np.array([0.70, 1.22, 2.11, 3.07, 3.25, 2.80, 2.11, 1.30, 0.70, 0.45, 0.88, 2.00, 3.25])
y_i = np.array([2.25, 1.77, 1.61, 1.75, 2.30, 2.76, 2.91, 2.76, 2.25, 1.37, 0.56, 0.08, 0.25])


def neville(x, y, xbar):
    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y
    for i in range(n):
        for j in range(1, i + 1):
            if x[i] == x[i - j]:
                raise ValueError(f"Division by zero detected at i={i}, j={j}")
            Q[i, j] = ((xbar - x[i - j]) * Q[i, j - 1] - (xbar - x[i]) * Q[i - 1, j - 1]) / (x[i] - x[i - j])
            #print(f"Q[{i}, {j}] = {Q[i, j]}")
    return Q[n - 1, n - 1]

# create a vector of t values raging from 0 to 12, 0.5 apart
no_of_points = 100
# Create a vector of t values ranging from 0 to 12, 0.5 apart
T = np.linspace(0, 12, no_of_points)
X = np.zeros(no_of_points)
Y = np.zeros(no_of_points)

# Calculate interpolated values for X and Y
for j, t in enumerate(T):
    X[j] = neville(t_i, x_i, t)
    Y[j] = neville(t_i, y_i, t)

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Plot x(t) vs t on the first subplot
ax[0].plot(T, X, 'b-', label='f(t)')
ax[0].plot(t_i, x_i, 'ro', label='Original x points')
ax[0].set_title("f(t) vs t")
ax[0].set_xlabel("t")
ax[0].set_ylabel("f(t)")
ax[0].legend()

# Plot y(t) vs t on the first subplot
ax[0].plot(T, Y, 'g-', label='g(t)')
ax[0].plot(t_i, y_i, 'ro', label='Original y points')
ax[0].set_title("f(t) and g(t) vs t")
ax[0].set_xlabel("t")
ax[0].set_ylabel("Values")
ax[0].legend()

# Plot y(t) vs x(t) on the second subplot
ax[1].plot(X, Y, 'b-', label='g(t) vs f(t)')
ax[1].plot(x_i, y_i, 'ro', label='Original points')
ax[1].set_title("g(t) Vs f(t)")
ax[1].set_xlabel("f(t)")
ax[1].set_ylabel("g(t)")
ax[1].legend()

plt.tight_layout()
plt.show()

print("--------------------------------------------\n")
# Question 3 solution
print("Question 3 solution")

# Given data points
t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
x = np.array([0.70, 1.22, 2.11, 3.07, 3.25, 2.80, 2.11, 1.30, 0.70, 0.45, 0.88, 2.00, 3.25])
y = np.array([2.25, 1.77, 1.61, 1.75, 2.30, 2.76, 2.91, 2.76, 2.25, 1.37, 0.56, 0.08, 0.25])

# Natural Cubic Splines for f(t) and g(t)
cs_f = CubicSpline(t, x, bc_type='natural')
cs_g = CubicSpline(t, y, bc_type='natural')

# Get the spline coefficients for f(t) and g(t)
coeffs_f = cs_f.c  # Coefficients for f(t)
coeffs_g = cs_g.c  # Coefficients for g(t)


# Print the coefficients of f(t)
print("Cubic Spline Coefficients for f(t):")
for i in range(len(t) - 1):
    print(f"Interval {i}: a = {coeffs_f[3, i]:.6f}, b = {coeffs_f[2, i]:.6f}, c = {coeffs_f[1, i]:.6f}, d = {coeffs_f[0, i]:.6f}")

# Print the coefficients of g(t)
print("\nCubic Spline Coefficients for g(t):")
for i in range(len(t) - 1):
    print(f"Interval {i}: a = {coeffs_g[3, i]:.6f}, b = {coeffs_g[2, i]:.6f}, c = {coeffs_g[1, i]:.6f}, d = {coeffs_g[0, i]:.6f}")

# Plot the splines along with the original data points
t_new = np.linspace(min(t), max(t), 200)

fx, ax = plt.subplots(1, 2, figsize=(15, 6))
# Plot f(t) and g(t) splines on one plot
ax[0].plot(t, x, 'o', label='Data points (f(t))')
ax[0].plot(t_new, cs_f(t_new), label='Cubic Spline f(t)')
ax[0].plot(t, y, 'x', label='Data points (g(t))')
ax[0].plot(t_new, cs_g(t_new), label='Cubic Spline g(t)')
ax[0].set_title('Natural Cubic Spline for f(t) and g(t)')
ax[0].set_xlabel('t')
ax[0].set_ylabel('Values')
ax[0].legend()

# Plot f(t) vs g(t)
ax[1].plot(cs_f(t_new), cs_g(t_new), label='Cubic Spline f(t) vs g(t)')
ax[1].plot(x, y, 'o', label='Original Data Points')
ax[1].set_title('g(t) vs f(t)')
ax[1].set_xlabel('f(t)')
ax[1].set_ylabel('g(t)')
ax[1].legend()

plt.show()

