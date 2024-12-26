import numpy as np
import matplotlib.pyplot as plt

J = np.array([
    [7.6520e-01, -1.7760e-01,  5.5812e-02],
    [4.4005e-01, -3.2757e-01, -9.7511e-02],
    [1.1490e-01,  4.6565e-02, -5.9712e-02],
    [1.9563e-02,  3.6483e-01,  9.2734e-02],
    [2.4766e-03,  3.9123e-01,  7.0840e-02],
    [2.4975e-04,  2.6114e-01, -8.1400e-02],
    [2.0938e-05,  1.3104e-01, -8.7121e-02],
    [1.5023e-06,  5.3376e-02,  6.0491e-02],
    [9.4223e-08,  1.8405e-02,  1.0405e-01],
    [5.2492e-09,  5.5202e-03, -2.7192e-02],
    [2.6306e-10,  1.4678e-03, -1.1384e-01]
])

print(J)
print("\n")

## forward Recursion
j_i_x = [
    [7.6519e-01, -1.7759e-01,  5.5812e-02],
    [4.4005e-01, -3.2757e-01, -9.7511e-02],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
    [0,  0, 0],
]
err = np.zeros((11, 3))

def beselfunction_forward(x_1, x_2, x_3, n):
    if n == 11:
        return (x_1[1], x_2[1], x_3[1])
    else:
        y_1 = [0, 0]
        y_2 = [0, 0]
        y_3 = [0, 0]
        y_1[0] = x_1[1]
        y_2[0] = x_2[1]
        y_3[0] = x_3[1]
        y_1[1] = 2 * (n-1) * x_1[1] - x_1[0]
        err[n][0] = abs(J[n][0] - y_1[1])
        y_2[1] = 2 * (n-1) * x_2[1] / 5 - x_2[0]
        err[n][1] = abs(J[n][1] - y_2[1])
        y_3[1] = 2 * (n-1) * x_3[1] / 50 - x_3[0]
        err[n][2] = abs(J[n][2] - y_3[1])
        n = n + 1
        return beselfunction_forward(y_1, y_2, y_3, n)

a, b, c = beselfunction_forward([7.6519e-01, 4.4005e-01], [-1.7759e-01, -3.2757e-01], [5.5812e-02, -9.7511e-02], 2)
print("1. {} {} {}".format(a, b, c))

n = np.arange(11)
##plt.plot(n, err[:, 0], label='Error 1')
plt.plot(n, err[:, 1], label='Error 2')
plt.plot(n, err[:, 2], label='Error 3')
plt.legend()
plt.show()

## Backward Recursion

j_i_x = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [5.2492e-09,  5.5202e-03, -2.7192e-02],
    [2.6306e-10,  1.4678e-03, -1.1384e-01],
]

err1 = np.zeros((11, 3))
def beselfunction_backward(x_1, x_2, x_3, n):
    if n == -1:
        return (x_1[0], x_2[0], x_3[0])
    else:
        y_1 = [0, 0]
        y_2 = [0, 0]
        y_3 = [0, 0]
        y_1[1] = x_1[0]
        y_2[1] = x_2[0]
        y_3[1] = x_3[0]
        y_1[0] = 2 * (n+1) * x_1[0] - x_1[1]
        err1[n][0] = abs(J[n][0] - y_1[0])
        y_2[0] = 2 * (n+1) * x_2[0] / 5 - x_2[1]
        err1[n][1] = abs(J[n][1] - y_2[0])
        y_3[0] = 2 * (n+1) * x_3[0] / 50 - x_3[1]
        err1[n][2] = abs(J[n][2] - y_3[0])
        n = n - 1
        return beselfunction_backward(y_1, y_2, y_3, n)

a, b, c = beselfunction_backward([5.2492e-09, 2.6306e-10], [5.5202e-03, 1.4678e-03], [-2.7192e-02, -1.1384e-01], 8)
print("2. {} {} {}".format(a, b, c))

n = np.arange(11)
plt.plot(n, err1[:, 0], label='Error 1')
plt.plot(n, err1[:, 1], label='Error 2')
plt.plot(n, err1[:, 2], label='Error 3')
plt.legend()
plt.show()
