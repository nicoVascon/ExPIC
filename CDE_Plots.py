import numpy as np
import matplotlib.pyplot as plt

def CDE(x, y, z, alpha, epsilon):
    
    target_mse = np.mean((x - y)**2)
    non_target_mse = np.mean((x - z)**2)
    
    return (1 - alpha) * target_mse + alpha * (1/(non_target_mse + epsilon))

def CDE_derivate(x, y, z, alpha, axis):    
    return (x[axis] - y[axis])*(1 - alpha) - ((4*alpha*(x[axis] - z[axis]))/(np.sum((x - z)**2))**2)

def calculate_newX_GradientDescent(x, y, z, alpha, epsilon, lr):
    numAxis = x.shape[-1]
    newX = np.zeros(x.shape)
    
    for axis in range(numAxis):
        newX[axis] = x[axis] - lr * CDE_derivate(x, y, z, alpha, axis)
        
    return newX

def compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations):
    x_points = np.zeros([iterations, x.shape[-1]])
    for i in range(iterations):
        x = calculate_newX_GradientDescent(x, y, z, alpha, epsilon, lr)
        x_points[i] = x
    return [x_points[:, i] for i in range(x.shape[-1])]

y = np.array([5, 5])*1e-5
z = np.array([2, 3])*1e-5
x = np.array([1, 1])*1e-5


alpha = 1e-25
epsilon = 1e-12

# CDE(x, y, z, alpha, epsilon)

iterations = 15000
lr = 1e-3
x_points = np.zeros([iterations, 2])

ax = plt.gca()

# # Plot x Point
# plt.plot(x=x[0], y=x[1], label="x")

# # Plot y point
# plt.plot(x=y[0], y=y[1], label="P_gamma")
# # Plot z point
# plt.plot(x=z[0], y=z[1], label="P_lambda")

ax.scatter(x=x[0], y=x[1], c = "blue", label="$x$")
ax.scatter(x=y[0], y=y[1], c = "green", label="$P_{\gamma}$")
ax.scatter(x=z[0], y=z[1], c = "red", label="$P_{\lambda}$")
# ax.scatter(x=z[0], y=z[1], c = "red", label="P ʎ")

# print(x_points)

# alpha = 1e-25
# for i in range(iterations):
#     x = calculate_newX_GradientDescent(x, y, z, alpha, epsilon, lr)
#     x_points[i] = x

# first_D = x_points[:, 0]
# second_D = x_points[:, 1]

x = np.array([1, 1])*1e-5
alpha = 0
first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# Plot CDE Optimal Path
plt.plot(first_D, second_D, color="orange", label="$\\alpha = 0$")
ax.scatter(x=first_D[-1], y=second_D[-1], c = "orange")

x = np.array([1, 1])*1e-5
alpha = 1e-20
first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# Plot CDE Optimal Path
plt.plot(first_D, second_D, color="fuchsia", label="$\\alpha = 10^{-20}$", linestyle="dashed")
ax.scatter(x=first_D[-1], y=second_D[-1], c = "fuchsia")

x = np.array([1, 1])*1e-5
alpha = 1e-18
first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# Plot CDE Optimal Path
plt.plot(first_D, second_D, color="darkturquoise", label="$\\alpha = 10^{-18}$", linestyle="dashdot")
ax.scatter(x=first_D[-1], y=second_D[-1], c = "darkturquoise")

x = np.array([1, 1])*1e-5
alpha = 1e-17
first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# Plot CDE Optimal Path
plt.plot(first_D, second_D, color="orangered", label="$\\alpha = 10^{-17}$", linestyle="dotted")
ax.scatter(x=first_D[-1], y=second_D[-1], c = "orangered")

plt.plot(y[0], y[1], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

plt.xlim(-3e-5, 11.5e-5)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])


plt.ylabel("i$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)
plt.xlabel("j$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)

import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='serif',
                                   style='italic', size=10)

plt.legend(loc="upper left", prop=font)

plt.savefig('CDE_3.png')

# import tikzplotlib

# tikzplotlib.save("test_2.tex")