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
        nearest_y_idx = np.argmin(np.sum((x - y)**2, axis=1))
        nearest_z_idx = np.argmin(np.sum((x - z)**2, axis=1))
        # print(nearest_z_idx)
        nearest_y = y[nearest_y_idx]
        nearest_z = z[nearest_z_idx] 
        x = calculate_newX_GradientDescent(x, nearest_y, nearest_z, alpha, epsilon, lr)
        x_points[i] = x
    return [x_points[:, i] for i in range(x.shape[-1])]

# y = np.array([[6, 3]])*1e-5
# z = np.array([[2, 3], [4, 2], [-1, 0], [10, 5]])*1e-5
# x = np.array([1, 1])*1e-5

import pickle
import torch
layer_num = 25
# trial_name = "Trial_Manual_1_FixBatchSizeAndLearningRate"
# trial_name = "Trial_Manual_1_FixBatchSizeAndLearningRate_Alpha=0"
# trial_name = "Trial_Manual_1_FixBatchSizeAndLearningRate_Alpha=1e-7"
trial_name = "Trial_Manual_1_FixBatchSizeAndLearningRate_AllLayers_Alpha=1e-7"


x_list = []
classes_list = []
for i in range(1,101):
    with open("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Study_Extra_NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_2c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNEvolving/%s/FineTuning_%d.weight_To_End_/T1/FV_FTLayer_%d.weight_Epoch_%d.pkl" % (trial_name, layer_num, layer_num, i),
            "rb") as file:
        x_epoch, classes = pickle.load(file)
    x_list.append(x_epoch.view(44, 2, 1))
    classes_list.append(classes.view(44, 1))

x = torch.cat(x_list, dim=2).numpy(force=True)[:, :, :]
classes = torch.cat(classes_list, dim=1).numpy(force=True)
    
with open("/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Study_Extra_NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_2c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNEvolving/%s/FineTuning_%d.weight_To_End_/T1/Prototypes_FTLayer_%d.weight.pkl" % (trial_name, layer_num, layer_num),
          "rb") as file:
    prototypes_dict = pickle.load(file)
    
y = torch.squeeze(prototypes_dict[0]).numpy(force=True)
z = torch.squeeze(prototypes_dict[1]).numpy(force=True)

print("y.shape", y.shape)
print("z.shape", z.shape)
print("x.shape", x.shape)

alpha = 1e-25
epsilon = 1e-12

# CDE(x, y, z, alpha, epsilon)

iterations = 15000
lr = 1e-3
x_points = np.zeros([iterations, 2])

fig, ax = plt.subplots()

# # Plot x Point
# plt.plot(x=x[0], y=x[1], label="x")

# # Plot y point
# plt.plot(x=y[0], y=y[1], label="P_gamma")
# # Plot z point
# plt.plot(x=z[0], y=z[1], label="P_lambda")

# ax.scatter(x=x[0], y=x[1], c = "blue", label="$x$")
ax.scatter(x=y[0][0], y=y[0][1], c = "green", label="$P_{\gamma}$")
for i in range(1,y.shape[0]):
    y_i = y[i]
    ax.scatter(x=y_i[0], y=y_i[1], c = "green")
ax.scatter(x=z[0][0], y=z[0][1], c = "red", label="$P_{\lambda}$")
for i in range(z.shape[0]):
    z_i = z[i]
    ax.scatter(x=z_i[0], y=z_i[1], c = "red")
# ax.scatter(x=z[0], y=z[1], c = "red", label="P ʎ")

# print(x_points)

# alpha = 1e-25
# for i in range(iterations):
#     x = calculate_newX_GradientDescent(x, y, z, alpha, epsilon, lr)
#     x_points[i] = x

# first_D = x_points[:, 0]
# second_D = x_points[:, 1]

import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())
colors = [colors[0]] + colors[4:]

print("Classes.shape:", classes.shape)
print(sum(classes == 0))

# x = np.array([[1, 1], [2, 2], [0, -2], [-3, 3]])*1e-5
# alpha = 1e-20
# for epoch in range(x.shape[2]):
line_list = []
scat_1_list = []
scat_2_list = []
for i in range(x.shape[0]):
    first_D, second_D = x[i][0], x[i][1] 
    # first_D, second_D = compute_optimal_path(x[i], y, z, alpha, epsilon, lr, iterations)
    # Plot CDE Optimal Path
    # color_idx = np.random.randint(low=0, high=len(colors))
    # line_list.append(ax.plot(first_D, second_D, color=colors[classes[i][0]]))
    scat_1_list.append(ax.scatter(x=first_D[0], y=second_D[0], c = colors[classes[i][0]]))
    # scat_2_list.append(ax.scatter(x=first_D[-1], y=second_D[-1], c = colors[classes[i][0]]))
    
def update(frame):
    # # for each frame, update the data stored on each artist.
    # x = t[:frame]
    # y = z[:frame]
    # # update the scatter plot:
    # data = np.stack([x, y]).T
    # scat.set_offsets(data)
    # # update the line plot:
    # line2.set_xdata(t[:frame])
    # line2.set_ydata(z2[:frame])
    for i in range(x.shape[0]):
        first_D, second_D = x[i][0], x[i][1] 
        # first_D, second_D = compute_optimal_path(x[i], y, z, alpha, epsilon, lr, iterations)
        # Plot CDE Optimal Path
        # color_idx = np.random.randint(low=0, high=len(colors))
        # line_list[i][0].set_xdata(first_D[:frame])
        # line_list[i][0].set_ydata(second_D[:frame])
        # scat_1_list[i].set_offsets(np.array([first_D[0], second_D[0]]))
        # scat_2_list[i].set_offsets(np.array([first_D[frame], second_D[frame]]))
        scat_1_list[i].set_offsets(np.array([first_D[frame], second_D[frame]]))
    # return line_list + scat_1_list + scat_2_list
    return scat_1_list

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=300)
# ani.save(filename="/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/pillow_example.gif", writer="pillow")
ani.save(filename="/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/ffmpeg_example.mp4", writer="ffmpeg")
import sys
sys.exit()

# x = np.array([1, 1])*1e-5
# alpha = 0
# first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# # Plot CDE Optimal Path
# plt.plot(first_D, second_D, color="orange", label="$\\alpha = 0$")
# ax.scatter(x=first_D[0], y=second_D[0], c = "blue")
# ax.scatter(x=first_D[-1], y=second_D[-1], c = "orange")

# x = np.array([1, 1])*1e-5
# alpha = 1e-18
# first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# # Plot CDE Optimal Path
# plt.plot(first_D, second_D, color="orangered", label="$\\alpha = 10^{-18}$", linestyle="dotted")
# ax.scatter(x=first_D[0], y=second_D[0], c = "blue")
# ax.scatter(x=first_D[-1], y=second_D[-1], c = "orangered")

# x = np.array([1, 1])*1e-5
# alpha = 1e-19
# first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# # Plot CDE Optimal Path
# plt.plot(first_D, second_D, color="darkturquoise", label="$\\alpha = 10^{-19}$", linestyle="dashdot")
# ax.scatter(x=first_D[-1], y=second_D[-1], c = "darkturquoise")

# x = np.array([1, 1])*1e-5
# alpha = 1e-20
# first_D, second_D = compute_optimal_path(x, y, z, alpha, epsilon, lr, iterations)
# # Plot CDE Optimal Path
# plt.plot(first_D, second_D, color="fuchsia", label="$\\alpha = 10^{-20}$", linestyle="dashed")
# ax.scatter(x=first_D[-1], y=second_D[-1], c = "fuchsia")

for i in range(y.shape[0]):
    y_i = y[i]
    plt.plot(y_i[0], y_i[1], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

for i in range(z.shape[0]):
    y_i = z[i]
    plt.plot(y_i[0], y_i[1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")

# plt.xlim(-3e-5, 11.5e-5)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])


plt.ylabel("i$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)
plt.xlabel("j$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)

import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='serif',
                                   style='italic', size=10)

# plt.legend(loc="upper left", prop=font)

plt.savefig('/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/CDE_Multi_2.png')

# import tikzplotlib

# tikzplotlib.save("test_2.tex")