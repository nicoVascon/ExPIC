import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os

# Define a custom sorting key function
def sort_key_function(filename):
    # Split the filename by "_" and extract the number part
    parts = filename.split("_")
    number_part = parts[-1].split(".")[0]
    # Convert the number part to an integer and use it as the sorting key
    return int(number_part)

# study_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Study_Extra_NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_2c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNEvolving"
study_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/NeuralNetJournal_Dataset_Exp/Prototype-Based_Training/From_xDNN_Offline/Optuna/Study_Extra_NeuralNetJournal_Prot-Based_TrainWithPCA_Torch_2c_CDEAlpha_Study_LowEpochs_New_Split_FVFromLastConv_xDNNOffline_RecalPCA"

# trial_name = "/Trial_Manual_1_FixBatchSizeAndLearningRate"
# trial_name = "/Trial_Manual_1_FixBatchSizeAndLearningRate_Alpha=0"
# trial_name = "/Trial_Manual_1_FixBatchSizeAndLearningRate_Alpha=1e-7"
# trial_name = "/Trial_Manual_1_FixBatchSizeAndLearningRate_AllLayers_Alpha=1e-7"
# trial_name = "/Trial_34"
trial_name = "/Trial_Manual_Alpha=0.0002029026841683182_vTestSet_v3"

layers_num = [29, 27, 25, 22, 20, 18, 15, 13, 11, 8, 6, 3, 1]
# layers_num = [29, 27]
# layers_num = [6, 3]


x_list = []
y_list = []
z_list = []
classes_list = []
layers_idx = []

for idx, layer_num in enumerate(layers_num):
    layer_dir = study_dir + trial_name + ("/FineTuning_%d.weight_To_End_/T1" % layer_num)
    FV_files = next(os.walk(layer_dir))[2]
    FV_files.sort()
    for file in FV_files:
        print(file)
    FV_files = FV_files[:-3]
    num_files = len(FV_files)
    print("num_files", num_files)
    print("(num_files/2): ", str((num_files/2)))
    # FV_files = FV_files[:int((num_files/2))]
    FV_files = FV_files[int((num_files/2)):]
    FV_files = sorted(FV_files, key=sort_key_function)
    for file in FV_files:
        # print(file)
        with open(layer_dir + "/" + file, "rb") as file:
            x_epoch, classes = pickle.load(file)
        x_list.append(x_epoch.view(-1, 2, 1))
        classes_list.append(classes.view(-1, 1))
        
    # for i in range(1,num_epochs+1):
    #     with open(layer_dir + "/FV_FTLayer_%d.weight_Epoch_%d.pkl" % (layer_num, i), "rb") as file:
    #         x_epoch, classes = pickle.load(file)
    #     x_list.append(x_epoch.view(-1, 2, 1))
    #     classes_list.append(classes.view(-1, 1))
    #     epochs += [i-1]*x_epoch.shape[0]

    with open(layer_dir + "/Prototypes_FTLayer_%d.weight.pkl" % (layer_num, ), "rb") as file:
        prototypes_dict = pickle.load(file)
    
    y = torch.squeeze(prototypes_dict[0])
    z = torch.squeeze(prototypes_dict[1])
    # y = torch.unsqueeze(y, dim=-1)
    # z = torch.unsqueeze(z, dim=-1)
    y_list.append(y.numpy(force=True))
    z_list.append(z.numpy(force=True))
    layers_idx += [idx]*len(FV_files)
    
x = torch.cat(x_list, dim=-1).numpy(force=True)[:, :, :]
classes = torch.cat(classes_list, dim=-1).numpy(force=True)
# y = torch.cat(y_list, dim=-1).numpy(force=True)
# z = torch.cat(z_list, dim=-1).numpy(force=True)
y = y_list
z = z_list
layers_idx = np.array(layers_idx)

# print("y.shape", y.shape)
# print("z.shape", z.shape)
print("x.shape", x.shape)


fig, ax = plt.subplots(figsize=(10, 10))

import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.keys())
colors = [colors[0]] + colors[4:]

print("Classes.shape:", classes.shape)
print(sum(classes == 0))

prot0_scat_list = []
prot1_scat_list = []
for i in range(y[0].shape[0]):
    y_i, y_j = y[0][i][0], y[0][i][1]
    prot0_scat_list.append(ax.scatter(x=y_i, y=y_j, c = "green"))
for i in range(z[0].shape[0]):
    z_i, z_j = z[0][i][0], z[0][i][1]
    prot1_scat_list.append(ax.scatter(x=z_i, y=z_j, c = "red"))

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

# num_epochs_per_layer = x.shape[2]/len(y)  
last_layer_idx = 0   
ax.set_title("Layer %d\tEpoch:1" % layers_num[0], fontstyle='italic', fontfamily='serif', fontsize=16) 
def update(frame):
    global last_layer_idx
    
    layer_idx = layers_idx[frame]
    epoch = (layers_idx==layer_idx)[:frame+1].sum()
    ax.set_title("Layer %d  Epoch:%d" % (layers_num[layer_idx], epoch), fontstyle='italic', fontfamily='serif', fontsize=16)
    # ax.set_title("Stage %d  Epoch:%d" % (layer_idx + 1, epoch))
    if layer_idx != last_layer_idx:
        print("Layer:", layers_num[layer_idx])
        for scat in prot0_scat_list:
            scat.remove()    
        for scat in prot1_scat_list:
            scat.remove()
        prot0_scat_list.clear()
        prot1_scat_list.clear()
        
        last_layer_idx = layer_idx
        for i in range(y[layer_idx].shape[0]):
            y_i, y_j = y[layer_idx][i][0], y[layer_idx][i][1]
            prot0_scat_list.append(ax.scatter(x=y_i, y=y_j, c = "green"))
        for i in range(z[layer_idx].shape[0]):
            z_i, z_j = z[layer_idx][i][0], z[layer_idx][i][1]
            prot1_scat_list.append(ax.scatter(x=z_i, y=z_j, c = "red"))
            
        for scat in scat_1_list:
            scat.remove()
        scat_1_list.clear()
        for i in range(x.shape[0]):
            first_D, second_D = x[i][0], x[i][1]
            # scat_1_list.append(ax.scatter(x=first_D[layer_idx], y=second_D[layer_idx], c = colors[classes[i][layer_idx]]))
            scat_1_list.append(ax.scatter(x=first_D[frame], y=second_D[frame], c = colors[classes[i][layer_idx]]))

    
    for i in range(x.shape[0]):
        first_D, second_D = x[i][0], x[i][1]
        scat_1_list[i].set_offsets(np.array([first_D[frame], second_D[frame]]))
    
    ax.relim()      # make sure all the data fits
    ax.autoscale()  # auto-scale
    
    # return line_list + scat_1_list + scat_2_list
    return  prot0_scat_list + prot1_scat_list + scat_1_list

# ax.axes.get_xaxis().set_ticks(ax.get_xticklabels(),fontstyle='italic', fontfamily='serif', fontsize=16)
# ax.axes.get_yaxis().set_ticks(ax.get_yticklabels(),fontstyle='italic', fontfamily='serif', fontsize=16)

plt.ylabel("i$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)
plt.xlabel("j$^{th}$ - Dimension", fontstyle='italic', fontfamily='serif', fontsize=16)

import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='serif',
                                   style='italic', size=10)

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig=fig, func=update, frames=x.shape[2], interval=300)
# ani.save(filename="/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/pillow_example.gif", writer="pillow")
ani.save(filename="/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/ffmpeg_example_val.mp4", writer="ffmpeg")
