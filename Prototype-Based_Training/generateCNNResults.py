from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import os
import sys
sys.path.insert(0, '/nfs/home/nvasconcellos.it/softLinkTests/X-MAN/Prototype-Based_Training/src')

import X_MAN
import X_MAN.FineTuning.utils.Generic_Functions as gf
from X_MAN.Models.VGG16.Model.Converted_VGG16 import Converted_VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

root_dir = "/nfs/home/nvasconcellos.it/softLinkTests/Datasets/COVID-QU-Ex_Dataset/Lung_Segmentation_Data/Lung_Segmentation_Data/test"
resultsModel_dir = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Test_LossFunctionOptimization/results"

# Functions
def displayMetrics(x_labels, y_values):
    
    fig, ax = plt.subplots()
    tab2 = [['%.5f' % y] for x, y in zip(x_labels, y_values)]
    rcolors = plt.cm.BuPu(np.full(len(y_values), 0.1))
    ccolors = plt.cm.BuPu(np.full(1, 0.1))
    

    ytable = plt.table(cellText=tab2, rowLabels=x_labels, colLabels=['Metrics'], rowColours=rcolors,
                        colWidths=[.3]*5, loc='center', colColours=ccolors)
    ytable.set_fontsize(24)
    ytable.scale(1, 4)
    plt.gcf().set_size_inches(8, 6)
    plt.axis('off')    
    
    
    figure_name = resultsModel_dir + '/Performance_Metrics.jpg'
    plt.savefig(figure_name)
   
def displayConfMatrix(cf_matrix, class_names): 
    group_names = ['True Neg','False Pos','False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    # group_percentages = ["{0:.2%}".format(value) for value in
    #                     cf_matrix.flatten()/np.sum(cf_matrix)]
    group_percentages = []
    
    for i in range(cf_matrix.shape[0]):
        for value in (cf_matrix[i].flatten()/np.sum(cf_matrix[i])):
            group_percentages.append("{0:.2%}".format(value))

    # labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    #         zip(group_names,group_counts,group_percentages)]
    labels = [f"{v2}\n{v3}" for v2, v3 in
            zip(group_counts,group_percentages)]

    num_classes = len(class_names)
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(['Non-Covid', 'Covid'])
    # ax.yaxis.set_ticklabels(['Non-Covid', 'Covid'])
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)

    ## Display the visualization of the Confusion Matrix.  
    figure_name = resultsModel_dir + '/Confution_Matrix.jpg'
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(figure_name)



class DatasetSequence(Dataset):
    # def __init__(self, x_set, y_set):
    #     self.x = x_set
    #     self.y = y_set
    #     # self.transform = transforms.Compose([
    #     #     transforms.Resize(256),
    #     #     transforms.CenterCrop(224),# VGG-16 Takes 224x224 images as input, so we resize all of them
    #     #     transforms.ToTensor()
    #     #     # ,transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     #     ]
    #     # )
    
    def __init__(self, root_dir):
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.classes_as_indx = {self.classes[i]: i for i in range(len(self.classes))}
        
        self.x = []
        self.y = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.x.append(img_path)
                self.y.append(self.classes_as_indx[class_name])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):        
        
        img_path = self.x[idx]
        # img_raw = Image.open(img_path)
        # img = Image.new("RGB", img_raw.size)
        # img.paste(img_raw)
        # img = self.transform(img)

        # # x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        # x = Variable(img.float(), requires_grad=False)
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = torch.from_numpy(x.copy())
        
        # y = torch.Tensor(self.y[idx]).view(-1)
        
        # x = torch.rand(1, 224, 224, 3)
        return x, self.y[idx]

weights_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/FineTuning_VGG16_CB/COVID-QU-Ex_Dataset/Prototype-Based_Training/Test_LossFunctionOptimization/checkpoints/best_model.pt"
fe_layer = "last_fc"

model = Converted_VGG16(weights_path, fe_layer = fe_layer, numClasses = 3)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size=128

val_ds = DatasetSequence(root_dir)
val_dl = DataLoader(val_ds, batch_size=batch_size, 
                        num_workers=4)

val_acc = 0

y_pred = torch.zeros(1, len(val_ds.y)).to(device)

model.eval()
batch = 0
with torch.no_grad():            
    for x_batch, y_classes in gf.progressbar(val_dl, "Val. \tEpoch %d: " % (1), "Batches", 40):
        x_batch = x_batch.to(device)
        
        outputs = model(x_batch)
        outputs = outputs / torch.sqrt(outputs.pow(2).sum(axis=1, keepdim=True))
        y_pred_batch = torch.argmax(outputs, dim=1)
        acc = torch.sum(y_pred_batch == y_classes.to(device))/outputs.shape[0]
        val_acc += acc
        # print("y_pred.shape: " + str(y_pred.shape))
        # print("y_pred_batch.shape: " + str(y_pred_batch.shape))

        y_pred[0, range(batch, batch + y_pred_batch.shape[0])] = y_pred_batch.view(1, -1).to(torch.float32)
        batch += y_pred_batch.shape[0]
        
val_acc /= len(val_dl)
print('Val Accuracy: %f' % val_acc)

y_test_labels = np.array(val_ds.y).reshape(-1)
y_pred = y_pred.cpu().numpy().reshape(-1)
# y_test_labels = val_ds.y
average = 'macro'

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_labels, y_pred)
# accuracy = val_acc
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
# print("y_test_labels.shape: " + str(y_test_labels.shape))
# print("y_pred.shape: " + str(y_pred.shape))

# precision = precision_score(y_test_labels.reshape(1, -1) , y_pred.cpu().numpy().reshape(1, -1), average=average)
precision = precision_score(y_test_labels, y_pred, average=average)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test_labels , y_pred,average=average)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_labels , y_pred, average=average)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_test_labels , y_pred)
print('Cohens kappa: %f' % kappa)

# confusion matrix
cf_matrix = confusion_matrix(y_test_labels , y_pred)
print("Confusion Matrix:\n",cf_matrix)

metricsLabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cohens kappa']
metricsValues = [accuracy, precision, recall, f1, kappa]


displayConfMatrix(cf_matrix, val_ds.classes)
displayMetrics(metricsLabels, metricsValues)
# plt.show()
plt.close('all')


# Save Results in a Text File
f = open(resultsModel_dir + "/Experiment_Results.txt","w+")

# covid_samples = 1252
# non_covid_samples = 1229

f.write("Experiment Parameters:\r\n")
f.write('\tTest Size:\t\t\t\t%d Images\r\n' % y_test_labels.shape[-1])
# f.write(f'\tCovid Samples:\t\t\t{covid_samples} Images\r\n')
# f.write(f'\tNon-Covid Samples:\t\t{non_covid_samples} Images\r\n')
f.write('\tNum. Classes:\t\t\t\t%d\r\n' % len(val_ds.classes))
f.write('\tClasses Names:\t\t\t%s\r\n' % val_ds.classes)

f.write("\nExperiment Results:\r\n")
for metric, value in zip(metricsLabels, metricsValues):
    f.write('\t%s:\t\t\t\t%.6f\r\n' % (metric, value))