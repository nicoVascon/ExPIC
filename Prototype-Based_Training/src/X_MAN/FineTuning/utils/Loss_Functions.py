import torch
from torch import nn

import numpy as np

class CCE(nn.Module):
    def __init__(self, clusters, alpha=5.0, epsilon=1e-8, device=torch.device('cpu')):
        super(CCE, self).__init__()
        self.device = device
        for key in clusters.keys():
            if type(clusters[key]) != torch.Tensor:
                if type(clusters[key]) == np.ndarray:
                    # clusters[key] = torch.from_numpy(clusters[key]).to(self.device)
                    clusters[key] = torch.from_numpy(clusters[key])
                else:
                    clusters[key] = torch.tensor(clusters[key])
            else:
                clusters[key] = clusters[key]
            # Optimization Part
            numProt = clusters[key].shape[0]
            numFeat = clusters[key].shape[1]
            clusters[key] = clusters[key].view(1, numProt, 1, numFeat).to(self.device).to(torch.float32)
        self.clusters = clusters
        self.epsilon = epsilon
        self.alpha = alpha     

    def forward(self, outputs, target_classes):
        device = outputs.device

        self.actual_closest_prototypes = torch.zeros(outputs.shape).to(device)
        wrong_closest_prototypes = torch.zeros(outputs.shape).to(device)

        batch_size = outputs.shape[0]
        numFeatures = outputs.shape[1]

        # Optimization Part
        outputs_expanded = outputs.view(batch_size, 1, 1, numFeatures)
       
    
        min_dis_per_class = torch.zeros(len(self.clusters),batch_size).to(device)
        min_idx_per_class = torch.zeros(len(self.clusters),batch_size).to(device)
     
        for class_indx in self.clusters.keys():
            distances = torch.cdist(outputs_expanded, self.clusters[class_indx], p=2.0) # p=2 is the Euclidean distance
            distances_min_values, distances_min_idx = torch.min(distances, dim=1, keepdim=True)
            min_dis_per_class[class_indx] = distances_min_values.squeeze()
            min_idx_per_class[class_indx] = distances_min_idx.squeeze()
            

        out_idx = range(0, batch_size)
        actual_prot_idx = min_idx_per_class[target_classes, out_idx].view(-1)

        min_dis_per_class[target_classes.tolist(), list(out_idx)] = float("Inf")
        min_dis_wrong_classes_idx = torch.argmin(min_dis_per_class, dim=0, keepdim=False)
    
        wrong_prot_idx = min_idx_per_class[min_dis_wrong_classes_idx, out_idx].view(-1)
        # print(self.clusters[target_classes.detach().cpu().numpy()])
        # self.actual_closest_prototypes = self.clusters[target_classes.item()].squeeze()[actual_prot_idx.to(torch.int)]
        # wrong_closest_prototypes = self.clusters[int(w_class)].squeeze()[w_prot_idx.to(torch.int)]
        
        for i, (a_prot_idx, t_class, w_prot_idx, w_class) in enumerate(zip(actual_prot_idx, target_classes, wrong_prot_idx, min_dis_wrong_classes_idx)):
            self.actual_closest_prototypes[i] = self.clusters[int(t_class)].squeeze()[a_prot_idx.to(torch.int)]
            wrong_closest_prototypes[i] = self.clusters[int(w_class)].squeeze()[w_prot_idx.to(torch.int)]

        criterion = nn.MSELoss()
        target_loss = criterion(outputs, self.actual_closest_prototypes)
        non_target_loss = criterion(outputs, wrong_closest_prototypes)

        # print("\n\ttarget_loss = \t" + str(target_loss) + "\n")
        # print("\n\tnon_target_loss = \t" + str(non_target_loss) + "\n")
        # print("\n\t1 / (non_target_loss) = \t" + str(1 / (non_target_loss)) + "\n")

        # print("\n\n\t(1 - self.alpha) * target_loss = " + str((1 - self.alpha) * target_loss) + "\n")
        # print("\n\t(self.alpha) * (1 / (non_target_loss + self.epsilon)) = " + str((self.alpha) * (1 / (non_target_loss + self.epsilon))) + "\n")

        del min_dis_per_class
        del min_idx_per_class
        del wrong_closest_prototypes
        del outputs_expanded
        del distances
        del actual_prot_idx
        del distances_min_values
        del distances_min_idx

        import gc
        gc.collect()

        # return (self.alpha * target_loss + self.beta * (1 - (non_target_loss)))
        # return (self.alpha * target_loss + (1 - self.alpha) * (1 / (non_target_loss + self.epsilon)))
        return ((1 - self.alpha) * target_loss + (self.alpha) * (1 / (non_target_loss + self.epsilon)))
    
    def setClusters(self, clusters):
        for key in clusters.keys():
            if type(clusters[key]) != torch.Tensor:
                if type(clusters[key]) == np.ndarray:
                    clusters[key] = torch.from_numpy(clusters[key]).to(self.device)
                else:
                    clusters[key] = torch.tensor(clusters[key]).to(self.device)
            else:
                clusters[key] = clusters[key].to(self.device)
        self.clusters = clusters