import torch
from torch import nn

import numpy as np

class CCE(nn.Module):
    def __init__(self, clusters, alpha=5.0, beta=5.0, epsilon=1e-8, device=torch.device('cpu')):
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
            clusters[key] = clusters[key].view(1, numProt, 1, numFeat).to(self.device)
        self.clusters = clusters
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta        

    def forward(self, outputs, target_classes):
        device = outputs.device
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        f = r-a
        print("Reserved Memory: " + str(r))
        print("Allocated Memory: " + str(a))
        print("Free Memory: " + str(f))

        self.actual_closest_prototypes = torch.zeros(outputs.shape).to(device)
        wrong_closest_prototypes = torch.zeros(outputs.shape).to(device)

        batch_size = outputs.shape[0]
        numFeatures = outputs.shape[1]

        # Optimization Part
        outputs_expanded = outputs.view(batch_size, 1, 1, numFeatures)
        # print("outputs_expanded shape: " + str(outputs_expanded.shape))
        min_dis_per_class = []
        min_idx_per_class = []
        for class_indx in self.clusters.keys():
            # print("class_indx = " + str(class_indx))
            distances = torch.cdist(outputs_expanded.to(torch.float64), self.clusters[class_indx], p=2.0) # p=2 is the Euclidean distance
            # print("Distances shape: " + str(distances.shape))
            distances_min_values, distances_min_idx = torch.min(distances, dim=1, keepdim=True)
            min_dis_per_class.append(distances_min_values.squeeze())  
            # print("distances_min_values.shape = " + str(distances_min_values.squeeze().shape))          
            min_idx_per_class.append(distances_min_idx.squeeze())
            # print("distances_min_idx.shape = " + str(distances_min_idx.squeeze().shape))

        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        f = r-a
        print("Reserved Memory: " + str(r))
        print("Allocated Memory: " + str(a))
        print("Free Memory: " + str(f))

        min_dis_all_classes_stacked = torch.stack(min_dis_per_class)
        min_idx_all_classes_stacked = torch.stack(min_idx_per_class)

        out_idx = range(0, batch_size)
        actual_prot_idx = min_idx_all_classes_stacked[target_classes, out_idx]
        # print("actual_prot_idx.shape = " + str(actual_prot_idx.shape))
        min_dis_all_classes_stacked[target_classes, out_idx] = float("Inf")
        min_dis_wrong_classes_idx = torch.argmin(min_dis_all_classes_stacked, dim=0, keepdim=False)
        # print("min_dis_wrong_classes_idx.shape = " + str(min_dis_wrong_classes_idx.shape))
        wrong_prot_idx = min_idx_all_classes_stacked[min_dis_wrong_classes_idx, out_idx].view(-1)
        # print("wrong_prot_idx.shape = " + str(wrong_prot_idx.shape))
        
        for i, (a_prot_idx, t_class, w_prot_idx, w_class) in enumerate(zip(actual_prot_idx, target_classes, wrong_prot_idx, min_dis_wrong_classes_idx)):
            self.actual_closest_prototypes[i] = self.clusters[int(t_class)].squeeze()[a_prot_idx]
            wrong_closest_prototypes[i] = self.clusters[int(w_class)].squeeze()[w_prot_idx]

        
        # for i, (out, t_class) in enumerate(zip(outputs, target_classes)):
        #     t_class = t_class.item()
        #     target_clusters = self.clusters[t_class]
        #     actual_class_dists = torch.cdist(out.view(1, -1).to(torch.float64), target_clusters, p=2.0) # p=2 is the Euclidean distance
        #     protNumber = torch.argmin(actual_class_dists)
        #     target_closest_prototype = target_clusters[protNumber]
        #     self.actual_closest_prototypes[i] = target_closest_prototype
            
        #     min_distance = torch.Tensor([np.inf]).to(device)            
        #     for class_indx in self.clusters.keys():
        #         if class_indx != t_class:
        #             non_target_clusters = self.clusters[class_indx]
        #             wrong_class_dists = torch.cdist(out.view(1, -1).to(torch.float64), non_target_clusters, p=2.0) # p=2 is the Euclidean distance
        #             protNumber, distance = torch.argmin(wrong_class_dists), torch.min(wrong_class_dists)
        #             if distance < min_distance:
        #                 min_distance = distance
        #                 non_target_closest_prototype = non_target_clusters[protNumber]
        #                 self.wrong_closest_prototypes[i] = non_target_closest_prototype
        
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        f = r-a
        print("Reserved Memory: " + str(r))
        print("Allocated Memory: " + str(a))
        print("Free Memory: " + str(f))

        criterion = nn.MSELoss()
        target_loss = criterion(outputs, self.actual_closest_prototypes)
        non_target_loss = target_loss#criterion(outputs, wrong_closest_prototypes)

        outputs_expanded.detach().cpu()
        min_dis_all_classes_stacked.detach().cpu()
        min_idx_all_classes_stacked.detach().cpu()
        distances.detach().cpu()
        import gc
        gc.collect()

        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        f = r-a
        print("Reserved Memory: " + str(r))
        print("Allocated Memory: " + str(a))
        print("Free Memory: " + str(f))
        
        return (self.alpha * target_loss + self.beta * (1 - (non_target_loss)))
    
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