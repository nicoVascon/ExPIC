import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import math

from scipy.spatial.distance import cdist
from torch.nn.functional import softmax

import torch
import math
from scipy.spatial.distance import cdist


def xDNN(Input, Mode, **kwargs):
    
    # normalizedVec = torch.from_numpy(Input['Features'])
    normalizedVec = Input['Features'].to(torch.float32)
    normalizedVecNormUnit = normalizedVec / torch.sqrt((normalizedVec**2).sum(axis=1).reshape(-1, 1))
    Input['Features'] = normalizedVecNormUnit.to(torch.float32)

    if Mode == 'Learning':        
        # Images = torch.from_numpy(Input['Images'])
        Images = Input['Images']
        # Features = torch.from_numpy(Input['Features'])
        # Labels = torch.from_numpy(Input['Labels'])
        Features = Input['Features']
        Labels = Input['Labels']
        CN = torch.max(Labels).item()
            
        if kwargs.get("config_dic") is not None:
            config_dic = kwargs["config_dic"]
        else:
            config_dic = {
                "initial_Radius": torch.sqrt(2*(1 - torch.cos(math.pi/6))),
                "c_W_k": 0.5,
                "xDNN_Subprocesses": ["Evolving"]
            }
            
        Prototypes = PrototypesIdentification(Images, Features, Labels, CN, config_dic)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0, CN+1):
           MemberLabels[i]=Input['Labels'][np.where(Input['Labels']==i)].to(torch.float32)
        Output['xDNNParms']['CurrentNumberofClass']=CN+1
        Output['xDNNParms']['OriginalNumberofClass']=CN+1
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    elif Mode == 'Validation':
        Params=Input['xDNNParms']
        # datates=torch.from_numpy(Input['Features'])
        datates= Input['Features']
        Test_Results = DecisionMaking(Params, datates)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'], Output['EstLabs'])
        m = torch.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        # Output['ClassAcc'] = torch.sum(Output['ConfMa']*m(len(Output['ConfMa'])))/len(Input['Labels'])
        Output['ObtainedRules'] = Test_Results['ObtainedRules']
        return Output



def PrototypesIdentification(Image, GlobalFeature, LABEL, CL, config_dic):
    data = {}
    image = {}
    label = {}
    Prototypes = {}
    for i in range(0,CL+1):
        seq = torch.where(LABEL==i)
        # data[i]=GlobalFeature[seq,]
        data[i]=GlobalFeature[LABEL==i]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = torch.ones((len(seq),1))*i
    for i in range(0, CL+1):
        if "Evolving" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = xDNNclassifier(data[i], image[i], initial_Radius = config_dic['initial_Radius'])
        if "Offline" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = xDNNclassifier_Offline(data[i])
        if "Prot_Filtering" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = PrototypesFiltering(Prototypes[i], data[i], image[i], c_W_k = config_dic['c_W_k'])
        if "Outliers_Filtering" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = OutliersPrototypesFiltering(Prototypes[i], data[i], image[i])
        
    return Prototypes

def DecisionMaking(Params, datates):
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    VV = 1
    LTes=datates.shape[0]
    EstimatedLabels = torch.zeros((LTes))
    Scores=torch.zeros((LTes,CurrentNC))
    obtained_rules = []
    
    for i in range(1, LTes + 1):
        data = datates[i-1,]
        R=torch.zeros((VV,CurrentNC))
        Value=torch.zeros((CurrentNC,1))
        if_thenRules = []
        for k in range(0,CurrentNC):
            dts = torch.cdist(data.reshape(1, -1), PARAM[k]['Centre'], p=2)

            indxMin = torch.argsort(dts)[0,:4]
            # prot_dist = torch.round(dts[0,indxMin], decimals=4).tolist()
            prot_dist = dts[0,indxMin].tolist()
            prot_names = "N/A"
            distance=torch.sort(dts)[0]
            DataCloudsVariances = PARAM[k]["DataCloudsVariances"]
            DegreesOfSimilarity = 1 / (1 + ((data.reshape(1, -1) - PARAM[k]['Centre'])**2).sum(axis=1).reshape(-1, 1) / PARAM[k]['variance'])
            
            Value[k]=distance[0,0]
            
            if_thenRules.append((prot_names, prot_dist))

        obtained_rules.append(if_thenRules) 

        Value = softmax(torch.Tensor(-1*Value**2), dim=0).T
        Scores[i-1,] = Value
        Value = Value[0]
        Value_new = torch.sort(torch.Tensor(Value))[0][-1]
        indx = torch.argsort(torch.Tensor(Value))[-1]
        EstimatedLabels[i-1]=indx.item()

    LABEL1=torch.zeros((CurrentNC,1))

    for i in range(0,CurrentNC): 
        LABEL1[i] = torch.unique(torch.Tensor(LAB[i]))

    EstimatedLabels = EstimatedLabels.long()
    EstimatedLabels = LABEL1[EstimatedLabels]   

    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    dic['ObtainedRules'] = obtained_rules
    return dic



def xDNNclassifier(Data, Image, initial_Radius=30):
    print("------------- xDNN Classifier -------------")
    L, N, W = Data.shape

    radius = torch.sqrt(2 * (1 - math.cos(initial_Radius * (math.pi / 180))))

    data = torch.Tensor(Data.reshape(L, -1))
    Centre = torch.Tensor([data[0,]])
    Center_power = torch.pow(Centre, 2)
    X = torch.Tensor([[torch.sum(Center_power)]])
    Support = torch.Tensor([[1]])
    Noc = 1
    GMean = Centre.clone()
    QuadMean = (Centre.clone() ** 2).sum()
    Radius = torch.Tensor([[radius]])
    ND = 1
    VisualPrototype = {}
    VisualPrototype[1] = Image[0]
    sumOfSquareVectors = torch.Tensor([[(data[0,].view(1, -1) ** 2).sum()]])

    for i in range(2, L + 1):
        GMean = ((i - 1) / i) * GMean + data[i - 1,] / i
        QuadMean = 1
        variance = QuadMean - (GMean ** 2).sum()
        CentreDensity = 1 / (1 + ((Centre - torch.kron(torch.ones((Noc, 1)), GMean)) ** 2).sum(axis=1).view(Noc, 1) / variance)

        normalizedCenters = Centre / torch.sqrt((Centre ** 2).sum(axis=1).view(Noc, 1))

        CDmax = CentreDensity.max()
        CDmin = CentreDensity.min()

        DataDensity = 1 / (1 + ((data[i - 1,] - GMean) ** 2).sum() / variance)

        distance = cdist(data[i - 1,].view(1, -1).numpy(), Centre.numpy(), 'euclidean')[0]
        value, position = distance.min(0), distance.argmin(0)

        if DataDensity >= CDmax or DataDensity <= CDmin or value >= Radius[position]:

            Centre = torch.vstack((Centre, data[i - 1,]))
            sumOfSquareVectors = torch.vstack((sumOfSquareVectors, (data[i - 1,].view(1, -1) ** 2).sum()))
            Noc = Noc + 1
            VisualPrototype[Noc] = Image[i - 1]
            X = torch.vstack((X, ND))
            Support = torch.vstack((Support, 1))
            Radius = torch.vstack((Radius, radius))
        else:
            Centre[position,] = Centre[position,] * (Support[position] / (Support[position] + 1)) + data[i - 1] / (Support[position] + 1)
            sumOfSquareVectors[position] += (data[i - 1,].view(1, -1) ** 2).sum()
            Support[position] = Support[position] + 1
            Radius[position] = torch.sqrt(((Radius[position] ** 2) + (1 - (Centre[position,] ** 2).sum())) / 2)
            VisualPrototype[position + 1] = torch.vstack((VisualPrototype[position + 1], Image[i - 1]))

    dic = {}
    dic['Noc'] = Noc
    dic['Centre'] = Centre
    dic['Support'] = Support
    dic['Radius'] = Radius
    dic['GMean'] = GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] = L
    dic['X'] = X
    dic["variance"] = variance
    dic["DataCloudsVariances"] = (sumOfSquareVectors / Support) - (Centre ** 2).sum(axis=1).view(-1, 1)
    return dic


def xDNNclassifier_Offline(Data):
    print("---------- xDNNclassifier_Offline ----------")
    K = Data.shape[0]
    data = Data.clone()
    GMean = data.sum(dim=0) / K
    variance = ((data**2).sum() / K) - (GMean**2).sum()
    DataDensities = 1/(1 + ((data.view(K, -1) - torch.ones((K, 1)).to(data.device) * GMean)**2).sum(dim=1).view(K, 1)/variance)
    globalDataDensity = DataDensities.sum()/K
    Typicalities = DataDensities / globalDataDensity
    m_star = torch.argmax(Typicalities)
    u = data.view(K, -1)
    u_idx = list(range(0, K))
    z_idx = []
    z_idx.append(m_star.item())
    Typicalities_z_star = []
    Typicalities_z_star.append(Typicalities[m_star].item())
    u_idx_position = m_star

    while True:
        r = u[z_idx[-1]]
        u_idx.pop(u_idx_position)
        print(f"Remaining Feature Vectors: {len(u_idx)}/{K}", )
        if len(u_idx) == 0:
            break
        distances = torch.cdist(r.view(1, -1), u[u_idx,].view(len(u_idx), -1))
        u_idx_position = torch.argmin(distances).item()

        n_star = u_idx[u_idx_position]
        z_idx.append(n_star)
        Typicalities_z_star.append(Typicalities[n_star].item())

    z_star_idx = []
    z_star_idx.append(z_idx[0])
    for i in range(1, len(z_idx)):
        if i == len(z_idx) - 1:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1]:
                z_star_idx.append(z_idx[i])
        else:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1] and Typicalities_z_star[i] > Typicalities_z_star[i + 1]:
                z_star_idx.append(z_idx[i])

    Noc = len(z_star_idx)
    print("Number of Clouds: ", Noc)
    z_star_np = u[z_star_idx].view(Noc, data.shape[-1])
    Support = torch.zeros((Noc, 1)).to(data.device)
    sumOfMembers = torch.zeros((Noc, data.shape[-1])).to(data.device)
    sumOfSquareVectors = torch.zeros((Noc, 1)).to(data.device)
    # Stage 2: Creating the Voronoi Tessellation
    for i in range(0, data.shape[0]):
        distance = torch.cdist(data[i,].view(1,-1),z_star_np)
        position = torch.argmin(distance).item()
        Support[position] += 1
        sumOfMembers[position,] += data[i].view(sumOfMembers[position].shape)
        sumOfSquareVectors[position] += (data[i]**2).sum()

    Centres = sumOfMembers / Support # Updating the Data Cloud Centers 
    DataCloudsVariances = (sumOfSquareVectors / Support) - (Centres**2).sum(dim=1).view(-1, 1)   

    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic["variance"] = variance
    dic["DataCloudsVariances"] = DataCloudsVariances
    return dic


def PrototypesFiltering(prototypes, Data, Image, c_W_k = 0.5):
    print("------------- Starting Prototypes Filtering -------------")
    device = Data.device
    Noc = prototypes['Noc']       
    Centres = prototypes['Centre'].clone()
    Support = prototypes['Support'].clone()
    GMean = prototypes['GMean'].clone()
    VisualPrototype = "N/A"
    variance = prototypes["variance"]
    DataCloudsVariances = prototypes["DataCloudsVariances"]
    data = Data.clone()
    FilteringIter = 0
    visualSupport = []
             
    while True:
        print("Filtering Iteration = ", FilteringIter)
        CentreDensitys = 1/(1 + ((Centres - torch.ones(Noc, 1, device=device) * GMean)).pow(2).sum(dim=1, keepdim=True)/variance)
        globalDataDensity = (Support * CentreDensitys).sum()
        CentreTypicalities = (Support * CentreDensitys) / globalDataDensity
        
        distances = torch.cdist(Centres, Centres)
        
        Y_k = distances.sum() / (Noc * (Noc-1))    
        M_Y = (distances < Y_k).sum().item()
        print("M_Y = ", M_Y)
        if M_Y == 0: # This wont probably happen
            break
        
        overline_W_k = (distances * (distances < Y_k)).sum() / M_Y
        M_ovline_W = ((distances < overline_W_k) * (distances > 0)).sum().item()
        print("M_ovline_W = ", M_ovline_W)
        if M_ovline_W == 0: # This wont probably happen
            break
        
        W_k = (distances * (distances < overline_W_k)).sum() / M_ovline_W
        
        C_star = (distances <= W_k*c_W_k) * (distances > 0)
        print("C_star = ", C_star.sum().item())
        if C_star.sum().item() == 0:
            break
        
        typicalities_C_star = (torch.ones(Noc, Noc, device=device)*CentreTypicalities.T) * C_star
        numGreaterLocalTypicalities = (typicalities_C_star > CentreTypicalities).sum(dim=1).reshape(-1, 1)
        newLocalMaxima_position = numGreaterLocalTypicalities == 0
        numNewLocalMaxima = newLocalMaxima_position.sum().item()
        newLocalMaxima = torch.zeros((numNewLocalMaxima, Centres.shape[1]), device=device, dtype=torch.float64)
        i = 0
        for centre, islocalMax in zip(Centres, newLocalMaxima_position):
            if islocalMax:
                newLocalMaxima[i] = centre            
                i += 1
                
        Support = torch.zeros((numNewLocalMaxima, 1), device=device)
        visualSupport = [None] * numNewLocalMaxima
        sumOfMembers = torch.zeros((numNewLocalMaxima, Centres.shape[1]), device=device)
        # Stage 2: Creating the Voronoi Tessellation
        for i in range(0, data.shape[0]):
            distance = torch.cdist(data[i,].view(1,-1),newLocalMaxima, p=2)
            position = torch.argmin(distance)
            Support[position] = Support[position] + 1

            if visualSupport[position] == None:
                visualSupport[position] = []
            # visualSupport[position].append(Image[i])
            sumOfMembers[position,] = sumOfMembers[position,] + data[i].view(sumOfMembers[position].shape)
            
        Centres = sumOfMembers / Support # Updating the Data Cloud Centers
        Noc = numNewLocalMaxima
        FilteringIter += 1
    
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic["variance"] = variance
    dic["FilteringIter"] = FilteringIter
    dic["visualSupport"] = visualSupport
    dic["DataCloudsVariances"] = DataCloudsVariances
    return dic


def OutliersPrototypesFiltering(prototypes, Data, Image):
    print("------------- Starting Outliers Prototypes Filtering -------------")
    Noc = prototypes['Noc']       
    Centres = prototypes['Centre'].clone()
    Support = prototypes['Support'].clone()
    GMean = prototypes['GMean'].clone()
    # VisualPrototype = prototypes['Prototype'] 
    VisualPrototype = "N/A"
    variance = prototypes["variance"]
    DataCloudsVariances = prototypes["DataCloudsVariances"]
    data = Data.clone()
    visualSupport = []             
    
    newLocalMaxima_position = Support != 1
    numNewLocalMaxima = newLocalMaxima_position.sum()
    newLocalMaxima = torch.zeros((numNewLocalMaxima, Centres.shape[1]))
    i = 0
    for centre, islocalMax in zip(Centres, newLocalMaxima_position):
        if islocalMax:
            newLocalMaxima[i] = centre            
            i += 1
            
    Support = torch.zeros((numNewLocalMaxima, 1))
    visualSupport = [None] * numNewLocalMaxima
    sumOfMembers = torch.zeros((numNewLocalMaxima, Centres.shape[1]))
    sumOfSquareVectors = torch.zeros((numNewLocalMaxima, 1))
    # Stage 2: Creating the Voronoi Tessellation
    for i in range(0, data.shape[0]):
        distance = cdist(data[i,].reshape(1,-1),newLocalMaxima,'euclidean')[0]
        position = distance.argmin(0)
        Support[position] += 1
        if visualSupport[position] == None:
            visualSupport[position] = []
        # visualSupport[position].append(Image[i])
        sumOfMembers[position,] += data[i].reshape(sumOfMembers[position].shape)
        sumOfSquareVectors[position] += (data[i]**2).sum()
        
    Centres = sumOfMembers / Support # Updating the Data Cloud Centers
    DataCloudsVariances = (sumOfSquareVectors / Support) - (Centres**2).sum(axis=1).reshape(-1, 1)
    Noc = numNewLocalMaxima
    
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres.to(torch.float32)
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic["variance"] = variance
    dic["visualSupport"] = visualSupport
    dic["DataCloudsVariances"] = DataCloudsVariances
    return dic