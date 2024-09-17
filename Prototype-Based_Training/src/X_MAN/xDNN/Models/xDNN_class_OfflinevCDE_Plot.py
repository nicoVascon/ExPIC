"""
Please cite:
    
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.
"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax 
from sklearn.metrics import confusion_matrix



def xDNN(Input,Mode, **kwargs):
    
    normalization = kwargs.get("Nomalization", "L2-Norm")
    print("Normalization: ", normalization)
    if normalization == "L2-Norm":
        normalizedVecNormUnit = Input['Features'] /np.sqrt((Input['Features']**2).sum(axis=1).reshape(-1, 1))
        Input['Features'] = normalizedVecNormUnit    
    
    classifier = kwargs.get("classifier", "xDNN")
    print("Classifier: ", classifier)
    if classifier == "kNN":
        k_neighbors = kwargs.get("k_neighbors")
        print("k_neighbors: ", k_neighbors)
        if k_neighbors is None:
            raise Exception("k_neighbors cannot be None if classifier is kNN")
    
    if kwargs.get("config_dic") is not None:
        config_dic = kwargs["config_dic"]
    else:
        config_dic = {
            "initial_Radius": 30,
            "c_W_k": 0.5,
            "xDNN_Subprocesses": ["Evolving"]
        }
    
    if "Mean-Shift" in config_dic['xDNN_Subprocesses']:
        bandwidth = kwargs.get("bandwidth")
        print("bandwidth: ", bandwidth)
        if bandwidth is None:
            raise Exception("bandwidth cannot be None if Mean-Shift is used")
        config_dic['Mean-Shift'] = {}
        config_dic['Mean-Shift']['bandwidth'] = bandwidth
    elif "Custom_Clustering" in config_dic['xDNN_Subprocesses']:
        custom_clustering = kwargs.get("custom_clustering")  
        config_dic['Custom_Clustering'] = custom_clustering
        
    
    ### End Modification
    if Mode == 'Learning':        
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
            
        Prototypes = PrototypesIdentification(Images,Features,Labels,CN, config_dic)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0,CN+1):
           MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        print("MemberLabels: ", MemberLabels)
        Output['xDNNParms']['CurrentNumberofClass']=CN+1
        Output['xDNNParms']['OriginalNumberofClass']=CN+1
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    elif Mode == 'Validation':
        Params=Input['xDNNParms']
        datates=Input['Features']
        if classifier == "xDNN":
            Test_Results = DecisionMaking(Params,datates)
        elif classifier == "kNN":
            Test_Results = DecisionMaking_kNN(Params,datates, k_neighbors)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        Output['ObtainedRules'] = Test_Results['ObtainedRules']
        return Output
    
def PrototypesIdentification(Image,GlobalFeature,LABEL,CL, config_dic):
    data = {}
    image = {}
    label = {}
    Prototypes = {}
    for i in range(0,CL+1):
        seq = np.argwhere(LABEL==i)
        data[i]=GlobalFeature[seq,]
        image[i] = {}
        for j in range(0, len(seq)):
            image[i][j] = Image[seq[j][0]]
        label[i] = np.ones((len(seq),1))*i
    for i in range(0, CL+1):
        if "Mean-Shift" == config_dic['xDNN_Subprocesses'][0]:
            Prototypes[i] = MeanShift_Clustering(data[i], config_dic['Mean-Shift']['bandwidth'])
        elif "Evolving" == config_dic['xDNN_Subprocesses'][0]:
            Prototypes[i] = xDNNclassifier(data[i],image[i], initial_Radius = config_dic['initial_Radius'])
        elif "Offline" == config_dic['xDNN_Subprocesses'][0]:
            Prototypes[i] = xDNNclassifier_Offline(data[i])
        elif "Custom_Clustering" == config_dic['xDNN_Subprocesses'][0]:
            Prototypes[i] = CustomClustering(data[i], config_dic['Custom_Clustering'])
        
        if "Prot_Filtering" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = PrototypesFiltering(Prototypes[i], data[i],image[i], c_W_k = config_dic['c_W_k'])
        if "Outliers_Filtering" in config_dic['xDNN_Subprocesses']:
            Prototypes[i] = OutliersPrototypesFiltering(Prototypes[i], data[i],image[i])
        
    return Prototypes

def OutliersPrototypesFiltering(prototypes, Data, Image):
    print("------------- Starting Outliers Prototypes Filtering -------------")
    Noc = prototypes['Noc']       
    Centres = prototypes['Centre'].copy()
    Support = prototypes['Support'].copy()
    GMean = prototypes['GMean'].copy()
    # VisualPrototype = prototypes['Prototype'] 
    VisualPrototype = "N/A"
    variance = prototypes["variance"]
    DataCloudsVariances = prototypes["DataCloudsVariances"]
    data = Data.copy()
    visualSupport = []             
    
    newLocalMaxima_position = Support != 1
    numNewLocalMaxima = newLocalMaxima_position.sum()
    newLocalMaxima = np.zeros((numNewLocalMaxima, Centres.shape[1]))
    i = 0
    for centre, islocalMax in zip(Centres, newLocalMaxima_position):
        if islocalMax:
            newLocalMaxima[i] = centre            
            i += 1
            
    Support = np.zeros((numNewLocalMaxima, 1))
    visualSupport = [None] * numNewLocalMaxima
    sumOfMembers = np.zeros((numNewLocalMaxima, Centres.shape[1]))
    sumOfSquareVectors = np.zeros((numNewLocalMaxima, 1))
    # Stage 2: Creating the Voronoi Tessellation
    for i in range(0, data.shape[0]):
        distance = cdist(data[i,].reshape(1,-1),newLocalMaxima,'euclidean')[0]
        position = distance.argmin(0)
        Support[position] += 1
        if visualSupport[position] == None:
            visualSupport[position] = []
        visualSupport[position].append(Image[i])
        sumOfMembers[position,] += data[i].reshape(sumOfMembers[position].shape)
        sumOfSquareVectors[position] += (data[i]**2).sum()
        
    Centres = sumOfMembers / Support # Updating the Data Cloud Centers
    DataCloudsVariances = (sumOfSquareVectors / Support) - (Centres**2).sum(axis=1).reshape(-1, 1)
    Noc = numNewLocalMaxima
    
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic["variance"] = variance
    dic["visualSupport"] = visualSupport
    dic["DataCloudsVariances"] = DataCloudsVariances
    return dic

def PrototypesFiltering(prototypes, Data, Image, c_W_k = 0.5):
    print("------------- Starting Prototypes Filtering -------------")
    Noc = prototypes['Noc']       
    Centres = prototypes['Centre'].copy()
    Support = prototypes['Support'].copy()
    GMean = prototypes['GMean'].copy()
    VisualPrototype = "N/A"
    variance = prototypes["variance"]
    DataCloudsVariances = prototypes["DataCloudsVariances"]
    data = Data.copy()
    FilteringIter = 0
    visualSupport = []
             
    while True:
        print("Filtering Iteration = ", FilteringIter)
        CentreDensitys = 1/(1 + ((Centres - np.kron(np.ones((Noc, 1)), GMean))**2).sum(axis=1).reshape(Noc, 1)/variance)
        globalDataDensity = (Support * CentreDensitys).sum()
        CentreTypicalities = (Support * CentreDensitys) / globalDataDensity
        
        distances = cdist(Centres, Centres,'euclidean')
        
        Y_k = distances.sum() / (Noc * (Noc-1))    
        M_Y = ((distances < Y_k) * (distances > 0)).sum()
        print("M_Y = ", M_Y)
        if M_Y == 0: # This wont probably happen
            break
        
        overline_W_k = (distances * (distances < Y_k)).sum() / M_Y
        M_ovline_W = ((distances < overline_W_k) * (distances > 0)).sum()
        print("M_ovline_W = ", M_ovline_W)
        if M_ovline_W == 0: # This wont probably happen
            break
        
        W_k = (distances * (distances < overline_W_k)).sum() / M_ovline_W
        
        C_star = (distances <= W_k*c_W_k) * (distances > 0)
        print("C_star = ", C_star.sum())
        if C_star.sum() == 0:
            break
        
        typicalities_C_star = (np.ones((Noc, Noc))*CentreTypicalities.T) * C_star
        numGreaterLocalTypicalities = (typicalities_C_star > CentreTypicalities).sum(axis=1).reshape(-1, 1)
        newLocalMaxima_position = numGreaterLocalTypicalities == 0
        numNewLocalMaxima = newLocalMaxima_position.sum()
        newLocalMaxima = np.zeros((numNewLocalMaxima, Centres.shape[1]))
        i = 0
        for centre, islocalMax in zip(Centres, newLocalMaxima_position):
            if islocalMax:
                newLocalMaxima[i] = centre            
                i += 1
                
        Support = np.zeros((numNewLocalMaxima, 1))
        visualSupport = [None] * numNewLocalMaxima
        sumOfMembers = np.zeros((numNewLocalMaxima, Centres.shape[1]))
        # Stage 2: Creating the Voronoi Tessellation
        for i in range(0, data.shape[0]):
            distance = cdist(data[i,].reshape(1,-1),newLocalMaxima,'euclidean')[0]
            position = distance.argmin(0)
            Support[position] += 1
            if visualSupport[position] == None:
                visualSupport[position] = []
            visualSupport[position].append(Image[i])
            sumOfMembers[position,] += data[i].reshape(sumOfMembers[position].shape)
            
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

def CustomClustering(Data, clustering_function : callable):  
    print("------------- Starting Custom Clustering -------------")  
    K = np.shape(Data)[0]
    data = Data.copy()
    GMean = data.sum(axis=0) / K
    variance = ((data**2).sum() / K) - (GMean**2).sum()
    data = np.squeeze(data)
    print("data.shape: ", data.shape)
    Centres = clustering_function(data)    
    Noc = Centres.shape[0]
    Support = np.zeros((Noc, 1))
    # Creating the Voronoi Tessellation
    for i in range(0, data.shape[0]):
        distance = cdist(data[i,].reshape(1,-1),Centres,'euclidean')[0]
        position = distance.argmin(0)
        Support[position] += 1
    
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic["variance"] = variance
    return dic

def MeanShift_Clustering(Data, bandwidth):
    from sklearn.cluster import MeanShift
    meanShift_seed = 1e-5
    
    K = np.shape(Data)[0]
    data = Data.copy()
    GMean = data.sum(axis=0) / K
    variance = ((data**2).sum() / K) - (GMean**2).sum()
    data = np.squeeze(data)
    print("data.shape: ", data.shape)
    # clustering = MeanShift(bandwidth=5, seeds=np.ones(data.shape)*meanShift_seed).fit(data)
    clustering = MeanShift(bandwidth=bandwidth).fit(data)
    
    Centres = clustering.cluster_centers_
    Noc = Centres.shape[0]
    labels = clustering.labels_
    Support = np.array([[(labels == class_idx).sum()] for class_idx in range(Noc)])

    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic["variance"] = variance
    return dic

def xDNNclassifier_Offline(Data):
    print("---------- xDNNclassifier_Offline ----------")
    K = np.shape(Data)[0]
    data = Data.copy()
    GMean = data.sum(axis=0) / K
    variance = ((data**2).sum() / K) - (GMean**2).sum()
    DataDensities = 1/(1 + ((data.reshape(K, -1) - np.kron(np.ones((K, 1)), GMean))**2).sum(axis=1).reshape(K, 1)/variance)
    globalDataDensity = DataDensities.sum()/K
    Typicalities = DataDensities / globalDataDensity
    m_star = np.argmax(Typicalities)
    # u = [x for x in data.copy().reshape(K, -1)]
    u = data.copy().reshape(K, -1)
    u_idx = list(range(0, K))
    # z = []
    # z.append(u[m_star])
    z_idx = []
    z_idx.append(m_star)
    Typicalities_z_star = []
    Typicalities_z_star.append(Typicalities[m_star])
    # n_star = m_star
    u_idx_position = m_star
    
    while True:
        # r = z[-1]
        r = u[z_idx[-1]]
        # u = np.delete(u, n_star, axis= 0)
        # u_idx.pop(n_star)
        # if len(u) == 0:
        u_idx.pop(u_idx_position)
        print(f"Remaining Feature Vectors: {len(u_idx)}/{K}", )
        if len(u_idx) == 0:
            break
        # distances = cdist(r.reshape(1, -1), u,'euclidean')
        distances = cdist(r.reshape(1, -1), u[u_idx,].reshape(len(u_idx), -1),'euclidean')
        u_idx_position = np.argmin(distances)
        # z.append(u[n_star])
        
        n_star = u_idx[u_idx_position]
        z_idx.append(n_star)
        Typicalities_z_star.append(Typicalities[n_star])
        
    # z_star = []
    # z_star.append(z[0])
    # for i in range(1, len(z)):
        # if i == len(z) - 1:
        #     if Typicalities_z_star[i] > Typicalities_z_star[i - 1]:
        #         z_star.append(z[i])
        # else:
        #     if Typicalities_z_star[i] > Typicalities_z_star[i - 1] and Typicalities_z_star[i] > Typicalities_z_star[i + 1]:
        #         z_star.append(z[i])    
    z_star_idx = []
    z_star_idx.append(z_idx[0])
    for i in range(1, len(z_idx)):
        if i == len(z_idx) - 1:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1]:
                z_star_idx.append(z_idx[i])
        else:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1] and Typicalities_z_star[i] > Typicalities_z_star[i + 1]:
                z_star_idx.append(z_idx[i])
    
    # Noc = len(z_star)
    # z_star_np = np.array([z_star]).reshape(Noc, data.shape[-1])
    Noc = len(z_star_idx)
    print("Number of Clouds: ", Noc)
    z_star_np = u[z_star_idx].reshape(Noc, data.shape[-1])
    Support = np.zeros((Noc, 1))
    sumOfMembers = np.zeros((Noc, data.shape[-1]))
    sumOfSquareVectors = np.zeros((Noc, 1))
    # Stage 2: Creating the Voronoi Tessellation
    for i in range(0, data.shape[0]):
        distance = cdist(data[i,].reshape(1,-1),z_star_np,'euclidean')[0]
        position = np.argmin(distance)
        Support[position] += 1
        sumOfMembers[position,] += data[i].reshape(sumOfMembers[position].shape)
        sumOfSquareVectors[position] += (data[i]**2).sum()
        
    Centres = sumOfMembers / Support # Updating the Data Cloud Centers 
    DataCloudsVariances = (sumOfSquareVectors / Support) - (Centres**2).sum(axis=1).reshape(-1, 1)   
    
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centres
    dic['Support'] =  Support
    dic['GMean'] =  GMean
    dic["variance"] = variance
    dic["DataCloudsVariances"] = DataCloudsVariances
    return dic

# def xDNNclassifier(Data,Image, initial_Radius = np.sqrt(2*(1 - math.cos(math.pi/6)))):
def xDNNclassifier(Data,Image, initial_Radius = 30):
    print("------------- xDNN Classifier -------------")
    L, N, W = np.shape(Data)
    ### Modification
    # radius = 1 - math.cos(math.pi/6)
    # radius = np.sqrt(2*(1 - math.cos(math.pi/6)))
    
    # radius = initial_Radius
    # radius = 0.77459666924
    
    # radius = 0.3
    # radius = np.sqrt(2*0.3) # Distance between 2 vectors with 45.572996º. Aprox. 0.77459666924
    
    radius = np.sqrt(2*(1 - math.cos(initial_Radius*(math.pi / 180))))
    
    print("Distance Criterium: Radius = " + str(radius))
    
    ### End Modification
    # data = Data.copy()
    data = Data.copy().reshape(L, -1)
    Centre = np.array([data[0,]])
    Center_power = np.power(Centre,2)
    X = np.array([np.sum(Center_power)])
    Support =np.array([1])
    Noc = 1
    GMean = Centre.copy()
    QuadMean = (Centre.copy()**2).sum()
    Radius = np.array([radius])
    ND = 1
    VisualPrototype = {}
    VisualPrototype[1] = Image[0]
    sumOfSquareVectors = np.array([[(data[0,].reshape(1, -1)**2).sum()]])
    
    
    for i in range(2,L+1):
        GMean = ((i-1)/i)*GMean + data[i-1,]/i
        ### Modification 
        # CentreDensity=np.sum((Centre-np.kron(np.ones((Noc,1)),GMean))**2,axis=1)
        # QuadMean = ((i-1)/i)*QuadMean + (data[i-1,]**2).sum()/i
        QuadMean = 1
        variance = QuadMean - (GMean**2).sum()
        CentreDensity= 1/(1 + ((Centre - np.kron(np.ones((Noc, 1)), GMean))**2).sum(axis=1).reshape(Noc, 1) / variance)
        
        normalizedCenters = Centre / np.sqrt((Centre**2).sum(axis=1).reshape(Noc, 1))
        # normalizedGMean =  GMean / np.sqrt((GMean**2).sum(axis=1))
        # variance = QuadMean - (normalizedGMean**2).sum()
        # CentreDensity= 1/(1 + ((normalizedCenters - np.kron(np.ones((Noc, 1)), normalizedGMean))**2).sum(axis=1).reshape(Noc, 1)/variance)
        
        # CentreDensity = CentreDensity * Support
        ### End Modification
        CDmax=max(CentreDensity)
        CDmin=min(CentreDensity)
        ### Modification
        # DataDensity=np.sum((data[i-1,] - GMean) ** 2)
        
        DataDensity= 1/(1 + ((data[i-1,] - GMean)**2).sum()/variance)
        # DataDensity= 1/(1 + ((data[i-1,] - normalizedGMean)**2).sum()/variance)
        ### End Modification
        
        if i == 2:
            distance = cdist(data[i-1,].reshape(1,-1),Centre.reshape(1,-1),'euclidean')[0]
        else:
            distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
            
        # if i == 2:
        #     distance = cdist((data[i-1,]/np.sqrt((data[i-1,]**2).sum())).reshape(1,-1),normalizedCenters.reshape(1,-1),'euclidean')[0]
        # else:
        #     distance = cdist((data[i-1,]/np.sqrt((data[i-1,]**2).sum())).reshape(1,-1),normalizedCenters,'euclidean')[0]
            
        ### Modification max -> min
        # value,position= distance.max(0),distance.argmax(0)
        # value=value**2
        value,position= distance.min(0),distance.argmin(0)
        
        # value=value**2
        ### End Modification
        
        
        ### Modification
        # if DataDensity >= CDmax or DataDensity <= CDmin or value > 2*Radius[position]:
        if DataDensity >= CDmax or DataDensity <= CDmin or value >= Radius[position]:
        # if DataDensity > CDmax or DataDensity < CDmin:
        ### End Modification
            Centre=np.vstack((Centre,data[i-1,])) # Concatena os centros das dataClouds
            sumOfSquareVectors = np.vstack((sumOfSquareVectors,(data[i-1,].reshape(1, -1)**2).sum())) 
            Noc=Noc+1
            VisualPrototype[Noc]=Image[i-1]
            X=np.vstack((X,ND))
            Support=np.vstack((Support, 1))
            Radius=np.vstack((Radius, radius))
        else:
            ### Modification
            # Centre[position,] = Centre[position,]*(Support[position]/Support[position]+1)+data[i-1]/(Support[position]+1)
            Centre[position,] = Centre[position,]*(Support[position]/(Support[position]+1))+data[i-1]/(Support[position]+1)
            sumOfSquareVectors[position] += (data[i-1,].reshape(1, -1)**2).sum()
            ### End Modification
            Support[position]=Support[position]+1
            ### Modification 
            # Radius[position]=0.5*Radius[position]+0.5*(1 -sum(Centre[position,]**2))/2
            Radius[position]=np.sqrt(((Radius[position]**2)+(1 - (Centre[position,]**2).sum()))/2)
            VisualPrototype[position + 1] = np.vstack((VisualPrototype[position +1], Image[i-1]))
            ### End Modification
    
        print(f"Analyzed Feature Vectors: {i}/{L+1}")
        
    dic = {}
    dic['Noc'] =  Noc
    dic['Centre'] =  Centre
    dic['Support'] =  Support
    dic['Radius'] =  Radius
    dic['GMean'] =  GMean
    dic['Prototype'] = VisualPrototype
    dic['L'] =  L
    dic['X'] =  X
    dic["variance"] = variance
    dic["DataCloudsVariances"] = (sumOfSquareVectors / Support) - (Centre**2).sum(axis=1).reshape(-1, 1)
    return dic  

def DecisionMaking_kNN(Params,datates, k):
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    LTes=np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores=np.zeros((LTes,CurrentNC))
    obtianed_rules = []
    
    # dts = cdist(PARAM[0]['Centre'],PARAM[0]['Centre'],'euclidean')
    # print("max distance class 0: ",dts.max())
    # dts[dts==0]=np.inf
    # print("min distance class 0: ",dts.min())
    # dts = cdist(PARAM[1]['Centre'],PARAM[1]['Centre'],'euclidean')
    # print("max distance class 1: ",dts.max())
    # dts[dts==0]=np.inf
    # print("min distance class 1: ",dts.min())
    
    for i in range(1,LTes + 1):
        data = datates[i-1,]
        Values= np.zeros((CurrentNC*k))
        Classes_idx = np.zeros((CurrentNC*k))
        if_thenRules = []
        for j in range(0,CurrentNC):            
            dts = cdist(data.reshape(1, -1),PARAM[j]['Centre'],'euclidean')
            
            indxMin = np.argsort(dts)[0,:k]
            dts_min = dts[0, indxMin]
            prot_dist = dts[0, indxMin].round(decimals=4).tolist()
            prot_names = "N/A"
            
            Values[j*k:(j+1)*k]= dts_min
            Classes_idx[j*k:(j+1)*k]= j       
            
            if_thenRules.append((prot_names, prot_dist))
        obtianed_rules.append(if_thenRules) 
        
        k_nearest_idx = np.argsort(Values)[:k]
        k_nearest_classes = Classes_idx[k_nearest_idx]
        
        classes, counts = np.unique(k_nearest_classes, return_counts=True)
        win_class_idx = np.argmax(counts)
        win_class = classes[win_class_idx]
        Scores[i-1,] = softmax(counts)
        EstimatedLabels[i-1]= win_class
    # LABEL1=np.zeros((CurrentNC,1))
    
    # for i in range(0,CurrentNC): 
    #     LABEL1[i] = np.unique(LAB[i])

    # EstimatedLabels = EstimatedLabels.astype(int)
    # EstimatedLabels = LABEL1[EstimatedLabels]   
    EstimatedLabels = EstimatedLabels.reshape(-1, 1)
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    dic['ObtainedRules'] = obtianed_rules
    return dic
    

def DecisionMaking(Params,datates):
    # print("------------- Stating Decision Making -------------")
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    VV = 1
    LTes=np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores=np.zeros((LTes,CurrentNC))
    obtianed_rules = []
    
    for i in range(1,LTes + 1):
        data = datates[i-1,]
        R=np.zeros((VV,CurrentNC))
        Value=np.zeros((CurrentNC,1))
        if_thenRules = []
        for k in range(0,CurrentNC):
            #dts = cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',p=0.5)
            
            dts = cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')
            
            # normalizedCenters = PARAM[k]['Centre'] / np.sqrt((PARAM[k]['Centre']**2).sum(axis=1).reshape(PARAM[k]['Centre'].shape[0], 1))
            # dts = cdist(data.reshape(1, -1),normalizedCenters,'euclidean')
            
            indxMin = np.argsort(dts)[0,:4]
            prot_dist = dts[0,indxMin].round(decimals=4).tolist()
            # prot_names = [PARAM[k]['Prototype'].get(key) for key in indxMin + 1]
            prot_names = "N/A"
            distance=np.sort(dts)[0]
            #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',p=6))[0]
            #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean'))[0]
            # DataCloudsVariances = PARAM[k]["DataCloudsVariances"]
            # # DataCloudsVariances = DataCloudsVariances + (DataCloudsVariances == 0)
            # # DegreesOfSimilarity = 1 / (1 + ((data.reshape(1, -1) - PARAM[k]['Centre'])**2).sum(axis=1).reshape(-1, 1) / DataCloudsVariances)
            # DegreesOfSimilarity = 1 / (1 + ((data.reshape(1, -1) - PARAM[k]['Centre'])**2).sum(axis=1).reshape(-1, 1) / PARAM[k]['variance'])
            
            
            Value[k]=distance[0]
            # Value[k]= DegreesOfSimilarity.max()
            
            if_thenRules.append((prot_names, prot_dist))
            # all_dis[k][i-1] = np.sort(dts)
        obtianed_rules.append(if_thenRules) 
          
        Value = softmax(-1*Value**2).T
        # Value = np.exp(-1*(Value**2)).T
        # Value = softmax(Value).T
        Scores[i-1,] = Value
        Value = Value[0]
        Value_new = np.sort(Value)[::-1]
        indx = np.argsort(Value)[::-1]
        EstimatedLabels[i-1]=indx[0]
    LABEL1=np.zeros((CurrentNC,1))
    
    for i in range(0,CurrentNC): 
        LABEL1[i] = np.unique(LAB[i])

    EstimatedLabels = EstimatedLabels.astype(int)
    EstimatedLabels = LABEL1[EstimatedLabels]   
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    dic['ObtainedRules'] = obtianed_rules
    return dic
         
###############################################################################

