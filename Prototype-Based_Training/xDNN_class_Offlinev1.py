"""
Please cite:
    
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.
"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax 



def xDNN(Input,Mode, **kwargs):
    ### Normalization by Feature Vector
    # means = Input['Features'].sum(axis=1).reshape(-1, 1)/Input['Features'].shape[1]
    # variance = ((Input['Features']**2).sum(axis=1).reshape(-1, 1)/Input['Features'].shape[1]) - (means**2)
    
    # standardizedVec = (Input['Features'] - means)/np.sqrt(variance)
    # minValues = standardizedVec.min(1).reshape(-1, 1)
    # maxValues = standardizedVec.max(1).reshape(-1, 1)
    # normalizedVec = (standardizedVec - minValues)/(maxValues - minValues)    
    # normalizedVecNormUnit = normalizedVec /np.sqrt((normalizedVec**2).sum(axis=1).reshape(-1, 1))
    
    # Input['Features'] = normalizedVecNormUnit
    ### End Modification
    
    ### Normalization by Feature
    # means = Input['Features'].sum(axis=0).reshape(1, -1)/Input['Features'].shape[0]
    # # variance = ((Input['Features']**2).sum(axis=0).reshape(1, -1)/Input['Features'].shape[0]) - (means**2)
    # variance = (((Input['Features'] - means)**2).sum(axis=0).reshape(1, -1)/Input['Features'].shape[0])
    # variance = variance + (variance == 0) # Variance = 1 for Null features
    
    # standardizedVec = (Input['Features'] - means)/np.sqrt(variance)
    # minValues = standardizedVec.min(axis=0).reshape(1, -1)
    # maxValues = standardizedVec.max(axis=0).reshape(1, -1)
    # FeatureRange = (maxValues - minValues)
    # FeatureRange = FeatureRange + (FeatureRange == 0) # FeatureRange = 1 for Null features
    # normalizedVec = (standardizedVec - minValues)/(FeatureRange)
    # normalizedVecNormUnit = normalizedVec /np.sqrt((normalizedVec**2).sum(axis=1).reshape(-1, 1))
    
    # Input['Features'] = normalizedVecNormUnit
    # # Input['Features'] = normalizedVec
    
    normalizedVec = Input['Features']
    normalizedVecNormUnit = normalizedVec /np.sqrt((normalizedVec**2).sum(axis=1).reshape(-1, 1))
    Input['Features'] = normalizedVecNormUnit
    ### End Modification
    if Mode == 'Learning':        
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        if "initialRadius" in kwargs.keys():
            initial_Radius=kwargs["initialRadius"]
        else:
            initial_Radius= np.sqrt(2*(1 - math.cos(math.pi/6)))
        
        if "c_W_k" in kwargs.keys():
            c_W_k_=kwargs["c_W_k"]
        else:
            c_W_k_=0.5
            
        Prototypes = PrototypesIdentification(Images,Features,Labels,CN, initial_Radius, c_W_k_)
        Output = {}
        Output['xDNNParms'] = {}
        Output['xDNNParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0,CN+1):
           MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['xDNNParms']['CurrentNumberofClass']=CN+1
        Output['xDNNParms']['OriginalNumberofClass']=CN+1
        Output['xDNNParms']['MemberLabels']=MemberLabels
        return Output
    
    elif Mode == 'Validation':
        Params=Input['xDNNParms']
        datates=Input['Features']
        Test_Results = DecisionMaking(Params,datates)
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        Output['ObtainedRules'] = Test_Results['ObtainedRules']
        Output['DegreesConfidence'] = Test_Results['DegreesConfidence']
        return Output
    
def PrototypesIdentification(Image,GlobalFeature,LABEL,CL, initialRadius, c_W_k):
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
        # Prototypes[i] = xDNNclassifier(data[i],image[i], initial_Radius = initialRadius)
        Prototypes[i] = xDNNclassifier_Offline(data[i])
        Prototypes[i] = PrototypesFiltering(Prototypes[i], data[i],image[i], c_W_k = c_W_k)
        Prototypes[i] = OutliersPrototypesFiltering(Prototypes[i], data[i],image[i])
        
    return Prototypes

def OutliersPrototypesFiltering(prototypes, Data, Image):
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
        CentreDensitys = 1/(1 + ((Centres - np.kron(np.ones((Noc, 1)), GMean))**2).sum(axis=1).reshape(Noc, 1)/variance)
        globalDataDensity = (Support * CentreDensitys).sum()
        CentreTypicalities = (Support * CentreDensitys) / globalDataDensity
        
        distances = cdist(Centres, Centres,'euclidean')
        
        Y_k = distances.sum() / (Noc * (Noc-1))    
        M_Y = ((distances < Y_k) * (distances > 0)).sum()
        
        overline_W_k = (distances * (distances < Y_k)).sum() / M_Y
        M_ovline_W = ((distances < overline_W_k) * (distances > 0)).sum()
        
        W_k = (distances * (distances < overline_W_k)).sum() / M_ovline_W
        
        C_star = (distances <= W_k*c_W_k) * (distances > 0)
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
    z = []
    z.append(u[m_star])
    Typicalities_z_star = []
    Typicalities_z_star.append(Typicalities[m_star])
    n_star = m_star
    
    while True:
        r = z[-1]
        u = np.delete(u, n_star, axis= 0)
        
        Typicalities = np.delete(Typicalities, n_star, axis= 0)
        
        if len(u) == 0:
            break
        distances = cdist(r.reshape(1, -1), u,'euclidean')
        n_star = np.argmin(distances)
        z.append(u[n_star])
        Typicalities_z_star.append(Typicalities[n_star])
        
    z_star = []
    z_star.append(z[0])
    for i in range(1, len(z)):
        if i == len(z) - 1:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1]:
                z_star.append(z[i])
        else:
            if Typicalities_z_star[i] > Typicalities_z_star[i - 1] and Typicalities_z_star[i] > Typicalities_z_star[i + 1]:
                z_star.append(z[i])
    
    Noc = len(z_star)
    z_star_np = np.array([z_star]).reshape(Noc, data.shape[-1])
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
    L, N, W = np.shape(Data)
    ### Modification
    # radius = 1 - math.cos(math.pi/6)
    # radius = np.sqrt(2*(1 - math.cos(math.pi/6)))
    
    # radius = initial_Radius
    # radius = 0.77459666924
    
    # radius = 0.3
    radius = np.sqrt(2*0.3) # Distance between 2 vectors with 45.572996º. Aprox. 0.77459666924
    
    # radius = np.sqrt(2*(1 - math.cos(initial_Radius*(math.pi / 180))))
    
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
    

def DecisionMaking(Params,datates, numProtAVRG=20, exp=4):
    PARAM=Params['Parameters']
    CurrentNC=Params['CurrentNumberofClass']
    LAB=Params['MemberLabels']
    VV = 1
    LTes=np.shape(datates)[0]
    EstimatedLabels = np.zeros((LTes))
    Scores=np.zeros((LTes,CurrentNC))
    obtianed_rules = []
    degrees_confidence = []
    
    for i in range(1,LTes + 1):
        data = datates[i-1,]
        R=np.zeros((VV,CurrentNC))
        Value=np.zeros((CurrentNC,1))
        if_thenRules = []
        distances = []
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
            distances.append(distance[:10])
            #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'minkowski',p=6))[0]
            #distance=np.sort(cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean'))[0]
            DataCloudsVariances = PARAM[k]["DataCloudsVariances"]
            # DataCloudsVariances = DataCloudsVariances + (DataCloudsVariances == 0)
            # DegreesOfSimilarity = 1 / (1 + ((data.reshape(1, -1) - PARAM[k]['Centre'])**2).sum(axis=1).reshape(-1, 1) / DataCloudsVariances)
            DegreesOfSimilarity = 1 / (1 + ((data.reshape(1, -1) - PARAM[k]['Centre'])**2).sum(axis=1).reshape(-1, 1) / PARAM[k]['variance'])
            
            
            Value[k]=distance[0]
            # Value[k]= DegreesOfSimilarity.max()
            
            if_thenRules.append((prot_names, prot_dist))
            # all_dis[k][i-1] = np.sort(dts)
        obtianed_rules.append(if_thenRules) 
          
        # Value = softmax(-1*Value**2).T
        Value = np.exp(-1*(Value**2)).T
        # Value = softmax(Value).T
        Scores[i-1,] = Value
        Value = Value[0]
        Value_new = np.sort(Value)[::-1]
        indx = np.argsort(Value)[::-1]
        EstimatedLabels[i-1]=indx[0]
        numOfProt = min(len(distances[indx[0]]), len(distances[1 - indx[0]]))        
        dist_dif = np.array(distances[indx[0]][:numOfProt]) - np.array(distances[1 - indx[0]][:numOfProt])
        degrees_confidence.append(dist_dif.sum(axis=0)/len(dist_dif))
    LABEL1=np.zeros((CurrentNC,1))
    
    ##### Code for distances analysis
    # import pickle
    # folder = '/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/results/WholeUnfreeze_copy/T1_Fine_Tuning_WholeModelUnfreeze_VGG16_model_2022-10-0719_02_24.157859/distances/'
    # with open(folder + 'covid_dis', 'wb') as file:
    #     pickle.dump(all_dis[0], file)
    # with open(folder + 'non_covid_dis', 'wb') as file:
    #     pickle.dump(all_dis[1], file)
        
    # import pandas as pd
    # df_covid_dis = pd.DataFrame(all_dis[0])
    # df_non_covid_dis = pd.DataFrame(all_dis[1])
    # df_covid_dis.to_csv(folder + 'csv_covid_dis.csv', header=False, index=False)
    # df_non_covid_dis.to_csv(folder + 'csv_non_covid_dis.csv', header=False, index=False)
    #####
    for i in range(0,CurrentNC): 
        LABEL1[i] = np.unique(LAB[i])

    EstimatedLabels = EstimatedLabels.astype(int)
    EstimatedLabels = LABEL1[EstimatedLabels]   
    dic = {}
    dic['EstimatedLabels'] = EstimatedLabels
    dic['Scores'] = Scores
    dic['ObtainedRules'] = obtianed_rules
    dic['DegreesConfidence'] = degrees_confidence
    return dic
         
###############################################################################

