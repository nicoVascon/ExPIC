# import plotly 
# import plotly.express as px
import plotly.graph_objects as go
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import cv2
# from math import log10, sqrt
# from skimage.metrics import structural_similarity as ssim
# path_excel = '/nfs/home/dnicolau.it/Verao_c_ciencia/data/bestResult_EUVIP_test_crop_20210713_320_lr0002-rnet15_ngf64-ba4n250ndec1300_cA10Acb10B_entr10-8x4x2x1/'
# excel_data = pd.read_excel(path_excel+"results_PCA_alfa_no_improv.xlsx")
# excel_data.loc[excel_data["Gain"] < -1, "Gain"] = -1

Accuracy = [
    0.93409636330072,
    0.9339117592763522,
    0.9320657190326749,
    0.9307734908621008,
    0.9281890345209526,
    0.9276352224478494,
    0.9272660143991139,
    0.9263429942772753,
    0.9239431419604948,
    0.9220971017168175,
    0.9217278936680819,
    0.9160051689126822,
    0.9158205648883145,
    0.9148975447664759,
    0.9027136791582057,
    0.9021598670851024,
    0.8938526859885545,
    0.8872069411113163,
    0.8849916928189034,
    0.8685619346501754,
    0.8517629684327118
]
Num_Prototypes = [
    2657,
    58,
    1205,
    2855,
    111,
    3397,
    2076,
    1128,
    2032,
    466,
    41,
    1401,
    1167,
    3385,
    41,
    162,
    3736,
    209,
    46,
    74,
    108
]
Alpha_value = [
    0.0004179657462531012,
    0.00017589139693168304,
    0.0003161523217823655,
    1.3048911546123239e-05,
    0.0006816169215513748,
    1.2230871803051272e-05,
    2.210575630440724e-05,
    7.599728457271897e-05,
    2.987648195316992e-05,
    2.4089433780107936e-05,
    0.0006944405363454479,
    3.075534478683447e-05,
    2.4807436708463776e-05,
    1.1949126747174556e-05,
    0.0010885090596442107,
    3.497809964263656e-05,
    1.020908624946497e-05,
    3.2189810932114854e-05,
    0.0015419771216157018,
    0.0010399967308279453,
    0.0013402935310840856
]

fig = go.Figure(data =[go.Scatter3d(x=Accuracy,
                                    y=Num_Prototypes,
                                    z=Alpha_value,
                                    mode = 'markers'
                                    # marker = dict(size = 12,color = excel_data['Gain'],colorscale ='Viridis',opacity = 0.8, colorbar=dict(thickness=40, title ="Compression Gain (%)")),
                                    )])

fig.update_layout(scene=dict(
                            xaxis_title="Accuracy",
                            yaxis_title="Num_Prototypes",
                            zaxis_title="Alpha_value"
                            ))
fig.show()
