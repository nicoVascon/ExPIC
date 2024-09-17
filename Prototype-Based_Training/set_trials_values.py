import optuna

studyName = "COVID-QU-Ex_Dataset_Prot-Based_TrainWithPCA_Torch_325c_CCEAlpha_Study_GridSearch_LowEpochs_v3"

study = optuna.create_study(directions=['maximize', 'minimize'],storage="sqlite:///xDNN_test.db",study_name=studyName, load_if_exists=True)

accuracy_values = [ 0.9062,
                    0.9214,
                    0.9044,
                    0.9083,
                    0.8929,
                    0.7624,
                    0.8898,
                    0.8885,
                    0.8874,
                    0.907,
                    0.8996,
                    0.914,
                    0.8826,
                    0.9188,
                    0.8844,
                    0.8926,
                    0.8726,
                    0.8817,
                    0.904,
                    0.8811,
                    0.9275,
                    0.8892,
                    0.8761,
                    0.8977,
                    0.8518,
                    0.8824]
numProt_values = [  3136,
                    2928,
                    3109,
                    1601,
                    1658,
                    2063,
                    1100,
                    1315,
                    1167,
                    3409,
                    1348,
                    3007,
                    1850,
                    2952,
                    1636,
                    3150,
                    1806,
                    1167,
                    2688,
                    1859,
                    2571,
                    1381,
                    1862,
                    2818,
                    1631,
                    2842
                    ]

alpha_values = [
1.2272948296189354e-07,
9.232677721683292e-06,
1.2840960082488333e-07,
3.652137714624658e-06,
3.3481509786245237e-07,
3.674416051513059e-06,
4.125016757352728e-07,
8.24202843551891e-06,
1.9383801059562176e-07,
1.3193056943517018e-07,
4.244862049990643e-06,
2.548158864972106e-07,
5.2327014012623145e-06,
1.1109437501574244e-07,
1.1348746173989806e-07,
3.180648056172755e-06,
1.039789281435725e-07,
8.386302804413458e-06
]

def obj(trial):
    trial_num = trial.number
    trial_CCE_Alpha = trial.suggest_float("CCE_Alpha", 1e-7, 1e-5, log=True)
    print("trial_CCE_Alpha = " + str(trial_CCE_Alpha))
    return (accuracy_values[trial_num], numProt_values[trial_num])

for i in range(0, 8):
    study.enqueue_trial({"CCE_Alpha": (1e-5)*(0.5**i)})

for i in range(8, 26):
    study.enqueue_trial({"CCE_Alpha": alpha_values[i-8]})

study.optimize(obj, n_trials=26)