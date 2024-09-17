import optuna
import FineTuning_VGG16_vProt_BasedDS_PyTorch as ft

def objective(trial):
    trial_batch_size = trial.suggest_int("batch_size",20,64,log=False)
    trial_lr = trial.suggest_float("learning_rate", 1e-6, 10e-6, log=True)
    
    exp_name = "Optuna/" + studyName +"/VGG16_Prot_Training_Trial_" + str(trial.number)
    config_file_path = "/nfs/home/nvasconcellos.it/softLinkTests/xDNN_Covid19_Det/xDNN---Python/FineTuning/Prototype-Based_Training/parameters.json"
    
    val_loss, _ = ft.run(config_file_path=config_file_path, batch_size=trial_batch_size, lr=trial_lr, exp_name=exp_name, trial=trial)
    return val_loss

studyName = 'Prototype_Based_Training'
study = optuna.create_study(direction='minimize',storage="sqlite:///xDNN_test.db",study_name=studyName, load_if_exists=True)

study.optimize(objective, n_trials=10)