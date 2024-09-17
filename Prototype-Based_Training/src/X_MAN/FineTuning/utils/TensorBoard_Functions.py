import tensorflow as tf

def write_scalars(writer, scalar_dict, step):
    with writer.as_default():
        for tag, value in scalar_dict.items():
            tf.summary.scalar(tag, value, step=step)

def write_Experiment_Setup(summary_dir : str, config_dic : dict):
    Experiment_Setup = """
### %s

**Direcotories**

| Name | Directory |
| --- | --- |
| Base Algorithm Dir | %s |

**Hyperparameters**

| Name | Value |
| --- | --- |
| Learning Rate | %e |
| Batch Size | %d |
| Epochs | %d |

**Dataset**

| Name | Value |
| --- | --- |
| Dataset Name | %s |
| Shuffle | %s |
| Seed | %d |

**xDNN**

| Name | Value |
| --- | --- |
| Base Prototypes Path | %s |

**Hardware**

| Name | Value |
| --- | --- |
| GPU | %s |

**ReduceLROnPlateau**

| Name | Value |
| --- | --- |
| Mode | %s |
| Factor | %e |
| Patience | %d |
| Threshold | %f |
| Threshold Mode | %s |
| Cooldown | %d |
| Min LR | %e |
| Epsilon | %e |
| Verbose | %s |

**EarlyStopping**

| Name | Value |
| --- | --- |
| Patience | %d |
| Verbose | %s |
| Delta | %f |

""" % (config_dic["General"]["Experiment_Name"],
        config_dic["Directories"]["base_algoritm_dir"],
        config_dic["Hyperparameters"]["learning_rate"],
        config_dic["Hyperparameters"]["batch_size"],
        config_dic["Hyperparameters"]["epochs"],
        config_dic["Dataset"]["Dataset_Name"],
        config_dic["Dataset"]["Shuffle"],
        config_dic["Dataset"]["Seed"],
        config_dic["xDNN"]["Base_Prototypes_path"],
        config_dic["Hardware"]["GPU"],
        config_dic["ReduceLROnPlateau"]["mode"],
        config_dic["ReduceLROnPlateau"]["factor"],
        config_dic["ReduceLROnPlateau"]["patience"],
        config_dic["ReduceLROnPlateau"]["threshold"],
        config_dic["ReduceLROnPlateau"]["threshold_mode"],
        config_dic["ReduceLROnPlateau"]["cooldown"],
        config_dic["ReduceLROnPlateau"]["min_lr"],
        config_dic["ReduceLROnPlateau"]["eps"],
        config_dic["ReduceLROnPlateau"]["verbose"],
        config_dic["EarlyStopping"]["patience"],
        config_dic["EarlyStopping"]["verbose"],
        config_dic["EarlyStopping"]["delta"])
    
    summary_writer = tf.summary.create_file_writer(summary_dir)
    with summary_writer.as_default():
        tf.summary.text("Experiment_Setup", Experiment_Setup, step=0)