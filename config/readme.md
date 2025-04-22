{
    "exp_name": "resnet",
    "mode": "train", “test" "5fold"
    "split":{
        "train":["00021", "00030", "00010", "00028", "00019", "00014", "00017", "00025", "00022", "00013", "00007", "00009", "00031", "00026", "00032", "00020", "00040", "00016", "00027", "00008", "00011"],
        "valid": ["00000", "00001", "00002", "00003", "00004", "00005", "00006"],
        "test": ["00023", "00024", "00018", "00029", "00015", "00012"],
        "5-Fold":{"Fold-1":["00000", "00001", "00002", "00003", "00004", "00005", "00006"],
                  "Fold-2":["00007", "00008", "00009", "00010", "00011","00012", "00013"],
                  "Fold-3":[ "00014" , "00015", "00016","00017", "00018", "00019", "00020"],
                  "Fold-4":["00021", "00022", "00023", "00024", "00025", "00026", "00027"],
                  "Fold-5":["00028", "00029", "00030", "00031", "00032", "00033", "00040"]
    }
    },
    "dataset":{
        "ring_type": "ring1",
        "input_type": ["ir-raw","ir-filtered","ir-standardized","ir-difference","ir-welch","ir-filtered-rr","ir-welch-rr","red-raw","red-filtered","red-standardized","red-difference","red-welch","red-filtered-rr","red-welch-rr"."ax-raw","ax-filtered","ax-standardized","ax-difference","ax-welch","ax-filtered-rr","ax-welch-rr","ay-raw","ay-filtered","ay-standardized","ay-difference","ay-welch","ay-filtered-rr","ay-welch-rr","az-raw","az-filtered","az-standardized","az-difference","az-welch","az-filtered-rr","az-welch-rr"],
        "label_type": ["hr", "spo2", "bvp_sdnn","resp_rr","samsung_hr","oura_hr","BP_sys","BP_dia"],
        "shuffle": true,
        "batch_size": 128,
        "quality_assessment": {
            "method": "elgendi",
            "th": 0.8
        },
        "target_fs": 100,
        "window_duration": 30,
        "experiment": ["Health", "Daily","Sport"],
        "task": ["sitting", "spo2", "deepsquat", "talking", "shaking_head", "standing", "striding"]
    },
    "seed": 42,
    "csv_path": "csv/resnet/resnet.csv",
    "img_path": "img/resnet",
    "task": ["hr","spo2"],
    "method": {
        "name": "resnet",
        "type": "ML", 
        "model_path": null,
        "params":{
            "in_channels": 1,  
            "base_filters": 32,
            "kernel_size": 5,
            "stride": 2,
            "groups": 1,
            "n_block": 8,
            "n_classes": 180,  
            "downsample_gap": 2,
            "increasefilter_gap": 4,
            "backbone": false
        }
    },
    "train":{
        "device": "1",
        "epochs": 200,
        "lr": 1e-3,
        "criterion": "mse",
        "optimizer": "adam",
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 30,
            "mode": "min"
        },
        "model_checkpoint": {
            "monitor": "val_loss",
            "mode": "min",
            "save_best_only": true
        }
    },
    "test":{
        "device": "1",
        "batch_size": 128,
        "metrics": ["mae", "rmse", "mape", "pearson"],
        "model_path": null,
        "model_name": null
    },
    "pretrain_model": "TODO"

}