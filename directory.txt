########### Directory structure description
├── README.md                   // help
├── preprocess                  // coding files for preprocessing app data
│   ├── pre4new_kw.py           // code for preprocessing pre-train data
│   ├── pre4new_bc.py           // code for preprocessing pre-train data
│   ├── pre4*_open.py           // code for preprocessing fine-tune data
│   ├── pre4*_unin.py           // code for preprocessing fine-tune data
│   └── pre4*_down.py           // code for preprocessing fine-tune & inference data
├── script                      // scripts for running code and settings
│   ├── data_pre.sh             // script for preprocessing data
│   ├── data_pre.sh             // script for preprocessing data
│   ├── draw_png.sh             // script for drawing t-SNE png to visualization
│   ├── draw.py                 // code for drawing t-SNE png to visualization
│   ├── mp_train.sh             // script for pre-train
│   ├── mp_finetune.sh          // script for fine-tune
│   ├── mp_predict.sh           // script for inference
│   ├── run_abcn.yaml           // config for pre-train & fine-tune
│   └── gen_abcn.yaml           // config for inference
├── downstream                  // coding files for downstream task
│   ├── data_process.py         // code for processing downstream dataset
│   ├── lgb.py                  // code for downstream model based on lightgbm
│   └── utils.py                // code for some utils and settings
└── src                         // coding files for model runing & constructing
    ├── run_*.py                // code for pre-train
    ├── finetune_*.py           // code for fine-tune
    ├── gen_*.sh                // code for inference
    ├── utils.py                // code for settings
    ├── autoinstall.py          // code for auto install some packages
    ├── base                    // coding files for basic model structure parts
    │   └── transformer         // coding files for transformer, for building transformer-based modules
    │   └── opt                 // coding files for early-stop and some operation
    |   └── merge_input.py      // code for merge temporal information in input
    |   └── register.py         // code for regist some model parts
    └── self_attention          // codding files for our model structure
        ├── activations.py      // code for some activation functions
        ├── config.py           // code for loading config files
        ├── core.py             // code for bert-based modules
        ├── dataset.py          // code for loading dataset
        ├── mlp.py              // code for a simple dnn, for testing fine-tune
        └── modelf.py           // code for our model
