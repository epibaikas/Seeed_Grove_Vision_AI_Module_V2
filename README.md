# Evolutionary Non-Volatile Memory Management for Incremental Learning at the Extreme Edge
Forked software repository for Seeed Grove Vision AI Module V2 board.

This fork containts the open-source implmentation for random, greedy and evolutionary subset selection methods used for discovering representative subsets of data examples that should be preserved within the non-volatile memory of an extreme edge device while learning incrementally. The implementation has been packaged as the scenario app `incr_learn` placed under:
```
EPII_CM55M_APP_S/app/scenario_app/incr_learn/ 
```

The C implmentation was initially developed for the Seeed Grove Vision AI Module V2 board, but has also been ported to Linux / MacOS host platforms. A dedicated makefile placed inside the `incr_learn` folder can be used to build the code for host platforms.

## Project folder structure
The original structure of the `incr_learn` app folder before generating experimental results is:
```
.
├── c_src       # C source code
├── config      # Experiment configuration files
├── inc         # C header files
├── py_src      # Python source files
├── scripts     # Bash scripts for generating experimental results
└── tests       # Unit tests
```

After running the scripts, the folder structure changes to: 
```
.
├── artifacts   # Results from intermediate computations
├── build       # Compiled C code for Linux / MacOS host platform
├── c_src       # C source code
├── config      # Experiment configuration files
├── datasets    # Dataset files downloaded by PyTorch library
├── inc         # C header files
├── log         # Experiment log files in xml and txt format
├── plots       # Plot figures
├── py_src      # Python source files
├── results     # Experimental results
├── scripts     # Bash scripts for generating experimental results
└── tests       # Unit tests
```

## Setup
### Setup the conda environment
Clone the repository and setup the conda environment that is necessary to run the experiments:

```
cd <path_to_incr_learn>
conda env create -f env.yml
conda activate incr_learn
```

## Build C code 
### For Linux / MacOS platform
if the code is built for MacOS, comment out line `CFLAGS += -D_GNU_SOURCE` from `makefile`. Then run:
```
make
```

### For Seeed Grove Vision AI Module V2 board
Switch back to folder `Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S` and run: 

```
make -j8
sh gen_img.sh
```

Connect the board to your system, find its serial port (use `ls /dev/*`), update it in `flash.sh` and run:
```
sh flash.sh
```

## Reproduce experiments
Switch to `EPII_CM55M_APP_S/app/scenario_app/incr_learn/`, if not there already:
```
cd EPII_CM55M_APP_S/app/scenario_app/incr_learn/ 
```

To reproduce experimental results, follow the steps described below in the exact order:

(1) Run 6-fold cross validation to determine the number of Nearest Neighbors for each dataset:
```
python py_src/k_fold_cross_validation.py MNIST
python py_src/k_fold_cross_validation.py FashionMNIST
python py_src/k_fold_cross_validation.py EMNIST
```

(2) Run the greedy algorithm to find the "low" and "high" accuracy sequences (please note that sequence finding for EMNIST might take several hours):
```
python py_src/greedy_sequence_finder.py MNIST
python py_src/greedy_sequence_finder.py FashionMNIST
python py_src/greedy_sequence_finder.py EMNIST
```

(3) Generate hyperparameter index file `artifacts/hyper_index.csv` 
```
python scripts/gen_hyperparameter_files.py
```

(4) Run incremental learning experiments for a specific dataset, subset selection function, balancing condition, ram and eeprom buffer sizes:
```
python scripts/run_sub_selection [dataset] [sub_sel_func] [bal] [ram_buf_size] [eeprom_buf_size] --hyperparam [hyperparam_set]
```

For example, to run greedy_bal() subset selection on FashionMNIST with a 32 kB RAM buffer size and a 64 kB EEPROM buffer size, use the following arguments:
```
python scripts/run_sub_selection FashionMNIST greedy 1 32 64 --hyperparam greedy_hyper02
```


To run on the Seeed board instead of the host platform, add the `--target_dev` flag at the end. 
Ensure that the device is connected to the system and update its serial port in `config/config_global.ini`


The codenames for the best hyperparameter sets determined for every dataset and function can be found in the following table. 
The exact hyperparameters corresponding to each codename can be found in `artifacts/hyper_index.csv`
|---------------|---------------|--------------------|
| Dataset       | Function      | Hyperparameter set |
| ------------- | ------------- | ------------------ |
| MNIST         | greedy_bal()  | greedy_hyper02     |
|               | evo_bal()     | evo_hyper29        |
|---------------|---------------|--------------------|
| FashionMNIST  | greedy_bal()  | greedy_hyper02     |
|               | evo_bal()     | evo_hyper29        |
|---------------|---------------|--------------------|
| EMNIST        | greedy_bal()  | greedy_hyper01     |
|               | evo_bal()     | evo_hyper05        |
|---------------|---------------|--------------------|
