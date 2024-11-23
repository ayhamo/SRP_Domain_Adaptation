## Worked on by: Ayham Treter (ayhmali123@gmail.com), Aseel Al-Qazqzy (alqazqzy.asseel@gmail.com), Nicholas Yegon (yegonicholas@gmail.com), Fatemeh Motevalian Naeini (motevalian@uni-hildesheim.de)

## Original Authors: [Huan He](https://hehuannb.github.io/) (hehuannb@gmail.com), [Owen Queen](https://www.linkedin.com/in/owen-queen-0348561a2/), [Teddy Koker](https://teddykoker.com/), [Consuelo Cuevas](https://consuelo-cuevas.editorx.io/home), [Theodoros Tsiligkaridis](https://github.com/mims-harvard/Raindrop)(ttsili@ll.mit.edu), [Marinka Zitnik](https://zitniklab.hms.harvard.edu/) (marinka@hms.harvard.edu)

## Raincoat Paper: [ICML 2023](https://arxiv.org/abs/2302.03133)

## Overview of Raincoat

The transfer of models trained on labeled datasets from a source domain to unlabeled target domains is facilitated by unsupervised domain adaptation (UDA). However, when dealing with complex time series models, transferability becomes challenging due to differences in dynamic temporal structures between domains, which can result in feature shifts and gaps in time and frequency representations. Additionally, the label distributions in the source and target domains can be vastly different, making it difficult for UDA to address label shifts and recognize labels unique to the target domain. Raincoat is a domain adaptation method for time series that can handle both feature and label shifts.

<p align="center">
<img src="https://zitniklab.hms.harvard.edu/img/Raincoat-method.png">
</p>

## Overview of The changes we made:
...

## Installation and Setup

### 1: Download the Repo

First, clone the GitHub repository

### 2: Set Up Environment

To install the core environment dependencies of Raincoat, 
First:
```
cd Umbrella
```

Second, for raincoat, install torch and torchvision:
```
pip3 install torch==2.4.1 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
for raindrop, install scatter:
```
pip3 install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```
then install requirments:
```
pip3 install -r requirement.txt
```
and finally SRP Package
```
pip3 install -e .
```


    
### 3: Download Datasets
Create a folder and download the pre-processed versions of the datasets [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B), [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ), [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO), and [Sleep-EDF](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9).

The structure of data files should in dictionary form as follows:
`train.pt = {"samples": data, "labels: labels}`, and similarly for `test.pt`.



### 4: Set up configurations
Next, add a class with the name NewData in the `configs/data_model_configs.py` file. 
You can find similar classes for existing datasets as guidelines. 
Also, you have to specify the cross-domain scenarios in `self.scenarios` variable.
Last, you have to add another class with the name NewData in the `configs/hparams.py` file to specify
the training parameters.


## Usage 

### Algorithm 
The algorithm & model can be found [here](Umbrella/raincoat.py). 


### Training a Model

The experiments are organised in a way such that:
- Experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different independent runs, which is determined by `--num_runs`.

To train a model:

```
python run.py  --experiment_description WISDM  \
                --dataset WISDM \
                --num_runs 5 \
                --device cuda
```


To see and/or modify the default hyperparameters, please see `configs/hparams.py` and `configs/data_model_configs.py`.

### Citation
If you find *Umbrella* useful for your research, please consider citing our work:

```

```

### Lisence
Raincoat codebase is under MIT license. For individual dataset usage, please refer to the dataset license found in the website.