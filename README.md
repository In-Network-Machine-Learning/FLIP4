# FLIP4
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction
FLIP4 is a federated learning-based framework for in-network traffic analysis. This repo is the artifact for the paper "Federated In-Network Machine Learning for Privacy-Preserving IoT Traffic Analysis" published on ACM Transactions on Internet Technology. [PDF](https://dl.acm.org/doi/pdf/10.1145/3696354)

## Prepare your environment
This setup is based on [BMv2](https://github.com/p4lang/behavioral-model) in Ubuntu 20.04. 
We recommend the following guideline to set up a [VM environment](https://github.com/p4lang/tutorials/tree/master/vm-ubuntu-20.04) with Ubuntu 20.04

Please make sure the following packages are installed by running: 
```
$ pip3 install -r ./requirements.txt
```
💡 To run the demo on Raspberry Pi, please follow the [P4Pi](https://github.com/p4lang/p4pi) guideline to configure the environment. 

## Run a demo
To run this code: 
1. Navigate to a use case folder (use case with dataset $CIC-IDS2017$ as an example here), compile and run the BMv2 environment:
```
$ cd use_case_cic
$ make clean && make run
```

2. Open a new terminal to run a server:
```  
$ python3 fedavg.py
```
3. Open a new terminal to run clients on switches:
```  
$ python3 FL_controller.py
```

* The source of the dataset: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## FLIP4 Workflow

![image](https://github.com/In-Network-Machine-Learning/FLIP4/blob/main/image/FLPI4_workflow.png)

### Aggregator
The file `fedavg.py` continuously receives model parameters from the end nodes (clients) and aggregates the received parameters to form a global model based on the FedAvg algorithm.  

### Control plane logic
The file `FL_controller.py` is responsible for model parameter sharing and communication between the local data plane and the aggregator for updates. 

### Data plane logic
The file `client_mapper.py` maps the ML model to the data plane for in-network inference. The model can be further replaced by other classical ML models by [Planter](https://github.com/In-Network-Machine-Learning/Planter)



## Citation

FLIP4 builds upon [Planter](https://github.com/In-Network-Machine-Learning/Planter) and [P4Pir](https://github.com/In-Network-Machine-Learning/P4Pir).

It is further inspired by [IIsy](https://github.com/cucl-srg/IIsy), [SwitchTree](https://github.com/ksingh25/SwitchTree), [pForest](https://arxiv.org/abs/1909.05680), [federated-boosted-dp-trees](https://github.com/Samuel-Maddock/federated-boosted-dp-trees).

If you find this code helpful, please cite: 

````
@article{zang2024flip4,
title = {Federated In-Network Machine Learning for Privacy-Preserving IoT Traffic Analysis},
author = {Zang, Mingyuan and Zheng, Changgang and Koziak, Tomasz and Zilberman, Noa and Dittmann, Lars},
journal = {ACM Trans. Internet Technol.},
year = {2024}
}
````

💡 If you are interested in further details and more use cases of In-Network Machine Learning inference and how an ML model is mapped to a programmable data plane, please refer to [Planter](https://arxiv.org/abs/2205.08824), [IIsy](https://arxiv.org/abs/2205.08243), [Linnet](https://dl.acm.org/doi/abs/10.1145/3546037.3546057):


## Acknowledgment
This work was partly supported by the Nordic University Hub on Industrial IoT (HI2OT) by NordForsk, VMware, and Innovate UK (Project No. 10056403) as part of the SmartEdge EU project (Grant Agreement No. 101092908).
