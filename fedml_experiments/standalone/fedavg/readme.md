## Experimental Tracking Platform (report real-time result to wandb.com, please change ID to your own)
wandb login 3da5dc700f370e170bbc9d4e272d783c9d87cdc6


## Experiment Scripts
Heterogeneous distribution (Non-IID) experiment:

Frond-end debugging:
``` 
## MNIST
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0

## shakespeare (LEAF)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 shakespeare ./../../../data/shakespeare rnn hetero 100 1 0.8 sgd 0

# fed_shakespeare (Google)
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 100 1 0.8 sgd 0

## Federated EMNIST
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 femnist ./../../../data/FederatedEMNIST cnn hetero 200 1 0.03 sgd 0

## Fed_CIFAR100
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_cifar100 ./../../../data/fed_cifar100 resnet18_gn hetero 200 1 0.03 adam 0

# Stackoverflow_LR
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_lr ./../../../data/stackoverflow lr hetero 200 1 0.03 sgd 0

# Stackoverflow_NWP
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 stackoverflow_nwp ./../../../data/stackoverflow rnn hetero 200 1 0.03 sgd 0

# CIFAR10
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 cifar10 ./../../../data/cifar10 resnet56 hetero 200 1 0.03 sgd 0
```

Please make sure to run on the background when you start training after debugging. An example to run on the background:
``` 
# MNIST
nohup sh run_fedavg_standalone_pytorch.sh 2 10 10 10 mnist ./../../../data/mnist lr hetero 200 20 0.03 sgd 0 > ./fedavg_standalone.txt 2>&1 &
```

For large DNNs (ResNet, Transformer, etc), please use the distributed computing (fedml_api/distributed). 


### Benchmark Results
We publish benchmark experimental results at wanb.com: \
https://app.wandb.ai/automl/fedml
