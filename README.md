# HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients
This is an implementation of [HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://arxiv.org/abs/2010.01264)
- Global model parametersWgare distributed to `m=6` local clients with `p=3` computation complexity levels.
<img src="/assest/HeteroFL.png">


## Requirements
 - see requirements.txt

## Instruction

 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 
 - fed.py contrains aggregation and separation of subnetworks

## Examples
 - Train MNIST dataset (IID) with CNN model, 100 users, active rate 0.1, model split 'Fix', model split mode 'a-b (20%-80%)', BatchNorm, Scaler (True) , Masked CrossEntropy (True)
    ```ruby
    python train_classifier_fed.py --data_name MNIST --model_name conv --control_name 1_100_0.1_iid_fix_a2-b8_bn_1_1
    ```
 - Train CIFAR10 dataset (Non-IID 2 classes) with ResNet model, 10 users, active rate 0.1, model split 'Dynamic', model split mode 'a-b-c (uniform)', GroupNorm, Scaler (False) , Masked CrossEntropy (False)
    ```ruby
    python train_classifier_fed.py --data_name CIFAR10 --model_name resnet18 --control_name 1_10_0.1_non-iid-2_dynamic_a1-b1-c1_gn_0_0
    ```
 - Test WikiText2 dataset with Transformer model, 100 users, active rate 0.01, model split 'Fix', model split mode 'a (50%), b(50%)', No Normalization, Scaler (True) , Masked CrossEntropy (False)
    ```ruby
    python test_transformer_fed.py --data_name WikiText2 --model_name transformer --control_name 1_100_0.01_iid_fix_a5-b5_none_1_0
    ```
## Results
- Interpolation experimental results for MNIST (IID) dataset between global model complexity ((a) a, (b) b, (c) c, (d) d) and various smaller model complexities.

![MNIST_interp_iid](/assest/MNIST_interp_iid.png)

- Interpolation experimental results for CIFAR10 (IID) dataset between global model complexity ((a) a, (b) b, (c) c, (d) d) and various smaller model complexities.

![CIFAR10_interp_iid](/assest/CIFAR10_interp_iid.png)

- Interpolation experimental results for WikiText2 (IID) dataset between global model complexity ((a) a, (b) b, (c) c, (d) d) and various smaller model complexities.

![WikiText2_interp_iid](/assest/WikiText2_interp_iid.png)

## Acknowledgement
*Enmao Diao  
Jie Ding  
Vahid Tarokh*
