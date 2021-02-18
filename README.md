# HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients
This is an implementation of [HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://arxiv.org/abs/2010.01264)
- Global model parametersWgare distributed to `m=6` local clients with `p=3` computation complexity levels.
<img src="/assest/HeteroFL.png">


## Requirements
 - see requirements.txt

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
