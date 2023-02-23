# FLDetector KDD22

The official code of KDD22 paper "FLDetecotor: Defending Federated Learning Against Model Poisoning Attacks via Detecting Malicious Clients" [[paper]](http://home.ustc.edu.cn/~zaixi/ZaixiZhang_files/FLDetector.pdf).

<div align=center><img src="https://github.com/zaixizhang/FLDetector/blob/main/fldetector.png" width="700"/></div>

Our implementation is based on [[MXNet]](https://mxnet.apache.org/versions/1.9.1/).

## Training with FLDetector

Use train_mnist.py, train_cifar.py, and train_femnist.py to implement FLDetector on MNIST, CIFAR10, and FEMNIST respectively.

## Reference
```
@inproceedings{zhang2022fldetector,
  title={FLDetector: Defending federated learning against model poisoning attacks via detecting malicious clients},
  author={Zhang, Zaixi and Cao, Xiaoyu and Jia, Jinyuan and Gong, Neil Zhenqiang},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2545--2555},
  year={2022}
}
```
