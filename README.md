# carla_cil_pytorch


The pytorch implementation to train the uncertain aware imitation learning policy in "Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective".

## Requirements
python 3.6    
pytorch > 0.4.0    
tensorboardX    
opencv    
imagaug    
h5py    

please  check ***docker/docker_build/Dockerfile*** for details.

## Train
**train-dir** and **eval-dir** should point to where the [Carla dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) located.
Please check our [paper](https://arxiv.org/abs/1903.00821) that how we split the train and eval dataset.
```
$ python main.py
  --batch-size 1000
  --workers 24
  --speed-weight 1
  --learning-rate 0.0001
  --lr-gamma 0.5
  --lr-step 10
  --train-dir "/path/to/the/training_data"
  --eval-dir "/path/to/the/eval_data"
  --net-structure 2
  --gpu 0
  --id name_of_policy
```
Check the training log through tensorboard.
```
$ tensorboard --logdir runs
```

## Docker
Revise the path of the dataset and this repo in ***docker/carla_cil_compose/docker-compose.yml***.    
docker-compose 2.3 and nvidia-docker 2 are required.

```
$ cd docker/carla_cil_compose
$ docker-compose up -d
```
We can still use tensorboard to check the log out of the docker.

## Dataset
Please check the original [dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) of Carla Imitation Learning.    
Please check this [issue](https://github.com/carla-simulator/imitation-learning/issues/1) for data augmentation.

## Benchmark
Please reference [carla_cil_pytorch_eval](https://github.com/onlytailei/carla_cil_pytorch_eval/blob/pytorch_eval/README.md).    
For the benchmark results, please check our paper [Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective](https://arxiv.org/abs/1903.00821).

## Reference
[carla-simulator/imitation-learning](https://github.com/carla-simulator/imitation-learning)    
[mvpcom/carlaILTrainer](https://github.com/mvpcom/carlaILTrainer)    
[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)    
[CARLA: An Open Urban Driving Simulator](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)    
[VR-Goggles for Robots: Real-to-sim Domain Adaptation for Visual Control](https://ram-lab.com/file/tailei/vr_goggles/index.html)    
[Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective](https://arxiv.org/abs/1903.00821)

The code for original "End-to-end Driving via Conditional Imitation Learning" and "CARLA: An Open Urban Driving Simulator" is in the [master branch](https://github.com/onlytailei/carla_cil_pytorch/tree/master). In the paper VR-Goggles, we also used the original setup to train the policy.

Please consider to cite our paper if thie repo helps:
```
@inproceedings{tai2019visual,
  author={Tai, Lei and Yun, Peng and Chen, Yuying and Liu, Congcong and Ye, Haoyang and Liu, Ming},
  title={Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2019}
}
@ARTICLE{zhang2019vrgoggles,
  author={Zhang, Jingwei and Tai, Lei and Yun, Peng and Xiong, Yufeng and Liu, Ming and Boedecker, Joschka and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters},
  title={VR-Goggles for Robots: Real-to-Sim Domain Adaptation for Visual Control},
  year={2019},
  volume={4},
  number={2},
  pages={1148-1155},
  keywords={Visualization;Training;Robots;Adaptation models;Semantics;Task analysis;Navigation;Deep learning in robotics and automation;visual-based navigation;model learning for control},
  doi={10.1109/LRA.2019.2894216},
  ISSN={2377-3766},
  month={April},}
```
