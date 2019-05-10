# CARLA_CIL_Pytorch
===============

A pytorch implementation to train the conditional imitation learning policy in "End-to-end Driving via Conditional Imitation Learning" and "CARLA: An Open Urban Driving Simulator]".

## Requirements
-------
python 3.6    
pytorch > 0.4.0    
tensorboardX    
opencv    
imagaug    
h5py    

please  check ***docker/docker_build/Dockerfile*** for details.

## Train
-------
```
$ python main.py --batch-size 1000 --workers 16
    --train-dir "path/to/AgentHuman/SeqTrain/"
    --eval-dir "path/to/AgentHuman/SeqVal/"
    --gpu 0
    --id training
```
Check the training log through tensorboard
```
$ tensorboard --logdir runs
```

## Docker
-------
Revise the path of the dataset and this repo in ***docker/carla_cil_compose/docker-compose.yml***.    
docker-compose 2.3 and nvidia-docker 2 are required.

```
$ cd docker/carla_cil_compose
$ docker-compose up -d
```
We can still use tensorboard to check the log out of the docker.

## dataset
-------
Please check the original [dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) of Carla Imitation Learning.    
Please check this [issue](https://github.com/carla-simulator/imitation-learning/issues/1) for data augmentation.

## Reference
-------
[carla-simulator/imitation-learning](https://github.com/carla-simulator/imitation-learning)    
[mvpcom/carlaILTrainer](https://github.com/mvpcom/carlaILTrainer)    
[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)    
[CARLA: An Open Urban Driving Simulator](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)
