# AllenAct Tutorials

**Note** The provided commands to execute these tutorials assume you have
[installed the full library](../installation/installation-allenact.md#full-library)
and the specific requirements for each used plugin.

We provide several tutorials to help ramp up researchers to the field of Embodied-AI as well as to the AllenAct framework.

## [Navigation in MiniGrid](../tutorials/minigrid-tutorial.md)

![MiniGridEmptyRandom5x5 task example](../img/minigrid_environment.png)

We train an agent to complete the `MiniGrid-Empty-Random-5x5-v0` task within the [MiniGrid](https://github.com/maximecb/gym-minigrid) environment. 

This tutorial presents:

* Writing an experiment configuration file with a simple training pipeline from scratch.
* Using one of the supported environments with minimal user effort.
* Training, validation and testing your experiment from the command line.

[Follow the tutorial here.](../tutorials/minigrid-tutorial.md)


## [PointNav in RoboTHOR](../tutorials/training-a-pointnav-model.md)

![RoboTHOR Robot](../img/RoboTHOR_robot.jpg)

We train an agent on the Point Navigation task within the RoboTHOR Embodied-AI environment.

This tutorial presents:

* The basics of the Point Navigation task, a common task in Embodied AI
* Using an external dataset
* Writing an experiment configuration file with a simple training pipeline from scratch.
* Use one of the supported environments with minimal user effort.
* Train, validate and test your experiment from the command line.
* Testing a pre-trained model

[Follow the tutorial here.](../tutorials/training-a-pointnav-model.md)


## [Swapping in a new environment](../tutorials/transfering-to-a-different-environment-framework.md)

![Environment Transfer](../img/env_transfer.jpg)

This tutorial demonstrates how easy it is modify the experiment config created in the RoboTHOR PointNav tutorial to work with the iTHOR and Habitat environments. 

[Follow the tutorial here.](../tutorials/transfering-to-a-different-environment-framework.md)


## [Using a pretrained model](../tutorials/running-inference-on-a-pretrained-model.md)

![Pretrained inference](../img/viz_pretrained_2videos.jpg)

This tutorial shows how to run inference on one or more checkpoints of a pretrained model and generate
visualizations of different types.

[Follow the tutorial here.](../tutorials/running-inference-on-a-pretrained-model.md)


## [Off-policy training](../tutorials/offpolicy-tutorial.md)

This tutorial shows how to train an Actor using an off-policy dataset with expert actions.

[Follow the tutorial here.](../tutorials/offpolicy-tutorial.md)


## [OpenAI gym for continuous control](../tutorials/gym-tutorial.md)

![gym task example](../img/lunar_lander_continuous_demo.png)

We train an agent to complete the `LunarLanderContinuous-v2` task from
[OpenAI gym](https://gym.openai.com/envs/LunarLanderContinuous-v2). 

This tutorial presents:

* A `gym` plugin fopr `AllenAct`. 
* A continuous control example with multiple actors using PPO.

[Follow the tutorial here.](../tutorials/gym-tutorial.md)


<!---
## [OpenAI gym for MuJoCo](../tutorials/gym-mujoco-tutorial.md)

In this tutorial, we train a four-legged creature,
"ant", to walk forward as fast as possible in [MuJoCo](http://www.mujoco.org/).

[Follow the tutorial here.](../tutorials/gym-mujoco-tutorial.md)
--->


## [Multi-node training for RoboTHOR ObjectNav](../tutorials/distributed-objectnav-tutorial.md)

![training speedup](../img/multinode_training.jpg)

We train an agent to navigate to an object in a fraction of the time
required for training in one node by distributing training across multiple
nodes. 

This tutorial presents:

1. The AllenAct API for training across multiple nodes, as well as
experimental scripts for distributed configuration, training start
and termination, and remote command execution.
2. The introduction of the headless mode for [AI2-THOR](https://ai2thor.allenai.org/) in `AllenaAct`.

[Follow the tutorial here.](../tutorials/distributed-objectnav-tutorial.md)
