<!-- <div align="center">
    <img src="docs/img/AllenAct.svg" width="350" />
    <br>
    <i><h3>An open source framework for research in Embodied AI</h3></i>
    </p>
    <hr/>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Documentation Status](https://img.shields.io/badge/docs-up%20to%20date-Green.svg)](https://allenact.org)
[![Latest Release](https://img.shields.io/github/v/release/allenai/allenact)](https://github.com/allenai/allenact/releases/latest)
[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![LGTM Grade: Python](https://img.shields.io/lgtm/grade/python/g/allenai/allenact.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/allenai/allenact/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AllenAct** is a modular and flexible learning framework designed with a focus on the unique requirements of Embodied-AI research. It provides first-class support for a growing collection of embodied environments, tasks and algorithms, provides reproductions of state-of-the-art models and includes extensive documentation, tutorials, start-up code, and pre-trained models.

AllenAct is built and backed by the [Allen Institute for AI (AI2)](https://allenai.org/). AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

## Quick Links

- [Website & Docs](https://www.allenact.org/)
- [Github](https://github.com/allenai/allenact)
- [Install](https://www.allenact.org/installation/installation-allenact/)
- [Tutorials](https://www.allenact.org/tutorials/)
- [AllenAct Paper](https://arxiv.org/abs/2008.12760)
- [Citation](#citation)

## Features & Highlights

* _Support for multiple environments_: Support for the [iTHOR](https://ai2thor.allenai.org/ithor/), [RoboTHOR](https://ai2thor.allenai.org/robothor/) and [Habitat](https://aihabitat.org/) embodied environments as well as for grid-worlds including [MiniGrid](https://github.com/maximecb/gym-minigrid).
* _Task Abstraction_: Tasks and environments are decoupled in AllenAct, enabling researchers to easily implement a large variety of tasks in the same environment.
* _Algorithms_: Support for a variety of on-policy algorithms including [PPO](https://arxiv.org/pdf/1707.06347.pdf), [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf), [A2C](https://arxiv.org/pdf/1611.05763.pdf), Imitation Learning and [DAgger](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf) as well as offline training such as offline IL.
* _Sequential Algorithms_: It is trivial to experiment with different sequences of training routines, which are often the key to successful policies.
* _Simultaneous Losses_: Easily combine various losses while training models (e.g. use an external self-supervised loss while optimizing a PPO loss).
* _Multi-agent support_: Support for multi-agent algorithms and tasks.
* _Visualizations_: Out of the box support to easily visualize first and third person views for agents as well as intermediate model tensors, integrated into Tensorboard.
* _Pre-trained models_: Code and models for a number of standard Embodied AI tasks.
* _Tutorials_: Start-up code and extensive tutorials to help ramp up to Embodied AI.
* _First-class PyTorch support_: One of the few RL frameworks to target PyTorch.
* _Arbitrary action spaces_: Supporting both discrete and continuous actions.

|Environments|Tasks|Algorithms|
|------------|-----|----------|
|[iTHOR](https://ai2thor.allenai.org/ithor/), [RoboTHOR](https://ai2thor.allenai.org/robothor/), [Habitat](https://aihabitat.org/), [MiniGrid](https://github.com/maximecb/gym-minigrid), [OpenAI Gym](https://gym.openai.com/)|[PointNav](https://arxiv.org/pdf/1807.06757.pdf), [ObjectNav](https://arxiv.org/pdf/2006.13171.pdf), [MiniGrid tasks](https://github.com/maximecb/gym-minigrid), [Gym Box2D tasks](https://gym.openai.com/envs/#box2d)|[A2C](https://arxiv.org/pdf/1611.05763.pdf), [PPO](https://arxiv.org/pdf/1707.06347.pdf), [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf), [DAgger](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf), Off-policy Imitation|

## Contributions
We welcome contributions from the greater community. If you would like to make such a contributions we recommend first submitting an [issue](https://github.com/allenai/allenact/issues) describing your proposed improvement. Doing so can ensure we can validate your suggestions before you spend a great deal of time upon them. Improvements and bug fixes should be made via a pull request from your fork of the repository at [https://github.com/allenai/allenact](https://github.com/allenai/allenact).

All code in this repository is subject to formatting, documentation, and type-annotation guidelines. For more details, please see the our [contribution guidelines](CONTRIBUTING.md).

## Acknowledgments
This work builds upon the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) library of Ilya Kostrikov and uses some data structures from FAIR's [habitat-lab](https://github.com/facebookresearch/habitat-lab). We would like to thank Dustin Schwenk for his help for the public release of the framework.

## License
AllenAct is MIT licensed, as found in the [LICENSE](LICENSE) file.

## Team
AllenAct is an open-source project built by members of the PRIOR research group at the Allen Institute for Artificial Intelligence (AI2). 

<div align="left">
    <a href="//prior.allenai.org/" target="_blank">
        <img src="docs/img/ai2-prior.svg" width="400">
    </a>
    <br>
</div>

## Citation
If you use this work, please cite our [paper](https://arxiv.org/abs/2008.12760):

```bibtex
@article{AllenAct,
  author = {Luca Weihs and Jordi Salvador and Klemen Kotar and Unnat Jain and Kuo-Hao Zeng and Roozbeh Mottaghi and Aniruddha Kembhavi},
  title = {AllenAct: A Framework for Embodied AI Research},
  year = {2020},
  journal = {arXiv preprint arXiv:2008.12760},
}
``` -->
Installation Instructions:

To begin, clone this repository locally
```bash
git clone -b rearrangement --single-branch git@github.com:allenai/embodied-clip.git embclip-rearrangement
```
Then run:
```bash
pip install -r requirements.txt ; pip install -r dev_requirements.txt
```

