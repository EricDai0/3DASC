# Transferable 3D Adversarial Shape Completion using Diffusion Models ECCV 2024
 
The code repository for our paper Transferable 3D Adversarial Shape Completion using Diffusion Models [arXiv](https://arxiv.org/abs/2407.10077)

# Installation

This repository is based on the official code from [PVD](https://github.com/alexzhou907/PVD/). 

1. Set up environments for the codes. Details please refer to the original Github code.

```
python==3.6
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
```

2. Download ShapeNet Completion Dataset (https://github.com/xiumingzhang/GenRe-ShapeHD).
   
3. Download the PVD model checkpoint (https://drive.google.com/drive/folders/1Q7aSaTr6lqmo8qx80nIm1j28mOHAHGiM?usp=sharing).

4. Download the ShapeNet pre-trained 3D point cloud classifiers (https://drive.google.com/file/d/1fgz4OPdPYnb6n5yyhGOjh-N2kKMThVQ4/view?usp=sharing). You can train the target models by following https://github.com/qiufan319/benchmark_pc_attack/tree/master/baselines.

# Usage

Simply run completion-attack_uncertain.py to conduct 3D point cloud attacks. Make sure to set the directory accordingly in the parameters.
```
python completion-attack_uncertain.py 
```

# Reference

Please cite our paper if you found any helpful information:


    @article{dai2024transferable,
    title={Transferable 3D Adversarial Shape Completion using Diffusion Models},
    author={Dai, Xuelong and Xiao, Bin},
    journal={arXiv preprint arXiv:2407.10077},
    year={2024}
    }
