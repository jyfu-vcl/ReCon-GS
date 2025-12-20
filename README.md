<div align="center">

# [NeurIPS 2025🔥] ReCon-GS: Continuum-Preserved Guassian Streaming for Fast and Compact Reconstruction of Dynamic Scenes

[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://github.com/jyfu-vcl/ReCon-GS/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.24325-b31b1b.svg)](https://arxiv.org/abs/2509.24325 )

[Jiaye Fu](https://scholar.google.com/citations?user=9qBFfMgAAAAJ&hl=en), Qiankun Gao, Chengxiang Wen, Yanmin Wu, Siwei Ma, Jiaqi Zhang, [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

National Engineering Research Center of Visual Technology, Peking University

</div>

----

## 🔥 Abstract

Online free-viewpoint video (FVV) reconstruction is challenged by slow per-frame optimization, inconsistent motion estimation, and unsustainable storage demands. To address these challenges, we propose the **Re**configurable **Con**tinuum Gaussian Stream, dubbed \textbf{ReCon-GS}, a novel storage-aware framework that enables high-fidelity online dynamic scene reconstruction and real-time rendering.
Specifically, we dynamically allocate multi-level Anchor Gaussians in a density-adaptive fashion to capture inter-frame geometric deformations, thereby decomposing scene motion into compact coarse-to-fine representations.
Then, we design a dynamic hierarchy reconfiguration strategy that preserves localized motion expressiveness through on-demand anchor re-hierarchization,  while ensuring temporal consistency through intra-hierarchical deformation inheritance that confines transformation priors to their respective hierarchy levels.
Furthermore, we introduce a storage-aware optimization mechanism that flexibly adjusts the density of Anchor Gaussians at different hierarchy levels, enabling a controllable trade-off between reconstruction fidelity and memory usage.
Extensive experiments on three widely used datasets demonstrate that, compared to state‐of‐the‐art methods, ReCon-GS improves training efficiency by approximately 15% and achieves superior FVV synthesis quality with enhanced robustness and stability. Moreover, at equivalent rendering quality, ReCon-GS slashes memory requirements by over 50% compared to leading state‑of‑the‑art methods. 


## 🚩 News

- 09.19 "ReCon-GS" is accepted by **NeurIPS 2025 poster**.
- 09.25 Release the paper
- 12.20 Release the code

## Instruction's to run the code

Our code is built based on official libgs repository. Please follow the instructions from [libgs](https://github.com/Awesome3DGS/libgs) to install the required library.

Please familiarize yourself with it before running the experiments.

1. Install dependencies

   ```bash
   pip install .
   ```

2. Run pipeline

   ```
   python main.py --config=config/dynerf.yaml --data.root=<PATH TO SCENE ROOT>
   ```

   The `config` directory contains pre-defined configurations for reproducing the results reported in the paper.


## Acknowledgement

We appreciate the releasing codes of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [libgs](https://github.com/Awesome3DGS/libgs). We also want to express our greatest gratitude to [Junoh Lee](cywon1997@gm.gist.ac.kr) for his assistance.

# Citation

```bibtex
@inproceedings{recongs2025,
   title={ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction of Dynamic Scenes}, 
   author={Jiaye Fu and Qiankun Gao and Chengxiang Wen and Yanmin Wu and Siwei Ma and Jiaqi Zhang and Jian Zhang},
   booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
   year = {2025}
}
```

[//]: # ()
[//]: # (---)

[//]: # (## ⭐️ Star History)

[//]: # ()
[//]: # ([![Star History Chart]&#40;https://api.star-history.com/svg?repos=Jiexuanz/AlignedGen&type=Date&#41;]&#40;https://www.star-history.com/#Jiexuanz/AlignedGen&Date&#41;)

[//]: # ()
