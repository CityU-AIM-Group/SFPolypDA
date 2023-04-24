# A Source-free Domain Adaptive Polyp Detection Framework with Style Diversification Flow

Code of [SMPT & SMPT++](https://ieeexplore.ieee.org/document/9709278) in TMI 2022.


#### Installation 

Please check [INSTALL.md](INSTALL.md) for installation instructions.

#### Data Preparation

1. Download datasets from the following sources:

    Source Domain:
    [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
    Target Domain:
    [Abnormal Symptoms](https://dl.acm.org/doi/10.1145/3343031.3356073)
    [ETIS-Larib](https://polyp.grand-challenge.org/EtisLarib/)
    [KID](https://mdss.uth.gr/datasets/endoscopy/kid/)

2. Change the masks to coco style. Please refer to [this link](https://github.com/chrise96/image-to-coco-json-converter) or write a script.

3. Change the dataset dir in [here](fcos_core/config/paths_catalog.py).

#### Training

    # The following command lines control different stages of training:

    # Train the source only model
    python tools/train_net_mcd.py --config-file configs/sf/source_only.yaml SOLVER.SFDA_STAGE 1

    # Train the SMPT stage
    python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 5

    # Train the SSDF stage, need to modify "ann_file" in ./fcos_core/engine/trainer
    python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 2

#### Testing the trained model 

    # Train the source only model
    python tools/train_net_mcd.py --config-file configs/sf/$YOUR YAML FILE$ SOLVER.SFDA_STAGE 5 SOLVER.TEST_ONLY True MODEL.WEIGHT $YOUR .pth WEIGHT$
    
##### Acknowledgement

The code is based on [FCOS](https://github.com/tianzhi0549/FCOS). For enquiries please contact 1155195604@link.cuhk.edu.hk.
