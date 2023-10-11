# A Source-free Domain Adaptive Polyp Detection Framework with Style Diversification Flow

:pushpin: This is an official PyTorch implementation of **[TMI 2022]** - [A Source-free Domain Adaptive Polyp Detection Framework with Style Diversification Flow](https://ieeexplore.ieee.org/document/9709278).

> [**A Source-free Domain Adaptive Polyp Detection Framework with Style Diversification Flow**]()
> [Xinyu Liu](https://xinyuliu-jeffrey.github.io/), [Yixuan Yuan](http://www.ee.cuhk.edu.hk/~yxyuan/)<br>The Chinese Univerisity of Hong Kong

# 💡 Installation 

Please check [INSTALL.md](INSTALL.md) for installation instructions.

# 🔥 Get Started 

## Step 1: Data Preparation 

### Option 1: Download data from official data link and convert to detection format.

1. Download datasets from the following sources:

    Source Domain:
    [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
    Target Domain:
    [Abnormal Symptoms](https://dl.acm.org/doi/10.1145/3343031.3356073)
    [ETIS-Larib](https://polyp.grand-challenge.org/EtisLarib/)
    [KID](https://mdss.uth.gr/datasets/endoscopy/kid/)

2. Change the masks to coco style. Please refer to [this link](https://github.com/chrise96/image-to-coco-json-converter) or write a script.

### Option 2: Use our preprocessed data 

1. Download datasets and corresponding coco format annotations in the following links:

    Source Domain:

    CVC-ClinicDB: [gDrive](https://drive.google.com/drive/folders/1A3HEVZjWEtR3h7_SPwixDfP9oz43POhc?usp=share_link)
    
    Target Domain:

    Abnormal Symptoms: [gDrive](https://drive.google.com/file/d/1NQC-xc7GB-mp9Zhq6lwH9kid4_NtPImF/view?usp=sharing)
    ETIS-Larib: [gDrive](https://drive.google.com/drive/folders/1Pq0g4n4QWg3o1oXjyBgrmjXWgyu_zWGf?usp=sharing)


### Change the dataset dir 
    
Change the dataset dir to your downloaded data path in [here](fcos_core/config/paths_catalog.py).


## Step 2: Prepare the source-only model

Download the Source-only model from [gDrive](https://drive.google.com/file/d/1vnSYyKxiKxbRF-RB8XYoHGWZX2oNA4bx/view?usp=sharing).

## Step 3: Training SMPT/SMPT++

    # Train the SMPT stage
    python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 5

    # Train the SSDF stage, need to modify "ann_file"(L348) in ./fcos_core/engine/trainer to the annotations of the training dataset.
    python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 2

## Step 4: Testing the trained model 

    # Test the trained model
    python tools/train_net_mcd.py --config-file configs/sf/$YOUR YAML FILE$ SOLVER.SFDA_STAGE 5 SOLVER.TEST_ONLY True MODEL.WEIGHT $YOUR .pth WEIGHT$
    
# 📝 Citation

If you find this work or codebase is useful for your research, please give it a star and citation. We sincerely appreciate for your acknowledgments.
```BibTeX
@article{liu2022source,
  title={A source-free domain adaptive polyp detection framework with style diversification flow},
  author={Liu, Xinyu and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={7},
  pages={1897--1908},
  year={2022},
  publisher={IEEE}
}
```

# 🤞 Acknowledgement

The code is based on [FCOS](https://github.com/tianzhi0549/FCOS). Thanks for the excellent framework. For enquiries please contact xinyuliu@link.cuhk.edu.hk.
