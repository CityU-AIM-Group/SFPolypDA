# SFPolypDA

#### Installation 

Please check [INSTALL.md](INSTALL.md) for installation instructions.

#### Training

## Training

The following command line controls different stages of training:

```bash
# Train the source only model
python tools/train_net_mcd.py --config-file configs/sf/source_only.yaml SOLVER.SFDA_STAGE 1

# Train the SMPT stage
python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 5

# Train the SSDF stage, need to modify "ann_file" in ./fcos_core/engine/trainer
python tools/train_net_mcd.py --config-file configs/sf/smpt_hcmus.yaml SOLVER.SFDA_STAGE 2
```