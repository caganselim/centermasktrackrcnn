# CenterMaskTrackRCNN

### Getting Models

__Run:__
* checkpoints/base_models/get_base_models.sh
* checkpoints/trained_models/get_trained_models.sh

### Reported Performances

*[Click here to download result zips](https://drive.google.com/drive/folders/1yZFOaPJV1iKs0D5m_GhXOmsmjnCdMm1T?usp=sharing)*

* __NO-LL:__ 30.99 mAP 
* __128-DIM Encoder:__ 31.12 mAP
* __512-DIM Encoder:__ 31.44 mAP

### Commands

Train something:
``
python train_net.py --config-file <config-file> --num-gpus 2
``


Eval 128-dim:
``
python train_net.py --config-file configs/V99_ytvis_128dim.yaml --eval-only
``

Eval 512-dim:
``
python train_net.py --config-file configs/V99_ytvis_512dim.yaml --eval-only
``