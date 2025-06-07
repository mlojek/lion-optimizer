# Lion optimizer
TODO introduction
Implementation of LION gradient optimizer from [this publication by Google](https://arxiv.org/pdf/2302.06675).

## Experiment plan
- due to limted resources, ill use fashionmnnist instead of imagenet. TODO describe both datasets
- in addition to test accuracy from paper, ill compare test loss and training time
- well compare optimizers SGD, AdamW and our implementation of Lion
- on resnet50 and vits
- despite using a different task, i wont do hyperparameter tuning due to limited resources and use hyperparameters from chapter 5 of the paper

## How to use
Use python 3.13, then run:
```bash
make install
```
Run training
```
python -m src.scripts.train configs/path/to/config.json
```

## Results

## Conclusions