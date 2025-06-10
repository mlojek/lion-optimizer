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
| Model     | Optimizer | Config                                | Test accuracy | Test loss     |
|-----------|-----------|---------------------------------------|---------------|---------------|
| ResNet-50 | SGD       | [here](configs/resnet50_sgd.json)     | TODO          | TODO          |
| ResNet-50 | AdamW     | [here](configs/resnet50_adamw.json)   | 88.18         | 0.3307        |
| ResNet-50 | Lion      | [here](configs/resnet50_lion.json)    | 91.84         | 0.2272        |
| Vit-B-16  | SGD       | [here](configs/vitb16_sgd.json)       | TODO          | TODO          |
| Vit-B-16  | AdamW     | [here](configs/vitb16_adamw.json)     | 82.69         | 0.4703        |
| Vit-B-16  | Lion      | [here](configs/vitb16_lion.json)      | 66.33         | 0.8865        |

## Conclusions