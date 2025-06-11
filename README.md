# Lion optimizer
TODO introduction

Implementation of LION gradient optimizer from [this publication by Google](https://arxiv.org/pdf/2302.06675).

## Experiment plan
To compare the performance of my implementation, I wanted to reproduce the experiments from the original publication. In those, AdamW and Lion were trained and evaluated on ImageNet dataset three times and then compared on the resulting classfication accuracies.

However due to limited computational resources I was not able to use ImageNet dataset. ImageNet is a collection of 14 million 224x224 RGB images representing 1000 object classes. Instead I opted for FashionMNIST dataset, consisting of 60 thousand train and 13 thousand test grayscale 28x28 images split into 10 classes, representing various articles of clothing.

In this comparison, ResNet-50 and Vit-B-16 models were compared. These were selected due to their relatively small size, meaning that the experiments could be ran on RTX 4060 GPU.

To ensure fair comparison, I used hyperparameters specified by authors in chapter 5 of the publication. These hyperparameters are the results of extensive hyperparameters tuning, although done on ImageNet dataset and not FashionMNIST. In addition to that, the train and val split is always done with the same random seed to ensure that the datasets are split identically for every run. The weights of the model are also initialized with a common random seed to ensure that the optimizers are performing the same exact optimization task.

Unfortunately, due to limited resources I was able to only run each model-optimizer pair once, which means that the results were not guaranteed to be reliable and statistically significant. To get statistically significant data, I would need to run each model 25 times and perform some sort of statistical tests, like Mann-Whitney U test.

## How to run this project
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
| ResNet-50 | AdamW     | [here](configs/resnet50_adamw.json)   | 88.18         | 0.3307        |
| ResNet-50 | Lion      | [here](configs/resnet50_lion.json)    | 91.84         | 0.2272        |
| Vit-B-16  | AdamW     | [here](configs/vitb16_adamw.json)     | 82.69         | 0.4703        |
| Vit-B-16  | Lion      | [here](configs/vitb16_lion.json)      | 66.33         | 0.8865        |

Although the results are not statistically reliable, a difference of around 3 percentage points in test accuracy can be observed on ResNet-50 model. This is a much more notable difference than that observed for ResNet-50 on ImageNet, where the difference was around 0.1 percentage point. A reverse result was observed for Vit-B-16, where the training stopped on a low accuracy value. This was due to the usage of an early stopping mechanism. The most propable cause of this low result was ill-fitted hyperparameter set and unfavourable starting weights of the model - in multiple runs of training this result would most likely be an outlier.

## Conclusions
- Lion optimizer was successfully reproduced.
- Further experiments are needed to gather statistically significant results.
- Experimenting on models this size requires a lot of computational power.
