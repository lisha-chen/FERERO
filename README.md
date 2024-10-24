# FERERO: A Flexible Framework for Preference-Guided Multi-Objective Learning

This repository contains the code for the paper's experiments: ["FERERO: A Flexible Framework for Preference-Guided Multi-Objective Learning"]().

In this work, we 



# Environment setup

1. Use the following command to install the dependencies
```
conda create -n moo python=3.8
conda activate moo
conda install pytorch torchvision==0.9.0 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy scipy seaborn tqdm
conda install -c conda-forge cvxpy
```


# Experiments

## Toy

## Multi-MNIST

## Emotion



## License

MIT license

## Citation

```
@inproceedings{chen2024FERERO,
  title={FERERO: A Flexible Framework for Preference-Guided Multi-Objective Learning},
  author={Chen, Lisha and Saif, AFM and Shen, Yanning and Chen, Tianyi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```


## Ackowledgement

- The toy example, multi-MNIST classification, and emotion recognition benchmark experiments use the code from [EPO](https://github.com/dbmptr/EPOSearch).
- The multi-lingual ASR experiment uses the code from [M2ASR](https://github.com/afmsaif/M2ASR) as a baseline.

We thank the authors for providing the code and data. Please cite their works and ours if you use the code or data.
