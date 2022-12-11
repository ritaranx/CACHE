# CACHE
This repo contains our code for paper Counterfactual and Factual Reasoning over Hypergraphs for Interpretable Clinical Predictions on EHR, in Proceedings of 2nd Machine Learning for Health symposium 2022 (ML4H 2022).

## Model Framework

![CACHE-Framework](docs/cache.png)

## Data
In order to facilitate the reproducibility, we provide two toy datasets in [data](data).
Note that although they're named as `mimic3` and `cradle` (the two datasets we mentioned in our paper), they're in fact randomly generated due to the privacy issue.
We include them in this repo only to show the format of the two datasts we used.
Thus, their experimental results should not reflect the performance we report in the paper.

## Package 
- PyTorch 1.4
- python 3.7
- tqdm
- torch-scatter 2.0.4
- torch-sparse 0.6.0
- torch-cluster 1.5.2
- torch-geometric 1.6.3
- sklearn

## Run the Code
Please use `run.sh` in [src](src) to run the code for the two toy datasets in [data](data).
It runs four experiments:
- CACHE for MIMIC-III dataset
- CACHE for CRADLE dataset
- vanilla backbone model for MIMIC-III dataset
- vanilla backbone model for CRADLE dataset

## Citation

If you find this paper useful for your research, please cite the following in your publication. Thanks!
```
@inproceedings{xu2022counterfactual,
  title={Counterfactual and Factual Reasoning over Hypergraphs for Interpretable Clinical Predictions on EHR},
  author={Xu, Ran and Yu, Yue and Zhang, Chao and Ali, Mohammed K and Ho, Joyce C and Yang, Carl},
  booktitle={Machine Learning for Health},
  pages={259--278},
  year={2022},
  organization={PMLR}
}
```

## Acknowledgement
We would like to thank the authors from [AllSet](https://github.com/jianhao2016/AllSet) for their open-source efforts.
