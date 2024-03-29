# Online Bilevel Optimization: Regret Analysis of Online Alternating Gradient Methods

This repository holds the official code for the manuscript
"[Online Bilevel Optimization: Regret Analysis of Online Alternating Gradient Methods](https://arxiv.org/pdf/2207.02829.pdf)".

### 🦸‍ Abstract
This paper introduces an _online bilevel optimization_ setting in which a sequence of time-varying bilevel problems are revealed one after the other. We extend the known regret bounds for single-level online algorithms to the bilevel setting. Specifically, we provide new notions of _bilevel regret_, develop an online alternating time-averaged gradient method that is capable of leveraging smoothness, and give regret bounds in terms of the path-length of the inner and outer minimizer sequences.

### 📝 Requirements

Before running the code, we need to deploy the environment.
A recommended way is to use conda to create the environment and install the related packages shown as follows.

```bash
conda create -n OAGD python=3.9
pip install -r requirements.txt
conda activate OAGD 
```

### 🔨 Usage
To run the code for either Hyperparameter-Optimization or Meta-Learning, 
please use 
```bash 
python main.py
``` 
with specific arguments in each folder.


### 📖 Reference
Our hyperparameter-optimization code is implemented based on [AutoBalance](https://proceedings.neurips.cc/paper_files/paper/2021/file/191f8f858acda435ae0daf994e2a72c2-Paper.pdf): 

https://github.com/ucr-optml/AutoBalance.

Our meta-learning code is implemented based on [iMAML](https://arxiv.org/pdf/1909.04630.pdf): 

https://github.com/prolearner/hypertorch/blob/master/examples/iMAML.py.

### 📭 Maintainers
[Bojian Hou](http://bojianhou.com) 
- ([bojian.hou@pennmedicine.upenn.edu](mailto:bojian.hou@pennmedicine.upenn.edu))
- ([hobo.hbj@gmail.com](mailto:hobo.hbj@gmail.com))
