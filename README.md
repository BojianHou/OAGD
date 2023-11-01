This is the official repository for the manuscript
"Online Bilevel Optimization: Regret Analysis of Online Alternating Gradient Methods"
submitted to AISTATS 2024.

### 🦸‍ Abstract
This paper introduces an \textit{online bilevel optimization} setting in which a sequence of time-varying bilevel problems are revealed one after the other. We extend the known regret bounds for single-level online algorithms to the bilevel setting. Specifically, we provide new notions of \textit{bilevel regret}, develop an online alternating time-averaged gradient method that is capable of leveraging smoothness, and give regret bounds in terms of the path-length of the inner and outer minimizer sequences.

### 📝 Requirements

Before running the code, we need to deploy the environment.
A recommended way is to use conda to create the environment and install the related packages.

```bash
conda create -n OAGD python=3.9
pip install -r requirements.txt
conda activate OAGD 
```

### 🔨 Usage
To run the code for either Hyperparameter-Optimization or Meta-Learning, 
please use ```bash python main.py``` with specific arguments in each folder.