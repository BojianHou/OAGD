This is the official repository for the manuscript
"Online Bilevel Optimization: Regret Analysis of Online Alternating Gradient Methods"
submitted to AISTATS 2024.



Before running the code, we need to deploy the environment.
A recommended way is to use conda to create the environment and install the related packages.

```bash
conda create -n OAGD python=3.9
pip install -r requirements.txt
conda activate OAGD 
```

To run the code for either Hyperparameter-Optimization or Meta-Learning, 
please use ```bash python main.py``` with specific arguments in each folder.