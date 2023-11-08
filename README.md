# Sparse Binary Transformers
#### Published at KDD 2023
https://dl.acm.org/doi/pdf/10.1145/3580305.3599508


Experimental runs contained in "run_*.sh" files

For example, to train the SMD anomaly detection model run the following: 
```
python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25
```