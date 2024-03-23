# Sparse Binary Transformers
#### Published at KDD 2023
https://dl.acm.org/doi/pdf/10.1145/3580305.3599508


Experiment scripts in "run_*.sh" files

For example, to train the SMD anomaly detection model run the following: 
```
python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25
```

Original paper uses two files, main_ts.py, which has single-step forecasting and anomaly detection, and main_ts_classification.py, which has classification.  
```
nohup python -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 > ~/results_forecasting/3.txt 2>&1 &

```

New project examines longer term forecasting.  I created main_ts_forecast_longterm.py
