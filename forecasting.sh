nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 > ~/results_anomaly/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_anomaly/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=3 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_anomaly/4.txt 2>&1 &
