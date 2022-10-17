nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 > ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=3 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/3.txt 2>&1 &



nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=4 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=5 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/6.txt 2>&1 &


nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=0 --attention='Sparse'> ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1 --attention='Sparse'> ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=3 --attention='Sparse'> ~/results_forecasting/3.txt 2>&1 &

