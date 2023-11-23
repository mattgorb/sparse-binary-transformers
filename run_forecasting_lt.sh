nohup python -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 > ~/results_forecasting/1.txt 2>&1 &
nohup python -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 > ~/results_forecasting/2.txt 2>&1 &
nohup python -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 > ~/results_forecasting/3.txt 2>&1 &



nohup python -u main_ts_new.py --config=configs/sparse_ts_elect_forecast_t_steps.yaml --gpu=0 --window_size=210 > ~/results_forecasting/3.txt 2>&1 &