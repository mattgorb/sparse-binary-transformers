nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=7 --dmodel=256 --window_size=300> ~/results_forecasting/2_new.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 --dmodel=64 > ~/results_forecasting/3.txt 2>&1 &


nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=256> ~/results_classification/4.txt 2>&1 &