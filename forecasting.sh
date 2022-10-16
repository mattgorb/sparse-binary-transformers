nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 > ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=3 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/3.txt 2>&1 &



nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=4 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=5 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_forecasting/6.txt 2>&1 &



nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=2 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/7.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=3 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/8.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/9.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/10.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/11.txt 2>&1 &
