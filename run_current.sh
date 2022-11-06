nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=7 --dmodel=256 --window_size=300> ~/results_forecasting/2_new.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 --dmodel=64 > ~/results_forecasting/3.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=256> ~/results_classification/4.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=64 --epochs=50> ~/results_classification/3.txt 2>&1 &






nohup python3 -u main_ts.py --config=configs/dense_ts_ettm1_forecast.yaml --gpu=5 --evaluate > ~/evaluate/dense_19.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=2  --evaluate > ~/evaluate/dense_22.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=3  --evaluate > ~/evaluate/dense_23.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=4 --evaluate > ~/evaluate/dense_24.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=5 --evaluate > ~/evaluate/19.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/22.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/23.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/24.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/26.txt 2>&1 &
