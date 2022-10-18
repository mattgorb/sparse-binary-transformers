

nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25 --evaluate > ~/evaluate/7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=7 --lin_prune_rate=0.25 --attention_prune_rate=0.25 --evaluate > ~/evaluate/8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=0 --lin_prune_rate=0.25 --attention_prune_rate=0.25 --evaluate > ~/evaluate/9.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=5 --evaluate > ~/evaluate/19.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=6 --evaluate > ~/evaluate/20.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=7 --evaluate > ~/evaluate/21.txt 2>&1 &

nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/22.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/23.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5--evaluate > ~/evaluate/24.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/25.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --evaluate > ~/evaluate/26.txt 2>&1 &







nohup python3 -u main_ts.py --config=configs/dense_ts_smd_anomaly.yaml --gpu=6 --evaluate > ~/evaluate/dense_7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_msl_anomaly.yaml --gpu=7  --evaluate > ~/evaluate/dense_8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_smap_anomaly.yaml --gpu=0 --evaluate > ~/evaluate/dense_9.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/dense_ts_ettm1_forecast.yaml --gpu=5 --evaluate > ~/evaluate/dense_19.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=6 --evaluate > ~/evaluate/dense_20.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=7 --evaluate > ~/evaluate/dense_21.txt 2>&1 &

nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=2  --evaluate > ~/evaluate/dense_22.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=3  --evaluate > ~/evaluate/dense_23.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=4 --evaluate > ~/evaluate/dense_24.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=5 --evaluate > ~/evaluate/dense_25.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_japan.yaml --gpu=6 --evaluate > ~/evaluate/dense_26.txt 2>&1 &
