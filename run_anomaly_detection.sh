nohup python3 -u main_ts.py --config=configs/dense_ts_msl_anomaly.yaml --gpu=0 > ~/results_anomaly/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_smd_anomaly.yaml --gpu=1 > ~/results_anomaly/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_smap_anomaly.yaml --gpu=2 > ~/results_anomaly/3.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=3 > ~/results_anomaly/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=4 > ~/results_anomaly/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=5 > ~/results_anomaly/6.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=6 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_anomaly/7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=7 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_anomaly/8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=0 --lin_prune_rate=0.25 --attention_prune_rate=0.25> ~/results_anomaly/9.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=1 --lin_prune_rate=0.1 --attention_prune_rate=0.1> ~/results_anomaly/10.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=2 --lin_prune_rate=0.1 --attention_prune_rate=0.1> ~/results_anomaly/11.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=3 --lin_prune_rate=0.1 --attention_prune_rate=0.1> ~/results_anomaly/12.txt 2>&1 &





nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=2   > ~/results_anomaly/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=0   > ~/results_anomaly/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=0  > ~/results_anomaly/3.txt 2>&1 &


nohup python3 -u main_ts.py --config=configs/sparse_ts_smd_anomaly.yaml --gpu=5  --ablation=True > ~/results_anomaly/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_msl_anomaly.yaml --gpu=3  --ablation=True > ~/results_anomaly/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_smap_anomaly.yaml --gpu=5  --ablation=True > ~/results_anomaly/6.txt 2>&1 &


nohup python3 -u main_ts.py --config=configs/dense_ts_smd_anomaly.yaml --gpu=5   > ~/results_anomaly/7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_msl_anomaly.yaml --gpu=5   > ~/results_anomaly/8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_smap_anomaly.yaml --gpu=5  > ~/results_anomaly/9.txt 2>&1 &