


nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5> ~/results_classification/1.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5> ~/results_classification/2.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=150> ~/results_classification/3.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=150> ~/results_classification/4.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_lsst.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=150> ~/results_classification/5.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_japan.yaml --gpu=6 > ~/results_classification/16.txt 2>&1 &


nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=7 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/6.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=0 --lin_prune_rate=0.75 --attention_prune_rate=0.75> ~/results_classification/7.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=1 --lin_prune_rate=0.75 --attention_prune_rate=0.75 --epochs=150> ~/results_classification/8.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=2 --lin_prune_rate=0.75 --attention_prune_rate=0.75 --epochs=150> ~/results_classification/9.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=3 --lin_prune_rate=0.75 --attention_prune_rate=0.75 --epochs=150> ~/results_classification/10.txt 2>&1 &

nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=4 --lin_prune_rate=0.9 --attention_prune_rate=0.9> ~/results_classification/11.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=5 --lin_prune_rate=0.9 --attention_prune_rate=0.9> ~/results_classification/12.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=6 --lin_prune_rate=0.9 --attention_prune_rate=0.9 --epochs=150> ~/results_classification/13.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=7 --lin_prune_rate=0.9 --attention_prune_rate=0.9 --epochs=150> ~/results_classification/14.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=0 --lin_prune_rate=0.9 --attention_prune_rate=0.9 --epochs=150> ~/results_classification/15.txt 2>&1 &