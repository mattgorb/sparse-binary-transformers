nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=32> ~/results_classification/100.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=1 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=128> ~/results_classification/101.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=1 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=256> ~/results_classification/102.txt 2>&1 &



nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=2   --dmodel=16> ~/results_classification_dense/4.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=32 --epochs=50> ~/results_classification/103.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=64 --epochs=50> ~/results_classification/104.txt 2>&1 &








nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=64 --evaluate > ~/evaluate/1_insect_new.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=400 --evaluate > ~/evaluate/3_insect_new.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=64 --evaluate > ~/evaluate/4_insect_new.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=400 --evaluate > ~/evaluate/6_insect_new.txt 2>&1 &