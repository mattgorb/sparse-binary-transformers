nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=0 --lin_prune_rate=0.9 --attention_prune_rate=0.9> ~/results_forecasting/7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=1 --lin_prune_rate=0.9 --attention_prune_rate=0.9> ~/results_forecasting/8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=2 --lin_prune_rate=0.9 --attention_prune_rate=0.9> ~/results_forecasting/9.txt 2>&1 &



nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=4 --lin_prune_rate=0.25 --attention_prune_rate=0.25   > ~/results_forecasting/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=5 --lin_prune_rate=0.25 --attention_prune_rate=0.25  --dmodel=32> ~/results_forecasting/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=4 --lin_prune_rate=0.25 --attention_prune_rate=0.25  --dmodel=32> ~/results_forecasting/6.txt 2>&1 &


nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=5 > ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=6 > ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=7 > ~/results_forecasting/3.txt 2>&1 &



nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=5 --dmodel=32 > ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=6  > ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=7 --dmodel=256 --window_size=300> ~/results_forecasting/2_new.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu= --dmodel=32 > ~/results_forecasting/3.txt 2>&1 &












nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=0 --ablation=True > ~/results_forecasting/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=2 --ablation=True > ~/results_forecasting/2.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=0 --ablation=True > ~/results_forecasting/3.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=5 > ~/results_forecasting/4.txt 2>&1 &

#no mask
nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=3 --ablation=True > ~/results_forecasting/5.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=3 --ablation=True > ~/results_forecasting/6.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=4 --ablation=True > ~/results_forecasting/7.txt 2>&1 &

nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=2 --ablation=True > ~/results_forecasting/8.txt 2>&1 &