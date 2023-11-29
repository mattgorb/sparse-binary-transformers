


#forecasting steps=1
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/33.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/34.txt 2>&1 &

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/31.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/30.txt 2>&1 & #adam


nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=25 --rerand_rate=0.1  > ~/results_forecasting/32.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/29.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=3  --rerand_rate=0.1 > ~/results_forecasting/33.txt 2>&1 & #adam



#forecasting steps=2
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/35.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/36.txt 2>&1 &

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/37.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/38.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/39.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 --rerand_epoch_freq=3  --rerand_rate=0.1 > ~/results_forecasting/40.txt 2>&1 & #adam



#forecasting steps=4
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/41.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/42.txt 2>&1 &

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/43.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/44.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/45.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=3  --rerand_rate=0.1 > ~/results_forecasting/46.txt 2>&1 & #adam
