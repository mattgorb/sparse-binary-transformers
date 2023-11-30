


#forecasting steps=1
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/1.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/2.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/3.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/4.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/5.txt 2>&1 & #adam


#forecasting steps=2
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/6.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/7.txt 2>&1 & #adam




#forecasting steps=4
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/8.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/9.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/10.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/11.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/12.txt 2>&1 & #adam






nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100  --model_runs=5 > ~/results_forecasting/13.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100 --model_runs=5 > ~/results_forecasting/14.txt 2>&1 & 


nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100  --model_runs=5 > ~/results_forecasting/15.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100 --model_runs=5 > ~/results_forecasting/16.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100  --model_runs=5 > ~/results_forecasting/17.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100 --model_runs=5 > ~/results_forecasting/18.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100  --model_runs=5 > ~/results_forecasting/19.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100 --model_runs=5 > ~/results_forecasting/20.txt 2>&1 & 


