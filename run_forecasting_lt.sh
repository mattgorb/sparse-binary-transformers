

#ECL
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

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100  --model_runs=5 >  ~/results_forecasting/19.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100 --model_runs=1 >  ~/results_forecasting/20.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100  --model_runs=5 >  ~/results_forecasting/21.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100 --model_runs=5 >  ~/results_forecasting/22.txt 2>&1 & 


#window_size=200+forecasting_steps


























#ettm1

#forecasting steps=1
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/21.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/22.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/23.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/24.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/25.txt 2>&1 & #adam


#forecasting steps=2
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/26.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/27.txt 2>&1 & #adam




#forecasting steps=4
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/28.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/29.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/30.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/31.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/32.txt 2>&1 & #adam


















#START HERE TOMORROW
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100  --model_runs=5 > ~/results_forecasting/33.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100 --model_runs=5 > ~/results_forecasting/34.txt 2>&1 & 


nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100  --model_runs=5 > ~/results_forecasting/35.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100 --model_runs=5 > ~/results_forecasting/36.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100  --model_runs=5 > ~/results_forecasting/37.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100 --model_runs=5 > ~/results_forecasting/38.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100  --model_runs=5 > ~/results_forecasting/39.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100 --model_runs=5 > ~/results_forecasting/40.txt 2>&1 & 



#WTH

#forecasting steps=1
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/41.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 > ~/results_forecasting/42.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/43.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/44.txt 2>&1 & #adam

nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=200 --forecasting_steps=1 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/45.txt 2>&1 & #adam


#forecasting steps=2
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/46.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=202 --forecasting_steps=2 --epochs=100 > ~/results_forecasting/47.txt 2>&1 & #adam




#forecasting steps=4
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/48.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 > ~/results_forecasting/49.txt 2>&1 & #adam
















nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=5  --rerand_rate=0.1 > ~/results_forecasting/50.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=10 --rerand_rate=0.1  > ~/results_forecasting/51.txt 2>&1 & #adam
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=204 --forecasting_steps=4 --epochs=100 --rerand_epoch_freq=20 --rerand_rate=0.1 > ~/results_forecasting/52.txt 2>&1 & #adam



#START HERE
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100  --model_runs=5 > ~/results_forecasting/53.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=208 --forecasting_steps=8 --epochs=100 --model_runs=5 > ~/results_forecasting/54.txt 2>&1 & 


nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100  --model_runs=5 > ~/results_forecasting/55.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=212 --forecasting_steps=12 --epochs=100 --model_runs=5 > ~/results_forecasting/56.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100  --model_runs=5 > ~/results_forecasting/57.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=216 --forecasting_steps=16 --epochs=100 --model_runs=5 > ~/results_forecasting/58.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100  --model_runs=5 > ~/results_forecasting/59.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100 --model_runs=5 > ~/results_forecasting/60.txt 2>&1 & 



#ECL
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100  --model_runs=5 > ~/results_forecasting/61.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100 --model_runs=5 > ~/results_forecasting/62.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100  --model_runs=5 > ~/results_forecasting/63.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100 --model_runs=5 > ~/results_forecasting/64.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100  --model_runs=5 > ~/results_forecasting/65.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_elect_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100 --model_runs=5 > ~/results_forecasting/66.txt 2>&1 & 

#ETTM1
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100  --model_runs=5 > ~/results_forecasting/67.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100 --model_runs=5 > ~/results_forecasting/68.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100  --model_runs=5 > ~/results_forecasting/69.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100 --model_runs=5 > ~/results_forecasting/70.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100  --model_runs=5 > ~/results_forecasting/71.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_ettm1_t_steps.yaml --gpu=0 --window_size=220 --forecasting_steps=20 --epochs=100 --model_runs=5 > ~/results_forecasting/72.txt 2>&1 & 

#WTH
nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100  --model_runs=5 > ~/results_forecasting/73.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=224 --forecasting_steps=24 --epochs=100 --model_runs=5 > ~/results_forecasting/74.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100  --model_runs=5 > ~/results_forecasting/75.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=228 --forecasting_steps=28 --epochs=100 --model_runs=5 > ~/results_forecasting/76.txt 2>&1 & 

nohup python -u main_ts_forecast_longterm.py --config=configs/dense_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100  --model_runs=5 > ~/results_forecasting/77.txt 2>&1 &
nohup python -u main_ts_forecast_longterm.py --config=configs/sparse_ts_forecast_wth_t_steps.yaml --gpu=0 --window_size=232 --forecasting_steps=32 --epochs=100 --model_runs=5 > ~/results_forecasting/78.txt 2>&1 & 
