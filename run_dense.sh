

#nohup python3 -u main_ts.py --config=configs/dense_ts_ettm1_forecast.yaml --gpu=0 --dmodel=16 > ~/results_forecasting_dense/1.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=1  --dmodel=16 > ~/results_forecasting_dense/2.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=2  --dmodel=16 > ~/results_forecasting_dense/3.txt 2>&1 &

#nohup python3 -u main_ts.py --config=configs/dense_ts_ettm1_forecast.yaml --gpu=3 --dmodel=32 > ~/results_forecasting_dense/4.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=4  --dmodel=32 > ~/results_forecasting_dense/5.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=5  --dmodel=32 > ~/results_forecasting_dense/6.txt 2>&1 &


#nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=6  --dmodel=64 > ~/results_forecasting_dense/7.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=7  --dmodel=64 > ~/results_forecasting_dense/8.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=0  --dmodel=128 > ~/results_forecasting_dense/9.txt 2>&1 &
nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=1  --dmodel=256 > ~/results_forecasting_dense/10.txt 2>&1 &

nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=128 --evaluate > ~/evaluate/25.txt 2>&1 &


#nohup python3 -u main_ts.py --config=configs/dense_ts_ettm1_forecast.yaml --gpu=3 --dmodel=128 > ~/results_forecasting_dense/11.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=4  --dmodel=400 > ~/results_forecasting_dense/12.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_elect_forecast.yaml --gpu=5  --dmodel=450 > ~/results_forecasting_dense/13.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=6  --dmodel=128 > ~/results_forecasting_dense/14.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/dense_ts_wth_forecast.yaml --gpu=7  --dmodel=256 > ~/results_forecasting_dense/15.txt 2>&1 &



#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=2   --dmodel=8> ~/results_classification_dense/1.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=3   --dmodel=16> ~/results_classification_dense/2.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=4   --dmodel=32> ~/results_classification_dense/3.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_arabic.yaml --gpu=5   --dmodel=128> ~/results_classification_dense/20.txt 2>&1 &


These are done
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=5   --dmodel=16> ~/results_classification_dense/4.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=6   --dmodel=64> ~/results_classification_dense/5.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=0   --dmodel=32> ~/results_classification_dense/6.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_fd.yaml --gpu=0   --dmodel=128> ~/results_classification_dense/6_new.txt 2>&1 &

#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_japan.yaml --gpu=6   --dmodel=8> ~/results_classification_dense/25.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_japan.yaml --gpu=7  --dmodel=16> ~/results_classification_dense/26.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_japan.yaml --gpu=0  --dmodel=64> ~/results_classification_dense/27.txt 2>&1 &



#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=1   --epochs=64 --dmodel=8 --epochs=50> ~/results_classification_dense/7.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=2   --epochs=64 --dmodel=16 --epochs=50> ~/results_classification_dense/8.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=3   --epochs=64 --dmodel=32 --epochs=50> ~/results_classification_dense/9.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=4   --epochs=64 --dmodel=64 --epochs=50> ~/results_classification_dense/21.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_hb.yaml --gpu=5   --epochs=64 --dmodel=128 --epochs=50> ~/results_classification_dense/22.txt 2>&1 &










#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=4   --dmodel=64> ~/results_classification_dense/10.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=7   --dmodel=128> ~/results_classification_dense/11.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=7   --dmodel=32> ~/results_classification_dense/12.txt 2>&1 &



#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=0   --dmodel=200> ~/results_classification_dense/23.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/dense_ts_classification_insect.yaml --gpu=1   --dmodel=400> ~/results_classification_dense/24.txt 2>&1 &






all sparse runs 1/30


ecl=350
weather=100
ett 64

#nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=0 --dmodel=16 > ~/results_forecasting/1.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1  --dmodel=16 > ~/results_forecasting/2.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=2  --dmodel=16 > ~/results_forecasting/3.txt 2>&1 &

#nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=3 --dmodel=32 > ~/results_forecasting/4.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=4  --dmodel=32 > ~/results_forecasting/5.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=5  --dmodel=32 > ~/results_forecasting/6.txt 2>&1 &


#nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=6  --dmodel=64 > ~/results_forecasting/7.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=7  --dmodel=64 > ~/results_forecasting/8.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=0  --dmodel=128 > ~/results_forecasting/9.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=1  --dmodel=256 > ~/results_forecasting/10.txt 2>&1 &



These are running, along with all dense
#nohup python3 -u main_ts.py --config=configs/sparse_ts_ettm1_forecast.yaml --gpu=3 --dmodel=128 > ~/results_forecasting/11.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=4  --dmodel=400 > ~/results_forecasting/12.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_elect_forecast.yaml --gpu=5  --dmodel=450 > ~/results_forecasting/13.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=16> ~/results_classification/2.txt 2>&1 &

#nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=6  --dmodel=128 > ~/results_forecasting/14.txt 2>&1 &
#nohup python3 -u main_ts.py --config=configs/sparse_ts_wth_forecast.yaml --gpu=7  --dmodel=256 > ~/results_forecasting/15.txt 2>&1 &




arabic 64
fd 128
hb 64
insect 256
japan 32

#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=8> ~/results_classification/1.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=16> ~/results_classification/2.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=32> ~/results_classification/3.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_arabic.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=128> ~/results_classification/1.txt 2>&1 &

#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=16> ~/results_classification/4.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=64> ~/results_classification/5.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=32> ~/results_classification/6.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_fd.yaml --gpu=0 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=128> ~/results_classification/6_new.txt 2>&1 &


#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=1 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=8 --epochs=50> ~/results_classification/7.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=16 --epochs=50> ~/results_classification/8.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=32 --epochs=50> ~/results_classification/9.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=2 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=64 --epochs=50> ~/results_classification/9.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_hb.yaml --gpu=3 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --epochs=64 --dmodel=128 --epochs=50> ~/results_classification/7.txt 2>&1 &


#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=4 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=64> ~/results_classification/10.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=7 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=128> ~/results_classification/11.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=7 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=32> ~/results_classification/12.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=5 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=200> ~/results_classification/12.txt 2>&1 &
nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_insect.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=400> ~/results_classification/11.txt 2>&1 &


#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=8> ~/results_classification/13.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=16> ~/results_classification/14.txt 2>&1 &
#nohup python3 -u main_ts_classification.py --config=configs/sparse_ts_classification_japan.yaml --gpu=6 --lin_prune_rate=0.5 --attention_prune_rate=0.5 --dmodel=64> ~/results_classification/14.txt 2>&1 &
