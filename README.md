# DCRNN
A simplified PyTorch implementation of the paper https://arxiv.org/abs/1707.01926 based on [xlwang233/pytorch-DCRNN](https://github.com/xlwang233/pytorch-DCRNN) and [chnsh/DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch)

The original tensorflow implementation: [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)

Welcome to open issues for any problems/questions!

## Requirements

- scipy=1.2.1
- numpy=1.16.2
- pandas=0.24.2
- PyTorch>=1.1.0
- tqdm
- pytable

## Training

Firstly, you need to pre-process the data by using ```generate_dataset.py```:

```bash
# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

 Then execute the training script (default setup on METR-LA):

```bash
python train.py
```

## Testing

Run single-step testing on trained model from horizon 1 to 12:

```bash
python single_step_test.py
```

Run multi-step testing on trained model from horizon 1 to 12:

```bash
python multi_step_test.py
```

## Results

Single-step testing results on 10 epochs:

```
--------single-step testing results--------
Horizon 01, MAE: 2.43, MAPE: 0.0580, RMSE: 4.21
Horizon 02, MAE: 2.83, MAPE: 0.0700, RMSE: 5.26
Horizon 03, MAE: 3.15, MAPE: 0.0800, RMSE: 6.04
Horizon 04, MAE: 3.41, MAPE: 0.0887, RMSE: 6.67
Horizon 05, MAE: 3.67, MAPE: 0.0967, RMSE: 7.23
Horizon 06, MAE: 3.91, MAPE: 0.1044, RMSE: 7.75
Horizon 07, MAE: 4.15, MAPE: 0.1117, RMSE: 8.22
Horizon 08, MAE: 4.37, MAPE: 0.1188, RMSE: 8.66
Horizon 09, MAE: 4.59, MAPE: 0.1255, RMSE: 9.08
Horizon 10, MAE: 4.80, MAPE: 0.1320, RMSE: 9.47
Horizon 11, MAE: 5.00, MAPE: 0.1383, RMSE: 9.83
Horizon 12, MAE: 5.20, MAPE: 0.1446, RMSE: 10.18
```

Multi-step forecasting on 10 epochs:

```
--------multi-step testing results--------
Horizon 01, MAE: 2.43, MAPE: 0.0580, RMSE: 4.21
Horizon 02, MAE: 2.63, MAPE: 0.0640, RMSE: 4.77
Horizon 03, MAE: 2.80, MAPE: 0.0693, RMSE: 5.22
Horizon 04, MAE: 2.96, MAPE: 0.0742, RMSE: 5.62
Horizon 05, MAE: 3.10, MAPE: 0.0787, RMSE: 5.98
Horizon 06, MAE: 3.23, MAPE: 0.0830, RMSE: 6.31
Horizon 07, MAE: 3.36, MAPE: 0.0871, RMSE: 6.61
Horizon 08, MAE: 3.49, MAPE: 0.0910, RMSE: 6.90
Horizon 09, MAE: 3.61, MAPE: 0.0949, RMSE: 7.18
Horizon 10, MAE: 3.73, MAPE: 0.0986, RMSE: 7.44
Horizon 11, MAE: 3.85, MAPE: 0.1022, RMSE: 7.69
Horizon 12, MAE: 3.96, MAPE: 0.1057, RMSE: 7.92
```

