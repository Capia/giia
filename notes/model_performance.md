# Model Performance

Any major version change should be accompanied by a hyperparameter tuning job to give us insight for the best parameters
 for that model's architecture.

- MAPE: Mean Absolute Percentage Error. Unit is percentage, which makes it easy to compare across different models.
- RMSE: Root Mean Squared Error. Unit is the same as the target variable (value of USD/ETH), which helps set 
expectations of the models's accuracy.

| Version | Commit #                                 | MAPE              | RMSE               | # of Parameters | Training set size | Testing set size | epochs | link                                                                                                                  |
|---------|------------------------------------------|-------------------|--------------------|-----------------|-------------------|------------------|--------|-----------------------------------------------------------------------------------------------------------------------|
| 1.0.3   | 5662f487039ccf7aef35f9b560d293f3d8ea99f9 | 2.503210258483887 | 1.5821536772652292 | 3419139         | 1142136           | 489488           | 20     | https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/mxnet-training-2023-02-11-14-31-11-050 |

```

#015  0%|          | 0/100 [00:00<?, ?it/s]INFO:gluonts.trainer:Number of parameters in TransformerTrainingNetwork: 33143
#015  1%|          | 1/100 [01:35<2:36:53, 95.08s/it, epoch=1/5, avg_epoch_loss=8.26]
#015  2%|▏         | 2/100 [02:46<2:12:19, 81.01s/it, epoch=1/5, avg_epoch_loss=8.54]
#015  3%|▎         | 3/100 [04:22<2:21:52, 87.75s/it, epoch=1/5, avg_epoch_loss=8.49]
#015  4%|▍         | 4/100 [06:05<2:30:06, 93.82s/it, epoch=1/5, avg_epoch_loss=8.38]
#015  5%|▌         | 5/100 [07:39<2:28:44, 93.95s/it, epoch=1/5, avg_epoch_loss=8.35]
#015  6%|▌         | 6/100 [09:02<2:21:31, 90.33s/it, epoch=1/5, avg_epoch_loss=8.38]
#015  7%|▋         | 7/100 [10:26<2:16:54, 88.33s/it, epoch=1/5, avg_epoch_loss=8.3] 
#015  8%|▊         | 8/100 [11:43<2:09:31, 84.47s/it, epoch=1/5, avg_epoch_loss=8.25]
#015  9%|▉         | 9/100 [13:10<2:09:35, 85.44s/it, epoch=1/5, avg_epoch_loss=8.21]
#015 10%|█         | 10/100 [14:42<2:11:04, 87.39s/it, epoch=1/5, avg_epoch_loss=8.18]
#015 11%|█         | 11/100 [15:49<2:00:28, 81.22s/it, epoch=1/5, avg_epoch_loss=8.16]
#015 12%|█▏        | 12/100 [17:21<2:04:02, 84.57s/it, epoch=1/5, avg_epoch_loss=8.12]
#015 13%|█▎        | 13/100 [18:57<2:07:43, 88.08s/it, epoch=1/5, avg_epoch_loss=8.09]
#015 14%|█▍        | 14/100 [20:12<2:00:25, 84.02s/it, epoch=1/5, avg_epoch_loss=8.06]
#015 15%|█▌        | 15/100 [21:21<1:52:43, 79.57s/it, epoch=1/5, avg_epoch_loss=8.05]
#015 16%|█▌        | 16/100 [22:51<1:55:25, 82.45s/it, epoch=1/5, avg_epoch_loss=8.03]
#015 17%|█▋        | 17/100 [24:06<1:51:14, 80.42s/it, epoch=1/5, avg_epoch_loss=8.01]
#015 18%|█▊        | 18/100 [25:35<1:53:23, 82.97s/it, epoch=1/5, avg_epoch_loss=7.99]
#015 19%|█▉        | 19/100 [27:12<1:57:42, 87.19s/it, epoch=1/5, avg_epoch_loss=7.96]
#015 20%|██        | 20/100 [28:27<1:51:12, 83.40s/it, epoch=1/5, avg_epoch_loss=7.94]
#015 21%|██        | 21/100 [29:48<1:48:50, 82.67s/it, epoch=1/5, avg_epoch_loss=7.91]
#015 22%|██▏       | 22/100 [31:24<1:52:57, 86.89s/it, epoch=1/5, avg_epoch_loss=7.9] 
#015 23%|██▎       | 23/100 [32:32<1:44:14, 81.23s/it, epoch=1/5, avg_epoch_loss=7.89]
#015 24%|██▍       | 24/100 [33:59<1:44:47, 82.72s/it, epoch=1/5, avg_epoch_loss=7.86]
```
