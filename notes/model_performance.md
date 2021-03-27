# Model Performance

Any major version change should be accompanied by a hyperparameter tuning job to give us insight for the best parameters
 for that model's architecture.

| Version | Commit # | Accuracy (loss) | dropout_rate | epochs | num_layers | prediction_length | link |
|---|---|---|---|---|---|---|---|
| 0.3.3 | ab77d3e33f2641fd4dc09203da97402069dd4068 | 9867.0594 | 0.02688249593944054 | 6 | 4 | 13 | https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs/mxnet-training-200414-0823?region=us-east-1&tab=bestTrainingJob |
| 0.5.3 | 8a54ad378ba48a98638870ff9181f02cb7c69db4 | 56489.1328 | 0.1844849452340674 | 30 | 6 | 15 | https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/hyper-tuning-jobs/mxnet-training-210326-1446?tab=bestTrainingJob |
