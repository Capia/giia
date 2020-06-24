# Model Performance

Any major version change should be accompanied by a hyperparameter tuning job to give us insight for the best parameters
 for that model's architecture.

| Version | Commit # | Accuracy (loss) | dropout_rate | epochs | num_layers | prediction_length | link |
|---|---|---|---|---|---|---|---|
| 0.3.3 | ab77d3e33f2641fd4dc09203da97402069dd4068 | 9867.0594 | 0.02688249593944054 | 6 | 4 | 13 | https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs/mxnet-training-200414-0823?region=us-east-1&tab=bestTrainingJob |
