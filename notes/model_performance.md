# Model Performance

Any major version change should be accompanied by a hyperparameter tuning job to give us insight for the best parameters
 for that model's architecture.

- MAPE: Mean Absolute Percentage Error. Unit is percentage, which makes it easy to compare across different models.
- RMSE: Root Mean Squared Error. Unit is the same as the target variable (value of USD/ETH), which helps set 
expectations of the models's accuracy.

| Version | Commit #                                 | MAPE              | RMSE               | # of Parameters | Training set size | Testing set size | epochs | link                                                                                                                  |
|---------|------------------------------------------|-------------------|--------------------|-----------------|-------------------|------------------|--------|-----------------------------------------------------------------------------------------------------------------------|
| 1.0.3   | 5662f487039ccf7aef35f9b560d293f3d8ea99f9 | 2.503210258483887 | 1.5821536772652292 | 3419139         | 1142136           | 489488           | 20     | https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/mxnet-training-2023-02-11-14-31-11-050 |
