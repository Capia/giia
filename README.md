# Giia
Giia is a backward acronym that stands for:
- A: Artificial
- I: Intelligence
-    for
- I: Investment
- G: Growth

Giia is a family of financial services owned by Capia Inc. Its goal is to provide predictions for long, medium, and 
short-term outlooks for any given cyrpto. 

This encompasses using MXNet + GluonTS to create an AI model based on time series data. This repo leverages AWS 
SageMaker for its robust API and scalability.

## Model Architecture
Currently the model is using DeepAR. However, other models listed below should be tested for their effectiveness. 
LSTNet is likely the most promising.

- **DeepAR** is a supervised learning algorithm for forecasting scalar time series using recurrent neural networks 
  (RNN)
- **SFeedFwd** (Simple Feedforward) is a supervised learning algorithm where information moves in only one 
  direction—forward—from the input nodes, through the hidden nodes (if any) and to the output nodes in the forward 
  direction
- **LSTNet** (Long- and Short-term Time-series network) is a multivariate time series forecasting model that uses the 
  combination of Convolution Neural Network (CNN) and the Recurrent Neural Network (RNN) to find short-term local 
  dependency patterns among variables and them find long-term patterns for time series trends
- **Seq2Seq** (Sequence-to-sequence learning) is a method to train models to convert sequences from one domain to 
  sequences in another domain

## Getting Started
First, set up a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

Then install the dependencies with:
```
# Note that there are a few dependencies commented out inside the requirements.txt file. You need to install those 
#  manually
pip install -r ./src/requirements.txt
```

Finally, set the jupyter notebook to use the virtual environment:
```
python -m ipykernel install --user --name=capia --display-name="Capia (venv)"
```

### Datasets
To download training and test datasets, read `./src/freqtrade/README.md`

### Running in AWS
This repo makes use of AWS Sagemaker's SDK which allows you to test and iterate quickly on your dev machine, before 
running a training session. To enable local execution change the `train_instance_type` of your estimator to `local`. 
Running locally requires Docker, make sure that it is installed and running.

### Metrics to Care About
For instance, the lower the Root-Mean-Squared Error (RMSE) the better - a value of 0 would indicate a perfect fit to the
data. But RMSE is dependent on the scale of the data being used. Dividing the RMSE by the range of the data, gives an
average error as a proportion of the data's scale. This is called the Normalized Root-Mean-Squared Error (NRMSE).
However, the RMSE and NRMSE are very sensitive to outliers.

### Developer Workflow
Lastly, jupyter cell output is distracting when looking at diffs and MRs. To remove this, we use a tool called 
`nbstripout` and git filters. I recommend installing it globally with `nbstripout --install --global`. Which ever way 
you choose, ensure you check it is installed with `nbstripout --status`

## Docker

### Build Image
First, acquire the serialized model `model.tar.gz` and note the location. This can be found locally if you ran the 
training job locally, or it can be found in S3 if it was ran in AWS SageMaker. Additionally, there are notebooks to 
help download the serialized model.

Then, to build the image run:
```
export IMAGE_TAG="0.0.9"
export MODEL_PATH="./out/giia-1.0.3/models/mxnet-training-2021-06-23-12-36-55-617/output/model.tar.gz"
./scripts/build_pdi_image.sh ${IMAGE_TAG} ${MODEL_PATH}
./scripts/build_ft_image.sh ${IMAGE_TAG}
```

Finally, to push the image:
```
export IMAGE_TAG="0.0.9"
./scripts/push_pdi_image.sh ${IMAGE_TAG}
./scripts/push_ft_image.sh ${IMAGE_TAG}
```

### Test Locally
In many cases, you will want to test your image locally before deploying. You can do this manually by first building
the image from above, then running:
```
docker run -p 9000:8080 --rm -it ${IMAGE_NAME}
```
This will make the function available at `http://localhost:9000/2015-03-31/functions/function/invocations`. Don't ask 
me where the "2015-03-31" comes from, it was in the docs and it works.

## Backlog
### TODO
- Rolling time series (unreleased version of gluonts)
- Feature time series (multivariate?)
- Visualize model (tensorboard like)
- Visualize model's performance (tensorboard like)
- Calculate model's confidence
  - This can be based of spread of precentiles, RMSE, and other model output.
- Calculate risk factor feature
  - Here an algorithm can determine the level of risk associated with a trade. Then another algorithm will sort
and identify the most favorable trades based on risk vs reward.

### Other Ideas for Input Parameters
- tick vs time based candles https://towardsdatascience.com/advanced-candlesticks-for-machine-learning-i-tick-bars-a8b93728b4c5
- Google Trends
- Foreign markets
- Futures
- Weather?
- Recessions indicator
- Exponential Moving Average
- Percent of presidential cycle
- Ask/bid spread
- Sector based performance
- DCA
  - Allow/prefer investing as a DCA function
  - Only if short term is negative, though long term is positive. This helps ease our way into a position without try to time it perfectly
  - Bucket Annealing
    - Each bucket is 100 shares
    - Buy 50 when max price to pay is hit
- Inflation rates
  - https://www.usinflationcalculator.com/inflation/current-inflation-rates/
- Go through past 5 years of spikes and determine what caused them. Then determine a way to evaluate it

## Other Notes
### References
https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-now-supports-random-search-and-hyperparameter-scaling/
https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html
https://github.com/aws-samples/amazon-sagemaker-time-series-prediction-using-gluonts/blob/master/notebooks/part3/twitter_volume_forecast.ipynb
https://aws.amazon.com/blogs/machine-learning/creating-neural-time-series-models-with-gluon-time-series/
https://aws.amazon.com/blogs/industries/novartis-ag-uses-amazon-sagemaker-and-gluonts-for-demand-forecasting/
https://aws.amazon.com/blogs/machine-learning/training-debugging-and-running-time-series-forecasting-models-with-the-gluonts-toolkit-on-amazon-sagemaker/
https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
https://github.com/awslabs/sagemaker-deep-demand-forecast
https://www.freqtrade.io/en/latest/configuration/
https://towardsdatascience.com/aws-sagemaker-endpoint-as-rest-service-with-api-gateway-48c36a3cf46c

### Market Viability
While this will largely be used by the founders of Capia, we believe we can market and sell its predictions to other 
users. Below are a few ideas to make Giia profitiable:
1) Sell tokens to a limited number of people (this number should stay below 50). Here are a few different pricing models
  - Cost per request, which is uniquely identified by the token. This works well because it is reoccuring and is 
  proportional to the usage and cost to run the infrastructure. We should round up this number and market it as buckets
  - One time cost for token. This is not viable as there is a large upfront cost that will disaude customers. Also, 
  since it is not reoccurring, it does not help support the infrastructure needed to provide the predictions
2) Set up a chat room and channels whose access requires a $79(?) per month per user. This is a proven model, see
boilingroomtrading https://boilerroomtrading.co, though we can uniquely market this a very effective AI based solution.
Marketing material should pit it against other AI model stock predictors
