# Gia

Gia - Unethically making money from AI analysis on time series data 🤑. The purpose of this repo is to establish a
interatively built model. Adding new parameters and fine tuning

## Getting Started

This repo is meant to be run in a AWS SageMaker environment. There is nothing really special besides the ease of 
use and scalability, so an environment that has python 3.6 and jupyter installed should suffice.

### Repo Structure
The `./notebooks/` directory contains the notebooks used to test new parameters and algorithms. 
The `./gia_forecast/` directory contains runs of training. SafeMaker's Forecast feature?

## Input Parameters to O

- ~~Stock price~~ 
- Sentiment value
  - Twitter
  - News articals
  - Reddit
- Google Trends
- Weather?
- Recessions indicator
- Exponential Moving Average
- Time of day
- Day of week
- Week of year
- Percent of presidential cycle
- Ask/bid spread
- Sector based performance

## TODO: Calculate model's confidence
This can be based of spread of precentiles, RMSE, and other model output.

## TODO: Calculate risk factor feature
Here an algorithm can determine the level of risk associated with a trade. I.e. 

Then another algorithm will sort 
and identify the most favorable trades based on risk vs reward.

## Testing

A few datasets are provided to test implemented models.

## Other notes

### Metrics to care about

For instance, the lower the Root-Mean-Squared Error (RMSE) the better - a value of 0 would indicate a perfect fit to the
data. But RMSE is dependent on the scale of the data being used. Dividing the RMSE by the range of the data, gives an 
average error as a proportion of the data's scale. This is called the Normalized Root-Mean-Squared Error (NRMSE). 
However, the RMSE and NRMSE are very sensitive to outliers.

https://github.com/aws-samples/amazon-sagemaker-time-series-prediction-using-gluonts/blob/master/notebooks/part3/twitter_volume_forecast.ipynb

