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

## Getting Started
First, set up a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

Then install the dependencies with:
```
pip install -r ./src/requirements.txt
```
and
```
# Install this separately from the requirements file, which the reasoning is explained within it. You will need to have 
#  the TA-Lib library installed on your machine 
pip install freqtrade
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

## Backlog
### TODO
- Rolling time series (unreleased version of gluonts)
- Visualize model (tensorboard like)
- Visualize model's performance (tensorboard like)
- Calculate model's confidence
  - This can be based of spread of precentiles, RMSE, and other model output.
- Calculate risk factor feature
  - Here an algorithm can determine the level of risk associated with a trade. Then another algorithm will sort
and identify the most favorable trades based on risk vs reward.

### Other Ideas for Input Parameters
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
https://www.freqtrade.io/en/latest/configuration/

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
