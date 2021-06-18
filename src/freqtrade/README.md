# FreqTrade
This repo uses an open source project called [FreqTrade](https://github.com/freqtrade/freqtrade) to:
- download data
  - to train the giia algorithm
  - to backtest the giia algorithm
- backtest and execute dry-runs
- automate buys/sells of cryptocurrencies

## Getting Started
Use this link to get started with FreqTrade https://www.freqtrade.io/en/latest/docker_quickstart/#docker-quick-start. 
It is advised that you read more into the documentation while you are there. You can find some helpful cli options like 
`-vvv`, which prints varying levels of logging (like cURL). 

The first thing you should do is build the image wrapper around the freqtrade image:
```
docker-compose build --no-cache
```

At this point you should have FreqTrade configured the way you like and a local docker image with all the necessary 
dependencies. Now you are ready to download some data:
```
docker-compose run -w /capia/src/freqtrade --rm freqtrade download-data --timerange 20170101- -v
```

## Back testing
There are two ways to run back testing. The first is the most development friendly way as it allows you to iterate 
quickly. Familiarize yourself with the `src/freqtrade/user_data/notebooks/strategy_analysis.ipynb` notebook.
This notebook enables you too easily:
- test different indicators 
- test the process of making a prediction
- visualize the data
- develop your strategy
- back test your strategy
- visualize your strategies performance

The relevant piece of this notebook is being able to generate the indicators and predictions separately from running a 
back test. Thus, you can generate your data _only once_ (which is the most compute intensive), then back test based on 
that data. Now you can test changes to the strategy without waiting for the same data to generate each time.

You can also run back testing through freqtrade's CLI, however, it generates indicators and predictions every time it 
runs. So if you are iterating on the strategy, without make changes to the indicator or model, then this method is 
inefficient as it is repeating most of the heavy work. Nevertheless, here are several common commands one can use to 
back test your strategy:
```
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=20210501- -v

# 1 hour worth of data for back testing
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=1619830800-1619834400 -v

# 24 hours worth of data for back testing
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=1619830800-1619917200 -v

# 1 week worth of data for back testing
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=20210501-20210508 -v

# 2 week bear market after ATH
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=20210514-20210530 -v

# 1 day bear market after ATH
docker-compose run -w /capia/src/freqtrade --rm freqtrade backtesting --export trades --timerange=20210516-20210517 -v
```
