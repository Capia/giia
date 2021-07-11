# FreqTrade

This repo uses an open source project called [FreqTrade](https://github.com/freqtrade/freqtrade) to:

- download data
    - to train the giia algorithm
    - to backtest the giia algorithm
- backtest and execute dry-runs
- automate buys/sells of cryptocurrencies

## Getting Started

### Understanding the Bot and Strategy Relationship

The freqtrade _bot_ executes trades based on the _strategy's_ buy and sell indicators. Each have a diverse set of
controls, both native and custom, that you can use to manipulate trades. These controls are useful for different
scenarios such as backtesting, hyperparameter testing, or running in prod.

The Freqtrade bot can be controlled through telegram or through the web UI. You can do things like start/stop the bot,
check in on performance, and view open trades. You can find out more information in `./user_data/config.json` where they
are configured.

As for the strategy controls, they are found in the strategy's python class itself (`./user_data/strategies/*`). Though
most of the configurations can be overridden in `./user_data/config.json` as well. The strategy has to been configured
at runtime. You can control how it interprets candle stick data, set buy/sell indicators, and return predictions already
generated. The latter, for example, is helpful when the bot is going through many iterations of hyperparameter testing
that use the same indicators, thus allowing us to cache the expensive prediction responses.

### Local Development

Use this link to get started with FreqTrade https://www.freqtrade.io/en/latest/docker_quickstart/#docker-quick-start. It
is advised that you read more into the documentation while you are there. You can find some helpful cli options like
`-vvv`, which prints varying levels of logging (like cURL).

The first thing you should do is build the image wrapper around the freqtrade image:

```
docker-compose build --no-cache
```

NOTE:
Using docker-compose locally allows us to use volumes so a build is not required each time a local file is changed.
However, when running freqtrade and our strategy in AWS, we do not have this luxury. Thus, we copy the source files into
the docker image on build.

At this point you should have FreqTrade configured the way you like and a local docker image with all the necessary
dependencies. Now you are ready to download some data:

```
docker-compose run -w /capia/src/freqtrade --rm freqtrade download-data --timerange 20170101- -v
```

Lastly, you can just start the container in one terminal and connect to it in another container like so:

```
# with docker-compose up (recommended)
docker-compose up
# or with docker-compose run
docker-compose run -p 8080:8080 --rm freqtrade

# then to connect to the container run:
docker exec -it freqtrade /bin/bash
```

## Back testing

There are two ways to run back testing. The first is the most development friendly way as it allows you to iterate
quickly. Familiarize yourself with the `src/freqtrade/user_data/notebooks/strategy_analysis.ipynb` notebook. This
notebook enables you too easily:

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
