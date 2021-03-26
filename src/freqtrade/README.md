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

Once FreqTrade is configured, download some data:
```
# it is assumed you are using binance
docker-compose run --rm freqtrade download-data -t 5m --timerange 20170101- -v
```

Some frequently used commands are:
```

```
