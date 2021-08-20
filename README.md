# Algorithmic Trading
---

This repository contains implementation of Algorithm Trading.  

Following are the files: 
* **algotrading_screener.ipynb** - This script scans entire universe of S&P500 and NASDAQ stocks. Computes momentum score, value score and social sentiment scores and finally computes composite scores from the above mentioned scores. Finally picks top n stocks based on the composite score. 
* **algotrading_classifier.ipynb** - This script evaluates 6 classifiers and select the best classifier that predicts future price actions such as short sell or long. You will need alpaca API keys to extract past price action data.
* **techanalysislib.py** - Library of custom built technical indicators which are used for feature engineering


## How to run the notebooks <br>
Clone the entire repository - "algo-trading"  into a local folder by issuing the following command from gitbash <br>
```
$git clone https://github.com/Roy-Tapas/algo-trading.git
```
Stay in the same gitbash directory and open Jupyter lab by issuing the following command from gitbash: <br>
```
$jupyter lab
```


<hr style="border:2px solid blue"> </hr>

## Tapas Roy

**Email:** rtapask@gmail.com# lstm-price-predictor