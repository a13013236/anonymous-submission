# Anonymous Code for Double-Blind Review
This repository contains the implementation of our research work "DYCOR: Dynamic Correlation for Stock Trend Prediction" currently under double-blind peer review.

## Components
Stock Encoding
Dynamic Stock Clustering
Intra-Stock Correlation
Stock-wise Inter-cluster Aggregation
Correlation-Aware Training

## Code Structure

dycor.py: Main model implementation integrating all components
encoding.py: Stock encoding module for processing historical features
clustering.py: Dynamic PCA-based clustering of stocks into market segments
intra_corr.py: Attention mechanism for modeling relationships within segments
aggregation.py: Combining representations from different market perspectives
loss.py: Implementation of correlation-aware and regression loss functions
main.py: Entry point for training and evaluation
train.py: Training loop with early stopping and model evaluation

## Datasets
We evaluate our model on three benchmark datasets:

NASDAQ: 1,026 stocks (Jan 2013 - Dec 2017)
NYSE: 1,737 stocks (Jan 2013 - Dec 2017)
S&P 500: 646 stocks (Jan 2003 - Dec 2023)

The NASDAQ and NYSE datasets are obtained from Temporal Relational Stock Ranking.

## Usage
bashpython main.py --market NASDAQ --gpu 0
Options:

--market: Market to train on (choices: NASDAQ, NYSE, SP500)
--seed: Random seed for reproducibility
--gpu: GPU device ID to use
