# Data

## Where it comes from
This is the "Default of Credit Card Clients" data from the UCI Machine Learning Repository.
It has 30,000 credit card customers in Taiwan, from 2005.
For each person it has their credit limit, some details about them,
and their bills and payments for 6 months.
What I want to predict: did they miss their payment the next month?

Page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Can I use it?
Yes. The license is CC BY 4.0. I can use it for anything,
but I have to say where it came from.

## How to cite it
Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

## How to get it
The data is not saved in Git. To download it, run this in the project folder:

    uv run python scripts/download_data.py

It downloads a zip file from UCI and unzips it into `data/raw/`.
If the files are already there, it does nothing.

## What is in data/raw
- `credit_default.zip`: the zip file from UCI.
- `default of credit card clients.xls`: the Excel file from inside the zip. I never change this file.