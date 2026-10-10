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

## Odd codes I found and what I did about them

Going by the dataset page, EDUCATION should only be 1–4 and MARRIAGE only 1–3.
When I counted the values I also found EDUCATION 0, 5 and 6 (345 customers)
and MARRIAGE 0 (54 customers). Nothing explains what these codes mean.

I merged them into the "others" group that already exists: EDUCATION 0, 5 and 6
become 4, and MARRIAGE 0 becomes 3. I didn't want to delete these customers, and
some of the groups are tiny (only 14 people have EDUCATION 0), so the model
couldn't learn much from them on their own. The catch is that I'm assuming these
unknown codes behave like "others", and I can't actually check that.

The raw file stays as it is. The merge happens in code when I prepare the data.

## The PAY_* status codes

The PAY_* columns are repayment status codes, not money. PAY_0 is September 2005,
PAY_2 is August, and so on back to PAY_6 in April. There's no PAY_1, the names
just skip from PAY_0 to PAY_2.

The dataset page says -1 means paid on time and 1 to 9 mean that many months late.
The data also has -2 and 0, which the page doesn't explain. The meaning most people
use (from discussions about the dataset, not the official source) is: -2 means no
credit used that month, and 0 means paid the minimum and carried the rest over. So
-2, -1 and 0 all mean "not late". Nobody has code 9, and PAY_5 and PAY_6 have no 1
at all, which I can't explain.

I kept these columns as numbers instead of one-hot encoding them, because the order
matters (higher means a later payment) and one-hot would have given me about 64 extra
columns, lots of them tiny. The catch is that the model treats every step as the same
size, so going from 0 to 1 counts the same as going from 7 to 8, which probably isn't true.