# pkt_features

#### Turns a text file into separate website objects with associative features.
- Optionally writes features to a CSV and creates CDF graphs of the features.
- Optionally outputs the features to a text file in a format to be used by machine learning.
- Optionally takes a sampling of the cumulative sums of packet sizes.

```
usage: pkt_features.py [-h] -i TXTFILE [-x X] [-y Y] [-z Z] [-csv] [-cdf] [-ml] [-s S] --zeros {True,False}

options:
  -h, --help            show this help message and exit
  -i TXTFILE            (required) input text file
  -x X                  (optional) Add first X number of total packets as features.
  -y Y                  (optional) Add first Y number of negative packets as features.
  -z Z                  (optional) Add first Z number of positive packets as features.
  -csv                  (optional) Write packet features to a CSV file.
  -cdf                  (optional) Create CDFs from CSV file.
  -ml                   (optional) Output to text file all websites in the format of websiteNumber1,feature1,feature2,...
  -s S                  (optional) Generate samples using size s.
  --zeros {True,False}  (required with -s flag) Specify whether or not to include packets of size zero in the sampling.
```
