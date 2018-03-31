import pandas
import random
import csv

filename = "all_data.csv"
# number of records in file (excludes header)
n = sum(1 for line in open(filename)) - 1
s = 5000  # desired sample size
# the 0-indexed header will not be included in the skip list
skip = sorted(random.sample(xrange(1, n + 1), n - s))
df = pandas.read_csv(filename, skiprows=skip)
df.to_csv("random_data.csv", sep=',', encoding='utf-8')
