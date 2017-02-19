from pandas2arff import *
import pandas as pd

data = pd.read_csv('train.csv')

pandas2arff(data, 'titanic_train')
