from pandas2arff import *
import pandas as pd

data = pd.read_csv('./data/tips.csv')
del data['UID']
del data['ID']
del data['Date']

pandas2arff(data, './data/tips.arff')