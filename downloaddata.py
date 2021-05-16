import twstock
import pandas as pd


stock = twstock.Stock('6269')
#stock = stock.fetch_from(2019, 1)
# data = stock.fetch_from(2020, 1)
data = stock.fetch(2020, 1)
print(data)

df = pd.DataFrame(data)
df.to_csv('6269.csv')

