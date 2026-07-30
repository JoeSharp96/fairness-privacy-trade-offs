import pandas as pd

data = [0,1,1,0,1,0,1]
df = pd.DataFrame(data)
value = df.where(df[0]==1).count()[0]
print(value)
