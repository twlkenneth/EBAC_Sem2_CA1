import pandas as pd

from q3_time_series.model import *
from q3_time_series.base import Base

def comparison() -> pd.DataFrame:
    uni = []
    for country in Base().df.columns:
        s = StepWiseArima().run(country)
        df = pd.DataFrame([s[country]])
        df['country'] = country
        df['model'] = 'univariate ARIMA'
        uni.append(df)
    univariate = pd.concat(uni)

    multi = []
    s = VectorAutoRegression().run()
    for content in s:
        df = pd.DataFrame([content[list(content.keys())[0]]])
        df['country'] = list(content.keys())[0]
        df['model'] = 'vector autoregression'
        multi.append(df)
    var = pd.concat(multi)

    leaderboard = pd.concat([univariate, var])

    return leaderboard

if __name__ == "__main__":
    comparison()