import requests
import pandas as pd
from collections import namedtuple

# def darkSkyCall():
#     API_KEY = '6dad852ea49ce601336822b43ea2e0c5'
#
#     from datetime import datetime as dt
#
#     darksky = DarkSky(API_KEY)
#     t = dt(2018, 12, 13, 17, 30)
#     print(t)
#
#     latitude = 55.816561
#     longitude = 10.636611
#     forecast = darksky.get_time_machine_forecast(
#         latitude, longitude,
#         extend=True,  # default `False`
#         lang=languages.ENGLISH,  # default `ENGLISH`
#         values_units=units.AUTO,  # default `auto`
#         time=t
#     )
#
#     json_data = json.loads(forecast.text)
#     print(json_data)

def darkSkyCall():
    #https://medium.com/analytics-vidhya/making-historical-weather-data-for-any-place-using-dark-sky-api-calls-d5876de0ec01
    df1 = []
    features = ['summary',
                'temperature',]
    DailySummary = namedtuple("DailySummary", features)
    unix = 1544486400
    res = pd.DataFrame()

    for i in range(350):
        BASE_URL = "https://api.darksky.net/forecast/6dad852ea49ce601336822b43ea2e0c5/55.816561,10.636611," + str(
            unix) + "?exclude=currently,flags,alerts"
        response = requests.get(BASE_URL)
        data = response.json()
        print(data)

        df = pd.DataFrame(data["hourly"]["data"])
        print(df)
        # df1.append(DailySummary(summary=df.at[0, 'summary'],
        #                         temperature=df.at[0, 'temperature'],))

        res = res.append(df)

        unix = unix + 86400

    print(res)
    res.to_csv('year_2019.csv', index=False)


def main():

    darkSkyCall()


if __name__ == "__main__":
    main()

