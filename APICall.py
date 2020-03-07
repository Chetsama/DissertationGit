from darksky.api import DarkSky, DarkSkyAsync
from darksky.types import languages, units, weather
import json
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
                'temperatureMin',
                'temperatureMax',
                'sunriseTime',
                'sunsetTime']
    DailySummary = namedtuple("DailySummary", features)
    unix = 1573996500
    BASE_URL = "https://api.darksky.net/forecast/6dad852ea49ce601336822b43ea2e0c5/55.816561,10.636611," + str(
        unix) + "?exclude=currently,flags,alerts,hourly"
    response = requests.get(BASE_URL)
    data = response.json()
    df = pd.DataFrame(data["daily"]["data"])
    df1.append(DailySummary(summary=df.at[0, 'summary'],
                            temperatureMin=df.at[0, 'temperatureMin'],
                            temperatureMax=df.at[0, 'temperatureMax'],
                            sunriseTime=df.at[0, 'sunriseTime'],
                            sunsetTime=df.at[0, 'sunsetTime']))
    res = pd.DataFrame(df1, columns=features)
    res.to_csv('year_2019.csv', index=False)

def main():

    darkSkyCall()


if __name__ == "__main__":
    main()

