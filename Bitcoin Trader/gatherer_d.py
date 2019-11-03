#=((B1+36000)/86400)+DATE(1970,1,1)

#imports
import time
import datetime
from dateutil import parser
import requests
import json
import urllib

historical = []
import csv
with open('data/historical.csv', newline='') as csvfile:
    #add rows to prices
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        historical.append(row)


def date_data(working_date):
    global historical
    params = (
        ('currency', 'USD'),
        ('date', working_date)
    )
    url='https://api.coinbase.com/v2/prices/spot'
    response = requests.get(url, params=params)
    data = response.json().get('data')
    print(data)

    historical.append([data.get('amount'), working_date])
    with open('data/historical.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([data.get('amount'), str(working_date)])

working_date = str(historical[-1][-1])
while True:
    working_date = str(historical[-1][-1])
    print(working_date)
    working_date = parser.isoparse(working_date)
    working_date = working_date + datetime.timedelta(hours=24)
    working_date = working_date.date()
    if working_date != datetime.date.today():
        date_data(working_date)
    else:
        print('Complete!')
        t = datetime.datetime.today()
        future = datetime.datetime(t.year,t.month,t.day,1,0)
        if t.hour >= 1:
            future += datetime.timedelta(days=1)
        print('sleeping for: ' + str((future-t).seconds))
        time.sleep((future-t).seconds)
