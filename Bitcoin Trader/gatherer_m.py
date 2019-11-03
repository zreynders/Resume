#=((B1+36000)/86400)+DATE(1970,1,1)

#imports
import time
import requests
import json
import urllib

prices = []
import csv
with open('data/prices.csv', newline='') as csvfile:
    #add rows to prices
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        prices.append(row)

#defs
looping = True
params = (
    ('currency', 'USD'),
)
url='https://api.coinbase.com/v2/prices/spot'
data = None

def updateLogic():
    global data
    response = requests.get(url, params=params)
    data = response.json().get('data')
    data['time'] = time.time()

    if data.get('amount') < starting_data.get('amount'):
        print("  *heartbeat* price: " + str(data.get('amount')) + " - ")
    elif data.get('amount') > starting_data.get('amount'):
        print("  *heartbeat* price: " + str(data.get('amount')) + " + ")
    else:
        print("  *heartbeat* price: " + str(data.get('amount')) + " . ")


#init code
print('starting...')

response = requests.get(url, params=params)
data = response.json().get('data')
starting_price = data.get('amount')

starttime = time.time()

print(' waiting for new price...')
#wait until a new amount appears
new_amount_found = False
while not new_amount_found:
    response = requests.get(url, params=params)
    data = response.json().get('data')
    if starting_price == data.get('amount'):
        time.sleep(0.01 - ((time.time() - starttime) % 0.01))
    else:
        new_amount_found = True

starttime = time.time()


print('done.')
print('starting session...')

#starting data to base up/down off
starting_data = data

#print starting_price
print(starting_data)
print(' starting price: ' + starting_data.get('amount'))


#main loop
while True:
    try:
        #get the new price
        updateLogic()

        #save to csv and update list
        prices.append([data.get('amount'), data.get('time')])
        with open('data/prices.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([data.get('amount'), data.get('time')])
        with open('data/current_price.txt', 'w', newline='') as textfile:
            writer = csv.writer(textfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([data.get('amount')])
        #sleep for the rest of the 30 second heartbeat
        time.sleep(30 - ((time.time() - starttime) % 0.1))
    except Exception as e:
        print("  heartbeat failed!")
