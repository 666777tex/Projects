from bs4 import BeautifulSoup
import requests
import json
import csv

url = "https://d15ldvyocwqu5y.cloudfront.net/jsonws/clm-20260318050351/991/cities-for-home-page"

r = requests.get(url)

data = r.json()["cityJson"]

with open("city.json", 'w') as json_file:
    json.dump(data, json_file, indent= 4)