import json
import requests
from bs4 import BeautifulSoup

# go through all the cities in UnC_Clean.json and then use the legal name 
# to scrape the attractions


def getAttractions(input):
    data = json.load(open(input))
    for i in range(len(data)):
        url = "https://www.visitacity.com/en/" + data[i]["legalName"] + "/attraction-by-type/all?attractionsCategory=top-attractions"
        page = requests.get(url)
        print("im getting the stuff for this url: ", url)
        print(page.text)
        break

getAttractions(r'jsons\UnC_Clean.json')