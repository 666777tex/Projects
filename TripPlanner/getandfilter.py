import json


# with open(r'jsons\city.json') as f:
#     data = json.load(f)

# getting the list of continents, the original author uses it to categorize it into europe, asia, us & canada, australia & new zealand, and central & south america

# continents = []
# for i in data:
#     if i["continent"] not in continents:
#         continents.append(i["continent"])

# i am just going to be working on US and canada for now (the author of the json file/data has canada also cateogorized into US as continents)
def getCities(path, cont):
    l = []
    with open(path, 'w') as f:
        for i in data:
            if i["continent"] == cont:
                l.append(i)

        json.dump(l, f, indent = 4)

path = r'jsons\USandCanada.json'
# getCities(path, "US")

# removes useless attributes
# keep: name, legalName, country, 
def removeAttr(input, output):
    filtered_data = []
    target = ["name", "legalName", "country"]
    content = json.load(open(input))
    for i in range(len(content)):
        fil_city = {
            "name": content[i]["name"],
            "legalName": content[i]["legalName"],
            "country": content[i]["country"]["name"]
        }
        filtered_data.append(fil_city)
    with open(output, "w") as f:
        json.dump(filtered_data, f, indent=4)
    
                

input = r'jsons\USandCanada.json'
output = r'jsons\UnC_Clean.json'
removeAttr(input, output)