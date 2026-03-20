attractions = {
    'london': ['buckingham palace', 'westminister abbey', 'trafalgar square'],
    'paris' : ['effel tower', 'notre dame', 'louvre museum'],
    'rome': ['the colosseum', 'the pantheon', 'piazza navona']
}

def getAttractions(place):
    if place in attractions:
        print(attractions[place])

input = input("Enter the place: ")
getAttractions(input)