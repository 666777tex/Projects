import xml.etree.ElementTree as ET
import pandas as pd 

tree = ET.parse(r'C:\Users\zhang\Projects\MusicOrganizer\songList.xml')
root = tree.getroot()

#first root (8 keys and dict for songs)
fr = root[0]

# 17 is dict which is where all the music is at
sr = fr[17]

#initializing songs dictionary
songs = {}

for n in range(1, len(sr), 2):
    name = ''
    artist = []
    name = sr[n][3].text
    temp = sr[n][5].text
    if ";" in temp:
        artist = temp.split("; ")
    elif "/" in temp:
        artist = temp.split("/")
    else:
        artist.append(temp)
    songs[name] = artist

df = pd.DataFrame(songs)
df.to_csv("songs.csv", index=False)