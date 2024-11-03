import pandas as pd

#Reading the file that i got from "https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics"
data = pd.read_csv('data/songs_with_attributes_and_lyrics.csv/songs_with_attributes_and_lyrics.csv')

data['lyrics_lower'] = data["lyrics"].str.lower()
print(data[['lyrics', 'lyrics_lower']].head())

