import os
import json
import pandas as pd

from stocktwits_collector.collector import Collector
sc = Collector()

# download last messages up to 30
# messages = sc.get_history({'symbols': ['TSLA'], 'limit': 4})
# download the messages from a date to today
messages = sc.get_history({'symbols': ['TSLA'], 'start': '2024-11-13T00:00:00Z'})
# save the messages on files splitted per chunk from a date to max ID
chunk = sc.save_history({'symbols': ['TSLA'], 'start': '2024-11-13T00:00:00Z', 'chunk': 'day'})

# load data from one file
# with open('history.20220404.json', 'r') as f:
#     data = json.loads(f.read())
# df = pd.json_normalize(
#     data,
#     meta=[
#         'id', 'body', 'created_at',
#         ['user', 'id'],
#         ['user', 'username'],
#         ['entities', 'sentiment', 'basic']
#     ]
# )
# twits = df[['id', 'body', 'created_at', 'user.username', 'entities.sentiment.basic']]

# load data from multiple files
# frames = []
# path = '.'
# for file in os.listdir(path):
#     filename = f"{path}/{file}"
#     with open(filename, 'r') as f:
#         data = json.loads(f.read())
#         frames.append(pd.json_normalize(
#             data,
#             meta=[
#                 'id', 'body', 'created_at',
#                 ['user', 'id'],
#                 ['user', 'username'],
#                 ['entities', 'sentiment', 'basic']
#             ]
#           )
#         )
# df = pd.concat(frames).sort_values(by=['id'])
# twits = df[['id', 'body', 'created_at', 'user.username', 'entities.sentiment.basic']]