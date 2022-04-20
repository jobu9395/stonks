import math
import os
import random
import re
import sys


wallPositions = [1, 3, 7]
wallHeights = [4, 3, 3]


# def maxHeight(wallPositions, wallHeights):
#
#     mud_position_list = []
#
#     # make a dict of positions with heights
#     position_height_dict = {}
#     for i in range(1, max(wallPositions) + 1):
#         if i not in wallPositions:
#             mud_position_list.append(i)
#         if i in wallPositions:
#             for j in range(0, len(wallHeights)):
#                 position_height_dict[i] = wallHeights[0]
#             wallHeights.pop(0)
#
#     print(position_height_dict)
#     print(mud_position_list)
#
#     for mud_position in mud_position_list:
#         if mud_position not in position_height_dict.keys():
#             position_height_dict[mud_position] = 0
#
#     for i in sorted(position_height_dict.keys()):
#         print(position_height_dict[i])
#         if i in mud_position_list:
#             position_height_dict[i] = max(position_height_dict[i - 1], position_height_dict[i + 1])
#
#     tallest_mud_segment = max(position_height_dict.values())
#     return tallest_mud_segment
#
#
# maxHeight(wallPositions, wallHeights)


import sys
import os
from urllib.request import Request
from urllib.request import urlopen
from urllib.error import URLError
import json
import requests

def  getMovieTitles(substr):
    response = requests.get(f"https://jsonmock.hackerrank.com/api/movies/search/?Title={substr}")

    try:
        response = requests.get(f"https://jsonmock.hackerrank.com/api/movies/search/?Title={substr}")
        response.raise_for_status()
        jsonResponse = response.json()

    except URLError as url_error:
        print(f'HTTP error occurred: {url_error}')
    except Exception as err:
        print(f'Other error occurred: {err}')

    Titles = []
    for nested_dict in jsonResponse["data"]:
        values = list(nested_dict.values())
        Titles.append(values[0])

    Titles = sorted(Titles)
    return Titles

getMovieTitles('spiderman')
