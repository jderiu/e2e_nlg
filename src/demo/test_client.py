import requests
import json
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)


test_mr = {'name': 'McDondals', 'eatType': 'coffee shop', 'priceRange': 'moderate', 'near': 'Orian', 'area': 'city centre', 'familyFriendly': 'no'}

r = requests.post("http://127.0.0.1:9999/generate_utterance", json=test_mr)
print(r.status_code)
response = json.loads(r.text)
for line in response['results']:
    print(line)
