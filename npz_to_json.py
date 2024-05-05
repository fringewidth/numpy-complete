import numpy as np
import json

model_npz = np.load('trained_parameters.npz')
model_json = {}

for key in model_npz.keys():
    model_json[key] = model_npz[key].tolist()

with open('trained_parameters.json', 'w') as f:
    json.dump(model_json, f, indent=1)