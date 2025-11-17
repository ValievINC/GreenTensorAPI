import requests
import json
import math
import zipfile
import io

# URL API
#url = "http://localhost:8000/generate-plot/"
url = "http://localhost:8000/generate-analysis/"

data = {
    "lens_config": {
        "wave_number": 4 * math.pi,
        "series_terms": 90,
        "layers_count": 4,
        "azimuth_angle": 90,
        "radii": [0.53, 0.75, 0.93, 1.0],
        "dielectric_constants": [1.86, 1.57, 1.28, 1.0],
        "magnetic_permeabilities": [1.0, 1.0, 1.0, 1.0]
    },
    "normalize": True
}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("lens_analysis.zip", "wb") as f:
        f.write(response.content)