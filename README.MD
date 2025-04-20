# API для GreenTensor

Апи полностью разработано на FastAPI и содержит в себе самоописываемую документацию на Swagger.

## Запуск

Чтобы запустить приложение, нужно установить некоторые зависимости:

```powershell
pip install uvicorn fastapi
```

После этого, в папке FastApiAPP:

```powershell
cd FastApiAPP
```

нужно запустить следующую команду:

```powershell
uvicorn main:app --reload
```

После всех процедур апи запустится по адресу http://127.0.0.1:8000.

## Использование

Можно зайти на http://127.0.0.1:8000/docs и посмотреть/потыкать по методам, доступным в API.

Если нужно получить какой-то результат из какого-нибудь языка программирования, то можно сделать классический request с передачей параметров.

Так это будет выглядеть на Python:

```python
import requests
import zipfile
from io import BytesIO

API_URL = "http://localhost:8000/generate-images/"

params = {
    "radiusRatio": 4,
    "layers_count": 6,
    "norm_radii": [0.1, 0.2, 0.3, 0.4, 0.5, 1],
    "dielectric_constants": [1e40, 1e40, 1e40, 1e40, 1e40, 1],
    "magnetic_permeabilities": [1, 1, 1, 1, 1, 1],
    "plot_type": "both"
}

response = requests.post(API_URL, json=params)

if response.status_code == 200:
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("extracted_images")
        print(f"Извлечено файлов: {len(zip_file.namelist())}")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)
```