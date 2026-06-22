# **energy-meter-bot-cv**

## **1. Клонируйте репозиторий и перейдите в папку:**
```
git clone <url-репозитория>
cd electric-energy-bot-ml
```

## **2. Создайте и активируйте виртуальное окружение:**
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

## **3. Установите все зависимости:**
```
pip install -r requirements.txt
```

## **4. Установите проект с dev-зависимостями:**
```
pip install -e .[dev]
```
Эта команда установит сам проект, а так же `ruff` и `pre-commit`.

## **4. Активируйте pre-commit хуки (делается ОДИН раз):**
```
pre-commit install
```
Проверьте все работает:
```
pre-commit run --all-files
```
