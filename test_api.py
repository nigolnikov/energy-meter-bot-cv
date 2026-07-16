from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict():
    with open("datasets/test/photo_screen_1903.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("meter.jpg", f, "image/jpg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "value" in data
    assert "status" in data


def test_predict_bad_file():
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
