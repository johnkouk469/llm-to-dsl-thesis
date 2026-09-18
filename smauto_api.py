"""This script sends requests to the SmAuto API endpoints.

Run it directly to validate one or more model files against the running API:
    python smauto_api.py experiments/smauto/exp1_complexity/simple/model.auto
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def validate(model: str) -> requests.Response:
    """Send a request to the /validate endpoint."""
    response = requests.post(
        "http://localhost:8080/validate",
        headers={"X-API-Key": os.getenv("SMAUTO_API_KEY")},
        json={"name": "someModel", "model": model},
        timeout=10,
    )
    return response


def gen(model: str) -> requests.Response:
    """Send a request to the /gen endpoint."""
    response = requests.post(
        "http://localhost:8080/generate/autos",
        headers={"X-API-Key": os.getenv("SMAUTO_API_KEY")},
        json={"model": model},
        timeout=10,
    )
    return response


def genv(model: str) -> requests.Response:
    """Send a request to the /genv endpoint."""
    response = requests.post(
        "http://localhost:8080/generate/ventities",
        headers={"X-API-Key": os.getenv("SMAUTO_API_KEY")},
        json={"model": model},
        timeout=10,
    )
    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smauto_api.py <model.auto> [<model.auto> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        with open(path, "r", encoding="utf-8") as file:
            model_text = file.read()
        validation = validate(model_text)
        print("The SmAuto validator response for " + path + " is:")
        print(validation.status_code)
        print(validation.text)
