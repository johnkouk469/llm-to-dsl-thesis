"""This script sends requests to the SmAuto API endpoints."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SMAUTO_EXAMPLES_PATH = "./smauto_examples_from_repo"
smauto_files = [f for f in os.listdir(SMAUTO_EXAMPLES_PATH) if f.endswith(".auto")]


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
    for auto in smauto_files:
        with open(SMAUTO_EXAMPLES_PATH + "/" + auto, "r", encoding="utf-8") as file:
            validation = validate(file.read())
            # print("The SmAuto's validator response for " + auto + " is:")
            # print(validation.status_code)
            if validation.status_code != 200:
                print("\n")
            else:
                generation = gen(file.read())
                print("The SmAuto's generator response for " + auto + " is:")
                print(generation.status_code)
                print(generation.text)
