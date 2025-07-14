"""This script sends requests to the Dflow API endpoints."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DFLOW_EXAMPLES_PATH = "./dflow_examples"
FEW_SHOT_MODELS_PATH = "./dflow_examples/llm_4_dflow"
DFLOW_GENERATED_MODELS_PATH = "./logs_dflow"
dflow_files = [
    f for f in os.listdir(DFLOW_GENERATED_MODELS_PATH) if f.endswith(".dflow")
]


def validate(model: str) -> requests.Response:
    """Send a request to the /validate endpoint."""
    response = requests.post(
        "http://localhost:8080/validate",
        headers={"X-API-Key": os.getenv("DFLOW_API_KEY")},
        json={"name": "someModel", "model": model},
        timeout=10,
    )
    return response


def gen(model: str) -> requests.Response:
    """Send a request to the /gen endpoint."""
    response = requests.post(
        "http://localhost:8080/generate",
        headers={"X-API-Key": os.getenv("DFLOW_API_KEY")},
        json={"name": "someModel", "model": model},
        timeout=10,
    )
    return response


if __name__ == "__main__":
    with open("./task1.dflow", "r", encoding="utf-8") as file:
        validation = validate(file.read())
        print(validation.status_code)
        print(validation.text)
    # for dflow in dflow_files:
    #     with open(
    #         DFLOW_GENERATED_MODELS_PATH + "/" + dflow, "r", encoding="utf-8"
    #     ) as file:
    #         validation = validate(file.read())
    #         print("The Dflow's validator response for " + dflow + " is:")
    #         print(validation.status_code)
    #         print(validation.text)
