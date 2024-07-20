"""Python module for AI Assistant writing SmAuto models."""

import os
import time
import logging
from typing import List, Tuple, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from requests import Response
import yaml

import smauto_api
import smauto_prompts

load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Create folders to log the results
timestamp = time.strftime("%Y%m%d-%H%M%S")
LOGS_FOLDER = "logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)
RESULTS_FOLDER = timestamp
os.makedirs(os.path.join(LOGS_FOLDER, RESULTS_FOLDER), exist_ok=True)

# Constants
MAX_REGENERATIONS = 5
CODE_PREFIX = "```smauto\n"
CODE_SUFFIX = "\n```"
SMAUTO_FILE_NAME_EXTENSION = ".auto"
RESULTS_PATH = os.path.join(LOGS_FOLDER, RESULTS_FOLDER)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")


def generate_smauto_model(
    user_utterance: str, history: Optional[List[Tuple[str, str]]] = None
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Generates an SmAuto model based on the user's utterance.

    Parameters:
    user_utterance (str): The input provided by the user to generate the model.
    history (Optional[List[Tuple[str, str]]]): A list to maintain the history of
    the conversation. Defaults to None.

    Returns:
    Tuple[str, List[Tuple[str, str]]]: A tuple containing the generated SmAuto
    model (str) and the updated conversation history (list).
    """
    try:
        save_user_utterance(user_utterance)
        if history is None:
            history = []

        smauto_model = invoke_model_generation(user_utterance, history)
        history.append(("user", format_user_message(user_utterance)))
        history.append(("assistant", smauto_model))

        save_model(smauto_model, "smauto_model" + SMAUTO_FILE_NAME_EXTENSION)

        if not validate_model(smauto_model):
            smauto_model, history = regenerate_invalid_model(smauto_model, history)

        return smauto_model, history
    except Exception as e:
        logger.error("Error generating SmAuto model: %s", e)
        raise


def regenerate_invalid_model(
    smauto_model: str, history: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Regenerates an invalid SmAuto model.

    Parameters:
    smauto_model (str): The invalid SmAuto model.
    history (List[Tuple[str, str]]): A list to maintain the history of the conversation.

    Returns:
    Tuple[str, List[Tuple[str, str]]]: A tuple containing the regenerated SmAuto model (str)
    and the updated conversation history (list).
    """
    try:
        for attempt in range(MAX_REGENERATIONS):
            validation_response = smauto_api.validate(strip_code_tags(smauto_model))
            if validation_response.status_code == 200:
                logger.info("The regenerated model is syntactically valid.")
                return smauto_model, history

            logger.info(
                "The regenerated model still has errors: %s", validation_response.text
            )
            smauto_model = invoke_model_regeneration(validation_response, history)
            history.append(
                ("user", format_invalid_model_message(validation_response.text))
            )
            history.append(("assistant", smauto_model))

            save_model(
                smauto_model,
                f"regenerated_smauto_model_{attempt + 1}{SMAUTO_FILE_NAME_EXTENSION}",
            )

            if not is_repeated_error(validation_response, smauto_model):
                break

        logger.info(
            "Max regeneration attempts reached or unable to fix error. Terminating process."
        )
        return smauto_model, history
    except Exception as e:
        logger.error("Error regenerating SmAuto model: %s", e)
        raise


def save_user_utterance(user_utterance: str) -> None:
    """Saves the user utterance to a file."""
    with open(
        os.path.join(RESULTS_PATH, "user_utterance.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(user_utterance)


def invoke_model_generation(user_utterance: str, history: List[Tuple[str, str]]) -> str:
    """Invokes the language model to generate the SmAuto model."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.CONSTRTUCT_SMAUTO_MODEL),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info(
        "Instructing the LLM to generate an SmAuto model based on the user input."
    )
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
            "user_utterance": user_utterance,
        }
    )


def format_user_message(user_utterance: str) -> str:
    """Formats the user message for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(smauto_prompts.CONSTRTUCT_SMAUTO_MODEL)
        .format(user_utterance=user_utterance)
        .pretty_repr()
    )


def save_model(model: str, filename: str) -> None:
    """Saves the generated or regenerated model to a file."""
    with open(os.path.join(RESULTS_PATH, filename), "w", encoding="utf-8") as file:
        file.write(strip_code_tags(model))


def validate_model(model: str) -> bool:
    """Validates the generated or regenerated SmAuto model."""
    validation = smauto_api.validate(strip_code_tags(model))
    if validation.status_code == 200:
        logger.info("The generated SmAuto model is syntactically valid.")
        return True
    logger.info("The generated SmAuto model is not syntactically valid.")
    return False


def invoke_model_regeneration(
    validation: Response, history: List[Tuple[str, str]]
) -> str:
    """Invokes the language model to regenerate the invalid SmAuto model."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.INVALID_MODEL),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    validation_message = extract_validation_message(validation)
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
            "validation_message": validation_message,
        }
    )


def format_invalid_model_message(validation_text: str) -> str:
    """Formats the validation message for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(smauto_prompts.INVALID_MODEL)
        .format(validation_message=validation_text)
        .pretty_repr()
    )


def is_repeated_error(validation: Response, model: str) -> bool:
    """Checks if the same validation error appeared two consecutive times."""
    validation_message = extract_validation_message(validation)
    validation_regen = smauto_api.validate(strip_code_tags(model))
    new_validation_message = extract_validation_message(validation_regen)
    return validation_message == new_validation_message


def extract_validation_message(validation: Response) -> str:
    """Extracts the validation message from the API response."""
    return validation.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1]


def strip_code_tags(model: str) -> str:
    """Removes the code prefix and suffix tags from the model."""
    return model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)


def read_yaml_file(yaml_file_path: str) -> Any:
    """
    Reads and parses the content of a YAML file at the given file path.

    Args:
        yaml_file_path (str): The path to the YAML file.

    Returns:
        Any: The parsed content of the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error processing the YAML file.
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.info("Read and parsed the YAML file: %s", config)
            return config
    except FileNotFoundError:
        logger.error("File not found: %s", yaml_file_path)
    except yaml.YAMLError as exc:
        logger.error("Error reading and parsing YAML file: %s", exc)


def main():
    """
    Main function to interact with the user via the terminal console.

    Prompts the user to choose between inputting an utterance or providing a YAML file.
    Based on the user's choice, the appropriate function is called to process the input.
    The user can also choose to exit the program.

    Returns:
        None
    """

    while True:
        # Get user input
        print("Choose an option to interact with the SmAuto assistant:")
        print("1. Input an utterance")
        print("2. Provide a YAML file")
        print("3. Exit")
        choice = input("Enter the number of your choice: ")

        # Process based on user's choice
        if choice == "1":
            utterance = input("Enter your utterance: ")
            generate_smauto_model(utterance)
        elif choice == "2":
            file_path = input("Enter the path to the YAML file: ")
            read_yaml_file(file_path)
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
