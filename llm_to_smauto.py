"""Python module for AI Assistant writing SmAuto models."""

import os
import time
import logging

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
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Create folders to log the results
timestamp = time.strftime("%Y%m%d-%H%M%S")
LOGS_FOLDER = "logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)
RESULTS_FOLDER = timestamp
os.makedirs(os.path.join(LOGS_FOLDER, RESULTS_FOLDER), exist_ok=True)
results_path = os.path.join(LOGS_FOLDER, RESULTS_FOLDER)

CODE_PREFIX = "```smauto\n"
CODE_SUFFIX = "\n```"
SMAUTO_FILE_NAME_EXTENSION = ".auto"

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")


def generate_smauto_model(user_utterance: str, history: list = None) -> tuple:
    """
    Generates an SmAuto model based on the user's utterance.

    This function takes the user's input, saves it to a file, and uses a
    language model to generate an initial SmAuto model. If the generated
    model is invalid, it attempts to regenerate it until it is valid or a
    maximum number of attempts is reached. The function maintains a history
    of interactions for context.

    Parameters:
    user_utterance (str): The input provided by the user to generate the model.
    history (list, optional): A list to maintain the history of the conversation.
                              Defaults to None.

    Returns:
    tuple: A tuple containing the generated SmAuto model (str) and the updated
           conversation history (list).
    """

    # Save the user utterance to a file
    with open(
        os.path.join(results_path, "user_utterance.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(user_utterance)
        file.close()

    if history is None:
        history = []

    # Instruct the LLM to generate a SmAuto model based on the user utterance

    write_full_model_prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.CONSTRTUCT_SMAUTO_MODEL),
        ]
    )

    smauto_model_chain = write_full_model_prompt_template | model | StrOutputParser()

    logger.info(
        "Instructing the LLM to generate an SmAuto model based on the user input."
    )

    smauto_model = smauto_model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
            "user_utterance": user_utterance,
        }
    )

    logger.info("An SmAuto model has been generated based on the user input.")

    # Add the user prompt to generate the model and the LLM's response to the conversation history
    history.append(
        (
            "user",
            HumanMessagePromptTemplate.from_template(
                smauto_prompts.CONSTRTUCT_SMAUTO_MODEL
            )
            .format(user_utterance=user_utterance)
            .pretty_repr(),
        )
    )
    history.append(("assistant", smauto_model))

    # Save the generated model to a file
    with open(
        os.path.join(results_path, "smauto_model" + SMAUTO_FILE_NAME_EXTENSION),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
        file.close()

    logger.info("The model has been saved on the smauto_model.auto file.")

    validation = smauto_api.validate(
        smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
    )

    # Validate the model and regenarate it if it is invalid
    if validation.status_code == 200:
        logger.info("The generated SmAuto model is syntactically valid.")
    else:
        smauto_model, history = regenerate_invalid_model(validation, history)

    return smauto_model, history


def regenerate_invalid_model(validation: Response, history: list = None) -> tuple:
    """
    Regenerates an invalid SmAuto model.

    This function uses a language model to regenerate an invalid SmAuto model
    based on validation error messages. It continues the regeneration process
    until the model is valid, the maximum number of regeneration attempts is
    reached, or it is determined that the same validation error cannot be fixed.

    Parameters:
    validation (Response): The response from the validation API indicating
                           validation errors.
    history (list, optional): A list to maintain the history of the conversation.
                              Defaults to None.

    Returns:
    tuple: A tuple containing the regenerated SmAuto model (str) and the updated
           conversation history (list).
    """

    if history is None:
        history = []

    invalid_model_prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.INVALID_MODEL),
        ]
    )

    invalid_model_chain = invalid_model_prompt_template | model | StrOutputParser()

    invalid_model_generations = 1

    while True:
        if invalid_model_generations == 1:
            logger.info("The generated SmAuto model is not syntactically valid.")
        else:
            logger.info("The regenarated model still has errors.")
        logger.info(
            "The SmAuto's validator response for the model is: %s", validation.text
        )
        logger.info("Instructing the LLM to regenerate the model with the error fixed.")

        smauto_model = invalid_model_chain.invoke(
            {
                "system_prompt": smauto_prompts.get_system_prompt(),
                "history": history,
                "validation_message": validation.json()
                .get("detail")
                .split(SMAUTO_FILE_NAME_EXTENSION)[1],
            }
        )

        # Add the user prompt to regenerate the model and the LLM's response
        # to the conversation history
        history.append(
            (
                "user",
                HumanMessagePromptTemplate.from_template(smauto_prompts.INVALID_MODEL)
                .format(validation_message=validation.text)
                .pretty_repr(),
            )
        )
        history.append(("assistant", smauto_model))

        # Save the regenerated model to a file
        with open(
            os.path.join(
                results_path,
                "regenerated_smauto_model_"
                + str(invalid_model_generations)
                + SMAUTO_FILE_NAME_EXTENSION,
            ),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
            file.close()
        logger.info(
            "The SmAuto model has been regenerated and saved at \
regenerated_smauto_model_%d%s file.",
            invalid_model_generations,
            SMAUTO_FILE_NAME_EXTENSION,
        )

        invalid_model_generations += 1

        # Exit the loop if the max number of iterations is reached
        if invalid_model_generations == 5:
            logger.info(
                "After %d attemps to regenerate the SmAuto model with the \
errors fixed the model remains invalid.",
                invalid_model_generations,
            )
            logger.info("Terminating the regeneration process.")
            break

        # Validate the regenerated model
        validation_regen = smauto_api.validate(
            smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
        )

        # Exit the loop if the regenerated model is syntactically valid
        # or if the same error appeared two consecutive times
        if validation_regen.status_code == 200:
            logger.info("All errors have been corrected successfully.")
            logger.info("The regenerated model is syntactically valid.")
            break
        if (
            validation.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1]
            == validation_regen.json()
            .get("detail")
            .split(SMAUTO_FILE_NAME_EXTENSION)[1]
        ):
            logger.info(
                "After the regeneration of the model, the same error was found. \
Therefore the assistant is unable to fix the error."
            )
            break
        validation = validation_regen
    return smauto_model, history


def process_yaml_file(file_path):
    """
    Processes a YAML file at the given file path.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error processing the YAML file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.info("Processing YAML file: %s", config)
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
    except yaml.YAMLError as exc:
        logger.error("Error processing YAML file: %s", exc)


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
            process_yaml_file(file_path)
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
