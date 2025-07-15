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


def generate_smauto_model_after_qna(
    history: List[Tuple[str, str]],
    qna_history: List[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Generates an SmAuto model based on the information gathered during the Q&A process.

    Parameters:
    history (List[Tuple[str, str]]): A list that maintains the history of the interactions
      up to the point where the Q&A process started.
    qna_history (List[Tuple[str, str]]): A list that maintains the history of the Q&A process.

    Returns:
    Tuple[str, List[Tuple[str, str]]]: A tuple containing the generated SmAuto
    model (str) and the updated conversation history (list).
    """
    try:

        history.extend(qna_history)

        smauto_model = invoke_model_generation_after_qna(history)

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

            if is_repeated_error(validation_response, smauto_model):
                logger.info("The same validation error appeared two consecutive times.")
                break

        logger.info(
            "Max regeneration attempts reached or unable to fix error. Terminating process."
        )
        return smauto_model, history
    except Exception as e:
        logger.error("Error regenerating SmAuto model: %s", e)
        raise


def analyze_user_utterance(
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

        analyzed_utterance = invoke_user_utterance_analysis(user_utterance, history)

        history.append(("system", smauto_prompts.IDENTIFY_USER_INTENT))
        history.append(("user", user_utterance))
        history.append(("assistant", analyzed_utterance))

        return analyzed_utterance, history
    except Exception as e:
        logger.error("Error analyzing the user's utterance: %s", e)
        raise


def qna_initialization(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Initializes the Q&A process to gather information from the user."""

    assistant_response = invoke_qna_initialization(history)

    return [("assistant", assistant_response)]


def qna_follow_up(
    user_response: str,
    history: List[Tuple[str, str]],
    qna_history: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Asks follow-up questions based on the user's response to gather additional information."""

    assistant_response = invoke_qna_follow_up(user_response, history, qna_history)

    qna_history.append(("user", user_response))
    qna_history.append(("assistant", assistant_response))

    return qna_history


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
            ("user", smauto_prompts.CONSTRUCT_SMAUTO_MODEL),
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


def invoke_model_generation_after_qna(history: List[Tuple[str, str]]) -> str:
    """Invokes the language model to generate the SmAuto model after the Q&A process."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("system", smauto_prompts.CONSTRUCT_SMAUTO_MODEL_AFTER_QA),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info(
        "Instructing the LLM to generate an SmAuto model after the Q&A process \
by using all the information gathered."
    )
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
        }
    )


def invoke_user_utterance_analysis(
    user_utterance: str, history: List[Tuple[str, str]]
) -> str:
    """Invokes the language model to analyze the user's utterance, identify missing information,
    and ask follow-up questions to gather all necessary details to write the SmAuto model.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("system", smauto_prompts.IDENTIFY_USER_INTENT),
            ("user", user_utterance),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info(
        "Instructing the LLM to to analyze the user's utterance, identify missing information, \
and ask follow-up questions to gather all necessary details to write the SmAuto model."
    )
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
        }
    )


def invoke_qna_initialization(history: List[Tuple[str, str]]) -> str:
    """Invokes the language model to initialize the Q&A process."""
    if history == []:
        logger.error(
            "The user utterance should be analyzed before initializing the Q&A process."
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("system", smauto_prompts.GATHER_INFOMATION),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info("Instructing the LLM to initializing the Q&A process.")
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
        }
    )


def invoke_qna_follow_up(
    user_response: str,
    history: List[Tuple[str, str]],
    qna_history: List[Tuple[str, str]],
) -> str:
    """Invokes the language model to ask follow-up questions based on the user's response."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history_with_analysis"),
            ("system", smauto_prompts.GATHER_INFOMATION),
            MessagesPlaceholder("qna_history"),
            ("user", user_response),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info(
        "Providing the LLM with the user's response to ask follow-up questions."
    )
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history_with_analysis": history,
            "qna_history": qna_history,
        }
    )


def format_user_message(user_utterance: str) -> str:
    """Formats the user message for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(smauto_prompts.CONSTRUCT_SMAUTO_MODEL)
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
    if validation_regen.status_code == 200:
        return False
    else:
        new_validation_message = extract_validation_message(validation_regen)
        return (
            new_validation_message[new_validation_message.index(".auto") :]
            == validation_message[validation_message.index(".auto") :]
        )


def extract_validation_message(validation: Response) -> str:
    """Extracts the validation message from the API response."""
    return validation.json().get("detail")


def strip_code_tags(model: str) -> str:
    """Removes the code prefix and suffix tags from the model."""
    return model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)


def conversation():
    """Initiates an interactive conversation with the SmAuto assistant."""

    conversation_history = []

    while True:
        utterance = input("Write your message to the SmAuto assistant or Exit to quit:")
        if utterance == "Exit":
            print("Exiting the conversation.")
            break

        logger.info("User: %s", utterance)

        smauto_model, conversation_history = generate_smauto_model(
            utterance, conversation_history
        )

        logger.info("SmAuto Assistant: %s", smauto_model)


def conversation_with_feedback():
    """Initiates an interactive conversation with the SmAuto assistant
    where the assistant will be analyzing the user's request identify missing information,
    and ask follow-up questions to gather all necessary details to write the SmAuto model
    before writing it."""

    conversation_history = []

    while True:
        utterance = input("Write your message to the SmAuto assistant or Exit to quit:")
        if utterance == "Exit":
            print("Exiting the conversation.")
            break

        logger.info("User: %s", utterance)

        analyzed_utterance, conversation_history = analyze_user_utterance(
            utterance, conversation_history
        )

        logger.info("SmAuto Assistant: %s", analyzed_utterance)

        qna_history = qna_initialization(conversation_history)

        logger.info("SmAuto Assistant: %s", qna_history[-1][1])

        while True:
            utterance = input(
                "Write your message to the SmAuto assistant or Exit to quit:"
            )
            if utterance == "Exit":
                print("Exiting the conversation.")
                break

            logger.info("User: %s", utterance)

            qna_history = qna_follow_up(utterance, conversation_history, qna_history)

            assistant_response = qna_history[-1][1]

            logger.info("SmAuto Assistant: %s", assistant_response)

            if "Q&A process complete." in assistant_response:
                smauto_model, conversation_history = generate_smauto_model_after_qna(
                    conversation_history, qna_history
                )

                logger.info("SmAuto Assistant: %s", smauto_model)

                break


def main():
    """
    Main function to interact with the user via the terminal console.
    """

    while True:
        # Get user input
        print("Choose an option to interact with the SmAuto assistant:")
        print("1. Have a conversation")
        print("2. Have converation with feedback")
        print("3. Exit")
        choice = input("Enter the number of your choice: ")

        # Process based on user's choice
        if choice == "1":
            conversation()
        elif choice == "2":
            conversation_with_feedback()
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
