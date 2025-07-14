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
os.makedirs(os.path.join(RESULTS_PATH, "exp3"), exist_ok=True)
EXP3_PATH = os.path.join(RESULTS_PATH, "exp3")


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


def generate_smauto_model_from_yaml(
    yaml_file_path: str, history: Optional[List[Tuple[str, str]]] = None
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Generates an SmAuto model based on the contents of a YAML file.

    Parameters:
    yaml_file_path (str): The path to the YAML file containing the input data.
    history (Optional[List[Tuple[str, str]]]): A list to maintain the history
    of the conversation. Defaults to None.

    Returns:
    Tuple[str, List[Tuple[str, str]]]: A tuple containing the generated SmAuto
    model (str) and the updated conversation history (list).
    """
    try:
        yaml_content = read_yaml_file(yaml_file_path)
        if history is None:
            history = []

        smauto_model = invoke_model_generation_from_yaml(yaml_content, history)
        history.append(("user", format_yaml_message(yaml_content)))
        history.append(("assistant", smauto_model))

        save_model(smauto_model, "smauto_model" + SMAUTO_FILE_NAME_EXTENSION)

        if not validate_model(smauto_model):
            smauto_model, history = regenerate_invalid_model(smauto_model, history)

        return smauto_model, history
    except Exception as e:
        logger.error("Error generating SmAuto model from YAML: %s", e)
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


def invoke_model_generation_from_yaml(
    yaml_content: Any, history: List[Tuple[str, str]]
) -> str:
    """Invokes the language model to generate the SmAuto model based on YAML content."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.CONSTRUCT_SMAUTO_MODEL_FROM_YAML),
        ]
    )
    model_chain = prompt_template | llm | StrOutputParser()
    logger.info(
        "Instructing the LLM to generate an SmAuto model based on the YAML content."
    )
    return model_chain.invoke(
        {
            "system_prompt": smauto_prompts.get_system_prompt(),
            "history": history,
            "yaml_content": yaml_content,
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


def format_yaml_message(yaml_content: Any) -> str:
    """Formats the YAML content for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(
            smauto_prompts.CONSTRUCT_SMAUTO_MODEL_FROM_YAML
        )
        .format(yaml_content=yaml_content)
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
        return None
    except yaml.YAMLError as exc:
        logger.error("Error reading and parsing YAML file: %s", exc)
        return None


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

    Prompts the user to choose between inputting an utterance or providing a YAML file.
    Based on the user's choice, the appropriate function is called to process the input.
    The user can also choose to exit the program.

    Returns:
        None
    """

    #     for i in range(10):
    #         utterance = """I'm looking to automate several parts of my smart home. Here's the setup. In the living room, I have a temperature and humidity sensor that publishes data every 30 seconds, a dimmable lamp, and an air purifier that supports both power and mode control. These sensors should use value generators to simulate readings over time, reflecting natural fluctuations in environmental conditions. The temperature readings should be generated using a Gaussian function with a mean of 22, a maximum value of 35, and a sigma of 1, combined with Gaussian noise with a mean of 0 and sigma of 1. The humidity readings should use a Gaussian generator with a mean of 45, a maximum of 70, and sigma of 2, along with Uniform noise between 0 and 1. The dimmable lamp should support a power attribute (on/off) and a brightness attribute represented as a float value ranging from 0.0 to 100.0. The air purifier should support the following modes: "auto", "eco", "turbo", and "off".
    # In the bedroom, there’s a temperature sensor that publishes data every one minute, a simple on/off lamp, and an air conditioner with adjustable power, temperature, and mode. The bedroom temperature sensor should also simulate readings using a Gaussian distribution with a mean of 20, a maximum value of 30, and sigma of 1, along with Gaussian noise with a mean of 0 and sigma of 1. The air conditioner should support power control, a temperature setting in degrees Celsius, and multiple modes, including "cool", "heat", "dry", "fan", and "off".
    # All devices communicate using an MQTT broker. This broker is hosted at mqtt.smarthome.local, operates on port 8883, uses SSL for secure connections, and requires authentication with the username casa_nova and the password TooSmart4U@home. For system monitoring, I’m using a Redis broker located at redis.monitor.local, with SSL enabled, running on port 6380, connected to database index 2, and secured with the API key redis-4a9f1e7b3c2d6f8912a3b4c5d678e9f0. The broker should be configured to publish monitoring events and logs using the namespace "casa.nova.universe", with events published to the "epic_events" topic and logs to the "chaotic_logs" topic.
    # Here’s what I want to automate: When the bedroom temperature drops below 18°C, the lamp should turn on; this should be a continuously active automation. In the living room, if it gets too hot and humid (above 25°C and 60%) and the lamp is off, the air purifier should turn on in auto mode; this condition should be checked every 90 seconds. Once the temperature and humidity return to normal levels (below 24°C and 50%), the purifier should turn off and its automation should stop running automatically. This condition should be checked every 120 seconds. These two automations should be linked so that each one starts the other upon completion, ensuring they remain mutually responsive.
    # In the bedroom again, if the average temperature over time rises above 26°C and is unstable, the AC should turn on in cool mode at 23°C. This condition should be checked every 180 seconds. Lastly, every night at exactly 10 PM, the living room lamp should automatically turn on at 30% brightness. For the described automations, if a frequency for checking the condition is not explicitly specified, then assume a default of 60 seconds.
    # I want the name of the model to be "smart_home_env_control" and its version "1.4". The author of the model should be me, Lucas Navarro, with a reference to my email nova@homehub.tech. Also, I want you to come up with an appropriate description for the model.
    # """
    #         conversation_history = []

    #         logger.info("Generating model #%d", i)

    #         smauto_model, conversation_history = generate_smauto_model(
    #             utterance, conversation_history
    #         )

    #         logger.info("---------------")

    #         with open(
    #             os.path.join(EXP3_PATH, str(i) + "_smauto_model.auto"),
    #             "w",
    #             encoding="utf-8",
    #         ) as file:
    #             file.write(strip_code_tags(smauto_model))

    while True:
        # Get user input
        print("Choose an option to interact with the SmAuto assistant:")
        print("1. Have a conversation")
        print("2. Provide a YAML file")
        print("3. Have converation with feedback")
        print("4. Exit")
        choice = input("Enter the number of your choice: ")

        # Process based on user's choice
        if choice == "1":
            conversation()
        elif choice == "2":
            file_path = input("Enter the path to the YAML file: ")
            generate_smauto_model_from_yaml(file_path)
        elif choice == "3":
            conversation_with_feedback()
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
