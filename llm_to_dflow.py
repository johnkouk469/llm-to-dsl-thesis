"""Python module for AI Assistant writing SmAuto models."""

import os
import time
from typing import List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

import dflow_api
import dflow_prompts

load_dotenv()

timestamp = time.strftime("%Y%m%d-%H%M%S")
LOGS_FOLDER = "logs_dflow"
os.makedirs(LOGS_FOLDER, exist_ok=True)

MAX_REGENERATIONS = 5
CODE_PREFIX = "```dflow\n"
CODE_SUFFIX = "\n```"

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")


def generate_dflow_model(
    user_utterance: str, history: Optional[List[Tuple[str, str]]] = None
) -> List[Tuple[str, str]]:
    """Generate a dFlow model for the given user utterance."""
    try:
        if history is None:
            history = []

        write_dflow_model_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", dflow_prompts.DFLOW_DESCRIPTION),
                ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
                ("system", dflow_prompts.SYSTEM_ROLE),
                dflow_prompts.get_few_shot_examples(),
                MessagesPlaceholder("history"),
                ("user", dflow_prompts.CONSTRTUCT_DFLOW_MODEL_PROMPT),
            ]
        )

        dflow_model_chain = (
            write_dflow_model_prompt_template | model | StrOutputParser()
        )

        dflow_model = dflow_model_chain.invoke(
            {
                "history": history,
                "input": str("Help me create a virtual assistant."),
                "user_utterance": user_utterance,
            }
        )

        history.append(("user", format_user_message(user_utterance)))
        history.append(("system", dflow_model))

        with open(
            os.path.join(LOGS_FOLDER, timestamp + "_dflow_model.dflow"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
            file.close()

        if not validate_model(dflow_model):
            history = regenerate_dflow_model(history)

        return history
    except Exception as e:
        print(e)
        raise


def regenerate_dflow_model(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Regenerates an invalid dFlow model."""
    try:
        for attempt in range(MAX_REGENERATIONS):
            validation_response = dflow_api.validate(
                history[-1][1].removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
            )
            if validation_response.status_code == 200:
                print("The regenerated model is syntactically valid.")
                return history

            print(
                "The regenerated model " + str(attempt) + " still has errors: ",
                validation_response.text,
            )

            write_dflow_model_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", dflow_prompts.DFLOW_DESCRIPTION),
                    ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
                    ("system", dflow_prompts.SYSTEM_ROLE),
                    dflow_prompts.get_few_shot_examples(),
                    MessagesPlaceholder("history"),
                    ("user", dflow_prompts.INVALID_MODEL_PROMPT),
                ]
            )

            dflow_model_chain = (
                write_dflow_model_prompt_template | model | StrOutputParser()
            )

            dflow_model = dflow_model_chain.invoke(
                {
                    "history": history,
                    "input": str("Help me create a virtual assistant."),
                    "validation_message": validation_response.json().get("detail"),
                }
            )

            history.append(
                ("user", format_invalid_model_message(validation_response.text))
            )
            history.append(("assistant", dflow_model))

            with open(
                os.path.join(
                    LOGS_FOLDER,
                    timestamp
                    + "_regenerated_dflow_model_"
                    + str(attempt + 1)
                    + ".dflow",
                ),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
                )
                file.close()

            validation_response_regen = dflow_api.validate(
                dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
            )

            print(validation_response_regen.text)
            if validation_response_regen.status_code != 200:
                old_validation_message = validation_response.json().get("detail")
                new_validation_message = validation_response_regen.json().get("detail")
                if (
                    old_validation_message[old_validation_message.index(".dflow") :]
                    == new_validation_message[new_validation_message.index(".dflow") :]
                ):
                    print("The same validation error appeared two consecutive times.")
                    break

            if attempt == MAX_REGENERATIONS - 1:
                print(
                    "The regenerated model still has errors: ",
                    validation_response_regen.text,
                )
                print("Max regeneration attempts reached.")

        print("Unable to fix the error. Terminating the regeneration process.")
        return history
    except Exception as e:
        print("Error regenerating SmAuto model: %s", e)
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
        if history is None:
            history = []

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", dflow_prompts.DFLOW_DESCRIPTION),
                ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
                ("system", dflow_prompts.SYSTEM_ROLE),
                dflow_prompts.get_few_shot_examples(),
                MessagesPlaceholder("history"),
                ("system", dflow_prompts.IDENTIFY_USER_INTENT_DFLOW),
                ("user", user_utterance),
            ]
        )
        model_chain = prompt_template | model | StrOutputParser()
        print(
            "Instructing the LLM to to analyze the user's utterance, identify missing information, \
    and ask follow-up questions to gather all necessary details to write the SmAuto model."
        )
        analyzed_utterance = model_chain.invoke(
            {
                "input": str("Help me create a virtual assistant."),
                "history": history,
            }
        )

        history.append(("system", dflow_prompts.IDENTIFY_USER_INTENT_DFLOW))
        history.append(("user", user_utterance))
        history.append(("assistant", analyzed_utterance))

        return analyzed_utterance, history
    except Exception as e:
        print("Error analyzing the user's utterance: %s", e)
        raise


def qna_initialization(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Initializes the Q&A process to gather information from the user."""
    if history == []:
        print(
            "The user utterance should be analyzed before initializing the Q&A process."
        )
        return []

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", dflow_prompts.DFLOW_DESCRIPTION),
            ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
            ("system", dflow_prompts.SYSTEM_ROLE),
            dflow_prompts.get_few_shot_examples(),
            MessagesPlaceholder("history"),
            ("system", dflow_prompts.GATHER_INFORMATION_DFLOW),
        ]
    )
    model_chain = prompt_template | model | StrOutputParser()
    print("Instructing the LLM to initializing the Q&A process.")

    assistant_response = model_chain.invoke(
        {
            "input": str("Help me create a virtual assistant."),
            "history": history,
        }
    )

    return [("assistant", assistant_response)]


def qna_follow_up(
    user_response: str,
    history: List[Tuple[str, str]],
    qna_history: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Asks follow-up questions based on the user's response to gather additional information."""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", dflow_prompts.DFLOW_DESCRIPTION),
            ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
            ("system", dflow_prompts.SYSTEM_ROLE),
            dflow_prompts.get_few_shot_examples(),
            MessagesPlaceholder("history_with_analysis"),
            ("system", dflow_prompts.GATHER_INFORMATION_DFLOW),
            MessagesPlaceholder("qna_history"),
            ("user", user_response),
        ]
    )
    model_chain = prompt_template | model | StrOutputParser()
    print("Providing the LLM with the user's response to ask follow-up questions.")

    assistant_response = model_chain.invoke(
        {
            "input": str("Help me create a virtual assistant."),
            "history_with_analysis": history,
            "qna_history": qna_history,
        }
    )

    qna_history.append(("user", user_response))
    qna_history.append(("assistant", assistant_response))

    return qna_history


def generate_dflow_model_after_qna(
    history: List[Tuple[str, str]],
    qna_history: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Generates a dFlow model based on the information gathered during the Q&A process."""
    try:
        history.extend(qna_history)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", dflow_prompts.DFLOW_DESCRIPTION),
                ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
                ("system", dflow_prompts.SYSTEM_ROLE),
                dflow_prompts.get_few_shot_examples(),
                MessagesPlaceholder("history"),
                ("system", dflow_prompts.CONSTRUCT_DFLOW_MODEL_AFTER_QA),
            ]
        )
        model_chain = prompt_template | model | StrOutputParser()
        print(
            "Instructing the LLM to generate a dFlow model after the Q&A process \
    by using all the information gathered."
        )

        dflow_model = model_chain.invoke(
            {
                "input": str("Help me create a virtual assistant."),
                "history": history,
            }
        )

        history.append(("assistant", dflow_model))

        with open(
            os.path.join(LOGS_FOLDER, timestamp + "_dflow_model_after_qa.dflow"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
            file.close()

        if not validate_model(dflow_model):
            dflow_model, history = regenerate_dflow_model(history)

        return history
    except Exception as e:
        print("Error generating SmAuto model: %s", e)
        raise


def validate_model(dflow_model: str) -> bool:
    """Validate the dFlow model."""
    validation = dflow_api.validate(
        dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
    )
    print(validation.text)
    if validation.status_code == 200:
        print("The dFlow model is syntactically valid.")
        return True
    print("The dFlow model is not syntactically valid.")
    return False


def format_user_message(user_utterance: str) -> str:
    """Formats the user message for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(
            dflow_prompts.CONSTRTUCT_DFLOW_MODEL_PROMPT
        )
        .format(user_utterance=user_utterance)
        .pretty_repr()
    )


def format_invalid_model_message(validation_text: str) -> str:
    """Formats the validation message for the conversation history."""
    return (
        HumanMessagePromptTemplate.from_template(dflow_prompts.INVALID_MODEL_PROMPT)
        .format(validation_message=validation_text)
        .pretty_repr()
    )


def conversation():
    """Initiates an interactive conversation with the dFlow assistant."""

    conversation_history = []

    while True:
        utterance = input("Write your message to the dFlow assistant or Exit to quit:")
        if utterance == "Exit":
            print("Exiting the conversation.")
            break

        conversation_history = generate_dflow_model(utterance, conversation_history)


def conversation_with_feedback():
    """Initiates an interactive conversation with the dFlow assistant
    where the assistant will be analyzing the user's request identify missing information,
    and ask follow-up questions to gather all necessary details to write the SmAuto model
    before writing it."""

    conversation_history = []

    while True:
        utterance = input("Write your message to the dFlow assistant or Exit to quit:")
        if utterance == "Exit":
            print("Exiting the conversation.")
            break

        analyzed_utterance, conversation_history = analyze_user_utterance(
            utterance, conversation_history
        )

        print("dFlow assistant: %s", analyzed_utterance)

        qna_history = qna_initialization(conversation_history)

        print("dFlow assistant: %s", qna_history[-1][1])

        while True:
            utterance = input(
                "Write your message to the dFlow assistant or Exit to quit:"
            )
            if utterance == "Exit":
                print("Exiting the conversation.")
                break

            print("User: %s", utterance)

            qna_history = qna_follow_up(utterance, conversation_history, qna_history)

            assistant_response = qna_history[-1][1]

            print("dFlow assistant: %s", assistant_response)

            if "Q&A process complete." in assistant_response:
                smauto_model, conversation_history = generate_dflow_model_after_qna(
                    conversation_history, qna_history
                )

                print("dFlow assistant: %s", smauto_model)

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
