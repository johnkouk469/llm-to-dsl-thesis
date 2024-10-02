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

        validate_model(dflow_model)

        return history
    except Exception as e:
        print(e)
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


def conversation():
    """Initiates an interactive conversation with the dFlow assistant."""

    conversation_history = []

    while True:
        utterance = input("Write your message to the dFlow assistant or Exit to quit:")
        if utterance == "Exit":
            print("Exiting the conversation.")
            break

        conversation_history = generate_dflow_model(utterance, conversation_history)


def main():
    """Main function."""
    conversation()


if __name__ == "__main__":
    main()
