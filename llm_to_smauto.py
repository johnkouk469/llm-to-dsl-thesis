"""Python module for AI Assistant writing SmAuto models."""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import smauto_api
import smauto_system_prompt

load_dotenv()

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

history = []

def generate_smauto_model(user_utterance: str) -> str:
    # Save the user utterance to a file
    with open(os.path.join(results_path, "user_utterance.txt"), "w", encoding="utf-8") as file:
        file.write(user_utterance)
        file.close()

    # Instruct the LLM to generate a SmAuto model based on the user utterance

    CONSTRTUCT_SMAUTO_MODEL_PROMPT = """
    Write brokers, entities, and automations, write the complete SmAuto model on the following description:
    {user_utterance}
    Define the Metadata and RTMonitor components as well.
    Follow the guidelines provided for each component to ensure the model is correctly structured.

    Do not use # to comment in the model. Use // for inline comments and /* */ for block comments.
    Use the appropriate operators for the conditions and actions in the automations for each type of attribute.
    As a reminder: **Boolean Operators**: `is`, `is not`


    Output only the SmAuto model code.
    Put the code inbetween the ```smauto and ``` tags."""

    write_full_model_prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            ("user", CONSTRTUCT_SMAUTO_MODEL_PROMPT),
        ]
    )

    smauto_model_chain = write_full_model_prompt_template | model | StrOutputParser()

    print("Instructing the LLM to generate an SmAuto model based on the list of devices.")

    smauto_model = smauto_model_chain.invoke(
        {"system_prompt": smauto_system_prompt.get_system_prompt(), "user_utterance": user_utterance}
    )

    # Add the user prompt to generate the model and the LLM's response to the conversation history
    history.append(("user", HumanMessagePromptTemplate.from_template(CONSTRTUCT_SMAUTO_MODEL_PROMPT).format(user_utterance=user_utterance).pretty_repr()))
    history.append(("assistant", smauto_model))

    # Save the generated model to a file
    with open(
        os.path.join(results_path, "smauto_model" + SMAUTO_FILE_NAME_EXTENSION), "w", encoding="utf-8"
    ) as file:
        file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
        file.close()
        
    print("An SmAuto model has been generated based on the list of devices and has been saved on the smauto_model.auto file.")

    # Validate the model and regenarate it if it is invalid

    validation = smauto_api.validate(
            smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
        )

    if validation.status_code == 200:
        print("The generated SmAuto model is syntactically valid.")
    else:
        
        INVALID_MODEL_PROMPT = """The model you have written is invalid.
        You should rewrite the model based on the guidelines and the error message.
        The error message contains the first error that was found in the model as it was parsed by the textX grammar.
        You will be provided with error message. 
        Each error is described with a specific format indicating the location and nature of the error.
        Your task is to identify and correct these errors. 
        The format of the error message is as follows:
        :<line>:<column>: <error_description>
        Where:
        <line> is the line number where the error occurs.
        <column> is the column number where the error starts.
        <error_description> is a detailed message describing the error.
        An * in the error description indicates the position of the error in the model.
        The error message is:
        {validation_message}
        Please make all the nessesary adjustments to the model based on the guidelines and the error message.
        Rewrite all the brokers, entities, and automations as nessesary to fix the error and improve the models functionality.
        Make sure that all the needed parentheses are included when combining conditions into a more complex ones using logical operators.
        Make sure that whenever you reference time it should using the system_clock.time variable and the time should be in the HH:MM format (24-hour clock).
        Output only the model code.
        Put the code inbetween the ```smauto and ``` tags.
        """

        invalid_model_prompt_template = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("system_prompt"),
                MessagesPlaceholder("history"),
                ("user", INVALID_MODEL_PROMPT),
            ]
        )

        invalid_model_chain = invalid_model_prompt_template | model | StrOutputParser()

        INVALID_MODEL_GENERATIONS = 1
        
        while True:
            if INVALID_MODEL_GENERATIONS == 1:
                print("The generated SmAuto model is not syntactically valid.")
            else:
                print("The regenarated model still has errors.")
            print("The SmAuto's validator response for the model is:", validation.text)
            print("Instructing the LLM to regenerate the model with the error fixed.")
            
            smauto_model = invalid_model_chain.invoke(
                {
                    "system_prompt": smauto_system_prompt.get_system_prompt(),
                    "history": history,
                    "validation_message": validation.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1],
                }
            )
            
            # Add the user prompt to regenerate the model and the LLM's response to the conversation history
            history.append(("user", HumanMessagePromptTemplate.from_template(INVALID_MODEL_PROMPT).format(validation_message=validation.text).pretty_repr()))
            history.append(("assistant", smauto_model))
            
            # Save the regenerated model to a file
            with open(
                os.path.join(
                    results_path,
                    "regenerated_smauto_model_" + str(INVALID_MODEL_GENERATIONS) + SMAUTO_FILE_NAME_EXTENSION,
                ),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
                file.close()
            print("The SmAuto model has been regenerated and saved at regenerated_smauto_model_" + str(INVALID_MODEL_GENERATIONS) + SMAUTO_FILE_NAME_EXTENSION + " file.")
            
            INVALID_MODEL_GENERATIONS += 1
            
            # Exit the loop if the max number of iterations is reached
            if INVALID_MODEL_GENERATIONS == 5:
                print("After " + str(INVALID_MODEL_GENERATIONS) + "attemps to regenerate the SmAuto model with the errors fixed the model remains invalid.")
                print("Terminating the regeneration process.")
                break
            
            # Validate the regenerated model
            validation_regen = smauto_api.validate(
                smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
            )
            
            # Exit the loop if the regenerated model is syntactically valid of if the same error appeared two consecutive times
            if validation_regen.status_code == 200:
                print("All errors have been corrected successfully.")
                print("The regenerated model is syntactically valid.")
                break
            else:
                if validation.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1] == validation_regen.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1]:
                    print("After the regeneration of the model, the same error was found. Therefore the assistant is unable to fix the error.")
                    break
            validation = validation_regen

if __name__ == "__main__":
    # User input
    print("Enter the description of the SmAuto model you would like to create:")
    user_input = input()
    generate_smauto_model(user_input)