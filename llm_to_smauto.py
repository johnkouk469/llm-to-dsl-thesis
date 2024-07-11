"""Python module for AI Assistant writing SmAuto models."""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from requests import Response

import smauto_api
import smauto_prompts

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

    write_full_model_prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            ("user", smauto_prompts.CONSTRTUCT_SMAUTO_MODEL),
        ]
    )

    smauto_model_chain = write_full_model_prompt_template | model | StrOutputParser()

    print("Instructing the LLM to generate an SmAuto model based on the list of devices.")

    smauto_model = smauto_model_chain.invoke(
        {"system_prompt": smauto_prompts.get_system_prompt(), "user_utterance": user_utterance}
    )

    # Add the user prompt to generate the model and the LLM's response to the conversation history
    history.append(("user", HumanMessagePromptTemplate.from_template(smauto_prompts.CONSTRTUCT_SMAUTO_MODEL).format(user_utterance=user_utterance).pretty_repr()))
    history.append(("assistant", smauto_model))

    # Save the generated model to a file
    with open(
        os.path.join(results_path, "smauto_model" + SMAUTO_FILE_NAME_EXTENSION), "w", encoding="utf-8"
    ) as file:
        file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
        file.close()
        
    print("An SmAuto model has been generated based on the list of devices and has been saved on the smauto_model.auto file.")
    
    validation = smauto_api.validate(
            smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
        )

    # Validate the model and regenarate it if it is invalid
    if validation.status_code == 200:
        print("The generated SmAuto model is syntactically valid.")
    else:
        smauto_model = regenerate_invalid_model(smauto_model, validation)
    
    return smauto_model
    

def regenerate_invalid_model(smauto_model: str, validation: Response) -> str:

    invalid_model_prompt_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("system_prompt"),
            MessagesPlaceholder("history"),
            ("user", smauto_prompts.INVALID_MODEL),
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
                "system_prompt": smauto_prompts.get_system_prompt(),
                "history": history,
                "validation_message": validation.json().get("detail").split(SMAUTO_FILE_NAME_EXTENSION)[1],
            }
        )
        
        # Add the user prompt to regenerate the model and the LLM's response to the conversation history
        history.append(("user", HumanMessagePromptTemplate.from_template(smauto_prompts.INVALID_MODEL).format(validation_message=validation.text).pretty_repr()))
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
    return smauto_model

if __name__ == "__main__":
    # User input
    print("Enter the description of the SmAuto model you would like to create:")
    user_input = input()
    generate_smauto_model(user_input)