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

timestamp = time.strftime("%Y%m%d-%H%M%S")
LOGS_FOLDER = "logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)
RESULTS_FOLDER = timestamp
os.makedirs(os.path.join(LOGS_FOLDER, RESULTS_FOLDER), exist_ok=True)
results_path = os.path.join(LOGS_FOLDER, RESULTS_FOLDER)

CODE_PREFIX = "```smauto\n"
CODE_SUFFIX = "\n```"

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

history = []

DEVICE_GENERATOR_PROMPT = """
Generate a list of {num_of_devices} devices that can be used in a smart environment.
The devices can be sensors, actuators, or any other device that can be used in a smart environment. 
Come up with the devices for each room of a three bedroom and two bathroom house seperately."""


generate_devices_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", smauto_system_prompt.SYSTEM_ROLE),
        ("user", DEVICE_GENERATOR_PROMPT),
    ]
)


devices_chain = generate_devices_prompt_template | model | StrOutputParser()

devices = devices_chain.invoke({"num_of_devices": 30})

with open(os.path.join(results_path, "devices.txt"), "w", encoding="utf-8") as file:
    file.write(devices)
    file.close()


## Construct the SmAuto model
CONSTRTUCT_SMAUTO_MODEL_PROMPT = """
Write brokers, entities, and automations, write the complete SmAuto model for the following smart enviroment:
{smart_enviroment}
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

smauto_model = smauto_model_chain.invoke(
    {"system_prompt": smauto_system_prompt.get_system_prompt(), "smart_enviroment": devices}
)

history.append(("user", HumanMessagePromptTemplate.from_template(CONSTRTUCT_SMAUTO_MODEL_PROMPT).format(smart_enviroment=devices).pretty_repr()))
history.append(("assistant", smauto_model))

with open(
    os.path.join(results_path, "smauto_model.auto"), "w", encoding="utf-8"
) as file:
    file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
    file.close()


## Validate the SmAuto model

INVALID_MODEL_PROMPT = """The model you have written is invalid.
You should rewrite the model based on the guidelines and the error message.
The error message contains the first error that was found in the model as it was parsed by the textX grammar.
An * indicates the position of the error in the model.
Make sure that all the needed parentheses are included when combining conditions into a more complex ones using logical operators.
Make sure that whenever you reference time it should using the system_clock.time variable and the time should be in the HH:MM format (24-hour clock).
The error message is:
{validation_message}
Please make all the nessesary adjustments to the model based on the guidelines and the error message.
Rewrite all the brokers, entities, and automations as nessesary to fix the error and improve the models functionality.
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
    validation = smauto_api.validate(
        smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
    )
    if validation.status_code == 200:
        print("The generated SmAuto model is syntactically valid.")
        break
    print("The SmAuto's validator response for the generated model is:", validation.text)
    smauto_model = invalid_model_chain.invoke(
        {
            "system_prompt": smauto_system_prompt.get_system_prompt(),
            "history": history,
            "validation_message": validation.text,
        }
    )
    history.append(("user", HumanMessagePromptTemplate.from_template(INVALID_MODEL_PROMPT).format(validation_message=validation.text)))
    history.append(("assistant", smauto_model))
    with open(
        os.path.join(
            results_path,
            "regenerated_smauto_model_" + str(INVALID_MODEL_GENERATIONS) + ".auto",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
        file.close()
    INVALID_MODEL_GENERATIONS += 1
    if INVALID_MODEL_GENERATIONS == 5:
        break
