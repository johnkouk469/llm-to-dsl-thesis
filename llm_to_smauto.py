"""Python module for AI Assistant writing SmAuto models."""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
CONSTRTUCT_FULL_SMAUTO_MODEL_PROMPT = """
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
        ("user", CONSTRTUCT_FULL_SMAUTO_MODEL_PROMPT),
    ]
)

full_smauto_model_chain = write_full_model_prompt_template | model | StrOutputParser()

full_smauto_model = full_smauto_model_chain.invoke(
    {"system_prompt": smauto_system_prompt.get_system_prompt(), "smart_enviroment": devices}
)

with open(
    os.path.join(results_path, "full_smauto_model.auto"), "w", encoding="utf-8"
) as file:
    file.write(full_smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
    file.close()


PLAN_TO_IMPROVE_MODEL = """
Read the SmAuto model you have written and think of ways to improve it.
Come up with ideas on how to make the model more efficient, scalable, and robust.
Think about additional features, optimizations, or changes that could enhance the functionality of the model.
Output only the plan to improve the model."""

plan_to_improve_model_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", smauto_system_prompt.SYSTEM_ROLE),
        ("user", PLAN_TO_IMPROVE_MODEL),
    ]
)

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

# invalid_model_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_ROLE),
#         ("system", DEFINE_BROKERS),
#         ("user", DEFINE_BROKERS_PROMPT),
#         ("assistant", brokers),
#         ("system", DEFINE_ENTITIES),
#         ("user", DEFINE_ENTITIES_PROMPT),
#         ("assistant", entities),
#         ("system", DEFINE_AUTOMATIONS),
#         ("system", SYSTEM_CLOCK_GUIDELINES),
#         ("user", DEFINE_AUTOMATIONS_PROMPT),
#         ("assistant", automations),
#         ("system", DEFINE_METADATA_RTMONITOR),
#         ("system", WRITE_SMAUTO_MODEL),
#         ("user", CONSTRTUCT_SMAUTO_MODEL_PROMPT),
#         ("assistant", smauto_model),
#         ("user", INVALID_MODEL_PROMPT),
#     ]
# )

# invalid_model_chain = invalid_model_prompt_template | model | StrOutputParser()

validation = smauto_api.validate(
    full_smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
)
print(validation.text)
# INVALID_MODEL_GENERATIONS = 0
# while True:
#     validation = smauto_api.validate(
#         smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
#     )
#     if validation.status_code == 200:
#         print("The generated SmAuto model is syntactically valid.")
#         break
#     print("The SmAuto's validator response for the model is:", validation.text)
#     smauto_model = invalid_model_chain.invoke(
#         {
#             "devices": devices,
#             "validation_message": validation.text,
#         }
#     )
#     print(smauto_model)
#     with open(
#         os.path.join(
#             results_path,
#             "smauto_" + str(INVALID_MODEL_GENERATIONS) + "_model.auto",
#         ),
#         "w",
#         encoding="utf-8",
#     ) as file:
#         file.write(smauto_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
#         file.close()
#     INVALID_MODEL_GENERATIONS += 1
#     if INVALID_MODEL_GENERATIONS == 5:
#         break
