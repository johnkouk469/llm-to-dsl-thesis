"""Python module for AI Assistant writing SmAuto models."""

import os
import time
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

# Load the files used to construct the prompts from the dflow paper
with open("./dflow_paper.txt", "r", encoding="utf-8") as file:
    dflow_paper = file.readlines()


generate_va_concept_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", dflow_prompts.SYSTEM_ROLE),
        ("system", dflow_prompts.DFLOW_USECASES),
        ("system", dflow_paper),
        ("user", dflow_prompts.VA_CONCEPT_GENERATOR_PROMPT),
    ]
)


va_concept_chain = generate_va_concept_prompt_template | model | StrOutputParser()

va_concept = va_concept_chain.invoke({"input": "Help me create a virtual assistant."})

with open(
    os.path.join(LOGS_FOLDER, timestamp + "_va_concept.txt"), "w", encoding="utf-8"
) as file:
    file.write(va_concept)
    file.close()




write_dflow_model_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", dflow_prompts.SYSTEM_ROLE),
        ("system", dflow_paper),
        ("system", dflow_prompts.DFLOW_USECASES),
        ("system", dflow_prompts.DFLOW_MODELING_GUIDELINES),
        ("system", dflow_prompts.WRITE_DFLOW_MODEL),
        dflow_prompts.get_few_shot_examples(),
        ("user", dflow_prompts.CONSTRTUCT_DFLOW_MODEL_PROMPT),
    ]
)

dflow_model_chain = write_dflow_model_prompt_template | model | StrOutputParser()

with open(
    os.path.join(dflow_api.DFLOW_EXAMPLES_PATH, "template.dflow"),
    "r",
    encoding="utf-8",
) as file:
    template_dflow = file.readlines()

dflow_model = dflow_model_chain.invoke(
    {
        "input": str("Help me create a virtual assistant."),
        "va_concept": va_concept,
        "template_dflow": template_dflow,
    }
)

with open(
    os.path.join(LOGS_FOLDER, timestamp + "_dflow_model.dflow"), "w", encoding="utf-8"
) as file:
    file.write(dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX))
    file.close()

validation = dflow_api.validate(
    dflow_model.removeprefix(CODE_PREFIX).removesuffix(CODE_SUFFIX)
)
print(validation.text)
