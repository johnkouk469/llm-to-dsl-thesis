"""Python module for AI Assistant writing SmAuto models."""

import os
import time
import json

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma

from dotenv import load_dotenv

import dflow_api

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

DFLOW_MODELING_GUIDELINES = """
dFlow Modeling Guidelines

These guidelines provide a detailed overview of the dFlow modeling language, covering the primary components and their syntax. The guidelines ensure you can create accurate and efficient dFlow models, which are then transformed into ready-to-deploy Rasa models.

Overview of dFlow Components

A dFlow model incorporates several key components: Entities, Synonyms, Triggers, EServices, Global Slots, and Dialogues. These components define the NLU (Natural Language Understanding) and NLG (Natural Language Generation) aspects of a Virtual Assistant (VA).

1. Entities
Entities are structured pieces of information extracted from user messages. They can represent real-world objects like persons, locations, organizations, products, etc. Entities are divided into two types:

a. Pre-trained Entities
These are standard entities that come with pre-trained models (e.g., SpaCy). They are defined using specific keywords:

```dflow
entities
    Entity PretrainedEntity
        PERSON,
        NORP,
        LOC
    end
end
```

b. Trainable Entities
These require examples for custom training:

```dflow
entities
    Entity CustomEntity
        example1,
        example2,
        example3
    end
end
```

2. Synonyms
Synonyms map various words or phrases to a single value. This is useful when users refer to the same concept in different ways:

```dflow
synonyms
    Synonym time_period
        day,
        week,
        month
    end
end
```

3. Triggers
Triggers define how a dialogue can be initiated, either by user intents or external events.

a. Intents
Intents represent user goals and require example phrases:

```dflow
triggers
    Intent greet
        "hello",
        "hi",
        "hey"
    end
end
```
b. Events
Events are system-initiated triggers:

```dflow
triggers
    Event reminder
        "bot/event/reminder"
    end
end
```

4. EServices
External services are REST endpoints used in the VA’s responses. They are defined globally and called dynamically within dialogues:

```dflow
eservices
    EServiceHTTP weather_service
        verb: GET
        host: "http://api.weather.com"
        path: "/current"
    end
end
```

5. Global Slots
Global slots store static information that the assistant can access across multiple dialogues:

```dflow
gslots
    location: str = "New York",
    temperature: int = 25
end
```
6. Dialogues
Dialogues define the conversational flows, consisting of triggers and responses. Responses can be forms or action groups.

a. Forms
Forms collect information from users:

```dflow
dialogues
    Dialogue GetWeather
        on: weather_intent
        responses:
            Form WeatherForm
                location: str = HRI("Please provide your location", [PE:LOC]),
                date: str = HRI("For which date?", [S:time_period])
            end,
            ActionGroup WeatherActions
                Speak("Getting weather for" WeatherForm.location "on" WeatherForm.date)
            end
    end
end
```

b. Action Groups
Action groups define actions like speaking, firing events, calling REST services, etc.:

```dflow
dialogues
    Dialogue GreetUser
        on: greet
        responses:
            ActionGroup GreetActions
                Speak("Hello! How can I help you today?")
            end
    end
end
```

7. Access Control (Optional)
Define user roles and permissions for executing certain actions:

```dflow
access_controls
    Roles
        admin,
        user
        default: user
    end
    Policies
        Policy1
            role: admin
            actions: [GetWeather, GreetUser]
        end
    end
    Users
        admin_user: admin,
        regular_user: user
    end
    Authentication
        basic_auth
    end
end
```

Example Models
Example 1: Greeting and Weather Inquiry

```dflow
entities
    Entity Location
        New York,
        Los Angeles,
        San Francisco
    end
end

synonyms
    Synonym time_period
        today,
        tomorrow,
        weekend
    end
end

triggers
    Intent greet
        "hello",
        "hi",
        "hey"
    end
    Intent ask_weather
        "What's the weather in TE:Location?",
        "How's the weather in TE:Location?"
    end
end

eservices
    EServiceHTTP weather_service
        verb: GET
        host: "http://api.weather.com"
        path: "/current"
    end
end

gslots
    location: str,
    date: str = "today"
end

dialogues
    Dialogue GreetUser
        on: greet
        responses:
            ActionGroup GreetActions
                Speak("Hello! How can I assist you today?")
            end
    end

    Dialogue GetWeather
        on: ask_weather
        responses:
            Form WeatherForm
                location: str = HRI("Which location?", [TE:Location]),
                date: str = HRI("For when?", [S:time_period])
            end,
            ActionGroup WeatherActions
                RESTCall(weather_service(query=[location=WeatherForm.location, date=WeatherForm.date])),
                Speak("Fetching weather for" WeatherForm.location "on" WeatherForm.date)
            end
    end
end
```

Example 2: Telling Jokes

```dflow
triggers
    Intent tell_joke
        "Tell me a joke",
        "Make me laugh",
        "Do you know any good jokes?"
    end
end

eservices
    EServiceHTTP joke_service
        verb: GET
        host: "http://api.jokes.com"
        path: "/random"
    end
end

dialogues
    Dialogue TellJoke
        on: tell_joke
        responses:
            ActionGroup JokeActions
                RESTCall(joke_service()),
                Speak("Here's a joke for you: <response from joke_service>")
            end
    end
end
```

These examples follow the detailed guidelines provided and demonstrate how dFlow models can be written for various functionalities of a Virtual Assistant. 
Ensure your models align with these structures to maximize the efficiency and effectiveness of your VA development.
"""

WRITE_DFLOW_MODEL = """
DFlow is a Domain Specific Language (DSL) designed for creating complex automation scenarios for dialogue systems, particularly for managing conversations, integrations, and access controls. Below are comprehensive guidelines to assist you in writing effective DFlow models.

General Structure
A DFlow model consists of several main components:

Metadata: Contains meta-information about the model.
Entities: Represents both trainable and pretrained entities.
Synonyms: Defines synonym sets for entities.
Triggers: Specifies the intents and events that trigger dialogues.
Dialogues: Defines the logic and responses for dialogues.
EServices: Describes external service integrations.
Access Controls: Defines roles, policies, and authentication methods.
Connectors: Defines integrations with external communication platforms.
Each component has its own syntax and set of properties. Follow the detailed instructions below to structure each part of your DFlow model correctly.
"""

SYSTEM_ROLE = """
I am an AI Assistant that can write dflow models. 
dFlow is a DSL designed for creating task-based dialogue flows, particularly suited for virtual assistants in smart environments.
Rasa Integration: It allows for the generation of complete Rasa models, which are used to build conversational AI applications.
Metamodel: The metamodel defines the language’s concepts, providing a structure for creating dialogue flows.
Grammar and Entities: The grammar includes entities, synonyms, services, global slots, triggers, dialogues, and actions, which are essential components for defining conversational logic.
Access Control: dFlow supports role-based access control, enabling different permissions for users and enforcing security within the dialogue flows.
This DSL facilitates the development of chatbots by providing a framework for defining complex conversational patterns and integrating with external services.
In order to write a dflow model, you need to define the entities, synonyms, services, global slots, triggers, dialogues, and actions to define the conversational logic.
I always follow the provided instructions and guidelines 
to ensure the model is valid and can be parsed by the provided textX grammar."""

DFLOW_USECASES = """
The dFlow DSL has several use cases, making it a valuable tool for creating conversational AI applications. Here are some scenarios where dFlow can be beneficial:

Virtual Assistants: dFlow is particularly well-suited for building virtual assistants that interact with users in natural language. Whether it’s answering questions, providing recommendations, or assisting with tasks, dFlow allows developers to define complex dialogue flows efficiently.
Smart Environments: In smart homes, offices, or other connected environments, dFlow can power voice-controlled interfaces. Users can interact with devices, request information, or perform actions using natural language commands.
Task Automation: dFlow can automate repetitive tasks by understanding user intent and executing relevant actions. For example, a virtual assistant built with dFlow could schedule meetings, order groceries, or control smart home devices.
Custom Chatbots: Developers can create custom chatbots tailored to specific domains or industries. Whether it’s customer support, healthcare, or finance, dFlow enables the design of chatbots with specialized knowledge.
Integration with External Services: dFlow allows seamless integration with external services, APIs, and databases. Developers can define actions that interact with these services, enhancing the capabilities of their virtual assistants.
Role-Based Access Control: With dFlow, access control can be enforced based on user roles. This is essential for maintaining security and ensuring that only authorized users can perform specific actions within the dialogue flows.
Remember that dFlow is a domain-specific language, so its primary focus is on creating effective dialogue flows. Developers can leverage its features to build powerful conversational experiences across various applications and platforms.
"""

VA_CONCEPT_GENERATOR_PROMPT = """
{input}
Think about a virtual assistant concept that you would like to create using the dFlow DSL. Give it a name and describe its functionality.
Write a detailed description of the virtual assistant, including its purpose, target users, key features, and any specific tasks it can perform.
Consider how the virtual assistant will interact with users, what services or information it will provide, and how it will handle different types of requests.

I'm providing you with small descriptions of other concepts to help you generate ideas for your virtual assistant concept.
Your description should be as it has been requested from you.

"greet": "I want to create a dialogue scenario for greeting a user. The scenario will consist of a generic intent and the assistant will respond with \"Hello there!\".",
"ask_weather": "I want to create a dialogue scenario for telling the weather forecast. The assistant will ask for the city the user wants to learn the forecast for and will call the API http://services.issel.ee.auth.gr/general_information/weather_openweather with city as a query parameter and will present the retrieved information to the user via a speak event. The retrieved information is located in the temp field of the response object. The response expression will be \"The weather forecast will be \" retrieved_temp \" for \" requested_city.",
"ask_weather_ac": "I want to create a dialogue scenario for telling the weather forecast. The assistant will ask for the city the user wants to learn the forecast for and will call the API http://services.issel.ee.auth.gr/general_information/weather_openweather with city as a query parameter and will present the retrieved information to the user via a speak event. The retrieved information is located in the temp field of the response object. The response expression will be \"The weather forecast will be \" retrieved_temp \" for \" requested_city. Retrieving the weather forecast should only be allowed to paid users and blocked for the other users. Protect this action using an explicit access control Policy and retrieve user's idenity from user_id. User-role mappings should be explicitly defined, where username1 is a paid user and username2 is a non-paid user.",
"book_appointment": "I want to create a dialogue scenario for a doctor appointment scheduling. The assistant should ask the user to provide the doctor name, date and time, and then use all these parameters to call the API. It is a post API is https://health.com/medical/book_appointment  that receives the three parameters as body params. Then, if successful it should respond to the user with \"Doctor <doctor_name> is waiting for you at <date>, <time>\".",
"book_appointment_ac": "I want to create a dialogue scenario for a doctor appointment scheduling. The assistant should ask the user to provide the doctor name, date and time, and then use all these parameters to call the API. It is a post API is https://health.com/medical/book_appointment  that receives the three parameters as body params. Then, if successful it should respond to the user with \"Doctor <doctor_name> is waiting for you at <date>, <time>\". Additionally, paid plan users should be informed that their appointment is covered by the insurance, and free plan users must be encouraged to upgrade their plan if they want insurance to cover their appointment. However, unregistered users should not be allowed to book an appointment. The users idenity should be extracted from a slack connector with token: xoxb-4883692765252-4884029447172-rH1b8v6PMj22OaTsaIQrtpfH, channel: doctor_assistant and signing_secret: 123456798. User-role mappings should be retrieved from /home/Desktop/users_db.txt",    
"remind_medicine": "Please write a dialogue for notifing a user for their medication of the day. The list is in https://health.com/profile/medication_list in response parameter medication. Create an intent with possible user utterances, call the api and state the message: \"Today you have to take\" <medication> .",
"open_window": "I want an assistant interaction for opening a smart windown. There will be an access control system that uses user's ID, and if it belongs to the user_parent role it will fire an event to uri: `/window` with message `open` and say `Sure, I am opening it right now`. If not, it will be a user_child role (which is also the default role), then it will say `I am sorry, you are not authorized`. Include the username-role mappings from the /home/Desktop/ac_policies.txt file.",
"user_profile": "Write an assistant interaction for registering a new user to our platform. We ask only for name and age and POST it to the https://platform.health.gr/user/regist API as query parameters. After that answer with 'Glad to meet you' and the name of the user.",
"user_profile_ac": "Write an assistant interaction for registering a new user to our platform. We ask only for name and age and POST it to the https://platform.health.gr/user/regist API as query parameters. After that answer with 'Glad to meet you' and the name of the user. If the user's name has already been registered, omit the post request and reply only with 'It is nice seeing you again' and the name of the user. The registered names should be retrieved from /home/Desktop/registered_users_db.txt",
"retrieve_steps": "Create a dialogue scenario for a user that wants to learn how many steps they have made in the current day. The steps are stored in the personal data registry in https://health.com/profile/steps under the 'steps' response object param. Retrieve it and say 'Today you have done <steps> steps so far.'",
"take_notes": "We want to support a note taking functionality verbally. We have an api that gets a `note` body parameter and stores it internally in the user profile. The api is a post api in https://services.issel.auth.gr/profile/notes. Ask user for what they want to save, send it to the API and respond 'OK, noted!'",
"buy_amazon": "Amazon buy assistant scenario, where user states a product (via an entity preferably) and the assistant calls the POST API: https://api.amazon.com/products/buy with body param `product`. Check the user's id if it is in parent or child role and if it a parent, do the process, send the product to the API and then say a successful response message. Otherwise, suggest them to ask their parents as they are not allowed to do that. the role username mappings are in the /home/Desktop/users.txt file.",
"log_food": "Craft a meal logging interaction. When users want to store the meal they ate, ask them the exact dish and volume they consumed, store it and send it to https://services.issel.auth.gr/profile/meal POST api as body params with names dish and volume. Respond with `Thanks for letting me know!`.",
"audiobook": "We want an audiobook assistant that users can use to listen to an audiobook whenever they greet the assistant and also change narrator's voice on demand. For now there are only two available audiobooks, Silmarilion by Tolkien and Dune by Frank Herbert, additionally there are only two available narrator voices, male and female. We distinguish the users into two roles, free (default) and paid. Free users are only given a sample of the audiobook, while paid users get the full audiobook. Additionally, free users cannot change narrator's voice and have to be encouranged to switch to a paid plan whenever they attempt to do so. To get the full/sample audiobook a GET request should be made at localhost:7777 and path full_book/sample_book respectively, with the name of the audiobook placed within the audiobook query parameter. To change narrator's voice, a PUT request at localhost:7777/change_narrator should be made, with the body parameter narrator containing the voice choice. User-role mappings should be retrieved from /home/Desktop/users.txt, while the identification of the users will be done via slack. The slack connector has the following parameters, token: xoxb-4883692765252-4884029447172-rH1b8v6PMj22OaTsaIQrtpfH, channel: audiobook_assistant and signing_secret: 123456798.",
"smart_car": "Write an assistant for a smart car that can start the engine of the vehicle and play music on demand. Only the driver should be able to start the engine, while all the passengers should be able to turn the radio on. To start the engine the event at /engine should be set to on, while to play music the event /radio should be set to on. Please inform the passengers whenever an action is performed. The default role should be the passenger. The user-role mappings must be retrieved from /home/Desktop/passengers.txt and the users must be authenticated by their user_id."
"""


generate_va_concept_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_ROLE),
        ("system", DFLOW_USECASES),
        ("system", dflow_paper),
        ("user", VA_CONCEPT_GENERATOR_PROMPT),
    ]
)


va_concept_chain = generate_va_concept_prompt_template | model | StrOutputParser()

va_concept = va_concept_chain.invoke({"input": "Help me create a virtual assistant."})

with open(
    os.path.join(LOGS_FOLDER, timestamp + "_va_concept.txt"), "w", encoding="utf-8"
) as file:
    file.write(va_concept)
    file.close()

CONSTRTUCT_DFLOW_MODEL_PROMPT = """
Write entities, synonyms, services, global slots, trtiggers, and dialogues, write the complete dflow model for the virtual assistant concept:
{va_concept}

You can use the following template of a dflow model to help you structure your model, you have to replace ## with the actual values:
{template_dflow} 

Follow the guidelines provided for each component to ensure the model is correctly structured.

Output only the dflow model code.
Put the code inbetween the ```dflow and ``` tags."""

FEW_SHOT_DESCRIPIONS = os.path.join(dflow_api.FEW_SHOT_MODELS_PATH, "descriptions.json")

try:
    with open(FEW_SHOT_DESCRIPIONS, "r", encoding="utf-8") as f:
        # Read the contents of the file
        descriptions = json.load(f)
except FileNotFoundError:
    print("File not found.")
except IOError:
    print("Error reading the file.")

files = descriptions.keys()

few_shot_examples = []

for file in files:

    try:
        with open(
            os.path.join(dflow_api.FEW_SHOT_MODELS_PATH, f"{file}.dflow"),
            "r",
            encoding="utf-8",
        ) as f:
            # Read the contents of the file
            data = f.read()
            few_shot_examples.append(({"input": descriptions[file], "output": data}))
    except FileNotFoundError:
        print("File not found.")
    except IOError:
        print("Error reading the file.")

to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shot_examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=4,
)

few_shot_prompt_template = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_prompt=ChatPromptTemplate.from_messages(
        [("user", "{input}"), ("assistant", "{output}")]
    ),
    examples=few_shot_examples,
)

write_dflow_model_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_ROLE),
        ("system", dflow_paper),
        ("system", DFLOW_USECASES),
        ("system", DFLOW_MODELING_GUIDELINES),
        ("system", WRITE_DFLOW_MODEL),
        few_shot_prompt_template,
        ("user", CONSTRTUCT_DFLOW_MODEL_PROMPT),
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
