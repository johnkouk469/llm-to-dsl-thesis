"""
Module that contains all the prompts that are going to be given to the LLM 
as intructions for dFlow model generation.
"""

import os
import json

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma

import dflow_api

DFLOW_MODELING_GUIDELINES = """
# dFlow Modeling Guidelines

These guidelines provide a detailed overview of the dFlow modeling language, covering the primary components and their syntax. The guidelines ensure you can create accurate and efficient dFlow models.
Pay attention to every little detail. You will be asked to follow these guidelines to write a dFlow model.

## Overview of dFlow Concepts

A task-oriented Virtual Assistant incorporates the following two components: the Natural Language Understanding (NLU) component, which is responsible for processing user utterances and interpreting the user’s goals or intents, and the Natural Language Generation (NLG) part, which is responsible for creating the most appropriate responses and actions. The root meta-model of dFlow consists of six concepts: Entity, Synonym, Trigger, EService, Global Slot, and Dialogues. Entity, Synonym and Trigger capture the NLU part of the assistant, EService and Global Slot define reusable features in general scope, and Dialogues describe the dialogue flows and the assistant responses, encompassing the entire NLG component. A dFlow model incorporates these concepts at the root scope and can be utilized to define the interactive part and include bot responses to predefined conditions (e.g., an internal intent is triggered).

### **Entities**
Entities are structured pieces of information inside a user message that can be extracted and used by the assistant. They can be real-world objects or meanings, such as a person, a location, an organization, or a product. DFlow can employ pre-trained Named Entity Recognition (NER) models that can efficiently extract those types of entities without further training. Frequently, Virtual Assistants need to detect use-case-specific information not supported by pre-trained NER models, such as types of food or fruits. In this case, Trainable Entities can be specified and trained during deployment given a set of entity examples.

The textX grammar for defining entities is as follows:

```
Entity: TrainableEntity | PretrainedEntity;

TrainableEntity:
    'Entity' name=ID
        words+=Words[',']
    'end'
;

PretrainedEntity:
    'PERSON'        |
    'NORP'          |
    'FAC'           |
    'ORG'           |
    'GPE'           |
    'LOC'           |
    'PRODUCT'       |
    'EVENT'         |
    'WORK_OF_ART'   |
    'LAW'           |
    'LANGUAGE'      |
    'DATE'          |
    'TIME'          |
    'PERCENT'       |
    'MONEY'         |
    'QUANTITY'      |
    'ORDINAL'       |
    'CARDINAL'
;

Words:
    /[-\w ]*\b/
;
```

a. Pre-trained Entities
These are standard entities that come with pre-trained models. They are defined using specific keywords:
PERSON: "People, including fictional",
NORP: "Nationalities or religious or political groups",
FAC: "Buildings, airports, highways, bridges, etc.",
ORG: "Companies, agencies, institutions, etc.",
GPE: "Countries, cities, states",
LOC: "Non-GPE locations, mountain ranges, bodies of water",
PRODUCT: "Objects, vehicles, foods, etc. (not services)",
EVENT: "Named hurricanes, battles, wars, sports events, etc.",
WORK_OF_ART: "Titles of books, songs, etc.",
LAW: "Named documents made into laws.",
LANGUAGE: "Any named language",
DATE: "Absolute or relative dates or periods",
TIME: "Times smaller than a day",
PERCENT: 'Percentage, including "%"',
MONEY: "Monetary values, including unit",
QUANTITY: "Measurements, as of weight or distance",
ORDINAL: '"first", "second", etc.',
CARDINAL: "Numerals that do not fall under another type",

b. Trainable Entities
In cases where the entity is domain or use-case specific such below, examples need to be given to train a new entity extractor. This is the case of a Trainable Entity, which is first defined and then included in the intents section.

An example of how to define entities in dFlow:
```dflow
entities
    Entity Doctor
        cardiologist,
        dentist,
        doc
    end
end
```

### **Synonyms**
Synonyms map various words or phrases to a single value. This is useful when users refer to the same concept in different ways. After defined, they are incorporated in the intent examples.

The textX grammar for defining synonyms is as follows:
```
Synonym:
    'Synonym' name=ID
        words+=Words[',']
    'end'
;
```

An example of how to define synonyms in dFlow:
```dflow
synonyms
    Synonym date_period
        day,
        week,
        month,
        tomorrow,
        now
    end
end
```

### **Triggers**
Triggers represent the two ways a dialogue can be initiated: when a user states a particular expression or Intent, or when an external Event is triggered, such as a reminder or a notification. In task-based dialogue systems, an Intent is a goal the user is trying to achieve or accomplish, such as retrieving specific information on the weather or setting a reminder. An Intent requires a set of phrase examples that are semantically similar to the expected user expressions. These examples can consist of combinations of text, pre-trained and trainable entities, as well as synonyms. Events are system-initiated and do not need any phrase examples.

The textX grammar for defining triggers is as follows:
```
Trigger: Intent | Event;
```

#### **Intents**
In a given user message, the thing that a user is trying to achieve or accomplish (e,g., greeting, ask for information) is called an Intent. An intent has a group of example user phrases with which an NLU model is trained, that consist of strings, references to Trainable Entities (TE), to Synonyms (S), and to Pretrained Entities (PE). Regarding the PEs, users can also give example words inside the brackets apart from the entity category (e.g., PE:PERSON["John"]).

The textX grammar for defining intents is as follows:
```
Intent:
    'Intent' name=ID
        phrases+=IntentPhraseComplex[',']
    'end'
;
IntentPhraseComplex: phrases+=IntentPhrase;

IntentPhrase:
    IntentPhraseStr |
    IntentPhraseSynonym |
    TrainableEntityRef |
    PretrainedEntityRef
;

IntentPhraseStr: STRING;

TrainableEntityRef: 'TE:' entity=[TrainableEntity|FQN|^entities*];

PretrainedEntityRef: 'PE:' entity=[PretrainedEntity|FQN|^entities*] ('[' refPreValues*=STRING[','] ']')?;

IntentPhraseSynonym: 'S:' synonym=[Synonym|FQN];
```

An example of how to define intents in dFlow:
In the code block below a simple intent called greet has been added, which contains example messages like "Hi", "Hey" and "Good morning", and a more complex one called find_person that uses all the possible references.

```dflow
triggers
    Intent greet
        "hey",
        "hello",
        "hi",
        "good morning",
        "good evening",
        "hey there",
        "Hey",
        "Hi there",
    end
    Intent find_person
        "I want" TE:name "please",
        TE:name "please!",
        "I want to call" TE:name,
        "I want call" TE:name S:date_period,
        "call" TE:name "now",
        "I want to call" PE:PERSON "immediately",
        "call" PE:PERSON["John"] "now"
    end
end
```

#### **Events**
Events are external triggers, such as IoT events, notifications or reminders. An event is defined by its name and the URI from which it is triggered.

The textX grammar for defining events is as follows:
```
Event:
    'Event' name=ID
        uri=STRING
    'end'
;
```

An example of how to define events in dFlow:
```dflow
triggers
  Event external_1
    "bot/event/external_1"
  end
end
```

### **EServices**
External services are REST endpoints that can be used as part of the VA’s responses. Their URL and HTTP method are defined globally as static attributes, while their parameters can be specified inside the dialogue section when called, in a more dynamic manner.

The textX grammar for defining external services is as follows:
```
EServiceDef: EServiceDefHTTP;

EServiceDefHTTP:
    'EServiceHTTP' name=ID
        (  'verb:' verb=HTTPVerb
          'host:' host=STRING
          ('port:' port=INT)?
          ('path:' path=STRING)?
        )#
    'end'
;

HTTPVerb:
    'GET'   |
    'POST'  |
    'PUT'
;
```

An example of how to define external services in dFlow:
```dflow
eservices
    EServiceHTTP weather_svc
        verb: GET
        host: 'r4a.issel.ee.auth.gr'
        port: 8080
        path: '/weather'
    end
end
```

### **Global Slots**
Slots are static information an assistant can access and use, offering multi-turn conversations, memory, and personalization. DFlow introduces the GSlot concept to define variables in the global scope so they can be accessed by various Dialogues, Forms, and Actions.

The textX grammar for defining global slots is as follows:
```
GlobalSlotValue: ParameterValue;
GlobalSlotType: ParameterTypeDef;
GlobalSlotRef: slot=[GlobalSlot|FQN|^gslots];
GlobalSlotIndex: FormParamRef('['ID('.'ID)*']')?;

GlobalSlot:
    name=ID ':' type=GlobalSlotType ('=' default=GlobalSlotValue)?
;
```

An example of how to define global slots in dFlow:
```dflow
gslots
    location: str = "New York",
    temperature: int = 25
end
```

### **Dialogues**
An important concept of the dFlow meta-model is the Dialogue. Dialogues are conversational flows the assistant supports in the form of trigger and response pairs, where each response is a sequence of Forms and ActionGroups in a one-turn conversation manner. Each trigger initiates only one dialogue. 

The textX grammar for defining dialogues is as follows:
```
Dialogue:
    'Dialogue' name=ID
        'on:' onTrigger+=[Trigger|FQN|^triggers][',']
        'responses:' responses+=Response[',']
    'end'
;

Response: ActionGroup | Form;
```

An example of how to define dialogues in dFlow:
```dflow
dialogues
    Dialogue DialA
        on: external_1
        responses:
            ActionGroup hey_answers
              Speak('Hello')
	      Speak('Hey there!!')
            end
    end

    Dialogue DialB
        on: find_doctor
        responses:
            Form AF1
                slot1: str = HRI('Give parameter 1', [PE:PERSON])
                slot2: str = HRI('Give parameter 2',
                    [find_doctor:True, external_1:False])
                slot3: str = HRI('Give parameter 3 you' AF1.slot1, [TE:Doctor])
            end,
            ActionGroup answers
              Speak('Hello' AF1.slot1 'how are you')
            end
    end
end
```

#### **Forms**
Regarding the responses, a Form is a conversational pattern to collect information and store it in form parameters or form slots, following business logic. Two interaction methods are supported by the dFlow DSL: Human-Robot Interaction (HRI) and External Services. Information can be collected via HRI, where the assistant sequentially collects the information from the user by requesting each slot using specified text and extracting data from the user expression. The expression can contain the entire text, an extracted entity, or a specific value set mapped to a particular intent. The second choice is the EServiceSource interaction, where the slot is filled with information received from an external service, previously defined as an EService in the dFlow model.
A Form is a conversational pattern to collect information and store it in form parameters or slots following business logic. Information can be collected via an HRI interaction, in which the assistant collects the information from the user. It requests each slot using a specific text and extracts the data from the user expression. It can contain the entire processed text (the extract variable is not filled), an extracted entity, or a specific value set in case the user states a particular intent. The second choice is the EServiceParamSource interaction, in which the slot is filled with information received from an external service, that is defined above. Each slot is of one of the 6 types: int, float, str, bool, list or dict.

The textX grammar for defining forms is as follows:
```
Form:
    'Form' name=ID
        params+=FormParam
    'end'
;

FormParam:
    name=ID ':' type=ParameterTypeDef '=' source=FormParamSource
;

FormParamRef: param=[FormParam|FQN|^dialogues*.responses.params];
FormParamIndex: FormParamRef('['ID('.'ID)*']')?;

FormParamSource: HRIParamSource | EServiceParamSource;

HRIParamSource:
    'HRI' '(' askSlot+=Text (',' '['extract+=ExtractionSource[','] ']')? ')'
;

ExtractionSource: ExtractFromEntity | ExtractFromIntent;
ExtractFromIntent: intent=[Trigger|FQN|^triggers*] ':' value=ParameterValue;
ExtractFromEntity: TrainableEntityRef | PretrainedEntityRef;

EServiceParamSource: EServiceCallHTTP;
```

#### **Action Groups**
An ActionGroup is a set of Actions. The dFlow language supports five different types of actions: the assistant can state a given phrase (SpeakAction), fire a broker event (FireEventAction), call a REST endpoint (RESTCallAction), set a global slot (SetGSlot) or form slot (SetFSlot) with particular parameters. Actions can also use real-time environment parameters and functions grouped as UserProperties and SystemProperties. User properties are user information stored locally on the device that the assistant can use, such as the name, surname, age, email, phone, city, and address. System properties are in-built system functions to get the current time, location, and a random integer or float. This allows the assistant to access data while being device-agnostic and offering more dynamic and personalized dialogues.

An action is an assistant response that can either:
Speak a specific text (SpeakAction)
Fire an Event (FireEventAction)
Call an HTTP endpoint (RESTCallAction)
Set a global or form slot with a specific value (SetFSlot and SetGSlot) (more on form slots in the forms section.)
An ActionGroup is a collection of actions that are executed sequentially.

Actions can also use real-time environment parameters, or data in general, grouped as user and system properties. User properties are user information stored locally in the device that the assistant can use, such as name, surname, age, email, phone, city and address. System properties are in-built system functions to get the current time, location, a random integer of float. That way the assistant has access to those data being device-agnostic on the same time.

The textX grammar for defining actions is as follows:
```
ActionGroup:
    'ActionGroup' name=ID
        actions+=Action
    'end'
;

Action:
    SpeakAction     |
    FireEventAction |
    RESTCallAction  |
    SetFormSlot     |
    SetGlobalSlot
;

SpeakAction:
    'Speak' '(' text+=Text ')'
;

SetFormSlot:
    'SetFSlot' '(' slotRef=FormParamRef ',' value=ParameterValue ')'
;

SetGlobalSlot:
    'SetGSlot' '(' slotRef=GlobalSlotRef ',' value=ParameterValue ')'
;

FireEventAction:
    'FireEvent' '(' uri+=Text ',' msg+=Text ')'
;

RESTCallAction: EServiceCallHTTP;


EServiceCallHTTP:
    eserviceRef=[EServiceDef|FQN|eservices]'('
        (
        ('query=' '[' query_params*=EServiceParam[','] ']' ',')?
        ('header=' '[' header_params*=EServiceParam[','] ']' ',')?
        ('path=' '[' path_params*=EServiceParam[','] ']' ',')?
        ('body=' '[' body_params*=EServiceParam[','] ']' ',')?
        )#
    ')' ('[' response_filter=EServiceResponseFilter ']')?
;

EServiceParam: name=ID '=' value=ParameterValue;

ParameterValue:
    INT                 |
    FLOAT               |
    STRING              |
    BOOL                |
    List                |
    Dict                |
    EnvPropertyDef		  |
    FormParamIndex      |
    GlobalSlotIndex     |
	Text
;

ParameterTypeDef:
    'int'   |
    'float' |
    'str'   |
    'bool'  |
    'list'  |
    'dict'
;

EServiceResponseFilter: ID('.'ID)*;

DictItem:
    name=ID ':' value=DictTypes
;

DictTypes:
  NUMBER | STRING | BOOL | Dict | List | FormParamIndex | GlobalSlotIndex
;

Dict:
    '{' items*=DictItem[','] '}'
;

List:
    '[' items*=ListElements[','] ']'
;

ListElements:
    NUMBER | STRING | BOOL | List | Dict | FormParamIndex | GlobalSlotIndex
;

Text: TextStr | EnvPropertyDef | FormParamIndex | GlobalSlotIndex;

TextStr: STRING;

EnvPropertyDef: UserPropertyDef | SystemPropertyDef;
UserPropertyDef: 'USER:' property=[UserProperty|FQN];
SystemPropertyDef: 'SYSTEM:' property=[SystemProperty|FQN];

UserProperty:
    'NAME'      |
    'SURNAME'   |
    'AGE'       |
    'EMAIL'     |
    'PHONE'		  |
    'CITY'		  |
    'ADDRESS'
;

SystemProperty:
	'TIME'			  |
	'LOCATION'		|
	'RANDOM_INT'	|
	'RANDOM_FLOAT'
;
```

### **Access Control**
DFlow supports a fully functional user access control mechanism integrated into the generated bots. Using a Role-Based Access Control methodology, the bot's developer can create roles with different permissions for executing the bot's actions and assign them to users. In this way, dFlow enables the enforcement of the least privilege principle by allowing the separation of user access levels to the bot's resources. It also empowers the development of complex dialogue flows, providing the means to differentiate the bot's behavior according to the user's role.

The textX grammar for defining access controls is as follows:
```
AccessControlDef:
    (
    roles=Roles
    policies*=Policy
    (path=Path)?
    (users=Users)?
    authentication=Authentication
    )#
;
```

An example of how to define access controls in dFlow:
```dflow
access_controls
    Roles
        role1,
        role2

        default: 
            role2
    end

    Users
        role1:
            role1@email.com

        role2: 
            role2@email.com
    end

    Policy all_actions_policy
        on: 
            all_actions
        roles: 
            role1
    end

    Path
        "/home/user/db/users/user_roles_policies.txt"
    end

    Authentication
        method: user_id
    end
end
```

#### **Roles**
Roles are granted permissions for accessing bot's actions. Each user can have one or multiple roles, inheriting all of their permissions. When defining the available roles, a default role should be provided, which is the role an unidentified/unauthenticated user will acquire.

The textX grammar for defining roles is as follows:
```
Roles:
    'Roles'
        words+=Words[',']
        'default:'
            default=Words
    'end'
;
```

#### **Users**
Each role can be assigned to one or many authenticated users. Users are dinstinguished by their unique id such as their email. To see the supported identifiers check the authentication section.

The textX grammar for defining users is as follows:
```
Users:
    'Users'
        roles+=UserRoles
    'end'
;
UserRoles: role=Word ':' users+=WordExt[','];
```
Users entity can be skipped if loading the user-role mappings from a file or a database is needed. This file should be specified using a path entity. DFlow supports user-role mappings imported from text files that conform with the following JSON format:
{
	"role1": ["user1_identifier"],
	"role2": ["user2_identifier", "user3_identifier"],
}

#### **Policies**
Access control policies serve as the basic layer of access control. They function to enforce a permit/deny access control on ActionGroups, ensuring that only the roles explicitly declared within the policy can execute the associated ActionGroup. It's important to note that each policy is dedicated to a single ActionGroup. In cases where there is no policy assigned to a particular ActionGroup, all roles gain the ability to execute it.

The textX grammar for defining policies is as follows:
```
Policy:
    'Policy' name=ID
        'on:' actions+=Words[',']
        'roles:' roles+=Words[',']
    'end'
;
```

##### **Inline Policies**
To enable enhanced access control capabilities, dFlow also supports inline policies. These enable the developer to control the flow of the conversation and differentiate the executed actions within an ActionGroup.

An inline policy example within a Dialogue is shown below:
```dflow 
ActionGroup inform_system_parameters
    Speak("System's HostName:" SYSTEM: HOSTNAME "\nSystem's public IP" SYSTEM: PUBLIC_IP)[roles=user_admin]
    Speak("Sorry, only admins are allowed to perform this action")[roles=user_paid]
end
```

Keep in mind that for the inline action policies to work, the user must first have access to the associated ActionGroup.

#### **Authentication**
For an authorization and access control mechanism to function properly, the physical users must first be assosciated with their digital identity. This is usually achieved via authentication. DFlow supports two different types of authentication schemes:

Third party authentication: This allows a third party connector to authenticate users. The default attribute used for user identification is their emails, fetched from the connector's channel.

User ID authentication: This allows the user to be authenticated by utilizing the sender_id variable of the RASA API.

Slot authentication: This allows the user to be authenticated using the content of a slot, which is filled during the conversation with the user, enabling the utilization of voice passwords.

The textX grammar for defining authentication is as follows:
```
Authentication:
    'Authentication'
        'method:' method=AuthMethods
        (
        'slot_name:' slot_name=Words
        )?
    'end'
;

AuthMethods: 'slot' | 'user_id' | 'slack' | 'telegram';
```

#### **Path**
Optional. User-role mappings are stored inside the file provided in the path entity. If the file doesn't exist it will be automatically generated. If no users entity is provided, dFlow will assume that the user-role mappings already exist within this file and attempt to load them.

The textX grammar for defining paths is as follows:
```
Path:
    'Path'
        path=STRING
    'end'
;
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

DFLOW_DESCRIPTION = """
dFlow is a Domain Specific Language (DSL) designed for creating complex automation scenarios for dialogue systems, particularly for managing conversations, integrations, and access controls.
This DSL facilitates the development of chatbots by providing a framework for defining complex conversational patterns and integrating with external services.
dFlow models consist of several components, including Entities, Synonyms, Triggers, Dialogues, EServices, Access Controls, and Global Slots, each with its own syntax and set of properties.
dFlow models can be used to create task-based dialogue flows, virtual assistants, and chatbots for various applications and platforms.
dFlow models are used to generate Rasa models, which are used to build conversational AI applications.

dFlow is a Domain-Specific Language (DSL) designed for creating Virtual Assistants (VAs) in a low-code, system-agnostic manner. The core objective of dFlow is to simplify VA development, making it accessible to both experienced developers and citizen developers (individuals with minimal programming experience). This DSL offers a textual interface and leverages open-source technologies to create task-specific VAs efficiently.

## Key Components of dFlow:

### Entities & Synonyms:

Entities: These represent structured pieces of information (e.g., person, location) extracted from user messages. dFlow supports both pre-trained Named Entity Recognition (NER) models for common entity types and trainable entities for domain-specific information.
Synonyms: Words or phrases that map to the same underlying value. They are used to ensure the assistant understands different expressions of the same concept.

### Triggers:

Triggers define how a dialogue is initiated. This could be a user statement (intent) or an external event (e.g., a reminder).
Intents: Goals or tasks that the user wants to achieve, such as asking for weather information. Each intent is associated with example phrases.
Events: System-driven triggers like notifications that don’t require user input.

### EServices:

External services (REST APIs) are defined as reusable components within the dFlow model. They are called dynamically in dialogues to fetch or send data as needed.

### Global Slots:

Static information stored and accessed by the assistant throughout the conversation. These enable multi-turn dialogues and maintain memory across interactions.

### Dialogues:

Dialogues represent conversational flows. They consist of triggers paired with responses, which can include actions such as speaking, making API calls, or setting slot values.
Forms: Used to gather information sequentially from the user or from external services.
ActionGroups: Collections of actions that the assistant performs, such as responding to the user or interacting with services.

### Access Controls:

dFlow is the first Domain-Specific Language (DSL) to integrate a fully functional user access control mechanism directly into the generated bots. This feature employs a Role-Based Access Control (RBAC) methodology, allowing developers to create distinct roles with specific permissions for executing the bot's actions and assigning these roles to users. This enables the least privilege principle, where users are only granted access to the resources they need, enhancing security and control. Additionally, dFlow’s access control allows the development of complex dialogue flows, where the bot’s behavior adapts dynamically based on the user's role.
"""

SYSTEM_ROLE = """
I am an AI Assistant that can write dflow models. 
dFlow is a DSL designed for creating task-based dialogue flows, particularly suited for virtual assistants in smart environments.
I can generate dFlow models based on user requirements and guidelines.
I generate dFlow models using the provided user utterance and following the dFlow modeling guidelines.
Before I generate a dFlow model, I am taking my time to understand the user requirements and guidelines to ensure the model is accurate and efficient.
Before I generate a dflow model, I am taking my time to figure out which of the dFlow concepts are needed to satisfy the user requirements.
I always follow the provided instructions and guidelines 
to ensure the model is valid and can be parsed by the provided textX grammar."""

CONSTRTUCT_DFLOW_MODEL_PROMPT = """
Write a dFlow model that will satisfy the following requirements:
{user_utterance} 

Follow the guidelines provided for each component to ensure the model is correctly structured.

Output only the dflow model code.
Put the code inbetween the ```dflow and ``` tags."""


def get_few_shot_examples():
    """Returns the few-shot examples for dFlow model generation."""
    descriptions_file_path = os.path.join(
        dflow_api.FEW_SHOT_MODELS_PATH, "descriptions.json"
    )

    try:
        with open(descriptions_file_path, "r", encoding="utf-8") as f:
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
                few_shot_examples.append(
                    ({"input": descriptions[file], "output": data})
                )
        except FileNotFoundError:
            print("File not found.")
        except IOError:
            print("Error reading the file.")

    to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        to_vectorize, embeddings, metadatas=few_shot_examples
    )

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

    return few_shot_prompt_template


def get_system_prompt():
    """Returns the system prompt containing the guidelines for dFlow model generation"""
    system_prompt = [
        ("system", DFLOW_DESCRIPTION),
        ("system", DFLOW_MODELING_GUIDELINES),
        ("system", SYSTEM_ROLE),
    ]
    return system_prompt
