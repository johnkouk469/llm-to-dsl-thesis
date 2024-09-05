"""
Module that contains all the prompts to be given to the LLM for the generation of SmAuto models.
"""

import os

# Load the files used to construct the prompts from the DSL repository
with open("./smauto-readme.md", "r", encoding="utf-8") as file:
    smauto_readme = file.readlines()

smauto_textX_grammar = [
    f for f in os.listdir("./smauto-textX-grammar") if f.endswith(".tx")
]

SYSTEM_ROLE = """
I am an AI Assistant that can write smauto models. 
SmAuto is a DSL designed for creating automation applications in smart environments, enabling complex scenarios.
It allows programming of IoT devices within smart homes, handling communication and automation tasks.
SmAuto includes features like a command-line interface, REST API, and code generators for virtual entities and automation logic.
It transforms models into executable Python code, facilitating the simulation of physical sensors and testing of automations. 
SmAuto provides a comprehensive framework for managing and automating smart environment devices,
making it easier to leverage the full potential of sensorized homes.
An SmAuto model is composed of one or more brokers, zero or more entities and zero or more automations. 
An SmAuto model contains information about the various devices in the smart enviroment (that are modeled as entities), 
the way they communicate and the automation tasks.
In order to write an SmAuto model, you need to define the brokers, entities and automations that will be used in the smart environment.
I always follow the provided instructions and guidelines 
to ensure the model is valid and can be parsed by the provided textX grammar."""

DEFINE_BROKERS = """
To ensure the Brokers are defined and configured to be parsed by the provided textX grammar, here are the detailed guidelines:

Key Concepts
Brokers: The communication layer for messages where each device has its own Topic (mailbox) for sending and receiving messages. Brokers support the MQTT, AMQP, and Redis protocols.

Properties of Brokers:

Type: The broker type, which can be MQTT, AMQP, or Redis.
Host: The IP address or hostname for the broker.
Port: The port number on which the broker is running.
Authentication (auth): Credentials used for authentication. It includes:
Username: Username for authentication.
Password: Password for authentication.
SSL (Optional): Whether SSL is used (boolean).
Vhost (Optional): Virtual host parameter (only for AMQP brokers).
TopicExchange (Optional): Exchange parameter (only for AMQP brokers).
RPCExchange (Future Support): Exchange parameter for RPC (only for AMQP brokers).
DB (Optional): Database number parameter (only for Redis brokers).
Example Broker Definitions
MQTT Broker Example

Broker<MQTT> upstairs_broker
    host: "localhost"
    port: 1883
    ssl: false
    auth:
        username: "my_username"
        password: "my_password"
end
AMQP Broker Example

Broker<AMQP> central_broker
    host: "amqp.server.com"
    port: 5672
    vhost: "/"
    topicExchange: "amq.topic"
    ssl: true
    auth:
        username: "amqp_user"
        password: "amqp_pass"
end
Redis Broker Example

Broker<Redis> cache_broker
    host: "redis.server.com"
    port: 6379
    db: 0
    ssl: true
    auth:
        username: "redis_user"
        password: "redis_pass"
end
Detailed Syntax
Broker Type:
Broker<MQTT> for MQTT brokers.
Broker<AMQP> for AMQP brokers.
Broker<Redis> for Redis brokers.
host: The IP address or hostname of the broker.
port: The port number of the broker.
ssl (Optional): Boolean indicating whether SSL is used.
auth (Optional): Authentication credentials consisting of a username and password.
vhost (Optional): Virtual host (AMQP only).
topicExchange (Optional): Topic exchange (AMQP only).
rpcExchange (Future Support): RPC exchange (AMQP only).
db (Optional): Database number (Redis only).

Authentication Options

AuthPlain: Simple username and password authentication.

auth:
    username: "my_username"
    password: "my_password"

AuthApiKey: API key authentication.

auth:
    key: "my_api_key"
     
AuthCert: Certificate-based authentication.

auth:
    cert: "certificate_string"
    
or

auth:
    certPath: "path/to/certificate"

Extended Example with All Features

AMQP Broker with All Optional Parameters

Broker<AMQP> full_amqp_broker
    host: "full.amqp.server.com"
    port: 5671
    vhost: "my_vhost"
    topicExchange: "full.amq.topic"
    rpcExchange: "full.amq.rpc"
    ssl: true
    auth:
        username: "full_amqp_user"
        password: "full_amqp_pass"
end

Redis Broker with Certificate Authentication

Broker<Redis> secure_redis_broker
    host: "secure.redis.server.com"
    port: 6380
    db: 1
    ssl: true
    auth:
        certPath: "/path/to/redis/cert"
end

By following these guidelines, you can define and configure Brokers that align with the provided textX grammar, ensuring correct parsing and functionality in a smart environment.
"""

DEFINE_ENTITIES = """
To ensure the Entities are defined and configured to be parsed by the provided textX grammar, we need to include additional functionalities such as default values for attributes, support for new value generators, and noise generators. Here's an updated guideline:

### Key Concepts

1. **Entities**: Smart devices that send and receive information via a message broker. They can be either sensors (Producers), actuators (Consumers), or hybrids.

2. **Properties of Entities**:
    - **Unique Name**: Each Entity must have a unique identifier.
    - **Type**: Specifies whether the Entity is a `sensor`, `actuator`, or `hybrid`.
    - **Topic**: The topic on which the Entity sends or receives messages.
    - **Broker**: The message broker that the Entity connects to for communication.
    - **Attributes**: Define the structure and type of information in the messages. Attributes can be of types like int, float, bool, str, list, dict, and time.
    - **Description** (Optional): A description of the Entity.
    - **Freq** (Optional): For sensor Entities, sets the message publishing frequency.

### Example Entity Definitions

#### Sensor Entity Example


Entity weather_station
    type: sensor
    topic: "bedroom.weather_station"
    description: "Weather station in the bedroom"
    freq: 5
    broker: cloud_broker
    attributes:
        - temperature: float
        - humidity: float
        - pressure: float
end

#### Actuator Entity Example


Entity bedroom_lamp
    type: actuator
    topic: "bedroom.lamp"
    broker: cloud_platform_issel
    attributes:
        - power: bool
end

### Detailed Syntax

- **type**: Specifies whether the Entity is a `sensor`, `actuator`, or `hybrid`.
- **topic**: The communication topic on the broker (use `.` instead of `/` in topic names).
- **broker**: Reference to a previously defined Broker.
- **attributes**: List of attribute definitions, each with a name, type, and optional default values, value generators, and noise.
- **description** (Optional): Description of the Entity.
- **freq** (Optional): For sensor Entities, sets the message publishing frequency.

### Supported Data Types for Attributes

- **int**: Integer values.
- **float**: Floating-point values.
- **bool**: Boolean values (true/false).
- **str**: String values.
- **time**: Time values (e.g., "01:25").
- **list**: List or array.
- **dict**: Dictionary (nested dictionaries are supported).

### Default Values and Value Generation for Attributes

Attributes can have default values and value generators, along with noise generators for simulating data.

#### Example with Value Generation


Entity weather_station
    type: sensor
    topic: "smauto.bme"
    freq: 5
    broker: home_mqtt_broker
    attributes:
        - temperature: float = 20.5 -> gaussian(10, 20, 5) with noise gaussian(1,1)
        - humidity: float -> linear(1, 0.2) with noise uniform(0, 1)
        - pressure: float -> constant(0.5)
end

### Supported Value Generators

- **Constant**: `constant(value)` - Generates a constant value.
- **Linear**: `linear(start, step)` - Generates values following a linear function.
- **Saw**: `saw(min, max, step)` - Generates values following a sawtooth wave pattern.
- **Gaussian**: `gaussian(value, maxValue, sigma)` - Generates values following a Gaussian distribution.
- **Sinus**: `sinus(dc, amplitude, step)` - Generates values following a sinusoidal function.
- **Replay**: `replay([values], times)` - Replays values from a list. `times` can specify iteration count; `-1` for infinite.
- **ReplayFile**: `replay(filepath)` - Replays data from a file.

### Supported Noise Generators

- **Uniform**: `uniform(min, max)` - Generates values with a uniform distribution.
- **Gaussian**: `gaussian(mu, sigma)` - Generates values with a Gaussian distribution.

### Extended Example with Default Values


Entity smart_thermostat
    type: hybrid
    topic: "home.thermostat"
    description: "Smart thermostat in the living room"
    broker: central_broker
    attributes:
        - current_temp: float = 22.0 -> gaussian(22, 30, 2) with noise uniform(0, 0.5)
        - target_temp: float = 24.0 -> constant(24)
        - mode: str = "auto"
        - status: bool = false
end

By following these updated guidelines, you can define and configure Entities that align with the provided textX grammar, ensuring correct parsing and functionality in a smart environment.
"""

DEFINE_AUTOMATIONS = """
To define and configure Automations that can be parsed by the provided textX grammar, follow these detailed guidelines:

### Key Concepts

1. **Automations**: Allow the execution of a set of actions when a condition is met. Actions are performed by sending messages to Entities.
2. **Conditions**: Used to determine if actions should be executed.
3. **Actions**: Messages sent to actuators in your setup when conditions are met.

### Properties of Automations

- **condition**: The condition used to determine if actions should be run.
- **enabled**: Whether the automation should be active.
- **continuous**: Whether the automation should remain enabled after its actions have been executed.
- **checkOnce**: The condition of the automation will run only once and then exit.
- **actions**: The actions to be run once the condition is met.
- **after**: The automation will not start until the listed dependencies are terminated.
- **starts**: Other automations to start after the current automation terminates.
- **stops**: Other automations to stop after the current automation terminates.
- **description**: An optional textual description of the automation.
- **freq**: An optional frequency (in seconds) for checking the condition.

### Example Automation Definitions

#### Simple Automation Example
Automation start_aircondition
    condition: 
        (
            (thermometer.temperature > 32) AND 
            (humidity.humidity > 30)
        ) AND (aircondition.on == true)
    enabled: true
    continuous: false
    actions:
        - aircondition.temperature: 25.0
        - aircondition.mode: "cool"
        - aircondition.on: true
end

#### Chained Automations Example
Automation start_humidifier
    condition:
        bedroom_humidity_sensor.humidity > 0.6
    enabled: true
    actions:
        - bedroom_humidifier.power: true
        - bedroom_humidifier.timer: -1
    starts:
        - stop_humidifier
end

Automation stop_humidifier
    condition:
        bedroom_humidity_sensor.humidity < 0.3
    enabled: false
    actions:
        - bedroom_humidifier.power: false
    starts:
        - start_humidifier
end

### Detailed Syntax

- **condition**: Conditions are defined using logical and comparison operators. They can reference entity attributes using their Fully-Qualified Name (FQN) in dot notation.

  condition:
      (corridor_temperature.temperature > 30) AND
      (kitchen_temperature.temperature > 30)
  

- **enabled**: Boolean value indicating if the automation is active.

  enabled: true
  

- **continuous**: Boolean value indicating if the automation remains active after execution.

  continuous: false
  

- **checkOnce**: Boolean value indicating if the condition should be checked only once.

  checkOnce: true
  

- **actions**: List of actions to perform when the condition is met. Each action follows the format `- entity_name.attribute_name: value`.

  actions:
      - aircondition.temperature: 25.0
      - aircondition.mode: "cool"
      - aircondition.on: true
  

- **after**: List of automations that must terminate before this one starts.

  after:
      - other_automation

- **starts**: List of automations to start after this one terminates.

  starts:
      - another_automation

- **stops**: List of automations to stop after this one terminates.

  stops:
      - some_automation

- **description**: Optional textual description of the automation.

  description: "This automation starts the air conditioner when it is too hot"

- **freq**: Optional frequency (in seconds) for checking the condition.

  freq: 60


### Writing Conditions

- **Basic Condition Example**:

  (bedroom_humidity.humidity < 0.3) AND (bedroom_humidifier.state == 0)


- **Advanced Condition with Built-in Functions**:

  condition:
      (mean(bedroom_temp_sensor.temperature, 10) > 28) AND
      (std(bedroom_temp_sensor.temperature, 10) > 1)
  

### Supported Operators and Functions

- **String Operators**: `~`, `!~`, `==`, `!=`, `has`, `in`, `not in`
- **Numeric Operators**: `>`, `>=`, `<`, `<=`, `==`, `!=`
- **Logical Operators**: `AND`, `OR`, `NOT`, `XOR`, `NOR`, `XNOR`, `NAND`
- **Boolean Operators**: `is`, `is not`
- **List and Dictionary Operators**: `==`, `!=`
- **Built-in Functions**: `mean`, `std`, `var`, `min`, `max`

When combining two conditions or more into a more complex one using logical operators. The general format of the Condition is:

(condition_1) LOGICAL_OP (condition_2)

Make sure to not forget the parenthesis!!! This is crucial for the correct parsing of the condition by the textX grammar

condition_1 AND condition_2 AND condition_3

will have to be rephrased to an equivalent like:

((condition_1) AND (condition_2)) AND (condition_3)

### Extended Example with All Features

#### Complex Automation with All Properties


Automation complex_automation
    condition:
        (thermometer.temperature > 32) AND 
        (humidity.humidity > 30)
    enabled: true
    continuous: true
    checkOnce: false
    actions:
        - aircondition.temperature: 25.0
        - aircondition.mode: "cool"
        - aircondition.on: true
    after:
        - some_other_automation
    starts:
        - another_automation
    stops:
        - an_additional_automation
    description: "This automation cools the room when the temperature and humidity are high."
    freq: 120
end

By following these guidelines, you can define and configure Automations that align with the provided textX grammar, ensuring correct parsing and functionality within your smart automation environment.
"""

SYSTEM_CLOCK_GUIDELINES = """
Guidelines for Using the System Clock in SmAuto Models

Introduction
The system_clock in SmAuto models provides the ability to trigger automations based on the system's current time. This allows for time-based control over smart devices, enabling automations to occur at specific times or intervals.

Example Model
Here is an example SmAuto model utilizing system_clock:

Broker<MQTT> bedroom_broker
    host: "192.168.1.12"
    port: 1883
    ssl: false
    auth:
        username: "myHome"
        password: "bgh-hjgk#"
end

Entity bedroom_lamp
    type: actuator
    topic: "bedroom.lamp"
    broker: bedroom_broker
    attributes:
        - power: bool
        - colorR: int
        - colorG: int
        - colorB: int
end

Automation motion_detected_self_start
    condition:
        system_clock.time >= 03:06
    enabled: true
    continuous: false
    actions:
        - bedroom_lamp.power: true
    starts:
        - motion_detected_self_start
end

Guidelines for Using System Clock

Defining System Clock Conditions:

Use system_clock.time to reference the current time of the system running the automation.
Time should be specified in the HH:MM format (24-hour clock).
Example:

condition:
    system_clock.time >= 03:06
    
Specifying Time-Based Conditions:

Automations can be triggered at or after a specific time using comparison operators such as >=, <=, ==, !=.
Ensure the time format is consistent and accurate to avoid errors.
Do not write time inside quotes. 06:00 is correct, "06:00" is incorrect.
When combining conditions using the system_clock.time, ensure the use of parentheses.
Example:

condition:
    system_clock.time == 06:00
    
Enabling and Disabling Automations:

The enabled property determines if the automation should be active.
Set enabled: true to activate the automation or enabled: false to deactivate it.
Example:

enabled: true

Setting Continuous Automations:

The continuous property specifies if the automation should remain enabled after executing its actions.
Set continuous: true for persistent checks or continuous: false for one-time execution.
Example:

continuous: false

Defining Actions:

Specify actions to be performed when the condition is met.
Actions should be defined in a list format, targeting specific attributes of entities.
Example:

actions:
    - bedroom_lamp.power: true
    
Chaining Automations:

Use the starts property to trigger subsequent automations after the current one completes.
This can create a chain of events based on time conditions.
Example:

starts:
    - another_automation

Practical Example
Consider a scenario where you want the bedroom lamp to turn on at 3:06 AM:

Automation turn_on_bedroom_lamp
    condition:
        system_clock.time >= 03:06
    enabled: true
    continuous: false
    actions:
        - bedroom_lamp.power: true
end

Best Practices
Synchronization: Ensure the system clock is synchronized accurately to avoid mismatches in time-based automations.
Testing: Test time-based conditions to confirm they trigger as expected, considering edge cases like daylight saving changes.
Logging: Implement logging in automations to monitor their execution, especially for debugging and verification purposes.
Security: Protect broker credentials and ensure secure communication (consider SSL if applicable) when defining brokers.
By following these guidelines, you can effectively use the system_clock in SmAuto models to create precise and reliable time-based automations for your smart environment.
"""

DEFINE_METADATA_RTMONITOR = """
Introduction
In SmAuto models, Metadata and RTMonitor sections provide essential information and configurations for model identification and runtime monitoring. These elements help manage and track the automation system, making it easier to understand, maintain, and monitor its performance.

Metadata
The Metadata section allows you to define meta-information about your SmAuto model, such as the model's name, version, author, and description. This information is crucial for documentation, version control, and collaboration.

Metadata Structure

Metadata
    name: <model_name>
    version: <model_version>
    description: <model_description>
    author: <author_name>
    email: <author_email>
end


Guidelines for Metadata

Name:

Provide a unique and descriptive name for your model.
The name should reflect the purpose or functionality of the model.
Make sure to NOT include the model name inside " "
Example:

name: SimpleHomeAutomation

Version:

Use semantic versioning (e.g., 1.0.0) to indicate the version of your model.
Update the version appropriately when making changes.
Example:

version: "1.0.0"

Description:

Include a brief description that explains the model's functionality and purpose.
This helps users understand the context and scope of the model.
Example:

description: "This model automates lighting based on motion detection and time conditions."

Author:

Specify the name of the individual or organization responsible for creating the model.
This is useful for accountability and collaboration.
Example:

author: "John Doe"

Email:

Provide a contact email address for the author or maintainer.
This allows users to reach out for support or questions.
Example:

email: "johndoe@example.com"

Extra Attributes (Optional):

Include any additional attributes that may be useful for your model.
These can be used for custom scripts or transformations.
RTMonitor
The RTMonitor section is used to define the monitoring parameters for the SmAuto runtime. This includes configuration for logging and event topics, as well as the broker used for monitoring messages.

RTMonitor Structure

RTMonitor
    broker: <broker_name>
    namespace: <namespace>
    eventTopic: <event_topic>
    logsTopic: <logs_topic>
end


Guidelines for RTMonitor

Broker:

Reference a previously defined broker that will handle monitoring messages.
Ensure the broker is configured correctly to handle the specified topics.
Example:

broker: bedroom_broker

Namespace:

Define a namespace to group related events and logs.
This helps organize messages and avoid conflicts.
Example:

namespace: "home_automation.bedroom"

Event Topic:

Specify the topic where events will be published.
Events can include state changes, triggers, and other runtime occurrences.
Example:

eventTopic: "events"

Logs Topic:

Define the topic where logs will be published.
Logs provide valuable information for debugging and monitoring the system.
Example:

logsTopic: "logs"
Extra Attributes (Optional):

Include any additional attributes for monitoring configurations.
These can be customized as per the requirements of the monitoring system.

Practical Example
Here is an example of a complete SmAuto model with both Metadata and RTMonitor sections:

Metadata
    name: SimpleHomeAutomation
    version: "1.0.0"
    description: "This model automates lighting based on motion detection and time conditions."
    author: "John Doe"
    email: "johndoe@example.com"
end

RTMonitor
    broker: bedroom_broker
    namespace: "home_automation.bedroom"
    eventTopic: "events"
    logsTopic: "logs"
end

Broker<MQTT> bedroom_broker
    host: "192.168.1.12"
    port: 1883
    ssl: false
    auth:
        username: "myHome"
        password: "bgh-hjgk#"
end

Entity bedroom_lamp
    type: actuator
    topic: "bedroom.lamp"
    broker: bedroom_broker
    attributes:
        - power: bool
        - colorR: int
        - colorG: int
        - colorB: int
end

Automation motion_detected_self_start
    condition:
        system_clock.time >= 03:06
    enabled: true
    continuous: false
    actions:
        - bedroom_lamp.power: true
    starts:
        - motion_detected_self_start
end

By following these guidelines, you can effectively use Metadata and RTMonitor to enhance the organization, documentation, and monitoring capabilities of your SmAuto models.
"""

WRITE_SMAUTO_MODEL = """
SmAuto is a Domain Specific Language (DSL) for creating complex automation scenarios in smart environments, particularly for IoT devices. Here are comprehensive guidelines to assist you in writing effective SmAuto models.

General Structure
An SmAuto model is composed of several main components:

Metadata: Contains meta-information about the model.
Brokers: Defines communication layers.
Entities: Represents smart devices (sensors and actuators).
Automations: Defines the logic for automated actions.
Conditions: Specifies the criteria under which automations are triggered.
RTMonitor: Defines runtime monitoring parameters.
Each component has its own syntax and set of properties.

These guidelines provide a comprehensive overview of writing SmAuto models. Ensure that you adhere to the syntax and structure specified for each component to create effective and efficient smart home automation scripts.
"""

CONSTRUCT_SMAUTO_MODEL = """
    Write brokers, entities, and automations, write the complete SmAuto model on the following description:
    {user_utterance}
    Define the Metadata and RTMonitor components as well.
    Follow the guidelines provided for each component to ensure the model is correctly structured.

    Do not use # to comment in the model. Use // for inline comments and /* */ for block comments.
    Use the appropriate operators for the conditions and actions in the automations for each type of attribute.
    As a reminder: **Boolean Operators**: `is`, `is not`


    Output only the SmAuto model code.
    Put the code inbetween the ```smauto and ``` tags."""

CONSTRUCT_SMAUTO_MODEL_FROM_YAML = """
    Write brokers, entities, and automations, write the complete SmAuto model on the following description:
    {yaml_content}
    Define the Metadata and RTMonitor components as well.
    Follow the guidelines provided for each component to ensure the model is correctly structured.

    Do not use # to comment in the model. Use // for inline comments and /* */ for block comments.
    Use the appropriate operators for the conditions and actions in the automations for each type of attribute.
    As a reminder: **Boolean Operators**: `is`, `is not`


    Output only the SmAuto model code.
    Put the code inbetween the ```smauto and ``` tags."""

INVALID_MODEL = """The model you have written is invalid.
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

IDENTIFY_USER_INTENT = """
    Your task is to analyze the user's input to determine the functionality they want the model to perform. The goal is to assess whether the provided information contains all the necessary details to create a complete SmAuto model. The model should include specific components: Metadata, Brokers, Entities, Automations, and RTMonitor.

    Begin by reviewing the user’s input to see if it specifies these components:

    Metadata: Includes the model's name, version, description, author, and email.
    Brokers: Defines the communication layers (MQTT, AMQP, or Redis brokers) with details such as type, host, port, SSL, authentication, and more.
    Entities: Represents smart devices (sensors and actuators) with attributes like type, topic, broker, and frequency.
    Automations: Outlines automated logic, including conditions, actions, dependencies, and descriptions.
    RTMonitor: Defines runtime monitoring parameters including broker, namespace, event topic, and logs topic.
    If any component is missing or incomplete, identify the gaps and engage the user in a conversation to extract the necessary details. Your goal is to ensure all the required components are fully defined to complete the SmAuto model.

    Process:

    Analyze the user’s input for completeness.
    Identify any missing or incomplete components.
    Generate specific questions to gather the missing information from the user.
    Proceed with the analysis and engage the user as needed to fill in the gaps.

    1. Metadata:

    Name:
    Version:
    Description:
    Author:
    Email:
    
    2. Brokers:

    Type (MQTT, AMQP, Redis):
    Host:
    Port:
    SSL (true/false):
    Authentication (username/password, cert, key):
    Vhost (AMQP only):
    TopicExchange (AMQP only):
    RPCExchange (AMQP only):
    DB (Redis only):
    
    3. Entities:

    Unique Name:
    Type (sensor, actuator, hybrid):
    Topic:
    Broker:
    Attributes (name and type, default values, value generators, noise generators):
    Description (optional):
    Frequency (for sensors):
    
    4. Automations:

    Condition:
    Enabled (true/false):
    Continuous (true/false):
    CheckOnce (true/false):
    Actions:
    After (dependencies):
    Starts (subsequent automations):
    Stops (terminating automations):
    Description (optional):
    Frequency (optional):
    
    5. RTMonitor:

    Broker:
    Namespace:
    Event Topic:
    Logs Topic:
    

    Condition:
    Based on this analysis, here are the missing or incomplete components:

    [Insert analysis results here]

    To complete the SmAuto model, please provide the following missing information:

    [Insert questions to gather missing information]
"""

GATHER_INFOMATION = """
    Your task is to behave as a Q&A bot. You will guide the user through the process of providing the necessary details to complete the SmAuto model by asking questions one at a time. You will also give the user the option to skip any question, in which case you will make reasonable assumptions to fill in the missing information.

    Follow these steps:

    1. **Question Generation:** Based on the previous analysis, you have identified specific missing or incomplete components required for the SmAuto model. You will ask the user one question at a time to gather this missing information.

    2. **Wait for Response:** After asking each question, wait for the user to provide an answer. If the user does not wish to answer, they may skip the question.

    3. **Skip and Assumption Option:** If the user chooses to skip a question, make a reasonable assumption based on typical usage patterns, best practices, or common defaults. Inform the user of the assumption you've made and let them confirm or adjust it.

    4. **Clarification:** If the user’s response is unclear or incomplete, ask follow-up questions to ensure you receive the necessary information. You can offer suggestions or common defaults if needed.

    5. **Confirmation:** Once all the required information for a specific component is gathered (either through user input or assumptions), briefly summarize it and confirm with the user before moving on to the next question.

    6. **Completion:** Continue this process until all the necessary details are obtained or assumed to complete the SmAuto model. Always confirm assumptions with the user as the conversation progresses.

    7. **Flexibility:** At any time, the user can review previous answers, clarify assumptions, or provide additional details if needed.

    Your goal is to systematically gather the information or make educated assumptions while keeping the process flexible and user-friendly.
    
    At the end of the Q&A process, do not generate or output the SmAuto model. Your sole task is to ask the questions, gather the necessary information, and confirm any assumptions made. Once all the required questions have been asked and answered, simply output "Q&A process complete." and only that, nothing else. Do not provide any further content or attempt to create the model itself.
    
    When the Q&A process is complete, you will output the following message as it is: "Q&A process complete."
"""

CONSTRUCT_SMAUTO_MODEL_AFTER_QA = """
    With all the information you have gathered from the user, you are now ready to construct the complete SmAuto model. Based on the responses provided during the Q&A process, you will write the complete SmAuto model.
    
    Follow the guidelines provided for each component to ensure the model is correctly structured.

    Do not use # to comment in the model. Use // for inline comments and /* */ for block comments.
    Use the appropriate operators for the conditions and actions in the automations for each type of attribute.
    As a reminder: **Boolean Operators**: `is`, `is not`

    Output only the SmAuto model code.
    Put the code inbetween the ```smauto and ``` tags."""


def get_system_prompt():
    """Returns the system prompt containing the guidelines for writing SmAuto models."""
    system_prompt = [
        ("system", SYSTEM_ROLE),
        ("system", DEFINE_BROKERS),
        ("system", DEFINE_ENTITIES),
        ("system", DEFINE_AUTOMATIONS),
        ("system", SYSTEM_CLOCK_GUIDELINES),
        ("system", DEFINE_METADATA_RTMONITOR),
        ("system", WRITE_SMAUTO_MODEL),
    ]
    return system_prompt
