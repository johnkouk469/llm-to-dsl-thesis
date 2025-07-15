# LLM-to-DSL-Thesis

This repository contains the source code for my integrated master's thesis:  
**"Generating Domain-Specific Language (DSL) Models from Natural Language Descriptions using Large Language Models (LLMs)"**.

The system allows users to generate valid models in two DSLs using natural language descriptions:
- **SmAuto** — for smart home automation.
- **dFlow** — for dialogue-based assistant logic.

Model generation is powered by GPT-based LLMs (via OpenAI API), while model validation is performed through locally hosted REST APIs for each DSL.

---

## 🚀 How to Run

To generate models from user input, run one of the following Python scripts:

- `llm_to_smauto.py` — generates a **SmAuto** model.
- `llm_to_dflow.py` — generates a **dFlow** model.

Each script:
1. Accepts a natural language description from the user.
2. Sends a prompt to the LLM (GPT-4).
3. Receives and saves the generated DSL model.
4. Sends the model to a local validation API.
5. If the model is invalid, triggers a correction loop using GPT.

---

## 🔌 Validator Requirements

To validate the generated models:

- The REST API for **SmAuto** and **dFlow** must be running locally at:
http://localhost:8080

- If needed, you can change the URL endpoints in:
- `smauto_api.py` (for SmAuto)
- `dflow_api.py` (for dFlow)

> ⚠️ If the API is not running or unreachable, the validation process will fail.

---

## 🔐 Environment Variables

The application requires certain API keys to function properly. These are provided in a `.env` file placed in the project’s root directory.

Here’s an example of the required `.env` file:

```env
OPENAI_API_KEY="sk-proj-******"
SMAUTO_API_KEY="API_KEY"
DFLOW_API_KEY="123"
```

### Default Keys
If you’re using the default local REST API setup for validation:

The default SmAuto API key is: API_KEY

The default dFlow API key is: 123

These values can be customized if you configure your validator servers differently.

## 📦 Installation
Clone the repository:

```bash
git clone https://github.com/johnkouk469/llm-to-dsl-thesis.git
cd llm-to-dsl-thesis
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```