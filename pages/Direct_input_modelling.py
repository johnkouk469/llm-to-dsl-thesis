"""
SmAuto Assistant Streamlit App

This module creates a Streamlit-based web application for interacting with the SmAuto Assistant, 
an AI-powered tool that generates SmAuto models based on user input. 
"""

import streamlit as st
import llm_to_smauto


def initialize_session_state():
    """
    Initializes the session state variables for the Direct Input Modeling scenario.

    This function ensures that the keys 'conversation_history' and 'display_history'
    are initialized in the Streamlit session state. These variables are essential for
    tracking the interaction between the user and the SmAuto Assistant within this
    specific modeling scenario.

    The difference between the 2 types of variables is that 'display_history' should
    contain all the messages as they are being displayed to the user and
    'converasation_history' should contain the same messages with some additional
    instructions passed through to the LLM.
    """
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "display_history" not in st.session_state:
        st.session_state.display_history = []


def handle_user_input(user_input):
    """
    Processes the user's input in the Direct Input Modeling scenario and generates a SmAuto model.

    This function handles the user's direct input, uses the llm_to_smauto module to generate a
    SmAuto model, and updates the display history with the model output. It is designed to
    facilitate interactive model creation based on user-provided specifications.

    Parameters:
    user_input (str): The input provided by the user through the chat interface.
    """
    st.session_state.display_history.append(("User", user_input))
    with st.spinner("Generating the model..."):
        smauto_model, st.session_state.conversation_history = (
            llm_to_smauto.generate_smauto_model(
                user_input, st.session_state.conversation_history
            )
        )
    st.session_state.display_history.append(("SmAuto Assistant", smauto_model))


def display_chat_history():
    """
    Displays the interaction history between the user and the SmAuto Assistant in the Direct
    Input Modeling scenario.

    This function renders the chat history in the UI, showing the sequence of user inputs and the
    corresponding responses from the SmAuto Assistant. It highlights the success or failure of the
    model generation and provides access to the generated model.
    """
    for sender, message in st.session_state.display_history:
        if sender == "User":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                if llm_to_smauto.validate_model(message):
                    st.success("A valid SmAuto model has been generated.")
                else:
                    st.warning(
                        "The generated model contains errors. \
Please check for missing or incorrect syntax."
                    )
                with st.expander("Generated model"):
                    st.markdown(message)


def main():
    """
    The main function that runs the SmAuto Assistant Streamlit app.

    This function sets up the title and sidebar instructions, initializes the session state,
    handles user input through a chat interface, and displays the conversation history.
    """
    st.title("SmAuto Assistant")
    st.sidebar.header("How to use the app.")
    st.sidebar.write(
        """
        1. Enter a command or query into the chat input box.
        2. The SmAuto Assistant will generate a model based on your input.
        3. If the model is valid, it will be displayed with a success message. 
           Otherwise, you'll be alerted to any syntactical errors.
        4. Use the 'Generated model' expander to view the full details of the model.
    """
    )

    initialize_session_state()

    user_input = st.chat_input("Enter your command or question here...")
    if user_input:
        handle_user_input(user_input)

    display_chat_history()


if __name__ == "__main__":
    main()
