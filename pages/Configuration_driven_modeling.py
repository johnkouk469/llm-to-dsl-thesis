# # Initialize session state variables if they don't exist
# if 'num_scenarios' not in st.session_state:
#     st.session_state.num_scenarios = 1
# if 'scenarios' not in st.session_state:
#     st.session_state.scenarios = ['']

# # Function to add a new textbox
# def add_scenario():
#     st.session_state.num_scenarios += 1
#     st.session_state.scenarios.append('')

# st.title("Home Automation Scenarios")

# # Display existing scenario textboxes
# for i in range(st.session_state.num_scenarios):
#     st.session_state.scenarios[i] = st.text_area(f'Scenario {i + 1}', value=st.session_state.scenarios[i], key=f'scenario_{i}')

# # Button to add a new textbox
# st.button('Add another scenario', on_click=add_scenario)

# # Create a form for submission
# with st.form(key='scenario_form'):
#     # Submit button
#     submitted = st.form_submit_button('Submit')
#     if submitted:
#         st.write("Scenarios submitted:")
#         for i, scenario in enumerate(st.session_state.scenarios, 1):
#             st.write(f"Scenario {i}: {scenario}")
