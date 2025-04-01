import streamlit as st
import requests

st.title("Wireframe")


import pandas as pd
import streamlit as st
# st.dataframe() and st.table() 

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame(
    {
    "First_col":[1,2,3,4],
    "Second_col":[10,20,30,40]
    }
   
))

st.write("streamlit session")
# st.text_input("Your name", key = "Name")

# You can access the value at any point with

"st.session_state_object:", st.session_state

st.session_state.name = "Sagar"

if "a_counter" not in st.session_state:
    st.session_state['a_counter'] = 0

if "boolean" not in st.session_state:
    st.session_state.boolean = True

# st.write(st.session_state)
del st.session_state.name
st.write("a counter is:",st.session_state["a_counter"])
st.write("a boolean is:",st.session_state.boolean)

for keys in st.session_state.keys():
    st.write(keys)

for values in st.session_state.values():
    st.write(values)

for items in st.session_state.items():
    st.write(items)


button = st.button("update state")

"before pressing button:", st.session_state

if button:
    st.session_state["a_counter"] += 1
    st.session_state.boolean = not st.session_state.boolean
    "after pressing button", st.session_state



# Work with all widgests?

number  = st.slider("A Number", 1,10, key = "slider")


"slider",st.session_state

col1 , col2 = st.columns(2)

option_name = [ "a", "b" , "c" ]

next = st.button("Next option")


if next:
    if st.session_state['radio_options'] == "a":
        st.session_state.radio_options="b" 

    elif st.session_state['radio_options'] == "b":
        st.session_state.radio_options="c"
    else:
         st.session_state.radio_options="a"

option = col1.radio("pick an option", option_name , key = "radio_options")

"radio",st.session_state




