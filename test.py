import streamlit as st
st.title('My First Streamlit App')

st.write('This is a text')

st.button("Reset",type='primary')
if st.button("Say Hello"):
              st.balloons()
              st.write("why hello there")
else:
    st.write("goodbye")