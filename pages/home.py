import streamlit as st

"""
The home page is a static image with some text. 
"""

def app(flag):
    # Only render the background image for home page
    if st.session_state.current_page == 'Home':
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background: url("https://cdn.pixabay.com/photo/2016/11/11/20/05/wave-1817646__340.png");
                 background-size: contain;
                 background-repeat: no-repeat;
                 -webkit-background-size: cover;
             }}
             </style>
             """,
            unsafe_allow_html=True
        )

        st.markdown("<h1 style='text-align: center; color: black;'>Phase Space Visualizer</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: grey; font-style: italic;'>Built by Alex Carney '22 for Skidmore students</h4>", unsafe_allow_html=True)
