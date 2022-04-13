
import streamlit as st

# Custom imports
from common.multipage import MultiPage
from pages import two_dimensions, home, one_dimension  # import your pages here

# Create an instance of the app
app = MultiPage()

# First command must be setting configs
# Initial Settings ------------------------
st.set_page_config(layout="wide")

# Title of the main page
st.title("Carney Phase Plane Visualizer")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("One ODE", one_dimension.app)
app.add_page("System of 2 Equations", two_dimensions.app)


# The main app
app.run()