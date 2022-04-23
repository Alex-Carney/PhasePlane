
import streamlit as st

# Custom imports
from common.multipage import MultiPage
from pages import two_dimensions, home, one_dimension, three_dimensions  # import your pages here




# Create an instance of the app
app = MultiPage()


# First command must be setting configs
# Initial Settings ------------------------
st.set_page_config(
    page_title="Phase Space",
    page_icon="sunglasses",
    layout="wide",
)

# Title of the main page
# st.title("Carney Phase Plane Visualizer")
#st.markdown("<h1 style='text-align: center; color: red;'>Carney Phase Plane Visualizer</h1>", unsafe_allow_html=True)



if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'





# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("One ODE", one_dimension.app)
app.add_page("System of 2 Equations", two_dimensions.app)
app.add_page("System of 3 Equations", three_dimensions.app)


# The main app
app.run()