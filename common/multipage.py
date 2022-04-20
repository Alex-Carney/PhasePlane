"""
Framework for generating multiple Streamlit applications with OOP

From https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030
"""

# Import necessary libraries
import streamlit as st


# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Dropdown to select the page to run
        page = st.sidebar.selectbox(
            'App Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )

        # run the app function
        print("Currently on " + str(page['title']) +
              " compared to state " + str(st.session_state.current_page))

        flag = page['title'] != st.session_state.current_page
        if flag:
            # the page has changed. Clear the cache
            print("Clearing cache...")
            st.session_state.clear()

        st.session_state.current_page = page['title']
        page['function'](flag)
