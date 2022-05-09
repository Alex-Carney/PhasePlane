import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sympy import lambdify, symbols
from util import carney_diff_eqs as ode44
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, \
    implicit_multiplication_application, convert_xor
import plotly.graph_objects as go
import plotly.figure_factory as ff
from streamlit_plotly_events import plotly_events
from common.components_callbacks import register_callback
import json
from common import variable as v
import scipy.integrate as si

"""
This is the application for one dimensional systems, with a single differential equation. 
"""


def app(flag):

    ##########################################################
    ######## GLOBAL DATA AND INITIAL STATE VALUES ############
    ##########################################################

    # Setup global vars
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    x = symbols('x')
    y = symbols('y')

    # Setup global constants
    global included_keys
    # Don't include variables in here. They have to be serialized separately
    included_keys = ['y_max', 'y_min', 'initial_conditions', 'step_max',
                     'x_max', 'dydx', 'x_segments',
                     'x_min', 'y_segments', 't_max']
    global float_included_keys
    float_included_keys = ['y_max', 'y_min', 'step_max', 'x_max', 'var_step_size',
                               'x_min', 't_max']
    global integer_included_keys
    integer_included_keys = ['y_segments', 'x_segments']

    global initial_condition_key
    initial_condition_key = "initial_conditions"

    global variable_key
    variable_key = "variables"

    global variable_letter_key
    variable_letter_key = "variable_letters"

    # Setup boundary values
    if 'x_segments' not in st.session_state:
        st.session_state.x_segments = 20
    if 'y_segments' not in st.session_state:
        st.session_state.y_segments = 20
    if 'x_min' not in st.session_state:
        st.session_state.x_min = -4
    if 'x_max' not in st.session_state:
        st.session_state.x_max = 4
    if 'y_min' not in st.session_state:
        st.session_state.y_min = -4
    if 'y_max' not in st.session_state:
        st.session_state.y_max = 4

    x_step: float = .1

    if 'initial_conditions' not in st.session_state:
        st.session_state['initial_conditions'] = []

    if 'variables' not in st.session_state:
        st.session_state['variables'] = []

    if 'variable_letters' not in st.session_state:
        st.session_state['variable_letters'] = []

    ##########################################################
    ######## CALLBACK FUNCTIONS ############
    ##########################################################

    def text_input_callback():
        """
        Fired when a user inputs a new equation. The input is parsed into an equation,
        then the plot scaffold is created. Finally, the plot is populated with points, then
        drawn
        """
        try:
            diff_eqn = generate_functions_from_input(st.session_state.dydx)
            x_partition, y_partition, x_grid, y_grid = generate_plot_scaffold(st.session_state.x_segments,
                                                                              st.session_state.y_segments,
                                                                              st.session_state.y_min,
                                                                              st.session_state.y_max,
                                                                              st.session_state.x_min,
                                                                              st.session_state.x_max)
            dx, dy, magnitude = generate_plot_output(diff_eqn, x_grid, y_grid)
            fig = draw_plot(dx, dy, magnitude, x_grid, y_grid)

            # For each line that the user had drawn, recalculate their positions
            for initial_condition in st.session_state.initial_conditions:
                x_domain, out = solve_ivp(initial_condition)
                fig.add_trace(go.Scatter(x=x_domain, y=out))

            render_plot(fig)
        except Exception as e:
            st.warning("An exception occurred while processing your equation. Try again, or refresh the page")

    def click_plot_input_callback():
        """
        Fired when a user clicks on the plot. First, the point the user clicked on is parsed.
        Then, the differential equation is numerically analyzed with RK. Finally, the trace
        from that line is added to the plot, and re-rendered.
        """
        # Get the point clicked. It is a string of JSON. Trim it, then parse to json for use
        str_without_brack = st.session_state.my_key[1:-1]
        result = json.loads(str_without_brack)
        # Extract initial conditions associated with clicking this point. Solve the IVP
        initial_conds = [result['x'], result['y']]
        x_domain, out = solve_ivp(initial_conds)
        st.session_state.initial_conditions.append(initial_conds)
        # Add the solution to the phase plane, re-render
        fig = st.session_state.phase_plane
        fig.add_trace(go.Scatter(x=x_domain, y=out))
        render_plot(fig)

    def manual_ivp_input_callback(init_x, init_y):
        """
        Fired when someone submits a form for an initial condition. Does the same
        as click plot input callback
        """
        try:
            initial_conds = [init_x, init_y]
            x_domain, out = solve_ivp(initial_conds)
            st.session_state.initial_conditions.append(initial_conds)
            fig = st.session_state.phase_plane
            fig.add_trace(go.Scatter(x=x_domain, y=out))
            render_plot(fig)
        except:
            st.warning("Can't do that right now. Try making a plot first")


    def clear_curves_callback():
        """
        Fired when a user wants to clear all curves from the plot
        """
        try:
            fig = st.session_state.phase_plane
            fig.data = [fig.data[0]]  # keep only the 0th trace (the arrows)
            st.session_state.initial_conditions = []
            render_plot(fig)
        except:
            st.warning("Can't use that right now. Try making a plot and adding some curves first.")

    def slider_change_callback(letter):
        """
        Fired when a user-created slider changes value
        The entire equation has changed, so all of the traces must be re-calculated.
        Good job figuring this out bro
        :param letter:
        :return:

        NOTE: When implementing this, we might not actually have to do much work, both the quivers
        and solution curves might already be in terms of the variables (* a * b whatever)? Check it out

        """
        text_input_callback()

    def import_session_state_callback(input_dict):
        """
        Uses the session state stored in JSON inside the input dict to load a session
        state, and redraw all necessary components.
        :param input_dict:
        """

        # Attempt to merge input dict into session state. Then reload everything
        try:
            for (key, val) in input_dict.items():
                # Different keys might require different mechanisms to load
                if key in float_included_keys:
                    st.session_state[key] = float(val)
                elif key in integer_included_keys:
                    st.session_state[key] = int(val)
                elif key == initial_condition_key:
                    st.session_state[initial_condition_key] = list(json.loads(val))
                elif key == variable_key:
                    # Variables are serialized as a list of variable JSON objects. Deserialize accordingly
                    for var_json in val:
                        st.session_state[variable_key].append(
                            v.Variable(var_json['letter'], var_json['min_value'], var_json['max_value'], var_json['step_size'])
                        )
                        st.session_state[variable_letter_key].append(var_json['letter'])
                else:
                    st.session_state[key] = val

        except Exception as ex:
            print("ERROR " + str(ex))
            st.warning("Invalid dictionary input")

    ##########################################################
    ######## HELPER FUNCTIONS ############
    ##########################################################

    def solve_ivp(initial_conditions):
        """
        Uses carney_diff_eqs RK4 implementation to solve the initial value problem
        :param initial_conditions: An array containing [X, Y] initial condition coordinates
        :return: A line of X,Y points that solves the equation
        """
        xInput = initial_conditions[0]
        yInput = initial_conditions[1]

        # Solve the equation twice, once forwards in time, once backwards.
        x_domainRight = np.r_[xInput:st.session_state.x_max * 1.2:x_step]
        x_domainLeft = np.r_[xInput:st.session_state.x_min * 1.2:-x_step]

        outRight = ode44.runge_kutta(st.session_state.diff_eqn, x_domainRight, yInput, x_step)
        outLeft = ode44.runge_kutta(st.session_state.diff_eqn, x_domainLeft, yInput, -x_step)

        # The X domain controls the X values of the output curves, the Y values are solved by the IVP
        x_domain = np.hstack((np.flip(x_domainLeft[1:]), x_domainRight))
        out = np.hstack((np.flip(outLeft[1:]), outRight))
        return x_domain, out


    def generate_functions_from_input(dydx_equation_input: str):
        # Parse equations into lambdas
        eqn_input = parse_expr(f'{dydx_equation_input}', transformations=transformations)
        # Convert SymPy objects into ones that numpy can use. Each variable is substituted in for its numerical value
        # Then, the equation is 'lambdified' into something workable.
        eqn_input_with_vars = eqn_input.subs([(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        diff_eqn = lambdify([x, y], eqn_input_with_vars, 'numpy')
        st.session_state.diff_eqn = diff_eqn
        return diff_eqn


    def generate_plot_scaffold(x_segments, y_segments, y_min, y_max, x_min, x_max):
        # Generate 'skeleton' for plot
        x_partition: ndarray = np.linspace(x_min, x_max, x_segments)
        y_partition: ndarray = np.linspace(y_min, y_max, y_segments)
        x_grid, y_grid = np.meshgrid(x_partition, y_partition)
        return x_partition, y_partition, x_grid, y_grid

    def generate_plot_output(diff_eqn,
                      x_grid: ndarray,
                      y_grid: ndarray):
        # Generate plot
        a = 1
        dy = diff_eqn(x_grid, y_grid)
        dx = np.ones(dy.shape)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)  # magnitude
        return dx, dy, magnitude

    def draw_plot(dx: ndarray,
                  dy: ndarray,
                  magnitude: int,
                  x_grid: ndarray,
                  y_grid: ndarray,
                  ):
        fig = ff.create_quiver(
            x_grid, y_grid, dx/magnitude, dy/magnitude,
            scale=.25, arrow_scale=.4, angle=0
        )
        font_dict = dict(family='Arial', size=26, color='black')
        fig.add_hline(0)
        fig.add_vline(0)
        fig.update_layout(font=font_dict,  # change the font
                          plot_bgcolor='white',  # get rid of the god awful background color
                          margin=dict(r=60, t=40, b=40),
                          )
        # x and y-axis formatting
        fig.update_yaxes(title_text='Y-axis',  # axis label
                         showline=True,  # add line at x=0
                         linecolor='black',  # line color
                         linewidth=2.4,  # line size
                         ticks='outside',  # ticks outside axis
                         tickfont=font_dict,  # tick label font
                         mirror='allticks',  # add ticks to top/right axes
                         tickwidth=2.4,  # tick width
                         tickcolor='black',  # tick color
                         )
        fig.update_xaxes(title_text='X-axis',
                         showline=True,
                         showticklabels=True,
                         linecolor='black',
                         linewidth=2.4,
                         ticks='outside',
                         tickfont=font_dict,
                         mirror='allticks',
                         tickwidth=2.4,
                         tickcolor='black',
                         )
        return fig


    def render_plot(fig):
        with col2:
            fig.update_layout(yaxis_range=[st.session_state.y_min, st.session_state.y_max])
            fig.update_layout(xaxis_range=[st.session_state.x_min, st.session_state.x_max])
            st.session_state.clicked_points = plotly_events(fig, key="my_key")
            st.session_state.phase_plane = fig

    def initialize_variable_sliders():
        for var in st.session_state.variables:
            var.slider = col2.slider(f"Slider for {var.letter}",
                                     key=f"{var.letter}",
                                     min_value=var.min_value,
                                     max_value=var.max_value,
                                     step=var.step_size,
                                     on_change=slider_change_callback,
                                     args=(var.letter)
                                     )

    ##########################################################
    ######## DISPLAY THE APPLICATION ITSELF ############
    ##########################################################

    # Setup screen ------------------------------------------
    col1, col2 = st.columns([1, 2])
    st.session_state.col2 = col2
    col1.header("Enter single ODE")
    col2.header("Output")

    # Setup widgets ------------------------------------------

    # Import session state expander
    import_expander = st.expander(label='Import Session')
    with import_expander:
        st.write("Paste in a session state. Then, click 'import' to load the saved session")
        import_form = st.form(key='import_state_form')
        with import_form:
            text_input = st.text_area("Paste session state here")
            import_submit = import_form.form_submit_button("Import")
            if text_input is not None and import_submit is True:
                try:
                    dict_input = json.loads(text_input)
                    import_session_state_callback(dict_input)
                except Exception as e:
                    print("ERROR IN LOADING JSON" + str(e))
                    st.warning("Something went wrong converting your input to a dictionary")

    # User equation input
    dydx_equation_input: str = col1.text_input("dy/dx = ", 'x', key="dydx", on_change=text_input_callback)

    # Clear curves button
    clear_curves = col1.button("Clear Curves", key="clear", on_click=clear_curves_callback)

    # Render plot button
    col1.button("Render Plot", on_click=text_input_callback)

    # Add a variable form
    var_form_expander = col1.expander(label='Add a Variable')
    with var_form_expander:
        var_form = st.form(key='add_var_form', clear_on_submit=True)
        with var_form:
            var_letter = st.text_input("Variable Letter", key='var_letter')
            var_min_value = st.number_input("Minimum Allowed Value", key='var_min_value')
            var_max_value = st.number_input("Maximum Allowed Value", key='var_max_value')
            var_step_size = st.number_input("Step Size (For Slider)", key='var_step_size') #TODO: Default value not zero
            var_submit = st.form_submit_button("Submit Variable")
            if var_letter is not None and var_submit is True:
                st.session_state.variables.append(v.Variable(var_letter, var_min_value, var_max_value, var_step_size))
                st.session_state.variable_letters.append(var_letter)


    initialize_variable_sliders()

    # Extra text for user
    col1.markdown("""---""")
    col1.markdown("""**Click on plot to draw solution curve. Or manually input initial conditions**""")


    # Initial Conditions Form --------------------------------
    initial_conditions_expander = col1.expander(label='Manual Initial Conditions')
    with initial_conditions_expander:
        ic_form = st.form(key='initial_conditions_form', clear_on_submit=True)
        with ic_form:
            init_x = st.number_input("Initial X")
            init_y = st.number_input("Initial Y")
            ic_submit = ic_form.form_submit_button("Submit IVP")
            if init_x is not None and init_y is not None and ic_submit is True:
                manual_ivp_input_callback(init_x, init_y)

    # Setup export plot state expander --------
    export_expander = st.expander(label='Export Session')
    with export_expander:
        st.write("Click here to export session state. Save the output to your clipboard, then import it later!")
        submit = st.button("Export", on_click=text_input_callback)
        if submit:
            export_dict = {key: val for key, val in st.session_state.items() if key in included_keys}
            # Variables have to be serialized separately
            # Serialize exported keys into JSON
            serialized_string = ('{%s' % ', '.join(['"%s": "%s"' % (key, val) for key, val in export_dict.items()]))
            # Serializing a list of variables is more difficult, and is done manually here
            serialized_string += ', "variables": '
            serialized_string += ('[%s]}' % ', '.join('%s' % var for var in st.session_state.variables))

            st.text_area("Copy and paste this", serialized_string)

    # Setup Options Expander ----------------------------------
    options_expander = st.expander(label='Plot Options')
    with options_expander:
        st.write("PLOT BOUNDS")
        options_col1, options_col2, options_col3 = st.columns(3)
        options_col1.number_input("xMin", step=1, value=-4, key='x_min', on_change=text_input_callback)
        options_col2.number_input("xMax", step=1, value=4, key='x_max', on_change=text_input_callback)
        options_col1.number_input("yMin", step=1, value=-4, key='y_min', on_change=text_input_callback)
        options_col2.number_input("yMax", step=1, value=4, key='y_max', on_change=text_input_callback)
        options_col3.number_input("xSegments", step=1, value=20, key='x_segments', on_change=text_input_callback)
        options_col3.number_input("ySegments", step=1, value=20, key='y_segments', on_change=text_input_callback)
        st.write("ARROW SCALING")
        st.checkbox("Normalize Arrows", value=True)


    # Streamlit does not support clicking on plotly plots as a valid event. Therefore, we use a "hack"
    # This has to be the last line of code as well
    register_callback("my_key", click_plot_input_callback)
