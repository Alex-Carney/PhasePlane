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


def app(flag):
    # Setup global vars
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    x = symbols('x')
    y = symbols('y')
    t = symbols('t')


    # Setup global constants
    global included_keys
    # Don't include variables in here. They have to be serialized separately
    included_keys = ['y_max', 'y_min', 'initial_conditions', 'step_max',
                     'x_max', 'solver', 'dydt', 'dxdt', 'x_segments',
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
    if 'solver' not in st.session_state:
        st.session_state.solver = 'RK45'
    if 't_max' not in st.session_state:
        st.session_state.t_max = 10
    if 'step_max' not in st.session_state:
        st.session_state.step_max = .1

    x_step: float = .1

    # Setup other global values
    t_step = .01
    h = .01
    time = np.r_[0:5:t_step]

    if 'initial_conditions' not in st.session_state:
        st.session_state['initial_conditions'] = []

    if 'variables' not in st.session_state:
        st.session_state['variables'] = []

    if 'variable_letters' not in st.session_state:
        st.session_state['variable_letters'] = []

    def text_input_callback():
        """
        Fired when a user inputs a new equation. The input is parsed into an equation,
        then the plot scaffold is created. Finally, the plot is populated with points, then
        drawn
        """
        try:
            (x_diff_eqn, y_diff_eqn) = generate_functions_from_input(st.session_state.dxdt, st.session_state.dydt)
            x_partition, y_partition, x_grid, y_grid = generate_plot_scaffold(st.session_state.x_segments,
                                                                              st.session_state.y_segments,
                                                                              st.session_state.y_min,
                                                                              st.session_state.y_max,
                                                                              st.session_state.x_min,
                                                                              st.session_state.x_max)
            dx, dy, magnitude = generate_plot_output(x_diff_eqn, y_diff_eqn, x_grid, y_grid)
            fig = draw_plot(dx, dy, magnitude, x_grid, y_grid, colormap=plt.cm.jet)

            # For each line that the user had drawn, recalculate their positions
            for initial_condition in st.session_state.initial_conditions:
                line = solve_ivp(initial_condition)
                fig.add_trace(go.Scatter(x=line[0, :], y=line[1, :]))

            render_plot(fig)
        except Exception as e:
            st.warning("An exception occurred while processing your equation. Try again, or refresh the page")
            print("ERROR WHILE PROCESSING EQUATION " + str(e))

    def click_plot_input_callback():
        """
        Fired when a user clicks on the plot. First, the point the user clicked on is parsed.
        Then, the differential equation is numerically analyzed with RK. Finally, the trace
        from that line is added to the plot, and re-rendered.
        """
        # text_input_callback()
        print(st.session_state.my_key)
        # Get the point clicked. It is a string of JSON. Trim it, then parse to json for use
        str_without_brack = st.session_state.my_key[1:-1]
        result = json.loads(str_without_brack)
        # split_str = st.session_state.my_key.split(':')

        # initial_conds = np.array([result['x'],
        #                           result['y']])
        initial_conds = [result['x'], result['y']]
        line = solve_ivp(initial_conds)
        st.session_state.initial_conditions.append(initial_conds)
        print(st.session_state.initial_conditions)
        fig = st.session_state.phase_plane
        fig.add_trace(go.Scatter(x=line[0, :], y=line[1, :]))
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
            print(st.session_state.initial_conditions)
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
            st.warning("Can't do that right now. Try making a plot first.")

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
        print(letter)
        print("Getting from session state: letter " + str(letter) + " val " + str(st.session_state[f"{letter}"]))
        text_input_callback()

    def import_session_state_callback(input_dict):
        """
        Uses the session state stored in JSON inside the input dict to load a session
        state, and redraw all necessary components.
        :param input_dict:
        """
        print("IMPORTING SAVED STATE " + str(input_dict))
        # Merge input dict into session state
        # try:
        #     print("TYPE OF SESSION STATE IS " + str(type(st.session_state)))
        #     st.session_state.update(input_dict)
        # except Exception as e:
        #     print("ERROR " + str(e))
        #     st.warning("Invalid dictionary input")

        # Attempt to merge input dict into session state. Then reload everything
        # try:
        for (key, val) in input_dict.items():
            # Different keys might require different mechanisms to load
            if key in float_included_keys:
                st.session_state[key] = float(val)
            elif key in integer_included_keys:
                st.session_state[key] = int(val)
            elif key == initial_condition_key:
                st.session_state[initial_condition_key] = list(json.loads(val))
            # elif key == variable_letter_key:
            #     for letter in val:
            #         st.session_state[variable_letter_key].append(letter)
            elif key == variable_key:
                # Variables are serialized as a list of variable JSON objects. Deserialize accordingly
                print("val total is " + str(val))
                for var_json in val:

                    print("Var json is " + str(var_json))
                    st.session_state[variable_key].append(
                        v.Variable(var_json['letter'], var_json['min_value'], var_json['max_value'], var_json['step_size'])
                    )
                    st.session_state[variable_letter_key].append(var_json['letter'])
            else:
                st.session_state[key] = val

                # st.session_state[key] = float(val) if key in float_included_keys \
                #     else int(val) if key in integer_included_keys else val

        # except Exception as ex:
        #     print("ERROR " + str(ex))
        #     st.warning("Invalid dictionary input")

        print("AFTER UPDATING, THIS IS SESSION STATE " + str(st.session_state))

        # initialize_variable_sliders()
        # text_input_callback()




    def solve_ivp(initial_conditions):
        """
        Uses carney_diff_eqs RK4 implementation to solve the initial value problem
        :param initial_conditions: An array containing [X, Y] initial condition coordinates
        :return: A line of X,Y points that solves the equation
        """

        print("ENTERING SOLVE IVP WITH SOVLER " + str(st.session_state.solver))
        # Required function by the implementation of si.solve_ivp
        def ode_sys(t, XY):
            dxdt = st.session_state.equation_system[0](XY[0], XY[1], t)
            dydt = st.session_state.equation_system[1](XY[0], XY[1], t)
            return [dxdt, dydt]

        """
        In order to provide the best user experience, we want the user to see
        the backwards and forwards solutions to their system, given their initial condition.
        I couldn't find a way to integrate forwards and backwards with a single method call (this would
        be a good thing to refactor), so instead the system is solved forwards in time,
        then backwards in time, then concatenated. 
        """


        if st.session_state.solver == 'Carney44':
            out = ode44.runge_kutta_second(
                st.session_state.equation_system,
                time,
                initial_conditions,
                h)
        else:
            out_forward = si.solve_ivp(
                ode_sys,
                np.array([0, st.session_state.t_max]),
                initial_conditions,
                method=st.session_state.solver,
                max_step=st.session_state.step_max)

            out_backwards = si.solve_ivp(
                ode_sys,
                np.array([0, -st.session_state.t_max]),
                initial_conditions,
                method=st.session_state.solver,
                max_step=st.session_state.step_max)
            out = np.hstack((np.flip(out_backwards.y, axis=1), out_forward.y))

        return out

    def generate_functions_from_input(dxdt_equation_input: str, dydt_equation_input: str):
        # Parse equations into lambdas
        x_input = parse_expr(f'{dxdt_equation_input}', transformations=transformations)
        y_input = parse_expr(f'{dydt_equation_input}', transformations=transformations)
        print("x_input = " + str(x_input))
        print("y_input = " + str(y_input))
        # Convert SymPy objects into ones that numpy can use

        x_input_with_vars = x_input.subs(
            [(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        y_input_with_vars = y_input.subs(
            [(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])

        x_diff_eqn = lambdify([x, y, t], x_input_with_vars, 'numpy')
        y_diff_eqn = lambdify([x, y, t], y_input_with_vars, 'numpy')
        print(x_diff_eqn)
        st.session_state.equation_system = np.array([x_diff_eqn, y_diff_eqn])

        return x_diff_eqn, y_diff_eqn

    def generate_plot_scaffold(x_segments, y_segments, y_min, y_max, x_min, x_max):
        # Generate 'skeleton' for plot
        x_partition: ndarray = np.linspace(x_min, x_max, x_segments)
        y_partition: ndarray = np.linspace(y_min, y_max, y_segments)
        x_grid, y_grid = np.meshgrid(x_partition, y_partition)
        return x_partition, y_partition, x_grid, y_grid

    def generate_plot_output(x_diff_eqn, y_diff_eqn,
                             x_grid: ndarray,
                             y_grid: ndarray):
        # Generate plot
        dx = x_diff_eqn(x_grid, y_grid, 0)
        dy = y_diff_eqn(x_grid, y_grid, 0)
        print("dx is " + str(dx))
        print("dy is " + str(dy))
        print("dx type is " + str(type(dx)))
        magnitude = np.sqrt(dx ** 2 + dy ** 2)  # magnitude
        return dx, dy, magnitude

    def draw_plot(dx: ndarray,
                  dy: ndarray,
                  magnitude: int,
                  x_grid: ndarray,
                  y_grid: ndarray,
                  colormap):
        print(str(np.shape(dx)))
        print(str(np.shape(x_grid)))

        fig = ff.create_quiver(
            x_grid, y_grid, dx / magnitude, dy / magnitude,
            scale=.25, arrow_scale=.4
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
            # st.plotly_chart(fig)
            print("this was called")
            fig.update_layout(yaxis_range=[st.session_state.y_min, st.session_state.y_max])
            fig.update_layout(xaxis_range=[st.session_state.x_min, st.session_state.x_max])
            st.session_state.clicked_points = plotly_events(fig, key="my_key")
            st.session_state.phase_plane = fig
            print(st.session_state.clicked_points)

    def initialize_variable_sliders():
        for var in st.session_state.variables:
            # if var.step_size != 0 and var.slider is None:
            print(var.letter)
            var.slider = col2.slider(f"Slider for {var.letter}",
                                     key=f"{var.letter}",
                                     min_value=var.min_value,
                                     max_value=var.max_value,
                                     step=var.step_size,
                                     on_change=slider_change_callback,
                                     args=(var.letter)
                                     )
            print(
                "this is a var: " + str(var) + " with value " + str(var.letter) + " and numerical value " + str(
                    var.slider))

    # # initialize all session keys
    # if 'dxdt' not in st.session_state:
    #     st.session_state['dxdt'] = '-x'
    # if 'dydt' not in st.session_state:
    #     st.session_state['dydt'] = 'y'

    # Setup screen
    col1, col2 = st.columns([1, 2])
    st.session_state.col2 = col2
    col1.header("Enter System of Equations")
    col2.header("Output")

    # Setup widgets ------------------------------------------

    import_expander = st.expander(label='Import Session')
    with import_expander:
        st.write("Paste in a session state. Then, click 'import' to load the saved session")
        import_form = st.form(key='import_state_form')
        with import_form:
            text_input = st.text_area("Paste session state here")
            import_submit = import_form.form_submit_button("Import")
            if text_input is not None and import_submit is True:
                # try:
                dict_input = json.loads(text_input)
                print("DICT INPUT IS " + str(dict_input))
                print("TYPE OF DICT INPUT IS " + str(type(dict_input)))
                import_session_state_callback(dict_input)
                # except Exception as e:
                #     print("EfRROR IN LOADING JSON" + str(e))
                #     st.warning("Something went wrong converting your input to a dictionary")




    dxdt_equation_input: str = col1.text_input("dx/dt = ", 'x', key="dxdt", on_change=text_input_callback)
    dydt_equation_input: str = col1.text_input("dy/dt = ", 'y', key="dydt", on_change=text_input_callback)

    # Clear curves
    clear_curves = col1.button("Clear Curves", key="clear", on_click=clear_curves_callback)

    # Render plot
    col1.button("Render Plot", on_click=text_input_callback)


    # Add a variable form
    var_form_expander = col1.expander(label='Add a Variable')
    with var_form_expander:
        var_form = st.form(key='add_var_form', clear_on_submit=True)
        with var_form:
            var_letter = st.text_input("Variable Letter", key='var_letter')
            var_min_value = st.number_input("Minimum Allowed Value", key='var_min_value')
            var_max_value = st.number_input("Maximum Allowed Value", key='var_max_value')
            var_step_size = st.number_input("Step Size (For Slider)",
                                            key='var_step_size')
            var_submit = st.form_submit_button("Submit Variable")
            print("inside form var submit is " + str(var_submit))
            print("hello i am here now")
            if var_letter is not None and var_submit is True:
                st.session_state.variables.append(
                    v.Variable(var_letter, var_min_value, var_max_value, var_step_size))
                st.session_state.variable_letters.append(var_letter)

    # for var in st.session_state.variables:
    #     # if var.step_size != 0 and var.slider is None:
    #     print(var.letter)
    #     var.slider = col2.slider(f"Slider for {var.letter}",
    #                              key=f"{var.letter}",
    #                              min_value=var.min_value,
    #                              max_value=var.max_value,
    #                              step=var.step_size,
    #                              on_change=slider_change_callback,
    #                              args=(var.letter)
    #                              )
    #     print(
    #         "this is a var: " + str(var) + " with value " + str(var.letter) + " and numerical value " + str(var.slider))

    initialize_variable_sliders()

    print("we are now here")
    col1.markdown("""---""")
    col1.markdown("""**Click on plot to draw solution curve. Or manually input initial conditions**""")

    # export_dict = {key: val for key, val in st.session_state.items() if key in included_keys}
    # print(str(export_dict))
    print(str(st.session_state))

    # Initial Conditions Form --------------------------------
    initial_conditions_expander = col1.expander(label='Manual Initial Conditions')
    with initial_conditions_expander:
        ic_form = st.form(key='initial_conditions_form')
        with ic_form:
            # form_col1, form_col2 = st.beta_columns(2)
            init_x = st.number_input("Initial X")
            init_y = st.number_input("Initial Y")
            print(init_x)
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

    # Setup Fine Tuning Expander --------------------------------
    fine_tuning_expander = st.expander(label='Fine Tuning')
    with fine_tuning_expander:
        st.write("Fine tune numerical solutions. "
                 "If solutions are not what you expect, "
                 "try messing around with these settings.  \n"
                 "Here are things you can try if you're having trouble:  \n"
                 "-- LSODA is a 'stiff' solver, "
                 "and good for numerically unstable systems.,  \n"
                 "-- If a solution line stops when it shouldn't, try increasing tMax.  \n"
                 "-- If a solution is diverging to infinity, try lowering tMax, or increasing the step size.  \n"
                 "-- If a solution looks jagged, lower the max step size, and potentially try DOP853.  \n"
                 "-- If you want solutions only forward in time, use Carney")
        tuning_col1, tuning_col2, tuning_col3 = st.columns(3)
        tuning_col1.selectbox("Numerical Solver",
                              (
                                  'RK45',
                                  'DOP853',
                                  'LSODA',
                                  'Carney44'
                              ), key='solver',
                              on_change=text_input_callback)
        tuning_col2.number_input("tMax", step=1, value=10, key='t_max', on_change=text_input_callback)
        tuning_col3.number_input("Max Step Size", step=.01, value=.1, key='step_max', on_change=text_input_callback)


    register_callback("my_key", click_plot_input_callback)

    print("please no")
    #
    # def my_callback():
    #     print("callback")
    #
    # if not 'my_key' in st.session_state:
    #     print("this happened once")
    #     register_callback("my_key", text_input_callback)

    # fig2 = px.line(x=[1], y=[1])
    # clicked_points = plotly_events(fig2, key="my_key")
    # print(clicked_points)
