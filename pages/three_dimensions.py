import numpy as np
from numpy import ndarray
from sympy import lambdify, symbols
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, \
    implicit_multiplication_application, convert_xor
import plotly.graph_objects as go
import json
from common import variable as v
import scipy.integrate as si



def app(flag):

    ##########################################################
    ######## GLOBAL DATA AND INITIAL STATE VALUES ############
    ##########################################################

    # Setup global vars
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    t = symbols('t')

    # Setup global constants
    global included_keys
    # Don't include variables in here. They have to be serialized separately
    included_keys = ['y_max', 'y_min', 'z_min', 'z_max', 'initial_conditions', 'step_max',
                     'x_max', 'solver', 'dydt', 'dxdt', 'dzdt', 'x_segments',
                     'x_min', 'y_segments', 'z_segments', 't_max']
    global float_included_keys
    float_included_keys = ['y_max', 'y_min', 'z_min', 'z_max', 'step_max', 'x_max',
                           'var_step_size', 'x_min', 't_max']
    global integer_included_keys
    integer_included_keys = ['y_segments', 'x_segments', 'z_segments']

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
    if 'z_segments' not in st.session_state:
        st.session_state.z_segments = 20
    if 'x_min' not in st.session_state:
        st.session_state.x_min = -4
    if 'x_max' not in st.session_state:
        st.session_state.x_max = 4
    if 'y_min' not in st.session_state:
        st.session_state.y_min = -4
    if 'y_max' not in st.session_state:
        st.session_state.y_max = 4
    if 'z_max' not in st.session_state:
        st.session_state.z_max = 4
    if 'z_min' not in st.session_state:
        st.session_state.z_min = -4

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
            (x_diff_eqn, y_diff_eqn, z_diff_eqn) = generate_functions_from_input(st.session_state.dxdt,
                                                                                 st.session_state.dydt,
                                                                                 st.session_state.dzdt)
            x_partition, y_partition, z_partition, x_grid, y_grid, z_grid = \
                generate_plot_scaffold(st.session_state.x_segments,
                                       st.session_state.y_segments,
                                       st.session_state.y_min,
                                       st.session_state.y_max,
                                       st.session_state.x_min,
                                       st.session_state.x_max)
            # replace grids with partitions
            dx, dy, dz, magnitude = generate_plot_output(x_diff_eqn, y_diff_eqn, z_diff_eqn, x_grid, y_grid, z_grid)
            fig = draw_plot(dx, dy, dz, magnitude, x_grid, y_grid, z_grid)

            # For each line that the user had drawn, recalculate their positions
            for initial_condition in st.session_state.initial_conditions:
                out = solve_ivp(initial_condition)
                fig.add_trace(go.Scatter3d(x=out[0, :], y=out[1, :], z=out[2, :]))
            render_plot(fig)
        except:
            st.warning("An exception occurred while processing your equation. Try again, or refresh the page")




    def manual_ivp_input_callback(init_x, init_y, init_z):
        """
        Fired when someone submits a form for an initial condition.
        There is no click plot callback for 3 dimensions because it happens
        too often, even when users are just trying to pan around the screen
        """
        try:
            initial_conds = [init_x, init_y, init_z]
            out = solve_ivp(initial_conds)
            st.session_state.initial_conditions.append(initial_conds)
            fig = st.session_state.phase_plane
            fig.add_trace(go.Scatter3d(x=out[0, :], y=out[1, :], z=out[2, :]))
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

    def solve_ivp(initial_conditions):
        """
        I wanted to use my own implementation of RK44 to solve 3D problems, but ran
        into an issue where most interesting problems in 3D are "stiff", or chaotic,
        meaning they are very sensitive to numerical instability. Therefore, I'm using
        an external library for stiff systems
        :param initial_conditions:
        :return:
        """

        def ode_sys(t, XYZ):
            eqn1 = st.session_state.equation_system[0]
            eqn2 = st.session_state.equation_system[1]
            eqn3 = st.session_state.equation_system[2]
            dxdt = eqn1(XYZ[0], XYZ[1], XYZ[2], t)
            dydt = eqn2(XYZ[0], XYZ[1], XYZ[2], t)
            dzdt = eqn3(XYZ[0], XYZ[1], XYZ[2], t)
            return [dxdt, dydt, dzdt]


        out = si.solve_ivp(
            ode_sys,
            np.array([0, st.session_state.t_max]),
            initial_conditions,
            method=st.session_state.solver
        )


        return out.y

    def generate_functions_from_input(dxdt_equation_input: str, dydt_equation_input: str, dzdt_equation_input: str):
        # Parse equations into lambdas
        x_input = parse_expr(f'{dxdt_equation_input}', transformations=transformations)
        y_input = parse_expr(f'{dydt_equation_input}', transformations=transformations)
        z_input = parse_expr(f'{dzdt_equation_input}', transformations=transformations)

        # Convert SymPy objects into ones that numpy can use
        x_input_with_vars = x_input.subs(
            [(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        y_input_with_vars = y_input.subs(
            [(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        z_input_with_vars = z_input.subs(
            [(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])

        x_diff_eqn = lambdify([x, y, z, t], x_input_with_vars, 'numpy')
        y_diff_eqn = lambdify([x, y, z, t], y_input_with_vars, 'numpy')
        z_diff_eqn = lambdify([x, y, z, t], z_input_with_vars, 'numpy')
        st.session_state.equation_system = np.array([x_diff_eqn, y_diff_eqn, z_diff_eqn])

        return x_diff_eqn, y_diff_eqn, z_diff_eqn

    def generate_plot_scaffold(x_segments, y_segments, y_min, y_max, x_min, x_max):
        # Generate 'skeleton' for plot
        x_partition: ndarray = np.linspace(st.session_state.x_min, st.session_state.x_max, int(st.session_state.x_segments))
        y_partition: ndarray = np.linspace(st.session_state.y_min, st.session_state.y_max, int(st.session_state.y_segments))
        z_partition: ndarray = np.linspace(st.session_state.z_min, st.session_state.z_max, int(st.session_state.z_segments))
        x_grid, y_grid, z_grid = np.meshgrid(x_partition, y_partition, z_partition)
        return x_partition, y_partition, z_partition, x_grid, y_grid, z_grid

    def generate_plot_output(x_diff_eqn, y_diff_eqn, z_diff_eqn,
                             x_grid: ndarray,
                             y_grid: ndarray,
                             z_grid: ndarray):
        # Generate plot
        dx = x_diff_eqn(x_grid, y_grid, z_grid, 0)
        dy = y_diff_eqn(x_grid, y_grid, z_grid, 0)
        dz = z_diff_eqn(x_grid, y_grid, z_grid, 0)
        magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # magnitude
        return dx, dy, dz, magnitude

    def draw_plot(dx: ndarray,
                  dy: ndarray,
                  dz: ndarray,
                  magnitude: int,
                  x_grid: ndarray,
                  y_grid: ndarray,
                  z_grid: ndarray,
                  ):

        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        dx_flat = dx.flatten()
        dy_flat = dy.flatten()
        dz_flat = dz.flatten()


        # Phase cube is created differently than the 1D and 2D phase planes
        fig = go.Figure(data=go.Cone(
            x=x_flat, y=y_flat, z=z_flat, u=dx_flat, v=dy_flat, w=dz_flat,
            colorscale='Jet',
            sizeref=2,
            name="Phase Cube",
            showlegend=True
        ))

        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                                     camera_eye=dict(x=1.2, y=1.2, z=0.6)))

        return fig

    def render_plot(fig):
        with col2:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[st.session_state.x_min, st.session_state.x_max]),
                    yaxis=dict(range=[st.session_state.y_min, st.session_state.y_max]),
                    zaxis=dict(range=[st.session_state.z_min, st.session_state.z_max])
                )
            )

            st.plotly_chart(fig)
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

    # Setup screen
    col1, col2 = st.columns([1, 2])
    st.session_state.col2 = col2
    col1.header("Enter Tri-System of Differential Equations")
    col2.header("Output")

    # Setup widgets ------------------------------------------


    # Import session state expander
    import_expander = st.expander(label='Import Session')
    with import_expander:
        st.write("Paste in a session state. Then, click 'import' to load the saved session")
        import_form = st.form(key='import_state_form', clear_on_submit=True)
        with import_form:
            text_input = st.text_area("Paste session state here")
            import_submit = import_form.form_submit_button("Import")
            if text_input is not None and import_submit is True:
                try:
                    dict_input = json.loads(text_input)
                    import_session_state_callback(dict_input)
                except Exception as e:
                    print("EfRROR IN LOADING JSON" + str(e))
                    st.warning("Something went wrong converting your input to a dictionary")

    # User equation input
    dxdt_equation_input: str = col1.text_input("dx/dt = ", '-x', key="dxdt", on_change=text_input_callback)
    dydt_equation_input: str = col1.text_input("dy/dt = ", "y", key="dydt", on_change=text_input_callback)
    dzdt_equation_input: str = col1.text_input("dz/dt = ", "z", key="dzdt", on_change=text_input_callback)

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
                                            key='var_step_size')  # TODO: Default value not zero
            var_submit = st.form_submit_button("Submit Variable")
            if var_letter is not None and var_letter != "" and var_submit is True:
                duplicate_key: bool = var_letter in st.session_state.variable_letters
                invalid_step: bool = var_step_size == 0
                invalid_range: bool = var_min_value >= var_max_value
                if duplicate_key or invalid_range or invalid_step:
                    st.warning("Invalid entry. Step cannot be 0. Min value cannot be greater than max,"
                               "cannot make duplicate variables. You did duplicate key error " + str(duplicate_key)
                               + " invalid step error " + str(invalid_step) + " invalid range error " + str(invalid_range)
                               + " if you don't understand why, refresh the page")
                else:
                    st.session_state.variables.append(
                        v.Variable(var_letter, var_min_value, var_max_value, var_step_size))
                    st.session_state.variable_letters.append(var_letter)

    initialize_variable_sliders()

    col1.markdown("""---""")
    col1.markdown("""**Click on plot to draw solution curve. Or manually input initial conditions**""")

    # Initial Conditions Form --------------------------------
    initial_conditions_expander = col1.expander(label='Manual Initial Conditions')
    with initial_conditions_expander:
        ic_form = st.form(key='initial_conditions_form')
        with ic_form:
            init_x = st.number_input("Initial X")
            init_y = st.number_input("Initial Y")
            init_z = st.number_input("Initial Z")
            ic_submit = ic_form.form_submit_button("Submit IVP")
            if init_x is not None and init_y is not None and init_z is not None and ic_submit is True:
                manual_ivp_input_callback(init_x, init_y, init_z)


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
        options_col1.number_input("zMin", step=1, value=-4, key='z_min', on_change=text_input_callback)
        options_col2.number_input("zMax", step=1, value=4, key='z_max', on_change=text_input_callback)
        options_col3.number_input("xSegments", step=1, value=20, key='x_segments', on_change=text_input_callback)
        options_col3.number_input("ySegments", step=1, value=20, key='y_segments', on_change=text_input_callback)
        options_col3.number_input("zSegments", step=1, value=20, key='z_segments', on_change=text_input_callback)

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
                              ), key='solver',
                              on_change=text_input_callback)
        tuning_col2.number_input("tMax", step=1, value=10, key='t_max', on_change=text_input_callback)
        tuning_col3.number_input("Max Step Size", step=.01, value=.1, key='step_max', on_change=text_input_callback)
