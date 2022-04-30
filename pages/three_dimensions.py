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
import plotly.express as px
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
    z = symbols('z')
    t = symbols('t')

    # Setup boundary values
    # x_segments: int = 20
    # y_segments: int = 20
    # z_segments: int = 20
    #
    # y_min: int = -4
    # y_max: int = 4
    # x_min: int = -4
    # x_max: int = 4
    #
    # z_min: int = -4
    # z_max: int = 4

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
    # Setup other global values
    t_step = .01
    h = .1
    time = np.r_[0:10:t_step]

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
            print("size of x " + str(np.shape(out[0, :])))
            print("size of y " + str(np.shape(out[1, :])))
            print("size of z " + str(np.shape(out[2, :])))


            print("x_input = " + str(max(out[0, :])))
            print("y_input = " + str(max(out[1, :])))
            print("z_input = " + str(max(out[2, :])))
            fig.add_trace(go.Scatter3d(x=out[0, :], y=out[1, :], z=out[2, :]))

        render_plot(fig)

    def manual_ivp_input_callback(init_x, init_y, init_z):
        """
        Fired when someone submits a form for an initial condition.
        There is no click plot callback for 3 dimensions because it happens
        too often, even when users are just trying to pan around the screen
        """


        # try:
        initial_conds = np.array([init_x, init_y, init_z])
        out = solve_ivp(initial_conds)
        st.session_state.initial_conditions.append(initial_conds)
        fig = st.session_state.phase_plane
        fig.add_trace(go.Scatter3d(x=out[0, :], y=out[1, :], z=out[2, :]))
        render_plot(fig)
        # except:
        # st.warning("Can't do that right now. Try making a plot first")

    def clear_curves_callback():
        """
        Fired when a user wants to clear all curves from the plot
        """
        fig = st.session_state.phase_plane
        fig.data = [fig.data[0]]  # keep only the 0th trace (the arrows)
        st.session_state.initial_conditions = []
        render_plot(fig)

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

    def solve_ivp(initial_conditions):
        """
        I wanted to use my own implementation of RK44 to solve 3D problems, but ran
        into an issue where most interesting problems in 3D are "stiff", or chaotic,
        meaning they are very sensitive to numerical instability. Therefore, I'm using
        an external library for stiff systems
        :param initial_conditions:
        :return:
        """
        print("ENTERING SOLVE_IVP BLOCK WITH ICS " + str(initial_conditions))
        print("TYPE OF " + str(type(st.session_state.equation_system[0])))

        print("EXAMPLE INPUTS ")
        print(st.session_state.equation_system[0](10, 10, 10, 10))
        print(st.session_state.equation_system[1](10, 10, 10, 10))
        print(st.session_state.equation_system[2](10, 10, 10, 10))

        ics = np.array([initial_conditions[0], initial_conditions[1], initial_conditions[2]])

        # r = 10
        # eqn1 = lambda t, x, y, z: 10 * (-x + y)
        # eqn2 = lambda t, x, y, z: r * x - y - x * z
        # eqn3 = lambda t, x, y, z: -(8 / 3) * z + x * y
        #
        # st.session_state.equation_system[0] = eqn1
        # st.session_state.equation_system[1] = eqn2
        # st.session_state.equation_system[2] = eqn3


        def ode_sys(t, XYZ):
            eqn1 = st.session_state.equation_system[0]
            eqn2 = st.session_state.equation_system[1]
            eqn3 = st.session_state.equation_system[2]


            dxdt = eqn1(XYZ[0], XYZ[1], XYZ[2], t)
            dydt = eqn2(XYZ[0], XYZ[1], XYZ[2], t)
            dzdt = eqn3(XYZ[0], XYZ[1], XYZ[2], t)
            return [dxdt, dydt, dzdt]
        t_span = np.array([0, 100])
        out = si.solve_ivp(ode_sys, t_span, ics, method='LSODA')

        xout = out.y[0, :]
        yout = out.y[1, :]
        zout = out.y[2, :]

        print("MAX OF XOUT IS " + str(max(xout)))
        print("MAX OF YOUT IS " + str(max(yout)))
        print("MAX OF ZOUT IS " + str(max(zout)))

        return out.y

        # return ode44.runge_kutta_any_order(
        #     st.session_state.equation_system,
        #     time,
        #     initial_conditions,
        #     h
        # )

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
        print(x_diff_eqn)
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
        a = 1
        dx = x_diff_eqn(x_grid, y_grid, z_grid, 0)
        dy = y_diff_eqn(x_grid, y_grid, z_grid, 0)
        dz = z_diff_eqn(x_grid, y_grid, z_grid, 0)
        print("dx is " + str(dx))
        print("dy is " + str(dy))
        print("dz is " + str(dy))
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

        print("dx is " + str(dx))
        print("dy is " + str(dy))
        print("dz is " + str(dz))
        print("xgrid is " + str(x_grid))
        print("ygrid is " + str(y_grid))
        print("zgrid is " + str(z_grid))

        print("dx is " + str(np.shape(dx)))
        print("dy is " + str(np.shape(dy)))
        print("dz is " + str(np.shape(dz)))
        print("xgrid is " + str(np.shape(x_grid)))
        print("ygrid is " + str(np.shape(y_grid)))
        print("zgrid is " + str(np.shape(z_grid)))

        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        dx_flat = dx.flatten()
        dy_flat = dy.flatten()
        dz_flat = dz.flatten()

        print("max val X is " + str(max(x_flat)))
        print("max val Y is " + str(max(y_flat)))
        print("max val Z is " + str(max(z_flat)))

        fig = go.Figure(data=go.Cone(
            # x=x_grid, y=y_grid, z=z_grid, u=dx/magnitude, v=dy/magnitude, w=dz/magnitude,
            x=x_flat, y=y_flat, z=z_flat, u=dx_flat, v=dy_flat, w=dz_flat,
            colorscale='Jet',
            sizeref=2
        ))

        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                                     camera_eye=dict(x=1.2, y=1.2, z=0.6)))

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.quiver(x_grid, y_grid, z_grid, dx/magnitude, dy/magnitude, dz/magnitude, length=.2)
        # st.pyplot(fig)

        # fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))
        font_dict = dict(family='Arial', size=26, color='black')
        # fig.add_hline(0)
        # fig.add_vline(0)
        # fig.update_layout(font=font_dict,  # change the font
        #                   plot_bgcolor='white',  # get rid of the god awful background color
        #                   margin=dict(r=60, t=40, b=40),
        #                   )
        # # x and y-axis formatting
        # fig.update_yaxes(title_text='Y-axis',  # axis label
        #                  showline=True,  # add line at x=0
        #                  linecolor='black',  # line color
        #                  linewidth=2.4,  # line size
        #                  ticks='outside',  # ticks outside axis
        #                  tickfont=font_dict,  # tick label font
        #                  mirror='allticks',  # add ticks to top/right axes
        #                  tickwidth=2.4,  # tick width
        #                  tickcolor='black',  # tick color
        #                  )
        # fig.update_xaxes(title_text='X-axis',
        #                  showline=True,
        #                  showticklabels=True,
        #                  linecolor='black',
        #                  linewidth=2.4,
        #                  ticks='outside',
        #                  tickfont=font_dict,
        #                  mirror='allticks',
        #                  tickwidth=2.4,
        #                  tickcolor='black',
        #                  )
        return fig

    def render_plot(fig):
        with col2:
            # st.plotly_chart(fig)
            print("this was called")
            # fig.update_layout(yaxis_range=[y_min, y_max])
            # fig.update_layout(xaxis_range=[x_min, x_max])
            # fig.update_layout(layout_zaxis_range=[z_min, z_max])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[st.session_state.x_min, st.session_state.x_max]),
                    yaxis=dict(range=[st.session_state.y_min, st.session_state.y_max]),
                    zaxis=dict(range=[st.session_state.z_min, st.session_state.z_max])
                )
            )

            st.plotly_chart(fig)
            # st.session_state.clicked_points = plotly_events(fig, key="my_key")
            st.session_state.phase_plane = fig
            # print(st.session_state.clicked_points)

    # Setup screen
    col1, col2 = st.columns([1, 2])
    st.session_state.col2 = col2
    col1.header("Enter Tri-System of Differential Equations")
    col2.header("Output")

    # Setup widgets ------------------------------------------
    dxdt_equation_input: str = col1.text_input("dx/dt = ", '-x', key="dxdt", on_change=text_input_callback)
    dydt_equation_input: str = col1.text_input("dy/dt = ", "y", key="dydt", on_change=text_input_callback)
    dzdt_equation_input: str = col1.text_input("dz/dt = ", "z", key="dzdt", on_change=text_input_callback)

    # Clear curves
    clear_curves = col1.button("Clear Curves", key="clear", on_click=clear_curves_callback)

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
            print("inside form var submit is " + str(var_submit))
            print("hello i am here now")
            if var_letter is not None and var_submit is True:
                st.session_state.variables.append(
                    v.Variable(var_letter, var_min_value, var_max_value, var_step_size, None))
                st.session_state.variable_letters.append(var_letter)

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
            "this is a var: " + str(var) + " with value " + str(var.letter) + " and numerical value " + str(var.slider))

    print("we are now here")
    col1.markdown("""---""")
    col1.markdown("""**Click on plot to draw solution curve. Or manually input initial conditions**""")

    # Initial Conditions Form --------------------------------
    initial_conditions_expander = col1.expander(label='Manual Initial Conditions')
    with initial_conditions_expander:
        ic_form = st.form(key='initial_conditions_form')
        with ic_form:
            # form_col1, form_col2 = st.beta_columns(2)
            init_x = st.number_input("Initial X")
            init_y = st.number_input("Initial Y")
            init_z = st.number_input("Initial Z")
            ic_submit = ic_form.form_submit_button("Submit IVP")
            if init_x is not None and init_y is not None and init_z is not None and ic_submit is True:
                manual_ivp_input_callback(init_x, init_y, init_z)

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
        st.write("ARROW SCALING")
        st.checkbox("Normalize Arrows", value=True)

    # Setup Plot Interactions Expander ------------------------
    plot_interactions_expander = st.expander(label='Plot Interactions')

    # register_callback("my_key", click_plot_input_callback)

    print("please no")
