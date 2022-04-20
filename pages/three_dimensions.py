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

def app(flag):
    # Setup global vars
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    t = symbols('t')

    # Setup boundary values
    x_segments: int = 20
    y_segments: int = 20
    z_segments: int = 20

    y_min: int = -4
    y_max: int = 4
    x_min: int = -4
    x_max: int = 4

    z_min: int = -4
    z_max: int = 4




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
        (x_diff_eqn, y_diff_eqn, z_diff_eqn) = generate_functions_from_input(st.session_state.dxdt, st.session_state.dydt, st.session_state.dzdt)
        x_partition, y_partition, z_partition, x_grid, y_grid, z_grid = \
            generate_plot_scaffold(x_segments, y_segments, y_min, y_max, x_min, x_max)
        #replace grids with partitions
        dx, dy, dz, magnitude = generate_plot_output(x_diff_eqn, y_diff_eqn, z_diff_eqn, x_grid, y_grid, z_grid)
        fig = draw_plot(dx, dy, dz, magnitude, x_grid, y_grid, z_grid)

        # For each line that the user had drawn, recalculate their positions
        # for initial_condition in st.session_state.initial_conditions:
        #     line = solve_ivp(initial_condition)
        #     fig.add_trace(go.Scatter(x=line[0, :], y=line[1, :]))

        render_plot(fig)

    def click_plot_input_callback():
        """
        Fired when a user clicks on the plot. First, the point the user clicked on is parsed.
        Then, the differential equation is numerically analyzed with RK. Finally, the trace
        from that line is added to the plot, and re-rendered.
        """
        #text_input_callback()
        # Get the point clicked. It is a string of JSON. Trim it, then parse to json for use
        str_without_brack = st.session_state.my_key[1:-1]
        result = json.loads(str_without_brack)



        initial_conds = np.array([result['x'],
                                  result['y']])
        line = solve_ivp(initial_conds)
        st.session_state.initial_conditions.append(initial_conds)
        print(st.session_state.initial_conditions)
        fig = st.session_state.phase_plane
        fig.add_trace(go.Scatter(x=line[0, :], y=line[1, :]))
        render_plot(fig)

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
        Uses carney_diff_eqs RK4 implementation to solve the initial value problem
        :param initial_conditions: An array containing [X, Y] initial condition coordinates
        :return: A line of X,Y points that solves the equation
        """
        return ode44.runge_kutta_second(
            st.session_state.equation_system,
            time,
            initial_conditions,
            h
        )


    def generate_functions_from_input(dxdt_equation_input: str, dydt_equation_input: str, dzdt_equation_input: str):
        # Parse equations into lambdas
        x_input = parse_expr(f'{dxdt_equation_input}', transformations=transformations)
        y_input = parse_expr(f'{dydt_equation_input}', transformations=transformations)
        z_input = parse_expr(f'{dzdt_equation_input}', transformations=transformations)
        print("x_input = " + str(x_input))
        print("y_input = " + str(y_input))
        print("z_input = " + str(z_input))
        # Convert SymPy objects into ones that numpy can use

        x_input_with_vars = x_input.subs([(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        y_input_with_vars = y_input.subs([(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])
        z_input_with_vars = z_input.subs([(this_v, st.session_state[f"{this_v}"]) for this_v in st.session_state.variable_letters])


        x_diff_eqn = lambdify([x, y, z, t], x_input_with_vars, 'numpy')
        y_diff_eqn = lambdify([x, y, z, t], y_input_with_vars, 'numpy')
        z_diff_eqn = lambdify([x, y, z, t], z_input_with_vars, 'numpy')
        print(x_diff_eqn)
        st.session_state.equation_system = np.array([x_diff_eqn, y_diff_eqn, z_diff_eqn])

        return x_diff_eqn, y_diff_eqn, z_diff_eqn


    def generate_plot_scaffold(x_segments, y_segments, y_min, y_max, x_min, x_max):
        # Generate 'skeleton' for plot
        x_partition: ndarray = np.linspace(x_min, x_max, x_segments)
        y_partition: ndarray = np.linspace(y_min, y_max, y_segments)
        z_partition: ndarray = np.linspace(z_min, z_max, z_segments)
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
            #st.plotly_chart(fig)
            print("this was called")
            # fig.update_layout(yaxis_range=[y_min, y_max])
            # fig.update_layout(xaxis_range=[x_min, x_max])
            # fig.update_layout(layout_zaxis_range=[z_min, z_max])


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
            var_step_size = st.number_input("Step Size (For Slider)", key='var_step_size') #TODO: Default value not zero
            var_submit = st.form_submit_button("Submit Variable")
            print("inside form var submit is " + str(var_submit))
            print("hello i am here now")
            if var_letter is not None and var_submit is True:
                st.session_state.variables.append(v.Variable(var_letter, var_min_value, var_max_value, var_step_size, None))
                st.session_state.variable_letters.append(var_letter)



    for var in st.session_state.variables:
        #if var.step_size != 0 and var.slider is None:
        print(var.letter)
        var.slider = col2.slider(f"Slider for {var.letter}",
                                 key=f"{var.letter}",
                                 min_value=var.min_value,
                                 max_value=var.max_value,
                                 step=var.step_size,
                                 on_change=slider_change_callback,
                                 args=(var.letter)
                                 )
        print("this is a var: " + str(var) + " with value " + str(var.letter) + " and numerical value " + str(var.slider))


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
            print(init_x)
            ic_submit = ic_form.form_submit_button("Submit IVP")

    # Setup Options Expander ----------------------------------
    options_expander = st.expander(label='Plot Options')
    with options_expander:
        st.write("PLOT BOUNDS")
        options_col1, options_col2, options_col3 = st.columns(3)
        options_col1.number_input("xMin", step=1)
        options_col2.number_input("xMax", step=1)
        options_col1.number_input("yMin", step=1)
        options_col2.number_input("yMax", step=1)
        options_col3.number_input("xSegments", step=1)
        options_col3.number_input("ySegments", step=1)
        st.write("ARROW SCALING")
        st.checkbox("Normalize Arrows", value=True)

    # Setup Plot Interactions Expander ------------------------
    plot_interactions_expander = st.expander(label='Plot Interactions')

    # register_callback("my_key", click_plot_input_callback)

    print("please no")
