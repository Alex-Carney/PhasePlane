import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sympy import Lambda, lambdify, symbols
import carney_diff_eqs as ode44
import streamlit as st
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, \
    implicit_multiplication_application, convert_xor, lambda_notation
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from streamlit_plotly_events import plotly_events
from components_callbacks import register_callback


# Setup global vars
transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
x = symbols('x')
y = symbols('y')
t = symbols('t')

# Setup boundary values
x_segments: int = 20
y_segments: int = 20
y_min: int = -3
y_max: int = 3
x_min: int = -4
x_max: int = 4

# global col1, col2, dxdt_equation_input, dydt_equation_input


def text_input_callback():
    # fired when a user enters a new equation
    (x_diff_eqn, y_diff_eqn) = generate_functions_from_input(st.session_state.dxdt, st.session_state.dydt)
    x_partition, y_partition, x_grid, y_grid = generate_plot_scaffold(x_segments, y_segments,
                                                                      y_min, y_max, x_min, x_max)
    dx, dy, magnitude = generate_plot_output(x_diff_eqn, y_diff_eqn, x_grid, y_grid)
    draw_plot(dx, dy, magnitude, x_grid, y_grid, colormap=plt.cm.jet)


def generate_functions_from_input(dxdt_equation_input: str, dydt_equation_input: str):
    # Parse equations into lambdas
    x_input = parse_expr(f'{dxdt_equation_input}', transformations=transformations)
    y_input = parse_expr(f'{dydt_equation_input}', transformations=transformations)
    # Convert SymPy objects into ones that numpy can use
    x_diff_eqn = lambdify([x, y, t], x_input, 'numpy')
    y_diff_eqn = lambdify([x, y, t], y_input, 'numpy')
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
    magnitude = np.sqrt(dx ** 2 + dy ** 2)  # magnitude
    return dx, dy, magnitude

def draw_plot(dx: ndarray,
              dy: ndarray,
              magnitude: int,
              x_grid: ndarray,
              y_grid: ndarray,
              colormap):
    fig = ff.create_quiver(
        x_grid, y_grid, dx/magnitude, dy/magnitude,
        scale=.25, arrow_scale=.4
    )
    font_dict = dict(family='Arial', size=26, color='black')
    fig.add_hline(0)
    fig.add_vline(0)
    fig.update_layout(font=font_dict,  # change the font
                      plot_bgcolor='white',  # get rid of the god awful background color
                      margin=dict(r=20, t=20, b=10)
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


    with col2:
        st.plotly_chart(fig)



def draw_matplotlib_plot(
        dx: ndarray,
        dy: ndarray,
        magnitude: int,
        x_partition: ndarray,
        y_partition: ndarray,
        colormap=plt.cm.jet):
    fig, ax = plt.subplots()
    ax.quiver(x_partition, y_partition, dx / magnitude, dy / magnitude, magnitude, cmap=colormap)
    ax.set_xlim(min(x_partition), max(x_partition))
    ax.set_ylim(min(y_partition), max(y_partition))
    ax.hlines(0, min(x_partition), max(x_partition), 'k')
    ax.vlines(0, min(y_partition), max(y_partition), 'k')
    col2.pyplot(fig)



# Setup screen
col1, col2 = st.columns(2)
col1.header("Enter System of Equations")
col2.header("Output")

# Setup widgets
dxdt_equation_input: str = col1.text_input("dx/dt = ", '-x', key="dxdt", on_change=text_input_callback)
dydt_equation_input: str = col1.text_input("dy/dt = ", 'y', key="dydt", on_change=text_input_callback)

print("this shouldn't happen")

def my_callback():
    print("new component value:", st.session_state.my_key)
    print("i happened")


register_callback("my_key", my_callback)

fig2 = px.line(x=[1], y=[1])
clicked_points = plotly_events(fig2, key="my_key")
print(clicked_points)