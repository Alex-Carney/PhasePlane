import plotly.express as px
import plotly.graph_objects as go
x= [1,2,3]
y= [4,5,6]
fig = go.Figure()
fig.add_trace(go.Scatter(x = x, y = y))
fig.show()