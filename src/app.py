import pandas as pd
import plotly.graph_objects as go
import os
from dash import Dash, html, dcc , callback , Output, Input
import dash_bootstrap_components as dbc

# Some color stuff.
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

# First, find the results file.
results_file = next(s for s in os.listdir() if s.endswith('.out'))
table_filename = 'table.txt'

# First filter out repeated rows and any rows without a tab, and also any rows that start with a tab.
lines_seen = set()
outfile = open(table_filename, 'w')
for line in open(results_file, 'r'):
    if '\t' in line and line[0] != '\t' and line not in lines_seen:
        outfile.write(line)
        lines_seen.add(line)
outfile.close()

df = pd.read_table(table_filename,delimiter="\t")
t_list = [t for t,_ in df.groupby('fevals')]
num_pred_parameters = df.iloc[0]['pred'].count(';')+1
num_prey_parameters = df.iloc[0]['prey'].count(';')+1

# App initialisation.
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# App layout.
app.layout = dbc.Container(fluid=True,children=[
    html.H1("Pareto fronts."),
    html.Hr(),
    dbc.Row(children=[
        dbc.Col([
            dcc.Slider(step=None,marks=dict([(i,str(t)) for i,t in enumerate(t_list)]),value=(len(t_list)-1),id='slider') ,
            dcc.Graph(figure={},id='pf-chart') ,
        ],width=6) ,
        dbc.Col([
            dbc.Row(children=[
                dbc.Col(width=2,children=[
                    dcc.Checklist(options=list(range(num_pred_parameters)),value=[],id='pred-param-checklist')
                ]) ,
                dbc.Col([dcc.Graph(figure={},id='pred-param-chart')])
            ]) ,
            dbc.Row(children=[
                dbc.Col(width=2,children=[
                    dcc.Checklist(options=list(range(num_prey_parameters)),value=[],id='prey-param-checklist')
                ]) ,
                dbc.Col([dcc.Graph(figure={},id='prey-param-chart')])
            ]) ,
        ],width=6)
    ],align='center')
])

@callback(
    Output(component_id='pf-chart',component_property='figure'),
    Input(component_id='slider', component_property='value')
)
def update_pf_chart(i):
    fig = go.Figure()
    t = t_list[i]
    colors = fig.layout['template']['layout']['colorway']

    # Plot the pareto front.
    pf_color = colors[0]
    df_pf = df[df.apply(lambda x:
        x['is_ea_point']==False and x['is_coea_point']==False and x['fevals']==t,
        axis=1)]
    pf = [ ( row['f1'] , row['f2'] ) for _,row in df_pf.iterrows()]
    x = [ p[0] for p in pf ]
    y = [ p[1] for p in pf ]
    x_line , y_line = [0] , []
    for p in pf:
        x_line += [p[0],p[0]]
        y_line += [p[1],p[1]]
    y_line += [0]
    fig.add_trace(go.Scatter(
        x=x_line,y=y_line,
        marker_color=pf_color,
        fill='tozeroy',fillcolor=f"rgba{(*hex_to_rgb(pf_color), 0.1)}",hoverinfo='none',mode='lines'))
    fig.add_trace(go.Scatter(
        x=x,y=y,
        marker_color=pf_color,
        mode='markers',
        showlegend=False))
    
    # Plot the coea performance.
    coea_color = colors[1]
    coea_row = df[df.apply(lambda x:
        x['is_ea_point']==False and x['is_coea_point']==True and x['fevals']==t,
        axis=1)].iloc[0]
    fig.add_trace(go.Scatter(
        x=[coea_row['f1']],y=[coea_row['f2']],
        marker_color=coea_color,mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(
        x=[coea_row['f1'],coea_row['f1']],y=[0,1], # Replace with upper and lower bounds. Do for other such traces too.
        hoverinfo='none',line=dict(color=f"rgba{(*hex_to_rgb(coea_color), 0.5)}",dash='dash'),showlegend=False
    ))

    # Plot the ea performance.
    ea_color = colors[2]
    ea_row = df[df.apply(lambda x:
        x['is_ea_point']==True and x['is_coea_point']==False and x['fevals']==t,
        axis=1)].iloc[0]
    fig.add_trace(go.Scatter(
        x=[ea_row['f1']],y=[ea_row['f2']],
        marker_color=ea_color,mode='markers',showlegend=False))
    fig.add_trace(go.Scatter(
        x=[0,1],y=[ea_row['f2'],ea_row['f2']], # Replace with upper and lower bounds. Do for other such traces too.
        hoverinfo='none',line=dict(color=f"rgba{(*hex_to_rgb(ea_color), 0.5)}",dash='dash'),showlegend=False
    ))

    # Add the line x=y to chart.
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],hoverinfo='none',line=dict(color='black',dash='dash'),showlegend=False))
    
    # Set scale
    buff = 0.1
    x_min , x_max = min(df['f1']) , max(df['f1'])
    y_min , y_max = min(df['f2']) , max(df['f2'])
    x_mid = ( x_min + x_max ) / 2
    y_mid = ( y_min + y_max ) / 2
    width = (1+buff) * max( x_max - x_min , y_max - y_min )

    # Layout
    fig.update_layout(xaxis_range=[ x_mid - width / 2 , x_mid + width / 2 ])
    fig.update_layout(yaxis_range=[ y_mid - width / 2 , y_mid + width / 2 ])
    fig.update_layout(width=1000,height=1000,showlegend=False)
    
    return fig

def update_param_chart(i,params,column_string):
    fig = go.Figure()
    t = t_list[i]
    colors = fig.layout['template']['layout']['colorway']
    df_pf = df[df.apply(lambda x:
        x['is_ea_point']==False and x['is_coea_point']==False and x['fevals']==t,
        axis=1)]
    def get_individual (row):
        s = row[column_string]
        for c in list(' []|'):
            s=s.replace(c,'')
        return s.split(';')
    individuals = [get_individual(row) for _,row in df_pf.iterrows()]
    for j in params:
        fig.add_trace(go.Scatter(
            x=list(range(len(individuals))),y=[float(pred[j]) for pred in individuals],
            mode='lines+markers',marker_color=colors[j%len(colors)]
        ))
    return fig

@callback(
    Output(component_id='pred-param-chart',component_property='figure'),
    Input(component_id='slider', component_property='value') ,
    Input(component_id='pred-param-checklist', component_property='value')
)
def update_pred_param_chart(i,params):
    return update_param_chart(i,params,'pred')

@callback(
    Output(component_id='prey-param-chart',component_property='figure'),
    Input(component_id='slider', component_property='value') ,
    Input(component_id='prey-param-checklist', component_property='value')
)
def update_pred_param_chart(i,params):
    return update_param_chart(i,params,'prey')





# Run the app
if __name__ == '__main__':
    app.run(debug=True)