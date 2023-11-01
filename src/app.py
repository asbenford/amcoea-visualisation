import pandas as pd
import plotly.graph_objects as go
import os
from dash import Dash, html, dcc , callback , Output, Input , State , ALL , MATCH , Patch
import dash_bootstrap_components as dbc

# Some color stuff.
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

# First, find the results file.
experiment_list = [s for s in os.listdir() if not s.endswith('.txt') and not s.endswith('.py') and s != '__pycache__']
home_dir = os.getcwd()
table_filename = 'table.txt'

def is_header(str) : return str.startswith('group')

df_dict = {}
# First filter out repeated rows and any rows without a tab, and also any rows that start with a tab.
for experiment in experiment_list:
    header_written = False
    os.chdir(home_dir+'/'+experiment)
    outfile = open(table_filename, 'w')
    file = [s for s in os.listdir() if s.endswith('.out')][0]
    for line in open(file, 'r'):
        if '\t' in line and is_header(line) and not header_written:
            outfile.write(line[:-1]+'\texperiment\n')
            header_written = True
        if '\t' in line and not is_header(line):
            outfile.write(line[:-1]+'\t'+experiment+'\n')
    df_dict[experiment] = pd.read_table(table_filename,delimiter="\t")
    outfile.close()
os.chdir(home_dir)

def t_list(dataframe) : return [t for t,_ in dataframe.groupby('f_evals')]

# App initialisation.
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server



# App layout.

def comparison_selector(i) :
    plot_types = ['pareto','average','worst-case','none']
    checkbox_label = 'plot sub-payoffs'
    if i == 0:
        default_checkbox_value = [checkbox_label]
    else:
        default_checkbox_value = []
    return dbc.Row(children=[
        dbc.Col([html.Div("Select comparison:")],width='auto') ,
        dbc.Col([
            dcc.Dropdown(options=[''],value='',id={'type':'group-dropdown','index':i})
        ],width=3) ,
        dbc.Col([ dcc.RadioItems(plot_types,plot_types[i%len(plot_types)],inline=True, id={'type':'plot-type-radio','index':i}) ]) ,
        dbc.Col([ dcc.Checklist(options=[checkbox_label],value=default_checkbox_value,id={'type':'subpayoff-checkbox','index':i})])
    ])

def parameter_chart(id,section_title,prefix,experiment):
    df = df_dict[experiment]
    col_names = [s for s in list(df.columns.values) if s.startswith(prefix)]
    group_list = [s for s,_ in df.groupby('group')]
    return dbc.Row(children=[dbc.Col([
        html.H3(section_title) ,
        dcc.Dropdown(options=group_list,value=group_list[0],id={'type':'param-dropdown','id':id}) ,
        dbc.Row(children=[
            dbc.Col(width=2,children=[
                dcc.Checklist(
                    options=[{'value':s,'label':s.removeprefix(prefix)} for s in col_names],value=[],
                    id={'type':'param-checklist','id':id})
            ]) ,
            dbc.Col([dcc.Graph(figure={},id={'type':'param-chart','id':id})])
        ]) 
    ])])

app.layout = dbc.Container(fluid=True,children=[
    html.H1("AMCoEA Experiment Results") ,
    html.Hr() ,
    dbc.Row(children=[
        dbc.Col([
            html.H3("Pareto front") ,
            dbc.Row(children=[
                dbc.Col([html.Div("Select experiment:")],width='auto') ,
                dbc.Col([
                    dcc.Dropdown(options=experiment_list,value=experiment_list[0],id='experiment-dropdown')
                ],width=3)
            ]) ,
            dbc.Row(children=[
                dbc.Col([html.Div("Select fevals:")],width='auto') ,
                dbc.Col([
                    dcc.Slider(step=None,marks=dict([(0,"0")]),value=0,id='slider')
                ])
            ]) ,
            dbc.Row(children=[
                dbc.Col([html.Button("Add comparison", id='add-comparison-btn',n_clicks=0)],width=2) ,
                dbc.Col([html.Button("Remove comparison", id='remove-comparison-btn',n_clicks=0)],width=2)
            ],justify='start'),
            html.Div(children=[comparison_selector(i) for i in range(3)],id='comparison-selector-container') ,
            dbc.Row(children=[dcc.Graph(figure={},style={'width': '70vh', 'height': '70vh'},id='pf-chart')]) ,
            dbc.Row(children=[dcc.Graph(figure={},style={'width': '70vh', 'height': '30vh'},id='subpayoff-chart')]) ,
            html.Hr() ,
            html.H3("File viewer") ,
            dbc.Row(children=[
                dbc.Col([html.Div("Select file:")],width='auto') ,
                dbc.Col([
                    dcc.Dropdown(options=[''],value='',id='file-viewer-select')
                ],width=3)
            ]) ,
            html.Div(
                id='file-viewer-display' ,
                style={'whiteSpace': 'pre-line','font-family':'monospace','background-color':'lightblue','maxHeight':'1000px','overflow':'scroll','white-space':'pre'} ,
                children=''
            )
        ],width=6) ,
        dbc.Col([],width=6,id='param-column')
    ],align='start')
])


# Dynamically add or remove comparisons.
@callback(
    Output(component_id='comparison-selector-container',component_property='children') ,
    Input(component_id='add-comparison-btn',component_property='n_clicks') ,
    Input(component_id='remove-comparison-btn',component_property='n_clicks')
)
def modify_comparison_selector_container(adds,removes):
    target = 3 + adds - removes
    return [comparison_selector(i) for i in range(target)]


# Experiment selection
@callback(
    Output(component_id='slider',component_property='marks'),
    Output(component_id='slider',component_property='value'),
    Output(component_id='file-viewer-select',component_property='options') ,
    Output(component_id='file-viewer-select',component_property='value') ,
    Output(component_id={'type':'group-dropdown','index':ALL},component_property='options') ,
    Output(component_id={'type':'group-dropdown','index':ALL},component_property='value') ,
    Output(component_id='param-column',component_property='children') ,
    Input(component_id='experiment-dropdown',component_property='value') ,
    State(component_id='comparison-selector-container',component_property='children')
)
def experiment_select(experiment,selector_holder):
    num_selectors = len(selector_holder)
    group_list = [s for s,_ in df_dict[experiment].groupby('group')]
    os.chdir(home_dir+'/'+experiment)
    available_files = os.listdir()
    os.chdir(home_dir)
    return (
        dict([(i,str(t)) for i,t in enumerate(t_list(df_dict[experiment]))]) ,
        (len(t_list(df_dict[experiment]))-1) ,
        available_files ,
        available_files[0] ,
        [group_list for _ in range(num_selectors)] ,
        [group_list[0] for _ in range(num_selectors)] ,
        [
            parameter_chart('pred','Controller parameters','pred_param:',experiment) ,
            html.Hr() ,
            parameter_chart('prey','Scenario parameters','prey_param:',experiment)
        ] )


# Generating the pareto front chart.
@callback(
    Output(component_id='pf-chart',component_property='figure'),
    Output(component_id='subpayoff-chart',component_property='figure'),
    Input(component_id='experiment-dropdown',component_property='value'),
    Input(component_id={'type':'group-dropdown','index':ALL},component_property='value') ,
    Input(component_id={'type':'plot-type-radio','index':ALL},component_property='value') ,
    Input(component_id={'type':'subpayoff-checkbox','index':ALL},component_property='value'),
    Input(component_id='slider', component_property='value')
)
def update_charts(experiment,groups,plot_types,plot_subpayoffs,i):
    pf_fig = go.Figure()
    sp_fig = go.Figure()
    df_exp = df_dict[experiment]
    t = t_list(df_exp)[i]
    colors = pf_fig.layout['template']['layout']['colorway']

    for j,group in enumerate(groups):

        # Obtain the pareto front.
        df_pf = df_exp[df_exp.apply(lambda x:
            x['group']==group and x['f_evals']==t,
            axis=1)]
        pf = [ ( row['f1'] , row['f2'] ) for _,row in df_pf.iterrows()]

        if plot_types[j] == 'pareto':
        # Plot the pareto front.
            x = [ p[0] for p in pf ]
            y = [ p[1] for p in pf ]
            x_line , y_line = [0] , []
            for p in pf:
                x_line += [p[0],p[0]]
                y_line += [p[1],p[1]]
            y_line += [0]
            pf_fig.add_trace(go.Scatter(
                x=x_line,y=y_line,
                marker_color=colors[j],
                fill='tozeroy',fillcolor=f"rgba{(*hex_to_rgb(colors[j]), 0.1)}",hoverinfo='none',mode='lines',
                name=group))
            pf_fig.add_trace(go.Scatter(
                x=x,y=y,
                marker_color=colors[j],
                mode='markers',
                showlegend=False))
            
        if plot_types[j] == 'average':
        # Plot the ea performance.
            pf_fig.add_trace(go.Scatter(
                x=[pf[0][0]],y=[pf[0][1]],
                marker_color=colors[j],mode='markers',showlegend=False))
            pf_fig.add_trace(go.Scatter(
                x=[0,1],y=[pf[0][1],pf[0][1]], # Replace with upper and lower bounds. Do for other such traces too.
                hoverinfo='none',line=dict(color=f"rgba{(*hex_to_rgb(colors[j]), 0.5)}",dash='dash'),
                name=group
            ))
        
        if plot_types[j] == 'worst-case':
        # Plot the coea performance.
            pf_fig.add_trace(go.Scatter(
                x=[pf[-1][0]],y=[pf[-1][1]],
                marker_color=colors[j],mode='markers',showlegend=False))
            pf_fig.add_trace(go.Scatter(
                x=[pf[-1][0],pf[-1][0]],y=[0,1], # Replace with upper and lower bounds. Do for other such traces too.
                hoverinfo='none',line=dict(color=f"rgba{(*hex_to_rgb(colors[j]), 0.5)}",dash='dash'),
                name=group
            ))
        
        if len(plot_subpayoffs[j]) > 0 :
        # Also plot monetary cost and customer satisfaction.
            sp_fig.add_trace(go.Scatter(
                x = [row['f1'] for _,row in df_pf.iterrows()] ,
                y = [row['desirability:monetary_cost_per_kwh'] for _,row in df_pf.iterrows()] ,
                mode='lines+markers',name='monetary_cost_per_kwh: '+group,
                line=dict(color=colors[j],dash='dash')
            ))
            sp_fig.add_trace(go.Scatter(
                x = [row['f1'] for _,row in df_pf.iterrows()] ,
                y = [row['desirability:customer_satisfaction'] for _,row in df_pf.iterrows()] ,
                mode='lines+markers',name='customer_satisfaction: '+group,
                line=dict(color=colors[j],dash='dot')
            ))

    # Add the line x=y to chart.
    pf_fig.add_trace(go.Scatter(x=[0,1],y=[0,1],hoverinfo='none',line=dict(color='black',dash='dash'),showlegend=False))
    
    # Set scale
    buff = 0.1
    x_min , x_max = min(df_exp['f1']) , max(df_exp['f1'])
    y_min , y_max = min(df_exp['f2']) , max(df_exp['f2'])
    x_mid = ( x_min + x_max ) / 2
    y_mid = ( y_min + y_max ) / 2
    width = (1+buff) * max( x_max - x_min , y_max - y_min )

    # Sizing
    pf_fig.update_layout(xaxis_range=[ x_mid - width / 2 , x_mid + width / 2 ])
    sp_fig.update_layout(xaxis_range=[ x_mid - width / 2 , x_mid + width / 2 ])
    pf_fig.update_layout(yaxis_range=[ y_mid - width / 2 , y_mid + width / 2 ])

    # Labels and legend
    pf_fig.update_layout(
        xaxis_title="worst-case controller payoff" ,
        yaxis_title="average controller payoff",
        showlegend=True,
        legend=dict(
            xanchor='left',yanchor='top',
            x=0,y=1
        ))
    
    sp_fig.update_layout(
        xaxis_title="worst-case controller payoff",
        showlegend=True,
        legend=dict(
            xanchor='left',yanchor='bottom',
            x=0,y=1
        ))


    
    return pf_fig , sp_fig

# Generating the parameter charts.
@callback(
    Output(component_id={'type':'param-chart','id':MATCH},component_property='figure'),
    Input(component_id='experiment-dropdown',component_property='value'),
    Input(component_id={'type':'param-dropdown','id':MATCH},component_property='value'),
    Input(component_id='slider', component_property='value') ,
    Input(component_id={'type':'param-checklist','id':MATCH}, component_property='value'),
    State(component_id={'type':'param-checklist','id':MATCH}, component_property='options')
)
def update_param_chart(experiment,group,i,params,options):
    fig = go.Figure()
    df_exp = df_dict[experiment]
    t = t_list(df_exp)[i]
    colors = fig.layout['template']['layout']['colorway']
    df_pf = df_exp[df_exp.apply(lambda x:
        str(x['experiment'])==experiment and x['group']==group and x['f_evals']==t,
        axis=1)]
    for j,opt in enumerate(options):
        if opt['value'] in params:
            fig.add_trace(go.Scatter(
                x=list(range(df_pf.shape[0])),y=df_pf[opt['value']],
                mode='lines+markers',marker_color=colors[j%len(colors)]
        ))
    return fig

@callback(
    Output(component_id='file-viewer-display',component_property='children') ,
    Input(component_id='experiment-dropdown',component_property='value') ,
    Input(component_id='file-viewer-select',component_property='value')
)
def view_file(experiment,filename):
    os.chdir(home_dir+'/'+experiment)
    with open(filename) as f:
        text_to_display = f.read()
    return text_to_display





# Run the app
if __name__ == '__main__':
    app.run(debug=True)