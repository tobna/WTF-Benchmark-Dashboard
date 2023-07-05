from dash import Dash, html, dcc, dash_table
import dash_daq as daq
from utils import prepare_table_info
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
import os


# --------------- Data loading --------------------------------
tbl_data, tbl_cols, tbl_tooltips = prepare_table_info()
point_metrics = ['image resolution (pretraining)', 'GPUS (pretraining)', 'lr (pretraining)',
                 'image resolution (finetuning)', 'GPUs (finetuning)', 'lr (finetuning)',
                 'inference VRAM @32', 'inference VRAM @128', 'inference VRAM @1', 'inference VRAM @64',
                 'total finetuning time', 'total validation time', 'throughput', 'throughput batch size',
                 'training VRAM', 'training VRAM (single GPU)', 'number of parameters', 'FLOPs',
                 'validation loss', 'training loss', 'top-5 validation accuracy', 'top-5 training accuracy',
                 'top-1 validation accuracy', 'top-1 training accuracy']
point_metrics = sorted(point_metrics)


# ------------ App definition ----------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = 'WTF Benchmark'
app.layout = html.Div([
    html.H1(f"What Transformer to Favor: Benchmark", style={'width': '75%', 'display': 'inline-block'}),
    daq.PowerButton(id='pareto-pwr-btn', label='Pareto front', labelPosition='top',
                    style={'width': '5%', 'display': 'inline-block'}, size=30),
    daq.ToggleSwitch(id='pareto-side-switch', style={'width': '10%', 'display': 'inline-block'},
                     label="left <-> right"),
    daq.ToggleSwitch(id='overall-switch', style={'width': '10%', 'display': 'inline-block'},
                     label="overall stats <-> per epoch stats"),
    dcc.Graph(id="graph", style={'width': '100vw', 'height': '80vh'}),
    html.Div([
        html.P("x:", style={'display': 'inline-block', 'width': '10%'}),
        dcc.Dropdown(id='x-picker', clearable=False, style={'width': '40%', 'display': 'inline-block'},
                     value='throughput', options=point_metrics),
        html.P("y:", style={'display': 'inline-block', 'width': '10%'}),
        dcc.Dropdown(id='y-picker', clearable=False, style={'width': '40%', 'display': 'inline-block'},
                     value='top-1 validation accuracy', options=point_metrics)
    ]),
    dash_table.DataTable(id='run-list', filter_action='native', columns=tbl_cols, data=tbl_data, tooltip=tbl_tooltips,
                         style_table={'overflow': 'scroll', 'width': '100%', 'maxHeight': '100%'},
                         fixed_rows={'headers': True}, style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
    dcc.Store(id='highlight-store', data=[]), dcc.Store(id='hidden-runs-store', data={'per epoch': [], 'global': []}),
    dcc.Store(id='legend-entries', data=[]), dcc.Store(id='graph-layout-store', data={}),
    html.H2('Paper'),
    dcc.Markdown('This data was collected for the paper [What Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers](LINK). '
                 'We opensource our code on [GitHub](https://gitfront.io/r/user-5921586/dmRcCBtFqbtK/WhatTransformerToFavor/).'),
    html.H4('Citation'),
    dcc.Markdown('```\nCitation...\n```'),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('This site uses cookies'), close_button=False),
        dbc.ModalBody('By clicking OK, you agree to our use of functionally necessary cookies!'),
        dbc.ModalFooter(dbc.Button('OK', id='cookie-ok-btn', n_clicks=0))
    ], id='cookie-modal', is_open=True, centered=True, backdrop='static'),
])



# ----------------------- Callbacks --------------------------------
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_figure'
    ),
    inputs=[
        Input('x-picker', 'value'),
        Input('y-picker', 'value'),
        Input('run-list', 'derived_virtual_data'),
        Input('pareto-pwr-btn', 'on'),
        Input('pareto-side-switch', 'value'),
        Input('overall-switch', 'value'),
        Input('highlight-store', 'data')
    ],
    state=[
        State('hidden-runs-store', 'data'),
        State('graph-layout-store', 'data')
    ],
    output=[
        Output('graph', 'figure'),
        Output('legend-entries', 'data')
    ]
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='set_highlight'
    ),
    inputs=[
        Input('graph', 'clickData'),
        Input('run-list', 'active_cell')
    ],
    state=[
        State('run-list', 'derived_virtual_data')
    ],
    output=[
        Output('highlight-store', 'data'),
        Output('run-list', 'style_data_conditional')
    ]
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='picker_options'
    ),
    inputs=[
        Input('overall-switch', 'value')
    ],
    output=[
        Output('x-picker', 'options'),
        Output('x-picker', 'value'),
        Output('y-picker', 'options'),
        Output('y-picker', 'value')
    ]
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='buttons_enabled'
    ),
    inputs=[
        Input('pareto-pwr-btn', 'on'),
        Input('overall-switch', 'value')
    ],
    output=[
        Output('pareto-side-switch', 'disabled'),
        Output('pareto-pwr-btn', 'disabled')
    ]
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='hidden_items_store'
    ),
    inputs=[
        Input('graph', 'restyleData')
    ],
    state=[
        State('hidden-runs-store', 'data'),
        State('legend-entries', 'data'),
        State('overall-switch', 'value')
    ],
    output=Output('hidden-runs-store', 'data')
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='plot_layout_store'
    ),
    inputs=[
        Input('graph', 'relayoutData'),
        Input('x-picker', 'value'),
        Input('y-picker', 'value')
    ],
    state=[
        State('graph-layout-store', 'data'),
    ],
    output=Output('graph-layout-store', 'data')
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='cookie_modal'
    ),
    inputs=[
        Input('cookie-ok-btn', 'n_clicks')
    ],
    output=Output('cookie-modal', 'is_open')
)

if __name__ == '__main__':
    app.run_server(debug='DEBUG' not in os.environ or not os.environ['DEBUG'])