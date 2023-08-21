import argparse
from dash import Dash, html, dcc, dash_table
import dash_daq as daq
from utils import prepare_table_info
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
import os
from time import sleep
import logging


parser = argparse.ArgumentParser()
parser.add_argument('-reload', action='store_true', help='reload current data every few seconds')

RELOAD = parser.parse_args().reload
RELOAD_FILE = 'data_tmp.json'

# --------------- Data loading --------------------------------
tbl_data, tbl_cols, tbl_tooltips = prepare_table_info()
point_metrics = ['throughput [ims/s]']
point_metrics = sorted(point_metrics)


# ------------ App definition ----------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = 'WTF Benchmark'
app.layout = html.Div([
    html.H1(f"What Transformer to Favor: Benchmark", style={'width': '75%', 'display': 'inline-block'}),
    daq.PowerButton(id='pareto-pwr-btn', label='Pareto front', labelPosition='top',
                    style={'width': '10%', 'display': 'inline-block'}, size=30),
    daq.ToggleSwitch(id='overall-switch', style={'width': '15%', 'display': 'inline-block'},
                     label="overall stats <-> per epoch stats"),
    dcc.Graph(id="graph", style={'width': '99vw', 'height': '80vh'}),
    html.Table([html.Tbody([html.Tr([
        html.Td(html.P("x:", style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center', 'lineHeight': '34px', 'fontSize': '1.5rem', 'fontFamily': 'var(--bs-body-font-family)'})),
        html.Td(dcc.Dropdown(id='x-picker', clearable=False, style={'width': '100%', 'display': 'inline-block'},
                     value='throughput [ims/s]', options=point_metrics)),
        html.Td(html.P("y:", style={'display': 'inline-block', 'width': '100%', 'textAlign': 'center', 'height': '34px', 'fontSize': '1.5rem', 'fontFamily': 'var(--bs-body-font-family)'})),
        html.Td(dcc.Dropdown(id='y-picker', clearable=False, style={'width': '100%', 'display': 'inline-block'},
                     value='top-1 validation accuracy', options=point_metrics))
    ])])], style={'width': '95%', 'margin': '10px'}),
    dash_table.DataTable(id='run-list', filter_action='native', columns=tbl_cols, data=tbl_data, tooltip=tbl_tooltips,
                         style_table={'overflow': 'scroll', 'width': '100%', 'maxHeight': '100%'}, sort_action='native',
                         fixed_rows={'headers': True}, style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
    dcc.Store(id='highlight-store', data=[]),
    dcc.Store(id='hidden-runs-store', data={'per epoch': [], 'global': []}),
    dcc.Store(id='legend-entries', data=[]),
    dcc.Store(id='graph-layout-store', data={}),
    dcc.Store(id='pareto-right', data=True),
] + ([dcc.Interval(id='update-data', interval=30*1000, n_intervals=0), dcc.Download('download-data'), html.Button("Download Data as CSV", 'download-btn')]
                           if RELOAD else [html.H2('Paper'),
    dcc.Markdown('This data was collected for the paper [What Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers](https://arxiv.org/abs/2308.09372). '
                 'Have fun playing around with it, and analyzing it deeper. '
                 'For more information on our methodology, checkout the paper and [code](https://github.com/tobna/WhatTransformerToFavor).'),
    html.H4('Citation'),
    dcc.Markdown('```\nCitation coming soon...\n```'),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('This site uses cookies'), close_button=False),
        dbc.ModalBody('By clicking OK, you agree to our use of functionally necessary cookies!'),
        dbc.ModalFooter(dbc.Button('OK', id='cookie-ok-btn', n_clicks=0))
    ], id='cookie-modal', is_open=True, centered=True, backdrop='static')]))



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
        Input('pareto-right', 'data'),
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
    output=Output('pareto-pwr-btn', 'disabled')
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
        function_name='pareto_right'
    ),
    inputs=[
        Input('x-picker', 'value'),
    ],
    output=Output('pareto-right', 'data')
)

if not RELOAD:
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

if RELOAD:
    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='download_data'
        ),
        inputs=[
            Input("download-btn", 'n_clicks')
        ],
        state=[
            State('run-list', 'derived_virtual_data')
        ],
        output=Output('download-data', 'data')
    )

if RELOAD:
    @app.callback([Output('run-list', 'data'), Output('run-list', 'columns')], Input('update-data', 'n_intervals'))
    def reload_data(n):
        while not os.path.isfile(RELOAD_FILE):
            sleep(10)

        data, cols, _ = prepare_table_info(RELOAD_FILE, order_by_date=True)
        logging.info('reloaded data')
        return data, cols


if __name__ == '__main__':
    debug = not 'DEBUG' in os.environ or os.environ['DEBUG']
    logging.basicConfig(format=f"%(asctime)s; %(levelname)s: %(message)s",
                        datefmt="%d.%m.%Y %H:%M:%S", level=logging.INFO,
                        handlers=[logging.StreamHandler()])
    logging.info(f'Debug={debug}')

    try:
        if RELOAD:
            import data_updating
            load_process = data_updating.start_data_process(n_workers=5, update_interval=30)

        app.run_server(debug=debug)
    except Exception as ex:
        if RELOAD:
            load_process.kill()
        raise ex
