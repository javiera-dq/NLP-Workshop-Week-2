import copy
import plotly.graph_objects as go

from dash import Dash, html
import dash_cytoscape as cyto
import json
# from sklearn.manifold import TSNE
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
# import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output

import requests

# Loading data for scatters of nodes.
with open("topic_modeling_data/job_topic_elm_list.json", "r") as f:
    startup_elms = json.load(f)
startup_elm_list = startup_elms["elm_list"]
startup_elm_list = [temp_dict for temp_dict in startup_elm_list if temp_dict['data']['topic_idx'] != -1]

# Preaparing data for visualizing token weights of each topic
with open("topic_modeling_data/job_topics.json", "r") as f:
    lda_topics = json.load(f)

topics_txt = [lda_topics[str(i - 1)] for i in range(len(lda_topics))]
topic_weight_tuple_list = [[(j.split("*")[1].replace('"', ""), j.split("*")[0]) for j in i]
                           for i in topics_txt]
topics_txt = [[j.split("*")[1].replace('"', "") for j in i] for i in topics_txt]
topics_txt = ["; ".join(i) for i in topics_txt]


# Returns information of selected node.
def get_node_data(node_data):
    # When nothing is selected
    if len(node_data) == 0:
        return None, -1, None

    node_idx = node_data[0]['id']
    text = node_data[0]['text']
    topic_idx = node_data[0]['topic_idx']
    topic_color = node_data[0]['color']
    return node_idx, topic_idx, topic_color




def generate_topic_bar_graph(topic_idx, color='rgb(248, 248, 249)'):
    topic_list = [temp[0] for temp in topic_weight_tuple_list[topic_idx]]
    weight_list = [float(temp[1]) for temp in topic_weight_tuple_list[topic_idx]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weight_list[::-1], y=topic_list[::-1],
        orientation='h',
        marker_color=color
    ))

    return fig

with open('topic_modeling_data/topic_color.txt') as f:
    lines = f.readlines()
topic_color = [line.rstrip('\n') for line in lines]

topics_html = list()
for topic_html in [
    html.Span(["Topic {}".format(i - 1) + ": " + topics_txt[i]],
              style={"color": topic_color[i]}
              # style={"color": 'grey'}

              )
    for i in range(len(topics_txt))
]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

def_stylesheet = [
    {
        # "selector": "node",
        "selector": 'node[topic_idx != -1]',
        "style": {
            "width": "data(node_size)",
            "height": "data(node_size)",
            "width": 0.25,
            "height": 0.25,
            'background-color': 'data(color)',
            'opacity': 0.1,
            'font-size': '1px'
        },
    },
]

col_swatch = px.colors.qualitative.Dark24


def take_dict_value(key, dictionary):
    try:
        return dictionary[key]
    except:
        return key


app = Dash(__name__)

# The main layout of the application
app.layout = dbc.Row([
    dbc.Row([
        cyto.Cytoscape(
            id='cytoscape_core',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '400px'},
            elements=startup_elm_list,
            # elements=[],
            stylesheet=def_stylesheet,
        ),
    ]),

    dbc.Row(
        [

            dbc.Col([
                dbc.Alert(
                    id="node-data",
                    children="Click on a node to see its details here",
                    color="secondary",
                    style={"fontSize": "32px"}  # Adjust the font size as desired

                ),
            ],
                width=6,
            ),

            dbc.Col([
                dcc.Graph(
                    figure=generate_topic_bar_graph(0),
                    id='topic_weight_bar'
                ),
            ],
                width=3,
            ),

        ]
    ),

])

# Callbacks for highlighting selected topic.
@app.callback(
    Output('cytoscape_core', 'stylesheet', ),
    [
        Input("cytoscape_core", "selectedNodeData"),
    ],
)
def generate_stylesheet(node_data,
                        ):
    if node_data is None or len(node_data)==0:
        return def_stylesheet

    updated_stylesheet = copy.deepcopy(def_stylesheet)

    selected_node_idx, selected_topic_idx, selected_topic_color = get_node_data(node_data)

    updated_stylesheet.append({
        "selector": 'node[topic_idx = {}]'.format(selected_topic_idx),
        "style": {
            "border-opacity": 0,
            "opacity": 0.5,
            "color": "black",
            "text-opacity": 0.8,
            "font-size": 5,
            'z-index': 9999,
            "text-wrap": "wrap"
        }
    }
    )

    updated_stylesheet.append({
        "selector": 'node[id = "{}"]'.format(selected_node_idx),
        "style": {
            # 'background-color': '#B10DC9',
            "border-color": "black",
            "border-width": 0.1,
            "border-opacity": 1,
            "opacity": 1,
            # "label": "data(label)",
            "color": "black",
            "text-opacity": 0.8,
            "font-size": 0.5,
            'z-index': 9999,
            "text-wrap": "wrap"
        }
    }
    )

    return updated_stylesheet


# Callbacks for showing text of each node.
@app.callback(
    Output("node-data", "children"), [Input("cytoscape_core", "selectedNodeData")]
)
def display_nodedata(datalist):
    contents = "Click on a node to see its details here"

    # When nothing is selected
    if datalist is None or len(datalist)==0:
        return contents

    text = datalist[0]['text']
    topic_idx = datalist[0]['topic_idx']

    contents = []
    contents.append(html.H3("Topic: {}".format(topic_idx)))
    contents.append(html.H3("Text: {}".format(text)))
    return contents

# Callbacks for visualizing token weights of each topic.
@app.callback(
    Output("topic_weight_bar", "figure"),
    [
        Input("cytoscape_core", "selectedNodeData"),
    ]
)
def update_topic_weight_barchart(node_data):
    if node_data is None:
        return go.Figure()

    selected_node_idx, selected_topic_idx, selected_topic_color = get_node_data(node_data)

    if selected_topic_idx == -1:
        return go.Figure()

    return generate_topic_bar_graph(selected_topic_idx, color=selected_topic_color)


if __name__ == '__main__':
    app.run_server(debug=True)