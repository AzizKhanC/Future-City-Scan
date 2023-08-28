# Import dash dependencies
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

# Import plotly dependencies
import plotly.express as px
import plotly.figure_factory as ff

# Import dependencies for io and data manipulation
import pandas as pd
import numpy as np
import math
import geopandas as gpd
import pathlib

def get_pandas_data(csv_filename: str) -> pd.DataFrame:
   '''
   Load data from /data directory as a pandas DataFrame
   using relative paths. Relative paths are necessary for
   data loading to work in Heroku.
   '''
#    PATH = pathlib.Path(__file__).parent
   PATH = pathlib.Path().parent
   DATA_PATH = PATH.joinpath("data").resolve()
   return pd.read_csv(DATA_PATH.joinpath(csv_filename))

def get_json_data(json_filename: str):
   '''
   Load data from /data directory as a pandas DataFrame
   using relative paths. Relative paths are necessary for
   data loading to work in Heroku.
   '''
   PATH = pathlib.Path().parent
   DATA_PATH = PATH.joinpath("data").resolve()
   json_data= gpd.read_file(DATA_PATH.joinpath(json_filename))
   return json_data

def get_asset_path_for_logo(logo_filename: str):
   '''
   Load data from /data directory as a pandas DataFrame
   using relative paths. Relative paths are necessary for
   data loading to work in Heroku.
   '''
   PATH = pathlib.Path().parent
#    DATA_PATH = PATH.joinpath("assets").resolve()
   DATA_PATH = PATH.joinpath("data").resolve()
   logo_filepath= DATA_PATH.joinpath(logo_filename)
   return logo_filepath

#--------------------------------------------------------------------#
#  Population data
#--------------------------------------------------------------------#

# Centroid
cos = pd.read_csv('data/csv_popdynamics_SSP2_2100.csv')
cos = cos.rename(columns={'popdynamics': 'Population'}) 
# cos = get_pandas_data('csv_Population_SSP2_2100.csv')
#  define earth radius (in meters) for hexagone resolution calculation)
earth_radius = 6.371e6
zoom=10
opacity=0.5

center_lat = cos[cos != 0.0].bfill(axis=1)['center_lat']
center_lon = cos[cos != 0.0].bfill(axis=1)['center_lon']
max= cos.loc[cos['Population'].idxmax()]['Population']
def roundup(x):
    return int(math.ceil(x / 10000)) * 10000
max=roundup(max)
label_max= int(math.ceil(max / 1000))


#  Population graph data
pop_df_scenrio_year = get_pandas_data('BGD_Chittagong_popdynamics.csv')
pop_df_scenrio_year["id"] = pop_df_scenrio_year.index
pop_df_scenrio_year = pd.wide_to_long(pop_df_scenrio_year, stubnames='', i=['id'], j='year').reset_index()
pop_df_scenrio_year.columns = [*pop_df_scenrio_year.columns[:-1], 'Population']
ssps= pop_df_scenrio_year['Scenario'].unique()
options=[{"value": x, "label": x} for x in ssps]
value= list(ssps) 

# Example map with OSM as basemap
url='panel_popdynamics.geojson'
pop_geo_df = get_json_data(url)
pop_geo_df = pop_geo_df.rename(columns={'popdynami': 'Population'}) 


#--------------------------------------------------------------------#
#  GDP data
#--------------------------------------------------------------------#


#  Population graph data
gdp_df_scenrio_year = get_pandas_data('BGD_Chittagong_gdp.csv')
gdp_df_scenrio_year["id"] = gdp_df_scenrio_year.index
gdp_df_scenrio_year = pd.wide_to_long(gdp_df_scenrio_year, stubnames='', i=['id'], j='year').reset_index()
gdp_df_scenrio_year.columns = [*gdp_df_scenrio_year.columns[:-1], 'GDP']
ssps= gdp_df_scenrio_year['Scenario'].unique()
options=[{"value": x, "label": x} for x in ssps]
value= list(ssps) 

# Example map with OSM as basemap
url='panel_gdp.geojson'
gdp_geo_df = get_json_data(url)
gdp_geo_df = gdp_geo_df.rename(columns={'gdp': 'GDP'}) 




#--------------------------------------------------------------------#
#  Logo
#--------------------------------------------------------------------#

crp_logo = "assets/crp_logo.png" #path to crp logo
# crp_logo = "crp_logo.png" #path to crp logo
# crp_logo = get_asset_path_for_logo(crp_logo)
###################################################
#############  Start of the Dash app #############
###################################################
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CERULEAN], # LUX FLATLY CERULEAN SPACELAB
                title='GFDRR.CRP  |  Future City Scan',
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, height=device-height, initial-scale=1, maximum-scale=1.2'}]
                )


###################################################
#############  GCP  #############
###################################################

server= app.server

###################################################
#############  Header /Navigation bar #############
###################################################
navbar = dbc.Navbar(
                [
                html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                                [   dbc.Col(html.Img(src=crp_logo, height="30px")),
                                    dbc.Col(dbc.NavbarBrand("   |  Future City Scan", className="ml-2")),
                                ],
                                align="center",
                                # no_gutters=True,
                                className="g-0",

                                ),
                        # href="https://www.gfdrr.org/en/crp",
                    ),
                ],
                color="dark",
                dark=True,
                className="mb-4",
)


###################################################
#############  Controls Population#################
###################################################


years= pop_geo_df.Year.unique().tolist()
scenarios= pop_geo_df.Scenario.unique().tolist()
control_tab = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Year"),
                    dcc.Dropdown(
                        id='year', 
                        options=[{"value": x, "label": x} for x in years],
                        value= years[0], #'RdYlGn',
                        style={'margin-bottom':'10px'}
                    ), 
                ]
            ),
         ),
        dbc.Card(   
            dbc.CardBody(
                [
                    html.H6("Scenario"),
                    dcc.Dropdown(
                        id="scenario",
                        options=[{"value": x, "label": x}  for x in scenarios],
                        value= scenarios[0], #'RdYlGn',
                        style={'margin-bottom':'10px'}
                    ) 
                ]
            )
        ),
    ]
)


###################################################
#############  Controls GDP#################
###################################################


years= gdp_geo_df.Year.unique().tolist()
scenarios= gdp_geo_df.Scenario.unique().tolist()
control_tab_gdp = dbc.CardGroup(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Year"),
                    dcc.Dropdown(
                        id='year_gdp', 
                        options=[{"value": x, "label": x} for x in years],
                        value= years[0], #'RdYlGn',
                        style={'margin-bottom':'10px'}
                    ), 
                ]
            ),
         ),
        dbc.Card(   
            dbc.CardBody(
                [
                    html.H6("Scenario"),
                    dcc.Dropdown(
                        id="scenario_gdp",
                        options=[{"value": x, "label": x}  for x in scenarios],
                        value= scenarios[0], #'RdYlGn',
                        style={'margin-bottom':'10px'}
                    ) 
                ]
            )
        ),
    ]
)


credits_tab = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            Developed by CRP: [CRP.GFDRR](https://www.gfdrr.org/en/crp)
        
            """
        ),
    ),
    className="mt-0",
)


intro_markdown= dcc.Markdown(
    """

    Demographic Dynamics context: The City Resilience Program (CRP) works to build resilient cities with the capacity to plan for and mitigate adverse impacts of disasters and climate change, thus enabling them to save lives, reduce losses, and unlock economic and social potential. The program is a partnership between the World Bank and GFDRR. A City Scan is a rapid geospatial assessment of the critical resilience challenges that cities face using the best publicly available global datasets and open source tools. The output is a package of maps, data visualizations and insights that integrate features of both the built and natural environments. 
    
    """
)


intro_markdown_gdp= dcc.Markdown(
    """

    GDP and overarching Economic Outlook: The City Resilience Program (CRP) works to build resilient cities with the capacity to plan for and mitigate adverse impacts of disasters and climate change, thus enabling them to save lives, reduce losses, and unlock economic and social potential. The program is a partnership between the World Bank and GFDRR. A City Scan is a rapid geospatial assessment of the critical resilience challenges that cities face using the best publicly available global datasets and open source tools. The output is a package of maps, data visualizations and insights that integrate features of both the built and natural environments. 
    
    """
)
###################################################
#############  Controls 2 ##########################
###################################################


credits_tab2 = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            This map estimates population numbers per 10,000 meter sq. grid cell. It provides a more consistent representation of population distributions across different landscapes than administrative unit counts. 
            
            """
        ),
    ),
    className="mt-02",
)


population_card = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            Statiscal or hot spot analysis shows: insert key message----->> This map estimates population numbers per 10,000 meter sq. grid cell. It provides a more consistent representation of population distributions across different landscapes than administrative unit counts. 
            
            """
        ),
    ),
    className="mt-03",
)


population_map_text = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            The map depicts projected demographic dynamics for Chitaagong. The current metro area population of Chittagong in 2023 is 5,380,000, a 2.42% increase from 2022. The metro area population of Chittagong in 2022 was 5,253,000, a 2.34% increase from 2021

            """
        ),
    ),
    className="mt-03",
)

population_map_text2 = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            USGS Sat Map overlay: Describe what the map shows for a given SSP and year. The map depicts projected demographic dynamics for Chitaagong.

            """
        ),
    ),
    className="mt-04",
)


gdp_map_text = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            GDP Map Description: USGS Sat Map overlay.  Describe what the map shows for a given SSP and year. The map depicts projected GDP dynamics for Chitaagong.

            """
        ),
    ),
    className="gdp-04",
)





credits_tab3 = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            The map illustrates monthly temporal changes from 2014 to 2022 in the emission of nighttime lights, indicating changes in economic activity. Positive values represent an increase in the intensity of nighttime light emission and, by proxy, economic activity.

            """
        ),
    ),
    className="mt-03",
)

# Add only description card for the fourth scatterplot
credits_tab4 = dbc.Card(
    dbc.CardBody(
        dcc.Markdown(
            """
            GDP growth measures how fast the local economy is growing. Positive GDP growth is typically attributed to government spending, personal consumption, business investment, construction, and net trade.
            """
        ),
    ),
    className="mt-04",
)

###################################################
#############  Right layout (=Maps) ###############
###################################################

data_density_map_component = dbc.Card(
    [
        dbc.CardHeader(
            html.H3("Section 1: Demographic Distribution")), 
            dbc.CardBody(
                [
                    intro_markdown,
                ],
            ) ,

            html.H6("USGS Basemap"), 
            dcc.Graph(
                    id='data_density' , 
                    style={'padding':'0.5'},
            ),  
            html.Br()  ,          
            html.Br()  ,          

            population_card,

            html.H6("OSM basemap"), 
                    dcc.Graph(
                        id='data_density_' , 
                        style={'padding':'0.5'},
                        ),  
            html.Br()  ,
            html.Br()  ,          
            html.Br()  ,          
            html.Br()  ,          
            html.H5('Demographic Projections'),
            dcc.Checklist(
                id="checklist",
                options= options, 
                value= value  , 
                inline=True
            ),    
            dcc.Graph(id="ssps_graph",),
    ], 
    className="my-2", 
    style={'height': 2000 , 'padding':'0.5'}
)

#-------------------------------------------------------------------------------------------------------#
#Section 2 GDP
#-------------------------------------------------------------------------------------------------------#

gdp_card_main = dbc.Card(
    [
        dbc.CardHeader(
            html.H3("Section 2: GDP Distribution")), 
            dbc.CardBody(
                [
                    intro_markdown_gdp,
                ],
            ) ,

            html.H6("USGS Basemap GDP"), 
            dcc.Graph(
                    id='data_density_gdp' , 
                    style={'padding':'0.5'},
            ),  
            html.Br()  ,          
            html.Br()  ,          

            population_card,

            html.H6("OSM basemap GDP"), 
                    dcc.Graph(
                        id='data_density_gdp_osm' , 
                        style={'padding':'0.5'},
                        ),  
            html.Br()  ,
            html.Br()  ,          
            html.Br()  ,          
            html.Br()  ,          
            html.H5('GDP Projections'),
            dcc.Checklist(
                id="checklist_gdp",
                options= options, 
                value= value  , 
                inline=True
            ),    
            dcc.Graph(id="ssps_graph_gdp",),
    ], 
    className="my-5", 
    style={'height': 2000 , 'padding':'0.5'}
)


#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

fig_tab = dbc.Tabs(
    [
        dbc.Tab(data_density_map_component, label="Section 1: Demographic Distribution",active_label_style={"font-weight":"800","color": "#00AEF9"}),
        dbc.Tab(gdp_card_main, label="Section 2: GDP Distribution",active_label_style={"font-weight":"800","color": "#00AEF9"}),

    ]
)

###################################
######  Layout of Dash app ########
###################################

app.layout = dbc.Container(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(  # left layout
                    [
                    dbc.CardHeader("Product"),
                    credits_tab,
                    html.Label('', style={'paddingTop': '17rem'}),
                    dbc.Row(
                            [
                            dbc.CardHeader("Select Scenario and Year"),
                            control_tab,
                            population_map_text
                            ], 
                            ),
                            dbc.Row(
                            [
                            dbc.CardHeader("Select Scenario and Year"),
                            control_tab_gdp,
                            gdp_map_text
                            ], 
                            ),
                    
                    html.Label('', style={'paddingTop': '30rem'}),
                    dbc.Row(
                            [
                            dbc.CardHeader("Graph Explanations"),
                            population_map_text2,
                            ], 
                            ),
                    ], 
                    #  

                        width=4),

                dbc.Col( # right layout
                    [
                    fig_tab,
                    ],
                    width=8,
                ),
            ]
        ),
    ],

    fluid=True,
)

###################################
######  Section 1: Demographic Distribution ########
###################################

@app.callback(
    Output('data_density', 'figure'),
    [
     Input("year", 'value'),
     Input("scenario", "value"),
    ]
)
def update_fig_data_density( y, s):
    dff= pop_geo_df.copy()
    dff = dff[ (dff["Year"] == y) & (dff["Scenario"] == s)]       
    fig_data_density = px.choropleth_mapbox(dff,
                           geojson=dff.geometry,
                           locations=dff.index,
                           color="Population",
                           color_continuous_scale='Reds', #scale
                           opacity=0.8,
                           center={"lat": center_lat[0], "lon": center_lon[0]}, #mapbox_style="open-street-map",
                           zoom=10,
                           )
    fig_data_density.update_traces(marker_line_width=0)
    # fig_data_density.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    s = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    fig_data_density.update_layout( 
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [s],
            }
        ],
    )

    fig_data_density.data[0].hovertemplate = "<span style='font-size:1.2rem; font-weight=400'>Population = %{z:,.0f}</span><br><br>"
    return fig_data_density

############################################
#Map 2 section osm

@app.callback(
    Output('data_density_', 'figure'),
    [
     Input("year", 'value'),
     Input("scenario", "value"),
    ]
)
def update_fig_data_density_( y, s):
    dff= pop_geo_df.copy()
    dff = dff[ (dff["Year"] == y) & (dff["Scenario"] == s)]    
    fig_data_density_ = px.choropleth_mapbox(dff,
                           geojson=dff.geometry,
                           locations=dff.index,
                           color="Population",
                           color_continuous_scale='Magma', #scale
                           opacity=0.8,
                           center={"lat": center_lat[0], "lon": center_lon[0]},
                           mapbox_style="open-street-map",
                           zoom=10,
                           )
    fig_data_density_.update_traces(marker_line_width=0)
    fig_data_density_.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig_data_density_.data[0].hovertemplate = "<span style='font-size:1.2rem; font-weight=400'>Population = %{z:,.0f}</span><br><br>"

    return fig_data_density_

############################################
############################################

@app.callback(
    Output("ssps_graph", "figure"), 
    Input("checklist", "value"))
def update_line_chart(Scenario):
    # df = px.data.gapminder() # replace with your own data source
    mask = pop_df_scenrio_year.Scenario.isin(Scenario)
    line_chart = px.line(pop_df_scenrio_year[mask], 
        x="year", y="Population", color='Scenario')
    return line_chart

# import plotly.express as px
# from textwrap import wrap
# named_colorscales = px.colors.named_colorscales()
# print("\n".join(wrap("".join('{:<12}'.format(c) for c in named_colorscales), 96)))
############################################
#####Section 2 GDP-->>>Interactive  ########
############################################


@app.callback(
    Output('data_density_gdp', 'figure'),
    [
     Input("year_gdp", 'value'),
     Input("scenario_gdp", "value"),
    ]
)
def update_fig_data_density( y, s):
    dff= gdp_geo_df.copy()
    dff = dff[ (dff["Year"] == y) & (dff["Scenario"] == s)]       
    fig_data_density = px.choropleth_mapbox(dff,
                           geojson=dff.geometry,
                           locations=dff.index,
                           color="GDP",
                           color_continuous_scale='Reds', #scale
                           opacity=0.8,
                           center={"lat": center_lat[0], "lon": center_lon[0]}, #mapbox_style="open-street-map",
                           zoom=10,
                           )
    fig_data_density.update_traces(marker_line_width=0)
    
    s = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    fig_data_density.update_layout( 
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [s],
            }
        ],
    )

    fig_data_density.data[0].hovertemplate = "<span style='font-size:1.2rem; font-weight=400'>Population = %{z:,.0f}</span><br><br>"
    return fig_data_density

############################################
#Map 2 section osm

@app.callback(
    Output('data_density_gdp_osm', 'figure'),
    [
     Input("year_gdp", 'value'),
     Input("scenario_gdp", "value"),
    ]
)
def update_fig_data_density_( y, s):
    dff= gdp_geo_df.copy()
    dff = dff[ (dff["Year"] == y) & (dff["Scenario"] == s)]    
    fig_data_density_ = px.choropleth_mapbox(dff,
                           geojson=dff.geometry,
                           locations=dff.index,
                           color="GDP",
                           color_continuous_scale='Magma', #scale
                           opacity=0.8,
                           center={"lat": center_lat[0], "lon": center_lon[0]},
                           mapbox_style="open-street-map",
                           zoom=10,
                           )
    fig_data_density_.update_traces(marker_line_width=0)
    fig_data_density_.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig_data_density_.data[0].hovertemplate = "<span style='font-size:1.2rem; font-weight=400'>Population = %{z:,.0f}</span><br><br>"

    return fig_data_density_

############################################
############################################

@app.callback(
    Output("ssps_graph_gdp", "figure"), 
    Input("checklist_gdp", "value"))
def update_line_chart(Scenario):
    # df = px.data.gapminder() # replace with your own data source
    mask = gdp_df_scenrio_year.Scenario.isin(Scenario)
    line_chart = px.line(gdp_df_scenrio_year[mask], 
        x="year", y="GDP", color='Scenario')
    return line_chart

############################################


if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server(debug=False, host="0.0.0.0", port=8080)
    # app.run_server()
