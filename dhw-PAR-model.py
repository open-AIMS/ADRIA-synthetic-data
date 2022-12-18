import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from deepecho import PARModel
from deepecho.models.basic_gan import BasicGANModel
from scipy.io import loadmat
from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdv.metrics.timeseries import LSTMDetection

from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


### ----------------Load site data and dhw data to synethesize------------------###
DHW_45 = loadmat(
    'C:/Users/rcrocker/Documents/datasets/data_packages/Moore/DHWs/NewDHWs/DHWexp_v2_0_Moore_45.mat')
site_data = pd.read_csv(
    "C:/Users/rcrocker/Documents/datasets/data_packages/Moore/site_data/MooreReefCluster_Spatial_w4.5covers.csv")
site_data_synth = pd.read_csv("site_data_Moore_numsamps_28.csv")

### ----------------Reshape data to fit PAR model------------------###
nyears = DHW_45['DHWdatacube'].shape[0]
nsites = DHW_45['DHWdatacube'].shape[1]
nreps = DHW_45['DHWdatacube'].shape[2]
size = nyears*nsites

entity_columns = ['Site']
context_columns = ['Lat', 'Long']
sequence_index = 'Year'
breakpoint()

lats = site_data.lat
longs = site_data.long
dhw_45_df = pd.DataFrame({"Year": [0]*size, "Lat": [0.0]*size,
                         "Long": [0.0]*size, "Site": [0]*size, "Dhw": [0.0]*size})

count = 0

for yr in range(nyears):
    for si in range(nsites):
        dhw_45_df['Year'][count] = str(yr+2025)
        dhw_45_df['Lat'][count] = lats[si]
        dhw_45_df['Long'][count] = longs[si]
        dhw_45_df['Site'][count] = si+1
        dhw_45_df['Dhw'][count] = DHW_45['DHWdatacube'][yr,
                                                        si, np.random.randint(nreps-1)]
        count += 1

data_types = {
    'Long': 'continuous',
    'Lat': 'continuous',
    'Site': 'categorical',
    'Dhw': 'continuous'
}
breakpoint()
old_years = dhw_45_df["Year"]
dhw_45_df["Year"] = pd.to_datetime(dhw_45_df["Year"], format='%Y')

### ----------------Set up and fit PAR model------------------###
model = PARModel(epochs=1024, cuda=False)


model.fit(data=dhw_45_df,
          context_columns=context_columns,
          entity_columns=entity_columns,
          data_types=data_types,
          sequence_index=sequence_index)

breakpoint()

### ----------------Sample data to synthesize------------------###
N_s = 200
new_data_dhw = model.sample(N_s)
new_years = [str(yr+2025) for yr in range(nyears)]
new_years = new_years*N_s
new_data_dhw['Year'] = new_years

# create metadata dictionary
metadata_dhw = {'fields': {'Year': {'type': 'datetime'},
                           'Lat': {'type': 'numerical', 'subtype': 'float'},
                           'Long': {'type': 'numerical', 'subtype': 'float'},
                           'Site': {'type': 'categorical'},
                           'Dhw': {'type': 'numerical', 'subtype': 'float'}},
                'context_columns': ['Site'],
                'entity_columns': ['Lat', 'Long'],
                'sequence_index': 'Year'}

# # Ml ability to detec difference between
# LogisticDetection.compute(dhw_45_df, new_data_dhw)
# LSTMDetection.compute(dhw_45_df, new_data_dhw, metadata_dhw)

breakpoint()
### ----------------Evaluiate data validity------------------###
outcomes_data = {'upper_25': [0.0]*nyears,
                 'lower_25': [0.0]*nyears,
                 'upper_50': [0.0]*nyears,
                 'lower_50': [0.0]*nyears,
                 'upper_75': [0.0]*nyears,
                 'lower_75': [0.0]*nyears,
                 'upper_95': [0.0]*nyears,
                 'lower_95': [0.0]*nyears,
                 'median': [0.0]*nyears, }
keys = [k for k in outcomes_data.keys()]
outcomes_synth = outcomes_data
quantiles = [97.5, 25, 87.5, 12.5, 75, 25, 62.5, 37.5]


for y in range(nyears):

    data_temp = dhw_45_df['Dhw'][old_years == str(y+2025)]
    synth_data_temp = new_data_dhw['Dhw'][np.transpose(
        new_years) == str(y+2025)]
    data_percentile_temp = np.percentile(data_temp, quantiles)
    synth_data_percentile_temp = np.percentile(synth_data_temp, quantiles)

    outcomes_data['median'][y] = np.median(data_temp)
    outcomes_synth['median'][y] = np.median(synth_data_temp)

    for j in range(len(keys)-1):
        outcomes_data[keys[j]][y] = data_percentile_temp[j]
        outcomes_synth[keys[j]][y] = synth_data_percentile_temp[j]


def create_timeseries(outcomes, label="", color_code="rgba(255, 0, 0, "):
    """Add summarized time series to given figure.
    Parameters
    ----------
    outcomes : summarized time series data
    n_scens : number of scenarios represented in the summarized data set
    label : text to use in legend
    color_code : rgba() color code as understood by Dash.
    Returns
    -------
    list, of plotly figure traces
    """
    no_outline = {"color": color_code + '0)'}

    def lower(y):
        return go.Scatter(y=y, mode="lines", showlegend=False, legendgroup=label, line=no_outline)

    def upper(y, fillcolor):
        return go.Scatter(y=y, mode="none", showlegend=False, legendgroup=label, fill="tonexty", fillcolor=fillcolor)

    fig_data = [
        lower(outcomes['lower_95']),
        upper(outcomes['upper_95'], color_code + "0.10)"),

        lower(outcomes['lower_75']),
        upper(outcomes['upper_75'], color_code + "0.20)"),

        lower(outcomes['lower_50']),
        upper(outcomes['upper_50'], color_code + "0.30)"),

        lower(outcomes['lower_25']),
        upper(outcomes['upper_25'], color_code + "0.40)"),

        go.Scatter(y=outcomes['median'], mode="lines", fillcolor=color_code + "1)",
                   name=str(label), line_color=color_code + "1)", legendgroup=label)
    ]

    return fig_data


breakpoint()
fig_data = create_timeseries(outcomes_data, label="Original DHW Data")
fig_data = go.Figure(fig_data)
fig_data.show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic DHW Data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()

breakpoint()
synth_lats = site_data_synth['lat']
synth_longs = site_data_synth['long']
samples = np.zeros((len(new_data_dhw['Lat']), 2))
site_data_vals = np.zeros((len(synth_lats), 2))

for l in range(len(new_data_dhw['Lat'])):
    samples[l][:] = [new_data_dhw['Lat'][l], new_data_dhw['Long'][l]]

breakpoint()
neigh = NearestNeighbors(n_neighbors=nyears)
neigh.fit(samples)
breakpoint()

for k in range(len(synth_lats)):
    site_data_vals[k][:] = [-synth_lats[k], synth_longs[k]]
breakpoint()
nearest_sites = neigh.kneighbors(site_data_vals, return_distance=False)
selected_dhws = np.zeros((len(nearest_sites), nyears))
for nn in range(len(nearest_sites)):

    nearest_sites[nn].sort()
    selected_dhws[nn, :] = new_data_dhw['Dhw'][nearest_sites[nn]]

breakpoint()
### ------------ test GAN model for time series for dhw data-------------###

sequences = []
for si in range(nsites):
    data_tmp = [[DHW_45['DHWdatacube']
                 [k, si, np.random.randint(nreps-1)] for k in range(nyears)]]
    dict_tmp = {
        'context': [si],
        'data': data_tmp
    }
    sequences.append(dict_tmp)

context_types = ['categorical']
data_types = ['continuous']
breakpoint()
model = BasicGANModel(epochs=1024, cuda=False)
model.fit_sequences(sequences, context_types, data_types)

breakpoint()
dhw_45_df = pd.DataFrame(
    {"Year": [0.0]*size, "Site": [0.0]*size, "Dhw": [0.0]*size})
dhw_45_df_synth = pd.DataFrame(
    {"Year": [0.0]*size, "Site": [0.0]*size, "Dhw": [0.0]*size})

count = 0

for si in range(nsites):
    temp_seq = model.sample_sequence([si], sequence_length=74)
    for yr in range(nyears):

        dhw_45_df['Year'][count] = str(yr+2025)
        dhw_45_df['Site'][count] = si+1
        dhw_45_df['Dhw'][count] = sequences[si]['data'][0][yr]

        dhw_45_df_synth['Year'][count] = str(yr+2025)
        dhw_45_df_synth['Site'][count] = si+1
        dhw_45_df_synth['Dhw'][count] = temp_seq[0][yr]

        count += 1

breakpoint()
evaluate(dhw_45_df_synth, dhw_45_df, metrics=['KSTest'], aggregate=False)
LogisticDetection.compute(dhw_45_df, dhw_45_df_synth)
LSTMDetection.compute(dhw_45_df, dhw_45_df_synth, metadata_dhw)

breakpoint()
