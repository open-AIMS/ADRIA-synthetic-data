import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_comparison_scatter(sample_sites,new_site_data,site_data,colour,size):
    """
    Plot scatter plot to compare synthetic and original data.

    :param dataframe sample_sites: conditionally sampled site data.
    :param dataframe new_site_data: synthetic site data.
    :param data_frame site_data: original site data.
    :param str color: site data variable to be represented by color (must be float).
    :param str size: site data variable to be represented by size (must be float).
    """
    fig, axes = plt.subplots(1, 3)
    axes[0].scatter(new_site_data['lat'], new_site_data['long'],
                            s=new_site_data[size], c=new_site_data[colour])
    axes[1].scatter(site_data['lat'], -site_data['long'],
                            s=site_data[size], c=site_data[colour])
    axes[2].scatter(sample_sites['lat'], -sample_sites['long'],
                            s=sample_sites[size], c=sample_sites[colour])
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[1].set(xlabel='lat', ylabel='long')
    axes[0].set(xlabel='lat', ylabel='long')
    axes[2].set(xlabel='lat', ylabel='long')
    fig.show()
    return fig, axes

def plot_comparison_hist(sample_sites,new_site_data,site_data,parameter):
    """
    Plot histograms to compare synthetic and original data.

    :param dataframe sample_sites: conditionally sampled site data.
    :param dataframe new_site_data: synthetic site data.
    :param data_frame site_data: original site data.
    :param str parameter: site data variable to be represented as histogram.
    """
    fig2, axes = plt.subplots(1, 3)
    axes[0].hist(new_site_data[parameter], bins=round(np.sqrt(new_site_data.shape[0])))
    axes[1].hist(site_data[parameter], bins=round(np.sqrt(site_data.shape[0])))
    axes[2].hist(sample_sites[parameter], bins=round(np.sqrt(sample_sites.shape[0])))
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[1].set(xlabel=parameter, ylabel='counts')
    axes[0].set(xlabel=parameter, ylabel='counts')
    axes[2].set(xlabel=parameter, ylabel='counts')
    fig2.show()

def get_data_quantiles(env_df,new_data_env,nyears,old_years,new_years,layer):
    """
    Plot environmental data as time series of quantiles.

    :param dataframe env_df: contains original env data.
    :param dataframe new_data_env: synthetic env data.
    :param int nyears: number of years to display.
    :param vec old_years: vector of years in original env dataset.
    :param vec new_years: vector of years in synthetic env dataset.
    :param str layer: environmental data layer type ('dhw' or 'wave').

    """
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

        data_temp = env_df[layer][old_years == str(y+2025)]
        synth_data_temp = new_data_env[layer][np.transpose(new_years) == str(y+2025)]
        data_percentile_temp = np.percentile(data_temp, quantiles)
        synth_data_percentile_temp = np.percentile(synth_data_temp, quantiles)

        outcomes_data['median'][y] = np.median(data_temp)
        outcomes_synth['median'][y] = np.median(synth_data_temp)

        for j in range(len(keys)-1):
            outcomes_data[keys[j]][y] = data_percentile_temp[j]
            outcomes_synth[keys[j]][y] = synth_data_percentile_temp[j]

    return outcomes_data, outcomes_synth 

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
    
    go.Figure(fig_data)
    return fig_data

def comparison_plots_site_data(sample_sites, new_site_data, site_data):
    fig1, axes = plot_comparison_scatter(sample_sites,new_site_data,site_data,'area','k')

    fig2, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'area')
    fig3, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'k')
    fig4, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'Reef')
    fig5, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'habitat')
    fig6, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'sitedepth')
    fig7, axes = plot_comparison_hist(sample_sites,new_site_data,site_data,'rubble')

    return [fig1,fig2,fig3,fig4,fig5,fig6,fig7]