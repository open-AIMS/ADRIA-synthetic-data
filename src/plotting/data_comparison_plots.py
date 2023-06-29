import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import seaborn as sns
from sklearn.decomposition import PCA


def plot_comparison_scatter(sample_sites, new_site_data, site_data):
    """
    Plot scatter plot to compare synthetic and original data.

    :param dataframe sample_sites: conditionally sampled site data.
    :param dataframe new_site_data: synthetic site data.
    :param data_frame site_data: original site data.
    :param str color: site data variable to be represented by color (must be float).
    :param str size: site data variable to be represented by size (must be float).
    """
    fig, axes = plt.subplots(1, 3)
    new_size = new_site_data["area"] * (new_site_data["k"] / 100)
    new_size = new_size / 1000
    axes[0].scatter(
        new_site_data["long"],
        -new_site_data["lat"],
        s=new_size,
        c=new_size,
    )
    orig_size = site_data["area"] * (site_data["k"] / 100)
    orig_size = orig_size / 1000
    axes[1].scatter(
        site_data["long"],
        site_data["lat"],
        s=orig_size,
        c=orig_size,
    )
    sample_size = sample_sites["area"] * (sample_sites["k"] / 100)
    sample_size = sample_size / 1000
    axes[2].scatter(
        sample_sites["long"],
        sample_sites["lat"],
        s=sample_size,
        c=sample_size,
    )
    axes[1].set_title("Original", fontsize=28)
    axes[0].set_title("Synthetic", fontsize=28)
    axes[2].set_title("Sampled", fontsize=28)
    axes[0].set_ylabel("latitude", fontsize=22)
    for ax_n in range(3):
        axes[ax_n].set_xlabel("longitude", fontsize=22)
        axes[ax_n].yaxis.set_tick_params(labelsize=15, rotation=25)
        axes[ax_n].xaxis.set_tick_params(labelsize=15, rotation=25)
        axes[ax_n].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        axes[ax_n].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    fig.show()
    return fig


def plot_comparison_hist_covers(cover_orig, cover_synth, cover_samp):
    """
    Plot scatter plot to compare synthetic and original data.

    :param np.array cover_orig: summed cover for original data.
    :param dataframe site_data: original site data.
    :param np.array cover_samp: summed cover for sampled synthetic data.
    :param dataframe site_data_samp: sampled synthetic site data.
    """
    fig, axes = plt.subplots(1, 3)

    axes[0].hist(
        cover_samp,
        bins=round(np.sqrt(cover_synth.shape[0])),
        color="skyblue",
        lw=0,
    )
    axes[1].hist(
        cover_orig,
        bins=round(np.sqrt(cover_orig.shape[0])),
        color="purple",
        lw=0,
    )
    axes[2].hist(
        cover_samp,
        bins=round(np.sqrt(cover_samp.shape[0])),
        color="pink",
        lw=0,
    )

    axes[1].set_title("Original", fontsize=28)
    axes[2].set_title("Sampled", fontsize=28)
    axes[0].set_title("Synthetic", fontsize=28)
    axes[0].set_ylabel("Number of sites", fontsize=22)
    for ax_n in range(3):
        axes[ax_n].set_xlabel("relative coral cover", fontsize=22)
        axes[ax_n].yaxis.set_tick_params(labelsize=15, rotation=25)
        axes[ax_n].xaxis.set_tick_params(labelsize=15, rotation=25)

    fig.show()
    return fig


def plot_comparison_hist(sample_sites, new_site_data, site_data, parameter, label_name):
    """
    Plot histograms to compare synthetic and original data.

    :param dataframe sample_sites: conditionally sampled site data.
    :param dataframe new_site_data: synthetic site data.
    :param data_frame site_data: original site data.
    :param str parameter: site data variable to be represented as histogram.
    """
    fig, axes = plt.subplots(1, 3)

    axes[0].hist(
        new_site_data[parameter],
        bins=round(np.sqrt(new_site_data.shape[0])),
        color="skyblue",
        lw=0,
    )
    axes[1].hist(
        site_data[parameter],
        bins=round(np.sqrt(site_data.shape[0])),
        color="purple",
        lw=0,
    )
    axes[2].hist(
        sample_sites[parameter],
        bins=round(np.sqrt(sample_sites.shape[0])),
        color="pink",
        lw=0,
    )
    axes[1].set_title("Original", fontsize=30)
    axes[0].set_title("Synthetic", fontsize=30)
    axes[2].set_title("Sampled", fontsize=30)
    axes[0].set_ylabel("counts", fontsize=26)
    for ax_n in range(3):
        axes[ax_n].set_xlabel(label_name, fontsize=26)
        axes[ax_n].yaxis.set_tick_params(labelsize=20, rotation=25)
        axes[ax_n].xaxis.set_tick_params(labelsize=20, rotation=25)

    fig.show()
    return fig


def get_data_quantiles(env_df, new_data_env, nyears, old_years, new_years, layer):
    """
    Plot environmental data as time series of quantiles.

    :param dataframe env_df: contains original env data.
    :param dataframe new_data_env: synthetic env data.
    :param int nyears: number of years to display.
    :param vec old_years: vector of years in original env dataset.
    :param vec new_years: vector of years in synthetic env dataset.
    :param str layer: environmental data layer type ('dhw' or 'wave').

    """
    outcomes_data = {
        "upper_25": [0.0] * nyears,
        "lower_25": [0.0] * nyears,
        "upper_50": [0.0] * nyears,
        "lower_50": [0.0] * nyears,
        "upper_75": [0.0] * nyears,
        "lower_75": [0.0] * nyears,
        "upper_95": [0.0] * nyears,
        "lower_95": [0.0] * nyears,
        "median": [0.0] * nyears,
    }

    keys = [k for k in outcomes_data.keys()]
    outcomes_synth = {
        "upper_25": [0.0] * nyears,
        "lower_25": [0.0] * nyears,
        "upper_50": [0.0] * nyears,
        "lower_50": [0.0] * nyears,
        "upper_75": [0.0] * nyears,
        "lower_75": [0.0] * nyears,
        "upper_95": [0.0] * nyears,
        "lower_95": [0.0] * nyears,
        "median": [0.0] * nyears,
    }

    quantiles = [97.5, 25, 87.5, 12.5, 75, 25, 62.5, 37.5]

    for y in range(nyears):
        data_temp = env_df[layer][old_years == str(y + 2025)]
        synth_data_temp = new_data_env[layer][np.transpose(new_years) == str(y + 2025)]
        data_percentile_temp = np.percentile(data_temp, quantiles)
        synth_data_percentile_temp = np.percentile(synth_data_temp, quantiles)

        outcomes_data["median"][y] = np.median(data_temp)
        outcomes_synth["median"][y] = np.median(synth_data_temp)

        for j in range(len(keys) - 1):
            outcomes_data[keys[j]][y] = data_percentile_temp[j]
            outcomes_synth[keys[j]][y] = synth_data_percentile_temp[j]

    return outcomes_data, outcomes_synth


def create_timeseries(outcomes, years, layer, label="", color_code="rgba(255, 0, 0, "):
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
    no_outline = {"color": color_code + "0)"}

    def lower(y):
        return go.Scatter(
            x=years,
            y=y,
            mode="lines",
            showlegend=False,
            legendgroup=label,
            line=no_outline,
        )

    def upper(y, fillcolor):
        return go.Scatter(
            x=years,
            y=y,
            mode="none",
            showlegend=False,
            legendgroup=label,
            fill="tonexty",
            fillcolor=fillcolor,
        )

    fig_data = [
        lower(outcomes["lower_95"]),
        upper(outcomes["upper_95"], color_code + "0.10)"),
        lower(outcomes["lower_75"]),
        upper(outcomes["upper_75"], color_code + "0.20)"),
        lower(outcomes["lower_50"]),
        upper(outcomes["upper_50"], color_code + "0.30)"),
        lower(outcomes["lower_25"]),
        upper(outcomes["upper_25"], color_code + "0.40)"),
        go.Scatter(
            x=years,
            y=outcomes["median"],
            mode="lines",
            fillcolor=color_code + "1)",
            name=str(label),
            line_color=color_code + "1)",
            legendgroup=label,
            showlegend=False,
        ),
    ]

    fig = go.Figure(fig_data)
    fig.update_layout(
        title=label,
        xaxis_title="Year",
        yaxis_title=layer,
        xaxis=dict(tickfont_size=28, title=dict(font=dict(size=30))),
        yaxis=dict(tickfont_size=28, title=dict(font=dict(size=30))),
    )
    return fig


def comparison_plots_site_data(sample_sites, new_site_data, site_data):
    fig1 = plot_comparison_scatter(sample_sites, new_site_data, site_data)

    fig2 = plot_comparison_hist(
        sample_sites, new_site_data, site_data, "area", r"area($m^2$)"
    )
    fig3 = plot_comparison_hist(sample_sites, new_site_data, site_data, "k", "k(%)")
    fig4 = plot_comparison_hist(sample_sites, new_site_data, site_data, "sand", "sand")
    fig5 = plot_comparison_hist(sample_sites, new_site_data, site_data, "rock", "rock")
    fig6 = plot_comparison_hist(
        sample_sites, new_site_data, site_data, "rubble", "rubble"
    )
    fig7 = plot_comparison_hist(
        sample_sites, new_site_data, site_data, "coral_algae", "coral algae"
    )
    fig8 = plot_comparison_hist(
        sample_sites, new_site_data, site_data, "depth_mean", "depth mean(m)"
    )
    fig9 = plot_comparison_hist(
        sample_sites, new_site_data, site_data, "zone_type", "zone type"
    )

    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]


def compared_cover_species_hist(cover_df, synth_cover, synth_sampled):
    synth_sampled_cover = [
        sum(synth_sampled.cover[synth_sampled.species == k]) / sum(synth_sampled.cover)
        for k in range(1, 37)
    ]
    synth_cover = [
        sum(synth_cover.cover[synth_cover.species == k]) / sum(synth_cover.cover)
        for k in range(1, 37)
    ]
    orig_cover = [
        sum(cover_df.cover[cover_df.species == k]) / sum(cover_df.cover)
        for k in range(1, 37)
    ]

    cover_species_df = pd.DataFrame(
        {
            "Species": np.array([str(k) for k in range(1, 37)]),
            "Original": orig_cover,
            "Synthetic": synth_cover,
            "Sampled": synth_sampled_cover,
        }
    )

    cover_species_df.plot(
        x="Species",
        kind="bar",
        stacked=True,
        color=["skyblue", "purple", "pink"],
    )
    plt.xticks(rotation="horizontal", fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("p", fontsize=30)
    plt.xlabel("species", fontsize=30)
    plt.legend(fontsize=30)
    plt.show()


def plot_pca(real, fake):
    """
    Plot the first two components of a PCA of real and fake data.
    :param fname: If not none, saves the plot with this file name.
    """

    pca_r = PCA(n_components=2)
    pca_f = PCA(n_components=2)

    real_t = pca_r.fit_transform(real)
    fake_t = pca_f.fit_transform(fake)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("First 2 components of PCA", fontsize=16)
    sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
    sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
    ax[0].set_title("Original")
    ax[1].set_title("Synthetic")

    plt.show()


def plot_mean_std(real, fake, ax=None):
    """
    Plot the means and standard deviations of each dataset.
    :param real: DataFrame containing the real data
    :param fake: DataFrame containing the fake data
    :param ax: Axis to plot on. If none, a new figure is made.
    :param fname: If not none, saves the plot with this file name.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Absolute Log Mean and STDs of numeric data\n", fontsize=16)

    ax[0].grid(True)
    ax[1].grid(True)
    real = real._get_numeric_data()
    fake = fake._get_numeric_data()
    real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
    fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
    min_mean = min(real_mean) - 1
    max_mean = max(real_mean) + 1
    line = np.arange(min_mean, max_mean)
    sns.lineplot(x=line, y=line, ax=ax[0])
    sns.scatterplot(x=real_mean, y=fake_mean, ax=ax[0])
    ax[0].set_title("Means of original and synthetic data")
    ax[0].set_xlabel("Original mean (log)")
    ax[0].set_ylabel("Synthetic mean (log)")

    real_std = np.log(np.add(real.std().values, 1e-5))
    fake_std = np.log(np.add(fake.std().values, 1e-5))
    min_std = min(real_std) - 1
    max_std = max(real_std) + 1
    line = np.arange(min_std, max_std)
    sns.lineplot(x=line, y=line, ax=ax[1])
    sns.scatterplot(x=real_std, y=fake_std, ax=ax[1])
    ax[1].set_title("Stds of original data")
    ax[1].set_xlabel("Original std (log)")
    ax[1].set_ylabel("Synthetic data std (log)")

    return fig
