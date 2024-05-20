import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import seaborn as sns
from sklearn.decomposition import PCA

from dython.nominal import associations


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
        new_site_data["lat"],
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


def plot_comparison_hist_covers(cover_orig, cover_samp):
    """
    Plot scatter plot to compare synthetic and original data.

    :param np.array cover_orig: summed cover for original data.
    :param np.array cover_samp: summed cover for sampled synthetic data.

    """
    fig, axes = plt.subplots(1, 2)

    axes[0].hist(
        cover_orig,
        weights=np.ones(len(cover_orig))/len(cover_orig),
        bins=round(np.sqrt(cover_orig.shape[0])),
        color="purple",
        lw=0,
    )
    axes[1].hist(
        cover_samp,
        weights=np.ones(len(cover_samp))/len(cover_samp),
        bins=round(np.sqrt(cover_samp.shape[0])),
        color="pink",
        lw=0,
    )

    axes[0].set_title("Original", fontsize=28)
    axes[1].set_title("Sampled", fontsize=28)
    axes[0].set_ylabel("density", fontsize=22)
    for ax_n in range(2):
        axes[ax_n].set_xlabel("relative coral cover", fontsize=22)
        axes[ax_n].yaxis.set_tick_params(labelsize=15, rotation=25)
        axes[ax_n].xaxis.set_tick_params(labelsize=15, rotation=25)
        axes[ax_n].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        axes[ax_n].xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        axes[ax_n].set_ylim([0.0, y_max+0.02*y_max])
        axes[ax_n].set_xlim([0.0, x_max+0.02*x_max])

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

    axes[1].hist(
        new_site_data[parameter],
        density=True,
        bins=round(np.sqrt(new_site_data.shape[0])),
        color="skyblue",
        lw=0,
    )
    axes[0].hist(
        site_data[parameter],
        density=True,
        bins=round(np.sqrt(site_data.shape[0])),
        color="purple",
        lw=0,
    )
    axes[2].hist(
        sample_sites[parameter],
        density=True,
        bins=round(np.sqrt(sample_sites.shape[0])),
        color="pink",
        lw=0,
    )
    axes[0].set_title("Original", fontsize=30)
    axes[1].set_title("Synthetic", fontsize=30)
    axes[2].set_title("Sampled", fontsize=30)
    axes[0].set_ylabel("density", fontsize=26)
    for ax_n in range(3):
        axes[ax_n].set_xlabel(label_name, fontsize=26)
        axes[ax_n].yaxis.set_tick_params(labelsize=20, rotation=25)
        axes[ax_n].xaxis.set_tick_params(labelsize=20, rotation=25)
        axes[ax_n].set_ylim([0.0, y_max+0.02*y_max])
        #axes[ax_n].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    fig.show()
    return fig


def get_data_quantiles(env_df, new_data_env, nyears, layer):
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
    old_years = np.array([str(year) for year in env_df["year"]])
    new_years = np.array([str(year) for year in new_data_env["year"]])
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
        for k in range(1, 7)
    ]
    synth_cover = [
        sum(synth_cover.cover[synth_cover.species == k]) / sum(synth_cover.cover)
        for k in range(1, 7)
    ]
    orig_cover = [
        sum(cover_df.cover[cover_df.species == k]) / sum(cover_df.cover)
        for k in range(1, 7)
    ]

    cover_species_df = pd.DataFrame(
        {
            "Species": np.array([str(k) for k in range(1, 7)]),
            "Original": orig_cover,
            "Synthetic": synth_cover,
            "Sampled": synth_sampled_cover,
        }
    )

    cover_species_df.plot(
        x="Species",
        kind="bar",
        color=["skyblue", "purple", "pink"],
    )
    plt.xticks(rotation="horizontal", fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("density", fontsize=30)
    plt.xlabel("species", fontsize=30)
    plt.legend(fontsize=30)
    plt.show()


def plot_pca(real, fake, samps):
    """
    Plot the first 2 components of a PCA of real and fake data.
    :param fname: If not none, saves the plot with this file name.
    """

    pca_r = PCA(n_components=2)
    pca_f = PCA(n_components=2)
    pca_s = PCA(n_components=2)

    real_t = pca_r.fit_transform(real)
    fake_t = pca_f.fit_transform(fake)
    samps_t = pca_s.fit_transform(samps)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    g1 = sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
    g2 = sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
    g3 = sns.scatterplot(ax=ax[2], x=samps_t[:, 0], y=samps_t[:, 1])
    ax[0].set_title("Original", size=20)
    ax[0].set_xlabel("1st component", size=20)
    ax[0].set_ylabel("2nd component", size=20)

    ax[1].set_title("Synthetic", size=20)
    ax[2].set_title("Sampled", size=20)
    ax[1].set_xlabel("1st component", size=20)
    ax[2].set_xlabel("1st component", size=20)
    for aa in range(3):
        ax[aa].set_xlim(-1, 3)
        ax[aa].set_ylim(-1, 3)

    g1.set_xticklabels(g1.get_xticks(), size=18)
    g1.set_yticklabels(g1.get_yticks(), size=18)
    ylabels = ["{:,.1f}".format(x) for x in g1.get_yticks()]
    g1.set_yticklabels(ylabels)
    xlabels = ["{:,.1f}".format(x) for x in g1.get_xticks()]
    g1.set_xticklabels(xlabels)

    g2.set_xticklabels(g2.get_xticks(), size=18)
    g2.set_yticklabels([])
    xlabels = ["{:,.1f}".format(x) for x in g2.get_xticks()]
    g1.set_xticklabels(xlabels)

    g3.set_xticklabels(g3.get_xticks(), size=18)
    g3.set_yticklabels([])
    xlabels = ["{:,.1f}".format(x) for x in g3.get_xticks()]
    g3.set_xticklabels(xlabels)

    plt.show()


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns the mean absolute percentage error between y_true and y_pred. Throws ValueError if y_true contains zero values.

    :param y_true: NumPy.ndarray with the ground truth values.
    :param y_pred: NumPy.ndarray with the ground predicted values.
    :return: Mean absolute percentage error (float).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def pca_correlation(real, fake):
    """
    Calculate the relation between PCA explained variance values.

    :param lingress: whether to use a linear regression, in this case Pearson's.
    :return: the correlation coefficient if lingress=True, otherwise 1 - MAPE(log(real), log(fake))
    """
    pca_r = PCA(n_components=5)
    pca_f = PCA(n_components=5)

    pca_r.fit(real)
    pca_f.fit(fake)

    results = pd.DataFrame(
        {
            "real": pca_r.explained_variance_ratio_,
            "fake": pca_f.explained_variance_ratio_,
        }
    )
    print(f"\nTop 5 PCA components:")
    print(results.to_string())

    pca_error = mean_absolute_percentage_error(
        pca_r.explained_variance_, pca_f.explained_variance_
    )

    return 1 - pca_error, pca_r, pca_f


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


def correlation_distance(real, fake, sampled, cat_cols) -> float:
    """
    Calculate distance between correlation matrices with certain metric.

    :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].
    :return: distance between the association matrices in the chosen evaluation metric. Default: Euclidean
    """

    real_corr = associations(
        real, nominal_columns=cat_cols, nom_nom_assoc="theil", compute_only=True
    )
    fake_corr = associations(
        fake, nominal_columns=cat_cols, nom_nom_assoc="theil", compute_only=True
    )
    samp_corr = associations(
        sampled, nominal_columns=cat_cols, nom_nom_assoc="theil", compute_only=True
    )

    return real_corr["corr"], fake_corr["corr"], samp_corr["corr"]


def correlation_heatmap(real_corr, fake_corr, samp_corr):
    f, (ax1, ax2, ax3, axcb) = plt.subplots(
        1, 4, gridspec_kw={"width_ratios": [1, 1, 1, 0.08]}
    )
    g1 = sns.heatmap(real_corr, cmap="YlGnBu", cbar=False, ax=ax1, vmin=0, vmax=1)
    g1.set_ylabel("to", size=20)
    g1.set_xlabel("from", size=20)
    g1.set_xticks(np.arange(1, real_corr.shape[0], 10))
    g1.set_yticks(np.arange(1, real_corr.shape[0], 10))
    g1.set_xticklabels(g1.get_xticks(), size=18)
    g1.set_yticklabels(g1.get_yticks(), size=18)

    g2 = sns.heatmap(fake_corr, cmap="YlGnBu", cbar=False, ax=ax2, vmin=0, vmax=1)
    g2.set_ylabel("")
    g2.set_xlabel("from", size=20)
    g2.set_yticks([])
    g2.set_xticks(np.arange(1, real_corr.shape[0], 10))
    g2.set_yticklabels(g2.get_yticks(), size=18)
    g2.set_xticklabels(g2.get_xticks(), size=18)

    g3 = sns.heatmap(samp_corr, cmap="YlGnBu", ax=ax3, cbar_ax=axcb, vmin=0, vmax=1)
    g3.set_ylabel("")
    g3.set_xlabel("from", size=20)
    g3.set_yticks(np.arange(1, samp_corr.shape[0], 10))
    g3.set_xticks(np.arange(1, samp_corr.shape[0], 10))
    g3.set_yticklabels(g3.get_yticks(), size=18)
    g3.set_xticklabels(g3.get_xticks(), size=18)

    plt.show()
    return

def correlation_diff_heatmap(real_corr, fake_corr):
    plt.plot()
    g1 = sns.heatmap(real_corr-fake_corr, cmap="YlGnBu", cbar=False, vmin=0, vmax=1)
    g1.set_ylabel("to", size=20)
    g1.set_xlabel("from", size=20)
    g1.set_xticks(np.arange(1, real_corr.shape[0], 10))
    g1.set_yticks(np.arange(1, real_corr.shape[0], 10))
    g1.set_xticklabels(g1.get_xticks(), size=18)
    g1.set_yticklabels(g1.get_yticks(), size=18)

    plt.show()
    return
