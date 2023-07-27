import sys

sys.path.append("..")

# from sdmetrics.reports.single_table import QualityReport
from sdv.evaluation.single_table import evaluate_quality
import sys
from sdv.metadata import SingleTableMetadata

sys.path.append("..")

from src.plotting.data_comparison_plots import comparison_plots_site_data
from src.models.site_data_fastML_model import site_data_model
from src.data_processing.package_synth_data import save_csv_plotting

### -----------------Use model to generate synthetic site data and sampled synthetic site data ---------------- ###
N1 = 200  # number of samples in unconditionalised sample
N2 = 130  # estimate of final number of sites (final number may be slightly less due to being generated outside of original domain)
N3 = 10  #  number of sites to generate positions in local radii around

orig_data_package = "Moore_2022-11-17"

(
    site_data,
    new_site_data,
    sample_sites,
    metadata_site,
    synth_site_data_fn,
) = site_data_model(orig_data_package, N1, N2, N3)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# report = QualityReport()
# report.generate(site_data, new_site_data, metadata_site)
# report.get_details(property_name="Column Shapes")
# report.get_details(property_name="Column Pair Trends")
quality_report = evaluate_quality(
    real_data=site_data, synthetic_data=new_site_data, metadata=metadata_site
)

cols = new_site_data.columns

### ------------------------------ Visual data comparison using plotting tools --------------------------------- ###
figs = comparison_plots_site_data(sample_sites, new_site_data, site_data)

file_name = synth_site_data_fn.split("\\")[10]
save_csv_plotting(site_data, new_site_data, sample_sites, file_name[:-5], "site_data")
