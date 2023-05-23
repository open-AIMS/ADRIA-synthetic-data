import sys

sys.path.append("..")

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

import sys

sys.path.append("..")

from src.plotting.data_comparison_plots import comparison_plots_site_data
from src.models.site_data_fastML_model import site_data_model
from src.data_processing.package_synth_data import save_csv_plotting

### -----------------Use model to generate synthetic site data and sampled synthetic site data ---------------- ###

orig_data_package = "Moore_2022-11-17"

(
    site_data,
    new_site_data,
    sample_sites,
    metadata_site,
    synth_site_data_fn,
) = site_data_model(orig_data_package, 500, 20, 4)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# evaluate utility measures
evaluate(new_site_data, site_data)
# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, new_site_data)

report = QualityReport()
report.generate(site_data, new_site_data, metadata_site)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

# evaluate K-S score
cols = new_site_data.columns
evaluate(sample_sites[cols], new_site_data)

# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, sample_sites[cols])

### ------------------------------ Visual data comparison using plotting tools --------------------------------- ###
figs = comparison_plots_site_data(sample_sites, new_site_data, site_data)

file_name = synth_site_data_fn.split("\\")[9]
save_csv_plotting(site_data, new_site_data, sample_sites, file_name, "site_data")
