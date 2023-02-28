from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

import sys
sys.path.append("..")
sys.path.append("..//src")
sys.path.append("..//original_data")
sys.path.append("..//synthetic_data")
breakpoint()
from src.plotting.data_comparison_plots import comparison_plots_site_data
from src.models.site_data_fastML_model import site_data_model

### -----------------Use model to generate synthetic site data and sampled synthetic site data ---------------- ###

orig_data_package = "Moore_2022-11-17"

site_data, new_site_data, sample_sites, metadata_site, synth_site_data_fn = site_data_model(orig_data_package, 300, 30, 10)

### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# evaluate utility measures
evaluate(new_site_data, site_data)
# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, new_site_data)

report = QualityReport()
report.generate(site_data, new_site_data, metadata_site)
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Pair Trends')

# evaluate K-S score
evaluate(sample_sites, new_site_data)

# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, sample_sites)

### ------------------------------ Visual data comparison using plotting tools --------------------------------- ###
figs = comparison_plots_site_data(sample_sites, new_site_data, site_data)