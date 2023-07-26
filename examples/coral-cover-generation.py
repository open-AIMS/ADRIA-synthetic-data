import sys

sys.path.append("..")

from src.models.coral_cover_TVAE_model import coral_cover_model
from src.data_processing.package_synth_data import save_csv_plotting

from sdv.metrics.tabular import LogisticDetection

# from sdmetrics.reports.single_table import QualityReport
from sdv.evaluation.single_table import evaluate_quality

### --------------------- Use model to generate synthetic covers and sampled synthetic cover ------------------- ###

root_original_file = "Moore_2022-11-17"
root_site_data_synth = "synth_2023-7-24_152038.csv"
N = 300  # initial sample of sites to draw

(
    cover_df,
    synth_cover,
    synth_sampled,
    metadata_cover,
    root_site_data_synth,
) = coral_cover_model(root_original_file, root_site_data_synth, N)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###
quality_report = evaluate_quality(
    real_data=cover_df, synthetic_data=synth_cover, metadata=metadata_cover
)

# report = QualityReport()
# report.generate(cover_df, synth_cover, metadata_cover)
# report.get_details(property_name="Column Shapes")
# report.get_details(property_name="Column Pair Trends")

save_csv_plotting(cover_df, synth_cover, synth_sampled, root_site_data_synth, "covers")
