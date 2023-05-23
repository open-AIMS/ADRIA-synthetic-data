import sys

sys.path.append("..")

from src.models.coral_cover_TVAE_model import coral_cover_model
from src.data_processing.package_synth_data import save_csv_plotting

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

### --------------------- Use model to generate synthetic covers and sampled synthetic cover ------------------- ###

root_original_file = "Moore_2022-11-17"
root_site_data_synth = "site_data_synth_19-5-2023_105441.csv"

(
    cover_df,
    synth_cover,
    synth_sampled,
    metadata_cover,
    root_site_data_synth,
) = coral_cover_model(root_original_file, root_site_data_synth, 300)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# evaluate K-S score
evaluate(cover_df, synth_cover)

LogisticDetection.compute(cover_df, synth_cover)

report = QualityReport()
report.generate(cover_df, synth_cover, metadata_cover)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

# evaluate K-S score
evaluate(synth_sampled, synth_cover)

save_csv_plotting(
    cover_df, synth_cover, synth_sampled, root_site_data_synth[10:], "covers"
)
