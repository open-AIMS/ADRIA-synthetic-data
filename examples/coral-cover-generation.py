import sys
sys.path.append("..")

from src.models.coral_cover_TVAE_model import coral_cover_model

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

### --------------------- Use model to generate synthetic covers and sampled synthetic cover ------------------- ###

root_original_file = 'Moore_2022-11-17'
root_site_data_synth = 'site_data_synth_21-3-2023_14478.csv'

cover_df, synth_cover, synth_sampled, metadata_cover, root_site_data_synth = coral_cover_model(root_original_file, root_site_data_synth, 300)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# evaluate K-S score
evaluate(cover_df,synth_cover)

LogisticDetection.compute(cover_df, synth_cover)

report = QualityReport()
report.generate(cover_df, synth_cover, metadata_cover)
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Column Pair Trends')

# evaluate K-S score
evaluate(synth_sampled, synth_cover)

synth_sampled_covers_spatial = [sum(synth_sampled.cover[synth_sampled.lat==lat]) for lat in np.unique(synth_sampled.lat)]
synth_covers_spatial = [sum(synth_cover.cover[synth_cover.lat==lat]) for lat in np.unique(synth_cover.lat)]
orig_covers_spatial = [sum(cover_df.cover[cover_df.lat==lat]) for lat in np.unique(cover_df.lat)]
covers_synth_sampled_df = pd.DataFrame({'lat':np.unique(synth_sampled.lat),'long':np.unique(synth_sampled.long),'covers':synth_sampled_covers_spatial})
covers_synth_df = pd.DataFrame({'lat':np.unique(synth_cover.lat),'long':np.unique(synth_cover.long),'covers':synth_covers_spatial})
covers_orig_df = pd.DataFrame({'lat':np.unique(synth_sampled.lat),'long':np.unique(synth_sampled.long),'covers':synth_sampled_covers_spatial})