# ADRIA-synthetic-data
Repository for the creation of synthetic input data layers for ADRIA.

# Set-up
Create the environment by running,
```
conda env create -f ADRIA_synth_data_env.yml
```
This environment can then be selected in your Python editor of choice.

Add the original data package you want to create synthetic data off to the `original_data` folder.

# Creating site data

Synthetic site data can be generated from the `site-data-generation.py` file in the `examples` folder. Add the name of your chosen original data package at the top of the file as `orig_data_package = "name of file"`. Adjust the parameters `N1`, `N2` and `N3` as desired also. `N1` is the number of unconditionalised samples to generate. `N2` is the final number of spatially conditionalised sites to generate. `N3` is the number of nodes to generate the final site positions in randomised radii around. Using the site data model automatically creates the synthetic data package and the package name will be given in the modal outputs as `synth_site_data_fn`.

# Creating initial coral cover data

Synthetic initial coal cover data can be generated from the `coral-cover-generation.py` file in the `examples` folder. Add the name of your chosen original data package at the top of the file as `orig_data_package = "name of file"` and the name of the synthetic site data file you want to base the cover data on: e.g. `root_site_data_synth = "synth_2023-7-24_152038.csv"`.

# Creating environmental data

Synthetic wave and DHW data can be generated from the `env-data-generation.py` file in the `examples` folder. Several inputs at the beginning of the file can be changed to adjust the output of the data model. An example is shown below:

```
layer = "Ub"
rcp = "45"
root_original_file = "name of file"
root_site_data_synth = "synth_2023-7-24_152038.csv"
nsamples = 10
nreplicates = 5
```

The `layer` variable designates the type of data to generate, so `dhw` for DHW data and `Ub` for wave data. `rcp` is the RCP to use to generate data from in the original data file. `nsamples` is the number of samples to generate from each climate replicate. `nreplicates` is the number of climate replicates to use from the original dataset. In the example above, the final dataset will have `10*5` replicates based on `5` replicates from the original dataset.

# Creating connectivity data

Synthetic connectivity data can be generated from the `connectivity-generation.py` file in the `examples` folder. Several inputs at the beginning of the file can be changed to adjust the output of the data model. An example is shown below:

```
root_original_file = "name of file"
root_site_data_synth = "synth_2023-7-24_152038.csv"
years = ["2015", "2016", "2017"]  # connectivity data years to use
num = ["1", "2", "3"]  # connectivity data sample number to use
model_type = "GAN"  # "GaussianCopula"
```
`years` designates how many years to base the synthestic connectivity dataset off, with the average being used to generate the final dataset. `num` designates any replicates to be used in each year (also averaged over). `model_type` designates whether to use the `GAN` model from tensorflow/keras (slower but better quality) or "Gaussian Copula" model from SDV (faster but lesser quality).

# Creating a synthetic JSON for the data package

A synthetic JSON file can be added to the synthetic data package using the file `data-package-json-creation.py`.

# Creating the whole data package

The whole data package can be created by running `data-package-creation.py`. The same inputs will need to be adjusted in this file as in the files described above, namely the sample number parameters and original dataset file name etc.
