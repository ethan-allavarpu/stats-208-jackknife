# stats-208-jackknife

Analysis of [Predictive inference with jackknife+](https://arxiv.org/abs/1905.02928)

## Extensions
### Simulations

### Real-World Data and Modeling

We gathered data from California Cooperative Oceanic Fisheries Invesitgations (CalCOFI) from the [CalCOFI website](https://calcofi.org/data/oceanographic-data/bottle-database/). The downloaded CSV files (for the bottle and cast data) should go in `data/raw/CalCOFI_Database_194903-202001_csv_22Sep2021/` to work with our written scripts. From here, we joined and processed the data to include fully-present observations across our response variable (salinity, denoted as `Salnty`) and 20 predictor variables: `Distance`, `Bottom_D`, `Wind_Spd`, `Depthm`, `T_degC`, `O2ml_L`, `STheta`, `O2Sat`, `Oxy_µmol/Kg`, `ChlorA`, `Phaeop`, `PO4uM`, `SiO3uM`, `NO2uM`, `NO3uM`, `NH3uM`, `DarkAs`, `MeanAs`, `R_DYNHT`, and `R_Nuts`. Please see the CalCOFI website for a codebook explaining each feature.

After processing, we were left with 6,102 complete observations. Similar to Barber, et al., we had a training set of 200 observations with the rest as our test set. We wanted to see how the jackknife+ would perform with a smaller number of predictors. Beyond this, though, we aimed to further test the generalizability of this method by constructing two different models: LASSO (with a hyperparameter value identical to the one proposed for the ridge regression simulations, and a boosting regressor (both used the default arguments in the model object from scikit-learn). Upon running these trials, we noticed that the performance remained similar across the models and interval types: the jackknife+ slightly outperformed the jackknife and met the coverage rate $1 - \alpha$ as proposed in the paper.
