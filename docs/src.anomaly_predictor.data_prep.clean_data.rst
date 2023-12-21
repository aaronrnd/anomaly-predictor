src.anomaly\_predictor.data\_prep.clean\_data
=============================================

Overview
----------
Data Cleaning is done via an instance of ``DataCleaner`` whose main method :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner.clean_data`
performs the following sequentially: 

- keeping features of interest and dropping unneeded features 
  (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._keep_features`)
- validating that values are non-negative 
  (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._check_negatives`)
- resampling data into hourly data points (via :func:`src.anomaly_predictor.utils.resample_hourly`)
- standardizing feature values (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._standardization`)
- detects outliers using neighboring points and a user-defined threshold 
  (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._check_outliers`)
- splits data into lists of smaller dataframes if there are any long missing periods 
  in between (via :func:`src.anomaly_predictor.utils.split_long_nans`)
- ensure that when "Speed" is 0, non-null values in "Bearing Condition" are changed 
  to null values for subsequent imputation. (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._preprocess_bearing_cond`); 
  this is to ensure consistency in behaviour of "Bearing Condition" across firmware updates.
- data imputation for points perceived as outliers, any interspersed missing values
  (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._impute_nans`) and specially for "Bearing Condition" 
  (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._impute_bearing_cond`) if any "Bearing Condition" values are still
  null even after prior imputation
- converting float to int values for user-defined columns (via :meth:`~src.anomaly_predictor.data_prep.clean_data.DataCleaner._float_to_int`),
  this is relevant for features that are expected to be integers but may have been
  imputed as floats, such as "Bearing Condition" and even for easier readability 
  for "Anomaly" column.
- dropping of any data points which still has any missing value(s) present

src.anomaly\_predictor.modeling.data\_prep.clean\_data
------------------------------------------------------
.. automodule:: src.anomaly_predictor.data_prep.clean_data
   :members:
   :undoc-members:
   :show-inheritance: