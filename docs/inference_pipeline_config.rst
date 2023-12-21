Inference pipeline configurations
=================================

There are two sets of inference pipeline configurations. One in the main ``conf/`` and another one in 
``conf/docker`` folder.  The configuration files is to be used for model training.


The configuration file in ``conf/`` folder is to be used in a local setting. Paths listed in the configuration  must be relative or absolute with respect to the local system. E.g ``C:\Users\ABB\AnomalyDetection``

However, the configuration file in ``conf/docker`` folder is to be used with the Docker containers. Its path listed
must be relative or absolute with respect to the Docker container. E.g ``/home/abb/``

``conf/docker/inference_pipeline.yml`` contains lesser parameters as certain parameters like clean_data,feature_engineering and data_loader's lookback_period. Such parameters are dependent on training's parameter and will be read and retain using train_pipeline.yml instead of exposing it. This will reduce errors arising from mismatched model parameters.

Configuration File - Artifacts
-------------------------------

Artifacts parameters contains the settings required for any inference run. The trained model directory and it's helper artifacts 
such as scalers and encoders will have to be instantiated here for any successful inference.

+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name | Parameter Data Type | Explanation                                                                                                                                                          |
+====================+=====================+======================================================================================================================================================================+
| artifacts:                                                                                                                                                                                                      |
+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_dir          | string              | Path of the model directory. Trained model and its relevant artifacts such as yaml files, precision recall curves and evaluation metrics will be saved to model_dir. |
+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_name         | string              | Defaults to "LSTMAE". Specifies the choice of trained model                                                                                                          |
+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| encoder_name       | string              | Defaults to "encoder". Specifies the name of the One-Hot Encoder used during training.                                                                               |
+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scaler_name        | string              | Defaults to "RobustScaler". Specifies the name of scaler used during training.                                                                                       |
+--------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Configuration File - Data Preparation
-------------------------------------

Data Preparation parameters allows the configurations of various settings such as input and output files, data ingestion cut off date. Unlike the training config file, data cleaning and feature engineering's configuration will not be exposed in the inference config file as these configurations will be extracted from the training's config file.

Data Preparation Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------+---------------------+--------------------------------------------------------------------------------------------+
| Configuration Name  | Parameter Data Type | Explanation                                                                                |
+=====================+=====================+============================================================================================+
| data_prep.pipeline:                                                                                                                    |
+---------------------+---------------------+--------------------------------------------------------------------------------------------+
| input_dir           | string              | Path of the input data directory. It should contains assets folder and annotations folder  |
+---------------------+---------------------+--------------------------------------------------------------------------------------------+
| interim_dir         | string              | Path of the interim data directory. Interim directory should contains the combined csv(s). |
+---------------------+---------------------+--------------------------------------------------------------------------------------------+
| processed_dir       | string              | Path of the processed data directory. CSV(s) should be cleaned and pre-processed.          |
+---------------------+---------------------+--------------------------------------------------------------------------------------------+

Data Ingestion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name     | Parameter Data Type | Explanation                                                                                                                                                                                              |
+========================+=====================+==========================================================================================================================================================================================================+
| data_prep.ingest_data:                                                                                                                                                                                                                                  |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| assets_dir             | string              | Defaults to "assets". Name of the folder containing the various assets' xlsx files.                                                                                                                      |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| annotation_list        | string              | A list containing the names of the filepath(s) of the annotation xlsx. It must be relative to the input directory. E.G. `["annotations/annotations_20211214.xlsx"]`                                      |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| time_col               | string              | Defaults to "MEASUREMENT_TAKEN_ON(UTC)". Name of the column which contains the time index.                                                                                                               |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| cut_off_date           | string              | Defaults to "01-7-21". Cut_off_date determines the earliest date to be kept and processed. Data before 1st July 2021 were deemed as outdated as major system upgrade occured, resulting in new features. |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| asset_timezone         | string              | Defaults to 0. Specifies the timezone of the assets' time index. 0 represents UTC, while 8 represents GMT+8.                                                                                             |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| annotation_timezone    | string              | Defaults to 0. Specifies the timezone of the annotations' time index.0 represents UTC, while 8 represents GMT+8.                                                                                         |
+------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



Data Cleaning 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Cleaning parameters will be left blank in ``conf/docker/inference_pipeline.yml``. 

+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name             | Parameter Data Type | Explanation                                                                                                                                                  |
+================================+=====================+==============================================================================================================================================================+
| data_prep.clean_data:                                                                                                                                                                                               |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| features_to_standardize        | list                | Specifies the list of features to standardize for outlier removal.                                                                                           |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| standardization_method         | string              | Defaults to "RobustScaler". Using specified standardization method to remove outliers.                                                                       |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| outlier_threshold              | integer             | Defaults to 99999. Pipeline will remove any outlier above the outlier threshold in the window.                                                               |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| outlier_window                 | integer             | Defaults to 5. Specifies the window size for outlier removal                                                                                                 |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| nan_range_threshold            | integer             | Defaults to 48. Data cleaning pipeline will split csv if number of consecutive NaNs exceeds nan_range threshold.                                             |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| impute_nan_window              | integer             | Defaults to 48. Data cleaning pipeline will impute NaNs according to rolling mean of impute_nan_window.                                                      |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| impute_nan_period              | integer             | Defaults to 1. Data cleaning pipeline will impute NaNs according to rolling mean of impute_nan_window if window contains more or equal to impute_nan_period. |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bearing_cond_fill              | integer             | Defaults to -1. Specifies the value to impute Bearing Condition if asset is not operating throughout data period.                                            |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| convert_to_int                 | list                | Specifies list of feature names to convert to integer after data cleaning process.                                                                           |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scalar_args.with_centering     | boolean             | Defaults to True. If True, center the data before scaling.                                                                                                   |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scalar_args.with_scaling       | boolean             | Defaults to True. If True, scale the data to interquartile range.                                                                                            |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scalar_args.copy               | boolean             | Defaults to True. If False, try to avoid a copy and do inplace scaling instead.                                                                              |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scalar_args.quantile_range     | list                | Defaults to [25.0, 75.0]. Quantile range used to calculate the scale. By default, [0.25, 75.0] is equal to IQR.                                              |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| scalar_args.unit_variance      | boolean             | Defaults to False. If True, scale data so that normally distributed features have a variance of 1.                                                           |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name             | Parameter Data Type | Explanation                                                                                                                                                  |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_prep.clean_data:          | null                | Parameter must be left blank. Configuration will be overridden by training parameters.                                                                       |
+--------------------------------+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

Feature Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Feature Engineering parameters will be left blank in ``conf/docker/inference_pipeline.yml``. 

+--------------------------------+---------------------+------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name             | Parameter Data Type | Explanation                                                                                                                  |
+================================+=====================+==============================================================================================================================+
| data_prep.feature_engineering:                                                                                                                                                      |
+--------------------------------+---------------------+------------------------------------------------------------------------------------------------------------------------------+
| min_data_points                | integer             | Defaults to 24. Specifies the number of minimum data points to determine if the motor of an asset is a variable speed motor. |
+--------------------------------+---------------------+------------------------------------------------------------------------------------------------------------------------------+
| motor_supply_freq              | string              | Defaults to "Motor Supply Frequency". Specifies the name of the feature which provides the motor supply frequency.           |
+--------------------------------+---------------------+------------------------------------------------------------------------------------------------------------------------------+


Configuration File - Model Inference 
------------------------------------

The inference configuration exposed are parameters which are not determined by training's parameters. Parameters related to the input and output folders and visualization parameters will be configurable. Other model parameters such as model architecture, dataloader's lookahead and lookback period will not be configurable and it is bounded to the model's training parameter.

Model Inference Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------+
| Configuration Name           | Parameter Data Type | Explanation                                                                                                          |
+==============================+=====================+======================================================================================================================+
| modeling.inference_pipeline:                                                                                                                                              |
+------------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------+
| inference_dir                | string              | Path of the input data directory. It should contains assets folder and annotations folder                            |
+------------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------+
| prediction_dir               | string              | Path of the inference output directory. Prediction results and visualizations will be saved to prediction directory. |
+------------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------+
| create_visualizations        | boolean             | Defaults to True. If True, create an d save visualization plots.                                                     |
+------------------------------+---------------------+----------------------------------------------------------------------------------------------------------------------+

Dataloader 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some of the data loader parameters do not exist in ``conf/docker/inference_pipeline.yml``. 

+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Name      | Parameter Data Type | Explanation                                                                                                                                                                                     |
+=========================+=====================+=================================================================================================================================================================================================+
| modeling.data_loader:                                                                                                                                                                                                                           |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.lookback_period    | integer             | Defaults to 336. Specifies the length of lookback period in hours. *Do not exist in ``conf/docker/inference_pipeline.yml``.*                                                                    |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.lookahead_period   | integer             | Defaults to 0. Specifies the length of lookahead period in hours. If 0, model will be a detection model. *Do not exist in ``conf/docker/inference_pipeline.yml``.*                              |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.batch_size         | integer             | Defaults to 336. Specifies the batch size for model training.                                                                                                                                   |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.pin_memory         | boolean             | Defaults to True. If True, feteched data Tensors will be in pinned memory, enabling faster data transfer to CUDA-enabled GPU.                                                                   |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.num_workers        | integer             | Defaults to 3. Specifies the number of subprocesses to use for data loading.                                                                                                                    |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| init.statistical_window | integer             | Defaults to 0. Specifies the window length to generate statistical features. If value = 0, no statistical features will be generated. *Do not exist in ``conf/docker/inference_pipeline.yml``.* |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| feature_to_standardize  | list                | Specifies the list of features to standardize. *Do not exist in ``conf/docker/inference_pipeline.yml``.*                                                                                        |
+-------------------------+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+---------------------+-----------------------------------------------------------------------------------+
| Configuration Name       | Parameter Data Type | Explanation                                                                       |
+==========================+=====================+===================================================================================+
| modeling.visualizations:                                                                                                           |
+--------------------------+---------------------+-----------------------------------------------------------------------------------+
| time_col                 | string              | Defaults to "MEASUREMENT_TAKEN_ON(UTC)". Specifies the name of time index column. |
+--------------------------+---------------------+-----------------------------------------------------------------------------------+
| plotting_features        | list                | Specifies the features to plot during visualizations.                             |
+--------------------------+---------------------+-----------------------------------------------------------------------------------+
