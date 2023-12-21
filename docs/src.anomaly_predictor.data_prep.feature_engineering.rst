src.anomaly\_predictor.data\_prep.feature_engineering
=====================================================

Overview
----------
For feature engineering, one can instantiate a ``FeatureEngineer`` object
and use its main method :meth:`~src.anomaly_predictor.data_prep.feature_engineering.FeatureEngineer.engineer_features` 
to engineer certain features for dataset. 

``engineer_features`` takes in a pandas DataFrame, along with its original
file name, from which the asset's application type is inferred. It then engineers
a new feature "Variable_speed" using private method 
:meth:`~src.anomaly_predictor.data_prep.feature_engineering.FeatureEngineer._determine_variable_speed_bool`
which uses the "Motor Supply Frequency" column of data to infer if asset's speed is 
fixed at a certain value or can vary. Another feature "Asset_Operating" is also created using
the same "Motor Supply Frequency" column to infer if asset is current operating
at that time point or is switched off. Finally, ``engineer_features`` also calls 
:meth:`~src.anomaly_predictor.data_prep.feature_engineering.FeatureEngineer.create_one_hot_encode_application` 
for onehot encoding of application type. During training, the full set of applications 
available would have been seen whilst doing data splitting. This same set of applications
is retained to fit the encoder in :meth:`~src.anomaly_predictor.data_prep.feature_engineering.FeatureEngineer.fit_encoder`. 
If this was during inference instead of training, then the saved encoder could be 
instantiated via :meth:`~src.anomaly_predictor.data_prep.feature_engineering.FeatureEngineer.load_encoder`.

src.anomaly\_predictor.modeling.data\_prep.feature\_engineering
---------------------------------------------------------------
.. automodule:: src.anomaly_predictor.data_prep.feature_engineering
   :members:
   :undoc-members:
   :show-inheritance: