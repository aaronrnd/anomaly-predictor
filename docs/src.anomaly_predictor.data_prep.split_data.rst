src.anomaly\_predictor.data\_prep.split\_data
=============================================

Overview
----------
For splitting data, one can instantiate a ``DataSplitter`` object, and use its 
main method :meth:`~src.anomaly_predictor.data_prep.split_data.DataSplitter.split_files`. 
This method expects to receive a directory in which
the data of various assets are held. The function reads through the directory to
extract the application type of each asset, using which to then split the assets
into train, validation and test partitions in a stratified manner. In the event 
that there is only 1 or 2 instances of an application, the code ensures that the 
assets of that application type would be included in the train, and potentially 
validation set, only. The proportion of validation and test sets will be the same, 
and is specified through the ``test_size`` argument. 

src.anomaly\_predictor.modeling.data\_prep.split\_data
------------------------------------------------------
.. automodule:: src.anomaly_predictor.data_prep.split_data
   :members:
   :undoc-members:
   :show-inheritance: