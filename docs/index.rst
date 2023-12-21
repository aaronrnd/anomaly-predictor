.. anomaly predictor documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AI-Enabled Application for ABB Smart Sensors
============================================

Purpose of this document
------------------------

Why are we writing this? Who are we writing for?

TODO: ensure all private methods docstrings are shown

Solution overview
-----------------

Description in prose, with diagrams as necessary

.. toctree::
   :maxdepth: 1
   :caption: Pipelines:

   src.anomaly_predictor.run_training
   src.anomaly_predictor.run_inference
   train_pipeline_config
   inference_pipeline_config

.. toctree::
   :maxdepth: 1
   :caption: Data Preparation:

   src.anomaly_predictor.data_prep
   src.anomaly_predictor.data_prep.ingest_data
   src.anomaly_predictor.data_prep.split_data
   src.anomaly_predictor.data_prep.clean_data
   src.anomaly_predictor.data_prep.feature_engineering

.. toctree::
   :maxdepth: 1
   :caption: Modeling:

   src.anomaly_predictor.modeling
   src.anomaly_predictor.modeling.data_loaders
   src.anomaly_predictor.modeling.models
   src.anomaly_predictor.modeling.evaluation
   src.anomaly_predictor.modeling.visualization

.. toctree::
   :maxdepth: 1
   :caption: Utilities:

   src.anomaly_predictor.utils