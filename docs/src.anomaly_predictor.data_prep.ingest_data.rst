src.anomaly\_predictor.data\_prep.ingest\_data
==============================================

Overview
----------
The class ``AnnotationIngestor`` actually accomplishes two objectives within the 
main method :meth:`~src.anomaly_predictor.data_prep.ingest_data.AnnotationIngestor.ingest_data_annotations`. 
It combines data for a single asset  into a single dataframe and ingests annotations 
to know which periods to ignore (and hence drop these ignored timepoints from the 
subsequent interim files), along with which periods were actually anomalous. 
``AnnotationIngestor`` has 2 modes: "training" and "inference", which dictates 
whether any "Anomaly" column is created.

The method first uses :meth:`~src.anomaly_predictor.data_prep.ingest_data.AnnotationIngestor._create_annotation_dictionary` 
to ingest the start and end times of anomalous and ignored periods. This dictionary is then used in 
:meth:`~src.anomaly_predictor.data_prep.ingest_data.AnnotationIngestor._xlsx2csv`
where features values for a single asset, spread across multiple excel sheets
and files, is combined into a single dataframe. Ignored periods are dropped. As 
for creation of the "Anomaly" column, if  ``AnnotationIngestor`` is in training mode, 
then any ingested anomalous periods are indicated in a newly created "Anomaly" column. 
However, if ``AnnotationIngestor`` is in inference mode, then no ingestion of anomalous
periods is done and no "Anomaly" column is created since ground truths are not known at 
the point of inference.

src.anomaly\_predictor.modeling.data\_prep.ingest\_data
-------------------------------------------------------
.. automodule:: src.anomaly_predictor.data_prep.ingest_data
   :members:
   :undoc-members:
   :show-inheritance: