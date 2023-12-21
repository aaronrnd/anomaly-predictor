# pylint: skip-file
import datetime
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.anomaly_predictor.data_prep.ingest_data import AnnotationIngestor

CORRECT_ANOMALY_LABEL = 1
TIME_COL = "MEASUREMENT_TAKEN_ON(UTC)"
CUT_OFF_DATE = "01-07-21"


@pytest.fixture
def datetime_checks():
    index = pd.to_datetime("05-01-2022", dayfirst=True)
    start_date = pd.to_datetime("01-01-2022", dayfirst=True)
    end_date = pd.to_datetime("10-01-2022", dayfirst=True)
    return index, start_date, end_date


@pytest.fixture(scope="session")
def annotation_xlsx(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("annotation.xlsx")
    column_names = [
        "Rename",
        "Start Date",
        "Start Time",
        "End Date",
        "End Time",
        "Description",
    ]
    anomaly = [
        "ABB",
        pd.to_datetime("5-Oct-21", dayfirst=True),
        0,
        pd.to_datetime("6-Oct-21", dayfirst=True),
        0,
        "ANOMALY",
    ]
    values = [[], column_names, anomaly]
    df1 = pd.DataFrame(values)
    df1.to_excel(fn, index=False, header=False)
    return str(fn)


@pytest.fixture
def correct_annotation_dictionary():
    dictionary = {
        "ABB": [
            {
                "start_date": pd.to_datetime("5-Oct-21", dayfirst=True),
                "start_time": datetime.time(0),
                "end_date": pd.to_datetime("6-Oct-21", dayfirst=True),
                "end_time": datetime.time(0),
                "description": "ANOMALY",
            }
        ]
    }
    return dictionary


@pytest.fixture
def df_name():
    return "ABB"


@pytest.fixture
def dataframe_without_anomaly():
    time_range = pd.date_range(start="01-01-2021", end="01-10-2022", freq="H")
    test_df = pd.DataFrame([time_range], index=[TIME_COL]).transpose()
    return test_df


@pytest.fixture(scope="session")
def combined_workbook(tmpdir_factory):
    df1 = pd.DataFrame({"Data": ["a", "b", "c", "d"]})
    df2 = pd.DataFrame({"Data": [1, 2, 3, 4]})
    df3 = pd.DataFrame({"Data": [1.1, 1.2, 1.3, 1.4]})
    fn = tmpdir_factory.mktemp("data").join("multiple.xlsx")
    writer = pd.ExcelWriter(fn, engine="openpyxl")
    df1.to_excel(writer, sheet_name="Sheeta")
    df2.to_excel(writer, sheet_name="Sheetb")
    df3.to_excel(writer, sheet_name="Sheetc")
    writer.save()
    return str(fn)


@pytest.fixture(scope="session")
def assets_dir(tmpdir_factory):
    assets_dir = tmpdir_factory.mktemp("assets")
    for i in range(0, 10, 2):
        asset_subdir = assets_dir / f"ABB_{i}"
        asset_subdir.mkdir()
        file = asset_subdir / f"ABB_{i}.xlsx"

        time_range = pd.date_range(
            start=f"{1+i}-01-2021", end=f"{i+3}-01-2021", freq="H"
        )
        test_df1 = pd.DataFrame([time_range], index=[TIME_COL]).transpose()
        test_df1["data"] = "a"
        test_df2 = pd.DataFrame([time_range], index=[TIME_COL]).transpose()
        test_df2["data"] = "b"
        test_df3 = pd.DataFrame([time_range], index=[TIME_COL]).transpose()
        test_df3["data"] = "c"
        writer = pd.ExcelWriter(file, engine="openpyxl")
        test_df1.to_excel(writer, sheet_name="Sheeta", index=False)
        test_df2.to_excel(writer, sheet_name="Sheetb", index=False)
        test_df3.to_excel(writer, sheet_name="Sheetc", index=False)
        writer.save()

    return assets_dir


@pytest.fixture(scope="session")
def resulting_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("resulting_dir"))


@pytest.fixture(scope="session")
def temp_folder_for_csv(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return str(fn)


def test_ingest_data_annotations(annotation_xlsx, assets_dir, resulting_dir):
    results = AnnotationIngestor().ingest_data_annotations(
        [annotation_xlsx], Path(assets_dir), TIME_COL, CUT_OFF_DATE, Path(resulting_dir)
    )
    assert isinstance(results, list)
    assert results == os.listdir(assets_dir)


def test_xlsx2csv(assets_dir, correct_annotation_dictionary, resulting_dir):
    resulting_dir = Path(resulting_dir)
    shutil.rmtree(resulting_dir)
    resulting_dir.mkdir(parents=True)

    xlsx_dir = Path(assets_dir) / "ABB_0"
    asset_name = AnnotationIngestor()._xlsx2csv(
        xlsx_dir, correct_annotation_dictionary, TIME_COL, CUT_OFF_DATE, resulting_dir
    )
    assert isinstance(asset_name, str)
    assert asset_name == xlsx_dir.name
    assert (resulting_dir / "ABB_0_NaT-NaT.csv").is_file()


def test_create_annotation_dictionary(annotation_xlsx, correct_annotation_dictionary):
    result = AnnotationIngestor()._create_annotation_dictionary([annotation_xlsx])
    assert result == correct_annotation_dictionary
    assert isinstance(result, dict)


def test_combine_sheets_to_df(combined_workbook):
    results = AnnotationIngestor()._combine_sheets_to_df(
        load_workbook(combined_workbook)
    )
    assert len(results.columns) == 3
    assert results["Sheetb"].sum() == 10
    assert results["Sheetc"].max() == 1.4
    assert isinstance(results, pd.DataFrame)


def test_generate_labels_and_ignore_period(
    annotation_xlsx, dataframe_without_anomaly, df_name
):
    annotation_dic = AnnotationIngestor()._create_annotation_dictionary(
        [annotation_xlsx]
    )
    # test inference
    inference_result = AnnotationIngestor(
        mode="inference"
    )._generate_labels_and_ignore_period(
        dataframe_without_anomaly, df_name, annotation_dic, TIME_COL, "01-07-2021"
    )
    assert inference_result.shape[1] == dataframe_without_anomaly.shape[1]

    # test training
    result = AnnotationIngestor()._generate_labels_and_ignore_period(
        dataframe_without_anomaly, df_name, annotation_dic, TIME_COL, "01-07-2021"
    )
    assert result["Anomaly"].cumsum().max() == 25
    assert result.shape[0] == 4632
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result["Anomaly"], pd.Series)
