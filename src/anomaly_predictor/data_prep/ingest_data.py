import datetime
import logging
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from openpyxl import load_workbook

logger = logging.getLogger(__name__)
dask.config.set(scheduler="multiprocessing")


class AnnotationIngestor:
    """Ingest Annotation into corresponding excel sheets and drop entries before 1st July 2021."""

    def __init__(self, mode="training"):
        self.mode = mode

    def ingest_data_annotations(
        self,
        annotation_list: list,
        assets_dir: str,
        time_col: str,
        cut_off_date: str,
        resulting_dir:str = None,
        offset_hours: int = None,
    ) -> list:
        """Loops through assets directory and perform the following steps:
            1. Creates anomaly dictionary with annotated anomalous period and asset Key:Value Pair.
            2. Combined all df with varying timeframes into 1 dataframe.
            3. Annotate anomaly Period.
            4. Dropped ignored periods (defaults to 1-7-21 if unspecified).

        Args:
            annotation_list (list): list of paths to annotations.
            assets_dir (str): path to assets directory.
            time_col (str): Column name of time-index in Dataframe.
            cut_off_date (str): Date to cut off dataset. Will only retain data after cut_off_date.
            resulting_dir ([type], optional): path of resulting directory. Defaults to None.
            offset_hours (int, optional): Timezone difference of annotation files in relation to
                asset file timezone.

        Returns:
            data_list (list): list of asset names that have been annotated.
        """
        # create annotation dictionary
        annotation_dictionary = self._create_annotation_dictionary(annotation_list)

        # check if resulting directory is given else default datestamped folder
        if not resulting_dir:
            path = (
                f"../data/interim/{str(datetime.datetime.today().strftime('%d-%m-%Y'))}"
            )
            resulting_dir = Path(path)

        # check if dir exists, else create
        Path(resulting_dir).mkdir(parents=True, exist_ok=True)

        # loop through given assets_dir
        subdirs = list(assets_dir.glob("**"))
        asset_list = []
        for subdir in subdirs[1:]:
            asset_name = dask.delayed(self._xlsx2csv)(
                subdir, annotation_dictionary, time_col, cut_off_date, resulting_dir, offset_hours
            )
            asset_list.append(asset_name)
        asset_list = dask.delayed(lambda x: x)(asset_list)

        return asset_list.compute()

    def _xlsx2csv(
        self,
        xlsx_dir: str,
        annotation_dictionary: dict,
        time_col: str,
        cut_off_date: str,
        resulting_dir=None,
        offset_hours:int = None,
    ) -> str:
        """Converts respective xlsx sheet(s) to a combined dataframe. Resultant
        dataframes to be annotated and only data after cut off date will be retained
        and saved.

        Args:
            xlsx_dir (str): Directory of folder containing respective asset's excel sheets.
            annotation_dictionary (dict): Annotation Dictionary containing anomalous period
                and asset Key:Value Pair.
            time_col (str): Column name of time-index in Dataframe.
            cut_off_date (str): Date to cut off dataset. Will only retain data after cut_off_date.
            resulting_dir ([type], optional): path of resulting directory. Defaults to None.
            offset_hours (int, optional): Timezone difference of annotation files in relation to
                asset file timezone. Defaults to None.

        Returns:
            asset_name (str): Asset name(s) that have been successfully combined and annotated.
        """
        xlsx_paths = list(xlsx_dir.glob("*.xlsx"))
        xlsx_paths.sort()

        # create combined dataframe (from varying timeframes)
        data = pd.DataFrame()
        for path in xlsx_paths:
            workbook = load_workbook(path)
            cur_df = self._combine_sheets_to_df(workbook)
            data = pd.concat([data, cur_df], axis=0)

        # reset index and convert time column to datetime format
        data.reset_index(inplace=True)
        data[time_col] = pd.to_datetime(data[time_col])

        # extract df name to save folder
        asset_name = Path(xlsx_dir).name

        # create labels + ignore period and save df.
        data = self._generate_labels_and_ignore_period(
            data,
            asset_name,
            annotation_dictionary,
            time_col,
            cut_off_date,
            save_directory=resulting_dir,
            offset_hours = offset_hours,
        )
        return asset_name

    def _create_annotation_dictionary(self, annotation_list: list) -> dict:
        """Creates annotation dictionary from a list of annotation paths.

        Args:
            annotation_list (list): list of paths to annotations xlsx.

        Returns:
            annotation_dic (dict): dictionary of annotations.
        """
        annotation_dic = {}
        for xlsx in annotation_list:
            annotations = pd.read_excel(
                xlsx, engine="openpyxl", header=1, usecols=lambda x: "Unnamed" not in x
            )
            description_list = ["ANOMALY", "IGNORE"]
            anomaly_annotations = annotations[
                annotations["Description"].isin(description_list)
            ]
            for _, row in anomaly_annotations.iterrows():
                if not isinstance(row["Start Time"], datetime.time):
                    start_time = datetime.time(int(row["Start Time"]))
                else:
                    start_time = row["Start Time"]

                if not isinstance(row["End Time"], datetime.time):
                    end_time = datetime.time(int(row["End Time"]))
                else:
                    end_time = row["End Time"]

                item_dict = {
                    "start_date": row["Start Date"],
                    "start_time": start_time,
                    "end_date": row["End Date"],
                    "end_time": end_time,
                    "description": row["Description"],
                }
                if row["Rename"] in annotation_dic:
                    annotation_dic[row["Rename"]].append(item_dict)
                else:
                    annotation_dic[row["Rename"]] = [item_dict]
        return annotation_dic

    @staticmethod
    def _combine_sheets_to_df(workbook:pd.DataFrame)-> pd.DataFrame:
        """Combines all sheets in a particular workbook into a single df.

        Args:
            workbook (pd.DataFrame): Excel workbook containing the various worksheet(s).

        Returns:
            main_df (pd.DataFrame): Dataframe consisting of the combined xlsx sheet(s).
        """
        first_sheet = True
        for sheet in workbook.worksheets:
            sheet_arr = np.array(list(sheet.values))[:, 0:2]
            if first_sheet:
                main_df = pd.DataFrame(
                    sheet_arr[1:, :], columns=[sheet_arr[0, 0], sheet.title]
                )
                main_df = main_df.set_index(main_df.columns[0])
                main_df = main_df[~main_df.index.duplicated(keep="first")]
                first_sheet = False

            else:
                cur_df = pd.DataFrame(
                    sheet_arr[1:, :], columns=[sheet_arr[0, 0], sheet.title]
                )
                cur_df = cur_df.set_index(cur_df.columns[0])
                main_df = main_df[~main_df.index.duplicated(keep="first")]
                main_df = main_df.merge(
                    cur_df, how="outer", left_index=True, right_index=True
                )

        return main_df[sorted(main_df.columns)]

    def _generate_labels_and_ignore_period(
        self,
        data: pd.DataFrame,
        asset_name: str,
        annotation_dic: dict,
        time_col: str,
        cut_off_date: str,
        save_directory: str = None,
        offset_hours: int = None,
    ) -> pd.DataFrame:
        """Generate "Anomaly" columns and drop ignore periods.

        Args:
            data (pd.DataFrame): DataFrame.
            asset_name (str): asset Name.
            annotation_dic (dict): annotation Dictionary.
            time_col (str, optional): name of time column. Defaults to global variable TIME_COL.
            cut_off_date (str): Date to cut off dataset. Will only retain data after cut_off_date.
            save_directory (str, optional): path directory to save data. Defaults to None.
            offset_hours (int, optional): Timezone difference of annotation files in relation to
            asset file timezone.

        Returns:
            data (pd.DataFrame): DataFrame with the dropped ignore periods and anomaly column.
        """
        earliest_date = pd.to_datetime(data[time_col].min())
        latest_date = pd.to_datetime(data[time_col].max())
        if self.mode == "training":
            data["Anomaly"] = 0
            if asset_name in annotation_dic:
                # loop through multiple annotations (multiple ignores or anomaly)
                for annotation in annotation_dic[asset_name]:
                    if annotation["description"] == "IGNORE":
                        ignore_date = datetime.datetime.combine(
                            annotation["end_date"], annotation["end_time"]
                        )
                        if offset_hours:
                            ignore_date = ignore_date + datetime.timedelta(hours=offset_hours)
                        ignore_date = pd.to_datetime(ignore_date, dayfirst=True)
                        data = data[data[time_col] > ignore_date]
                    elif annotation["description"] == "ANOMALY":
                        start_date_placeholder = datetime.datetime.combine(
                            annotation["start_date"], annotation["start_time"]
                        )
                        end_date_placeholder = datetime.datetime.combine(
                            annotation["end_date"], annotation["end_time"]
                        )
                        if offset_hours:
                            start_date_placeholder = start_date_placeholder + datetime.timedelta(
                                hours=offset_hours
                            )
                            end_date_placeholder = end_date_placeholder + datetime.timedelta(
                                hours=offset_hours
                            )
                        anomaly_start = pd.to_datetime(
                            start_date_placeholder, dayfirst=True
                        )
                        anomaly_end = pd.to_datetime(
                            end_date_placeholder, dayfirst=True
                        )
                        if anomaly_start < earliest_date or anomaly_end > latest_date:
                            logger.warning(
                                "Annotation exceeds date range of %s",
                                asset_name
                            )

                        anomaly_range = pd.date_range(
                            anomaly_start, anomaly_end, freq="s"
                        )
                        data.loc[data[time_col].isin(anomaly_range), "Anomaly"] = 1
        elif self.mode == "inference":
            if asset_name in annotation_dic:
                # loop through multiple annotations (multiple ignores or anomaly)
                for annotation in annotation_dic[asset_name]:
                    if annotation["description"] == "IGNORE":
                        ignore_date = datetime.datetime.combine(
                            annotation["end_date"], annotation["end_time"]
                        )
                        if offset_hours:
                            ignore_date = ignore_date + datetime.timedelta(hours=offset_hours)
                        ignore_date = pd.to_datetime(ignore_date, dayfirst=True)
                        data = data[data[time_col] > ignore_date]

        # default ignore period = cut_off_date
        ignore_date = pd.to_datetime(cut_off_date, dayfirst=True)
        data = data[data[time_col] > ignore_date]

        earliest_date = data[time_col].min()
        latest_date = data[time_col].max()

        # saving directory
        if save_directory:
            start_date = str(earliest_date.date()).replace("-", "")
            end_date = str(latest_date.date()).replace("-", "")
            csv_name = (
                str(save_directory / asset_name) + f"_{start_date}-{end_date}.csv"
            )
            data.to_csv(csv_name, index=False)

        return data
