import logging
import functools
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Setup the logger
# logger = Logger().get_logger()


class DataUtility:
    """
    DataUtility class which contains meta data, logic for tracking rows and column changes along with functional
    logic to aid in slicing and converting data all used for modelling.

    Legend speaks of Uttarakhand, India. If one wishes to find the true nature of this class, let your journey begin there.
    Ask for DataGuru.
    """

    class _Decorators(object):
        @classmethod
        def target_exists(cls, function):
            """decorator function used to test whether the target property of the DataUtility class has a value
            This is to ensure that functions making use of the target don't get called before it has a value.

            Args:
                function (objec): a function in DataUtility class which is dependant on the target property having a value
            """

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                # args[0].target = self.target = meta_data.target because 'self' is the first argument passed for any DataUtility method
                if args[0].target is None:
                    err_msg = f"The value of target for the DataUtility class is None, assign a value to the target property to use '{function.__name__}' function"
                    raise TargetNotDefinedError(err_msg)
                return function(*args, **kwargs)

            return wrapper

    excluded_cols = set()
    unique_sets = set()
    row_sample = {
        "train": set(),
        "test": set(),
        "valid": set(),
        "rec": set(),
        "oot": set(),
    }
    replacements = {}
    fillna_value = np.nan

    def __init__(
        self, data, target=None, convert_dtypes=False, default_subset="modelling"
    ):
        """
        Initializes the class by assigning default values derived from the dataset.

        Args:
            data (pandas Dataframe): The dataset used for modeling, which the DataUtility will be based off of.
            target (str, optional): The name of the target column. Defaults to None
            convert_dtypes (bool, optional): Flag to specify whether to use pandas convert_dtypes method to try
                                             and infer the data types of the different variables
        """

        self.default_subset = default_subset
        self.target = target
        if convert_dtypes:
            data = data.convert_dtypes()
        self.column_info = {
            col: {
                "data_type": data[col].dtype,
                "included": {"value": True, "reason": None},
                # Target should not be part of the default set
                "subset": default_subset if not (col == target) else "target",
                "force_type": get_force_type(data[col]),
                "index": index,
            }
            for col, index in zip(data.columns, range(len(data.columns)))
        }

    def __repr__(self):
        """Overwrite the repr dunder method to nicely print the object

        Returns
        -------
        str
            Formatted string of the dictionary
        """

        column_info_table = pd.DataFrame(self.column_info).transpose()
        column_info_table["included_value"] = column_info_table["included"].apply(
            lambda x: x["value"]
        )
        repr_str = f"______________________________________"
        repr_str = f"{repr_str}\n----------Force Type Summary----------"
        repr_str = f"{repr_str}\n{repr(column_info_table.pivot_table(index='force_type', columns='included_value', aggfunc=len, fill_value=0)['data_type'])}"
        repr_str = f"{repr_str}\n______________________________________"
        repr_str = f"{repr_str}\n----------Column Sets Summary---------"
        repr_str = f"{repr_str}\n{repr(column_info_table.pivot_table(index='subset', columns='included_value', aggfunc=len, fill_value=0)['data_type'])}"
        repr_str = f"{repr_str}\n______________________________________"
        repr_str = f"{repr_str}\n----------Row Sample Summary----------"
        repr_str = f"{repr_str}\n{repr({k: len(v) for k,v in self.row_sample.items()})}"

        return repr_str

    def update_properties(self):
        """
        Method that updates the excluded_cols and unique_sets properties if they were
        changed during an update. The idea is to have a set of the current excluded columns
        readily avaiable so it needs to be updated anytime a column is excluded.
        """
        self.excluded_cols = {
            col
            for col, col_info in self.column_info.items()
            if col_info["included"]["value"] == False
        }

        self.unique_sets = {val["subset"] for val in self.column_info.values()}

    def update(self, col_set, col_info, value):
        """Updates a specific information for a set of columns
        Example
            col_set = set('Logical_Record_Id', 'Insert_Date')
            col_info = 'subset'
            value = 'meta'
            will change the `subset` field of "Logical_Record_Id" and "Insert_Date"  columns to contain the value : 'meta'
            if col_info = subset and value = 'meta'...then all the `col_set` columns, subset fields will have
            Meaning they were classified as meta data (from a business logic perspective)
        Args:
            col_set (set(str)): The set for which the update should occure
            col_info (str): The type of information of the columns that's going to be changed,
            value (any): Can be a dict(str,str) or str or int depending on the field it's in reference to
        """
        if type(col_set) == list:
            # Convert colset to set if list was provided
            col_set = set(col_set)
        elif type(col_set) == str:
            col_set = {col_set}

        for col in col_set:
            self.column_info[col][col_info] = value

        if col_info == "included":
            # If the exluded list gets changed
            self.update_properties()

    def get(
        self,
        filter_dict={},
        data=None,
        row_sample_name=None,
        remove_excluded=True,
        include_target=False,
    ):
        """
            Gets all the included columns that meets the filter dictionary criteria,
            if a key isn't specified in the filter dictionary then that key is treated
            as a wild card (*), meaning any value for that key is allowed.
            Example if filter_dict only has {"subset": "bureau"} then all variables belonging
            to the 'bureau' subset will be returned, regardless of force_type, data_type.
            By default the excluded columns are removed from the set, but this can be changed with the remove_exlcuded flag.
            If filter_dict isn't specified then all columns are returned.

            Example: filter_dict={"subset": "bureau", force_type="numeric"} will return all included numeric bureau variables.
            If the data is provided then the sliced dataframe is returned (sliced based on columns that was filtered.)

        Args:
            filter_dict (dict(), optional):     A dictionary to filter the columns on, has the same format
                                                as a specific columns meta data. If you want to filter multiple values
                                                then you can provide it as a list, i.e. filter_dict={"subset":["bureau", "demographic"]} will
                                                return all bureau and demographic variables. Defaults to {}.

            data (pandas.DataFrame, optional):  A dataset containing the columns, if this is passed through then a dataframe
                                                with the selected columns will be returned. Defaults to None

            row_sample_name (str, optional):    The name of the sample set to filter rows, ex: "train". Defaults to None.

            remove_excluded (bool, optional)    Flag to spcify whether the objects excluded set should be removed from the
                                                returned set. Defaults to True.

            include_target (bool, optional):    Flag to specify the inclusion of the target to the set. Defaults to False.

        Returns:
            set(str):                           The set of columns belonging to the criteria specified
            or
            pandas.DataFrame:                   if a dataframe is passed through, it is then returned with only the subset columns
                                                selected
        """

        col_set = {
            # Return the column
            column
            # For every column in the data_utlity object
            for column, column_info in self.column_info.items()
            # If it meets all criteria specified in filter_dict
            if all(
                [
                    # If the column_info[key] is something like included which is a dictionary "included": {"value": False, "reason": "identical rate >= 0.95"}
                    # Then compare the dictionaries with == otherwise use the in operator for in case multiple sets was given as options
                    column_info[key] == filter_dict[key]
                    if type(filter_dict[key]) == dict
                    else column_info[key] in filter_dict[key]
                    # For every key that is in both dictionaries,
                    # i.e. if a key isn't in the filter dictionary then all values for that key is allowed
                    for key in set(column_info.keys()) & set(filter_dict.keys())
                ]
            )
        }

        target_set = {self.target} if include_target else set()
        # Excluded set should be populated with the objects excluded columns if remove_excluded==True and
        # filter_dict does not have the included key. If the filter_dict contains the included key, then that
        # should be used to decide which columns is included
        excluded_set = (
            self.excluded_cols
            if remove_excluded and "included" not in filter_dict
            else set()
        )

        return_set = col_set - excluded_set | target_set
        ordered_list = [col for col in self.column_info.keys() if col in return_set]

        if data is None:
            # TODO raise warning if sample is not None here, maybe return the ordered list here
            return ordered_list
        else:
            sample_idx = (
                set(data.index) & set(self.row_sample[row_sample_name])
                if not (row_sample_name is None)
                else set(data.index)
            )

            return data[ordered_list].loc[sample_idx]

    @_Decorators.target_exists
    def get_samples(self, data, row_samples=None, subsets=[]):
        """A wrapper function to get X, y and Xy datasets for a specific sample such as train, test.
        usage: X_train, y_train, Xy_train, X_test, y_test, Xy_test = get_samples(raw, ["train, test"]).

        Args:
            data (pandas.DataFrame): The data that needs to be sliced based on a row sample and columns subset
            row_samples (list(str) / str, optional): list of row sample names to return or name of row sample if it's just one.
                Defaults to None, if None then all rows are treated as the sample.
            subsets (list(str), optional):  A list of the subsets of columns that needs to be selected.
                Defaults to None which will include all.

        Returns:
            list(pd.Dataframe, pd.Series):  Returns a list of dataframes in the following order X, y, Xy where
                X: is a pd.DataFrame without target
                y: is a pd.Series (only target)
                Xy: is the pd.DataFrame with both target and variables
        """

        # If no subsets are specified then filter_dict should be empty as to
        # return all columns, else only the selected subsets should be returned
        filter_dict = {"subset": subsets} if subsets else {}

        if row_samples is None:
            row_samples = [None]

        if type(row_samples) == str:
            # samples is suppose to be a list, but this ensures if a string is passed that
            # it still functions the way it should since that can be intuitive for some users
            row_samples = [row_samples]

        sample_dfs = []
        for sample_name in row_samples:
            Xy_sample = self.get(
                filter_dict, data=data, row_sample_name=sample_name, include_target=True
            )
            ordered_X = [col for col in Xy_sample.columns if not(col in {self.target})]
            X_sample = Xy_sample[ordered_X]
            y_sample = Xy_sample[self.target]

            sample_dfs = sample_dfs + [X_sample, y_sample, Xy_sample]
        return sample_dfs


def get_force_type(column):
    """Gets the force type for a specific column of the dataset. Force type groups the column into
    one of three categories: numeric(ints, floats), mixed_other(objects) and string (strings). The classification
    is based purely on the dtype of the column of the data.

    This is created in a seperate function, so that when the classification changes, it's only changed in one place.

    Args:
        column (pd.Series): The column to be categorized
    Returns:
        (str): The correct force type classification
    """

    current_force_type = (
        "numeric"
        if is_numeric_dtype(column.dtype)
        else "mixed_other"
        if column.dtype == "O"
        else "string"
    )

    return current_force_type


class TargetNotDefinedError(Exception):
    """Exception raised if the target has not been defined"""

    pass
