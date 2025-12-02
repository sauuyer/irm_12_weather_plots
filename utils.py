# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# +
def md2vect(wspd, wdir):
    """Calculate north and east wind vectors from magnitude and direction

    Parameters
    ----------
    wspd: array-like
        Absolute wind speeds (m/s)
    wdir: array-like
        Wind directions (degrees). Values will be wrapped into [0, 360).

    Returns
    -------
    vx: numpy.ndarray
        Northerly wind component (same shape as wspd)
    vy: numpy.ndarray
        Easterly wind component (same shape as wspd)
    """
    # Ensure NumPy arrays (avoid pandas SettingWithCopyWarning)
    wspd = np.asarray(wspd, dtype=float)
    wdir = np.asarray(wdir, dtype=float)

    # Wrap angles into [0, 360)
    wdir = np.mod(wdir, 360.0)

    # Convert to radians
    wdir_rad = np.deg2rad(wdir)

    # Calculate northerly and easterly wind vectors
    vx = wspd * np.sin(wdir_rad)  # north component
    vy = wspd * np.cos(wdir_rad)  # east component

    return vx, vy


def vect2md(vx, vy):
    """Calculate the wind speed and direction from north and east wind vectors.

    Parameters
    ----------
    vx: array-like
        Northerly wind component
    vy: array-like
        Easterly wind component

    Returns
    -------
    wspd: numpy.ndarray
        Absolute wind speeds (m/s)
    wdir: numpy.ndarray
        Directions (degrees) in [0, 360)
        (direction the vector points TOWARD, clockwise from north)
    """
    # Ensure NumPy arrays
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    # Speed
    wspd = np.sqrt(vx**2 + vy**2)

    # Direction in degrees, range [-180, 180]
    wdir = np.degrees(np.arctan2(vx, vy))

    # Wrap into [0, 360)
    wdir = np.mod(wdir, 360.0)

    return wspd, wdir


def bpr_adjust(bpr,atmp,zbpr,znew):
    # constants
    g = 9.81 # gravity acceleration in m/s^-2
    Ra = 287.05 # gas constant for dry air in J Kg^-1 K^-1

    # barometric pressure
    atmp=atmp+273.15 # air temperature in K
    bprnew=bpr*np.exp(-g*(znew-zbpr)/(Ra*atmp));

    return bprnew


def rh2q(atmp, bpr, hrh, sflag=0):
    """Calculate specific humidity from relative humidity.

    Parameters
    ----------
    atmp: (float)
        air temperature in deg_C
    bpr: (float)
        barometric pressure in mbars
    hrh: (float)
        relative humidity expressed in percentage (so 95% rh = 95, not .95)
    sflag: (int, default==0)
        1 means at the sea interface (then atmp should be skin SST and hrh
        value will be replaced to 100%rh), 0= otherwise

    Return
    ------
    q: (float)
        Specific humidity in g/kg
    """

    # Calculate saturated vapor pressure es in mb, using Buck
    es=6.1121*np.exp(17.502*atmp/(atmp+240.97))*(1.0007+3.46E-6*bpr)

    # Saturated pressure is 2% less due to slat right at the sea interface
    es=(1.0-0.02*sflag)*es;

    # Second, by def of relative hymidity, vapor pressure is:
    e=es*hrh/100 # Where HRH is in %RH
    if sflag==1:
        e = es

    # Compute specific humidity
    q=(0.62197*e)/(bpr-0.378*e); # q in kg/kg
    q = q*1000; # q in g/kg

    return q


# -

class Ship():

    def __init__(self):

        self.heights = {
            "wind": 17.9,
            "atmp": 17.9,
            "bpr": 17.9,
            "sst": -4.5
        }

    def find_files(self, filepath, T1, T2):
        """Function to find metbk files which fall between two stated time periods.

        Parameters
        ----------
        filepath: (str)
            The path to the directory which has the ship met data files
        T1: (datetime)
            The time to start the file search.
        T2: (datetime)
            The end time for the file search.

        Returns
        -------
        ship_files: (list)
            A list of all the ship met files which fall between the two time periods (inclusive)
        """

        # ID desired ship files
        ship_files = []
        for file in os.listdir(filepath):
            if file.endswith(".csv"):
                file_date = pd.to_datetime(file[2:8], format="%y%m%d")
                if (file_date >= T1) & (file_date <= T2):
                    ship_files.append(f"{filepath}/{file}")
        return ship_files


    def parse_data(self, filepaths):
        """Parse the ship met data into a pandas DataFrame.

        Parameters
        ----------
        filepaths: (list)
            A list of files with absolute or relative paths which
            contain the data to parse

        Returns
        -------
        data: (pandas.DataFrame)
            A dataframe with the parsed ship met data
        """
        data = pd.DataFrame()
        file_df_list = []
        for file in filepaths:
            df = pd.read_csv(file, header=1)
            file_df_list.append(df)
            #data = data.append(pd.read_csv(file, header=1), ignore_index=True)
        data = pd.concat(file_df_list)
        data.rename(columns=lambda x: x.strip(), inplace=True)
        print(data)

        for col in data.columns:
            #print(col)
            data.rename(columns={col: col.strip()}, inplace=True)

        return data
        print(data)


    def parse_headers(self, filepath):
        """Parse the header information and attributes for the ship met data.

        Parameters
        ----------
        filepath: (str)
            The full filepath, including file name, of the header file with
            associated metadata for the ship met data

        Returns
        -------
        headers: (list)
            A list of the column headers of the ship met data
        attributes: (list)
            A list of dictionary objects containing the header name,
            a description, and the units for each column header.
        """
        # First, open the header file
        with open(filepath) as file:
            hdr = file.readlines()
            hdr = [h.replace("\n", "") for h in hdr]
            hdr = [h for h in hdr if len(h) > 0]

        # Parse the column headers and column attributes
        attributes = []
        for n, line in enumerate(hdr):
            # First line contains no relevant info
            if n == 0:
                pass
            # The second line contains the column names
            elif n == 1:
                headers = line.split(",")
                headers = [h.strip() for h in headers]
            # The remainder is the attribute information
            else:
                line = line.strip("\n")
                if len(line) == 0:
                    continue
                if len(line.split("-")) == 2:
                    name, desc = [x.strip() for x in line.split("-")]
                    units = ""
                else:
                    name, desc, units = [x.strip() for x in line.split("-")]
                    attributes.append({"header": name, "desc": desc, "units": units})

        # Return the data
        return headers, attributes


    def add_attributes(self, data, attributes):
        """Function which adds attributes to the ship met data.

        Parameters
        ----------
        data: (pandas.DataFrame)
            A pandas dataframe with the parsed ship met data and column headers
        attributes: (list)
            A list of dictionary objects containing the header name,
            a description, and the units for each column header.

        Returns
        -------
        data: (pandas.DataFrame)
            The dataframe of the parsed ship met data with the description and units
            added to the attributes of each column.
        """

        for item in attributes:
            name, desc, units = item.get("header"), item.get("desc"), item.get("units")
            try:
                data[name].attrs = {
                    "description": desc,
                    "units" : units,
                }
            except:
                pass

        return data


    def process_data(self, df):
        """
        Process the ship met data to get a single value from twinned sensors
        and rename the fields to match the metbk
        """

        #Remove leading and trailing white space from cell values
        #Reformat NAN strings to actual NaNs
        #Drop any NaNs from the data
        #df['WXTS_Ta'] = df['WXTS_Ta'].apply(lambda x: x.strip())

        print(df)
        df = df.replace(' NAN', 'NaN')
        df = df.replace('NAN', 'NaN')
        df = df.replace('NaN', np.nan)
        #df = df.dropna()
        print(df)
        df.to_csv('./metbk_ship_parsed_data_preprocessed.csv')

        #make a list of all object column names in df
        object_columns_list = list(df.select_dtypes(include=['object']).columns)
        print('OBJECT COLS LIST RESULTS')
        print(object_columns_list)
        #remove the date and time column names from this list, remove function only takes one arg at a time...?
        object_columns_list.remove('TIME_GMT')
        object_columns_list.remove('DATE_GMT')
        #make all remaining columns in the list to_numeric
        #df[object_columns_list] = pd.to_numeric(df[object_columns_list])
        df[object_columns_list] = df[object_columns_list].apply(pd.to_numeric, errors='coerce')

        # Generate a datetime column
        df["datetime"] = pd.to_datetime(df["DATE_GMT"] + " " + df["TIME_GMT"])
        df = df.drop(columns=["DATE_GMT", "TIME_GMT"])

        #df["WXTS_Ta"] = pd.to_numeric(df['WXTS_Ta'])
        print(df.dtypes)
        # Calculate an air temperature
        df["atmp"] = df[["WXTS_Ta", "WXTP_Ta"]].mean(axis=1)
        df = df.drop(columns=["WXTS_Ta", "WXTP_Ta"])

        # Calculate a barometric pressure & drop the "corrected" pressure
        #df["WXTS_Pa"] = pd.to_numeric(df['WXTS_Pa'])
        df["bpr"] = df[["WXTP_Pa", "WXTS_Pa"]].mean(axis=1)
        df = df.drop(columns=["WXTP_Pa", "WXTS_Pa", "BAROM_P", "BAROM_S"])

        # Rain intensity
        df["prc"] = df[["WXTP_Ri", "WXTS_Ri"]].mean(axis=1)
        df = df.drop(columns=["WXTP_Ri", "WXTS_Ri", "WXTP_Rc", "WXTS_Rc"])

        # Drop relative wind directions and speed in favor of true wind direction and speed
        df = df.drop(columns=["WXTP_Dm", "WXTS_Dm", "WXTP_Sm", "WXTS_Sm"])
        df["wspd"] = df[["WXTS_TS", "WXTP_TS"]].mean(axis=1)
        df["wdir"] = df[["WXTS_TD", "WXTP_TD"]].mean(axis=1)
        df = df.drop(columns=["WXTS_TS", "WXTP_TS", "WXTS_TD", "WXTP_TD"])

        # Relative humidity
        df["rhr"] = df[["WXTP_Ua", "WXTS_Ua"]].mean(axis=1)
        df = df.drop(columns=["WXTP_Ua", "WXTS_Ua"])

        # Shortwave radiation
        df["swr"] = df["RAD_SW"]
        df = df.drop(columns=["RAD_SW"])

        # Longwave radiation
        df["lwr"] = df["RAD_LW"]
        df = df.drop(columns=["RAD_LW"])

        # Surface water measurements
        df["sal"] = df["SBE45S"]
        df["sst"] = df["SBE48T"]
        df["par"] = df["PAR"]
        df["flr"] = df["FLR"]
        df = df.drop(columns=["SBE45S", "SBE48T", "PAR", "FLR"])

        # Ship DAta
        df["lon"] = df["Dec_LON"]
        df["lat"] = df["Dec_LAT"]
        df["spd"] = df["SPD"]
        df["sog"] = df["SOG"]
        df["cog"] = df["COG"]
        df["hdt"] = df["HDT"]

        # Drop remaining columns
        df = df.drop(columns=["Dec_LON", "Dec_LAT", "SPD", "SOG", "COG",
                              "SSVdslog", "HDT", "FLOW", "Depth12", "Depth35", "EM122"])

        print(df)
        return df


class Buoy():

    def __init__(self):

        self.heights = {
            "deck": 45/100,
            "wind": 5.74,
            "atmp": 5.25,
            "bpr": 5.05,
            "sst": -1.30
        }

    def parse_data(self, filepaths):
        """Parse the metbk data into a pandas DataFrame"""
        columns = ["date", "time", "bpr", "hrh", "atmp", "lwr", "prc", "sst",
                   "cond", "swr", "wnde", "wndn", "Vbatt1", "Vbatt2"]

        metbk = pd.DataFrame()
        file_df_list = []
        for file in filepaths:
            df = pd.read_csv(file, names=columns, delim_whitespace=True, on_bad_lines='skip')
            file_df_list.append(df)

            '''with open(file) as data:
                for line in data.readlines():
                    line = line.replace("\n","")
                    line = [x for x in line.split(" ") if len(x) > 0]
                    if len(line) != 14:
                        continue
                    metbk_line_dict = {
                        "date": line[0],
                        "time": line[1],
                        "bpr": line[2],
                        "hrh": line[3],
                        "atmp": line[4],
                        "lwr": line[5],
                        "prc": line[6],
                        "sst": line[7],
                        "cond": line[8],
                        "swr": line[9],
                        "wnde": line[10],
                        "wndn": line[11],
                        "Vbatt1": line[12],
                        "Vbatt2": line[13]}
                        '''

                    #file_df_line_list.append(metbk_line_dict)



                #print(metbk_line_dict)

                    #metbk = pd.DataFrame.from_dict(metbk_dict, orient='index').transpose()

        metbk = pd.concat(file_df_list)

        #print('METBK print result')
        #print(metbk)

        return metbk


    def find_files(self, filepath, T1, T2):
        """Function to find metbk files which fall between two stated time periods.

        Parameters
        ----------
        filepath: (str)
            The path to the directory which has the metbk data files
        T1: (datetime)
            The time to start the file search.
        T2: (datetime)
            The end time for the file search.

        Returns
        -------
        metbk_files: (list)
            A list of all the metbk files which fall between the two time periods (inclusive)
        """

        metbk_files = []
        for file in os.listdir(filepath):
            if file.endswith(".log"):
                file_date = pd.to_datetime(file[2:8], format="%y%m%d")
                if (file_date >= T1) & (file_date <= T2):
                    metbk_files.append(f"{filepath}/{file}")

        return metbk_files
