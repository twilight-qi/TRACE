
import networkx as nx
import numpy as np
import pandas as pd
import torch
import time
import os
import hashlib
import pathlib
import errno
import tempfile
from nltk import ngrams
from pandarallel import pandarallel
from scipy.sparse import csr_matrix, diags
from scipy.spatial.ckdtree import cKDTree
from sklearn.cluster import KMeans
from torch_geometric.utils import add_self_loops, to_dense_adj
from tqdm import tqdm
from abc import ABC, abstractmethod
from contextlib import contextmanager
from src.utils import RankedLogger

pandarallel.initialize(progress_bar=False, nb_workers=16, verbose=0)

log = RankedLogger(__file__, rank_zero_only=True)


class Preprocessor:
    """Preprocessor class for cleaning and filtering user check-in data.

    Parameters:
    -----------
    max_speed_kmph : float, optional
        Maximum allowed speed in km/h for users (default is 1200 km/h).
    min_nb_checkins_per_location : int, optional
        Minimum number of check-ins required per location (default is 10).
    min_nb_checkins_per_user : int, optional
        Minimum number of check-ins required per user (default is 10).

    Methods:
    --------
    _filter_max_speed_users(data: pd.DataFrame) -> pd.DataFrame:
        Filters out users who exceed the maximum allowed speed.
    _filter_small_number_of_checkins_per_location(df: pd.DataFrame) -> pd.DataFrame:
        Filters out locations with fewer check-ins than the specified minimum.
    _filter_small_number_of_checkins_per_user(df: pd.DataFrame) -> pd.DataFrame:
        Filters out users with fewer check-ins than the specified minimum.
    process_data(data: pd.DataFrame) -> pd.DataFrame:
        Applies all filtering steps to the input DataFrame.
    """

    def __init__(
        self,
        max_speed_kmph: float = 1200,  # km/h
        min_nb_checkins_per_location: int = 10,
        min_nb_checkins_per_user: int = 10,
        filter_invalid_speed: bool = True,
    ):
        """Initializes the Preprocessor with specified parameters.

        Parameters:
        -----------
        max_speed_kmph : float, optional
            Maximum allowed speed in km/h for users (default is 1200 km/h).
        min_nb_checkins_per_location : int, optional
            Minimum number of check-ins required per location (default is 10).
        min_nb_checkins_per_user : int, optional
            Minimum number of check-ins required per user (default is 10).
        """
        self.max_speed_kmph = max_speed_kmph
        self.min_nb_checkins_per_location = min_nb_checkins_per_location
        self.min_nb_checkins_per_user = min_nb_checkins_per_user
        self.filter_invalid_speed = filter_invalid_speed

    def _filter_max_speed_users(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filters out users who exceed the maximum allowed speed.

        Parameters:
        -----------
        data : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            DataFrame with users exceeding the maximum speed removed.
        """

        def geo_distance(lat1, lon1, lat2, lon2):
            """Calculates the great-circle distance between two points on the Earth's surface.

            Parameters:
            -----------
            lat1 : float
                Latitude of the first point in decimal degrees.
            lon1 : float
                Longitude of the first point in decimal degrees.
            lat2 : float
                Latitude of the second point in decimal degrees.
            lon2 : float
                Longitude of the second point in decimal degrees.

            Returns:
            --------
            float
                Distance between the two points in kilometers.
            """
            R = 6371  # km
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)
            a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi1) * np.cos(
                phi2
            ) * np.sin(delta_lambda / 2) * np.sin(delta_lambda / 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = R * c
            return d

        def time_delta(t1, t2):
            """Calculates the time difference between two timestamps in hours.

            Parameters:
            -----------
            t1 : pd.Timestamp
                First timestamp.
            t2 : pd.Timestamp
                Second timestamp.

            Returns:
            --------
            float
                Time difference in hours.
            """
            return (t2 - t1).total_seconds() / 3600  # hours

        origin_data_shape = data.shape
        origin_users = data["user_id"].nunique()
        # filter only one check in of user
        data = data[data.groupby("user_id").user_id.transform("size") > 1]
        more_then_one_checkin = data.shape
        more_then_one_users = data["user_id"].nunique()
        data = data.sort_values(by=["user_id", "local_time"], ascending=True)
        grouped = (
            data.groupby(by=["user_id"])
            .agg(
                latitudes=pd.NamedAgg("latitude", list),
                longitudes=pd.NamedAgg("longitude", list),
                times=pd.NamedAgg("local_time", list),
            )
            .reset_index()
        )

        def calculate_max_speed(row):
            speeds = [
                geo_distance(
                    row["latitudes"][i],
                    row["longitudes"][i],
                    row["latitudes"][i + 1],
                    row["longitudes"][i + 1],
                )
                / time_delta(row["times"][i], row["times"][i + 1])
                for i in range(len(row["latitudes"]) - 1)
            ]
            return max(speeds) if speeds else 0

        grouped["max_speed"] = grouped.parallel_apply(calculate_max_speed, axis=1)  # type: ignore
        valid_user = grouped[grouped["max_speed"] <= self.max_speed_kmph][
            "user_id"
        ].tolist()
        log.info(f"{len(valid_user)=} of {more_then_one_users=} in {origin_users=}")
        data = data[data["user_id"].isin(valid_user)]
        log.info(f"{origin_data_shape} -> {more_then_one_checkin} -> {data.shape}")
        return data

    def _filter_small_number_of_checkins_per_location(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters out locations with fewer check-ins than the specified minimum.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            DataFrame with locations having fewer check-ins removed.
        """
        location_count = df.groupby("location_id").size().reset_index(name="count")
        min_num = self.min_nb_checkins_per_location
        location_count = location_count[location_count["count"] >= min_num]
        df = df[df["location_id"].isin(location_count["location_id"])]
        return df

    def _filter_small_number_of_checkins_per_user(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filters out users with fewer check-ins than the specified minimum.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            DataFrame with users having fewer check-ins removed.
        """
        user_count = df.groupby("user_id").size().reset_index(name="count")
        min_num = self.min_nb_checkins_per_user
        user_count = user_count[user_count["count"] >= min_num]
        df = df[df["user_id"].isin(user_count["user_id"])]
        return df

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies all filtering steps to the input DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame after applying all filtering steps.
        """
        if not self.filter_invalid_speed:
            data = self._filter_max_speed_users(data)
        data = self._filter_small_number_of_checkins_per_location(data)
        data = self._filter_small_number_of_checkins_per_user(data)
        return data

class AlgorithmProcessor(ABC):
    """Abstract base class for algorithm-specific processors.

    Methods:
    --------
    process_data(data: pd.DataFrame) -> pd.DataFrame:
        Abstract method to be implemented by subclasses for processing data.
    """

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Processes the input DataFrame.

        Parameters:
        -----------
        data : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame.
        """
        pass

    def _base_process(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._build_trajectory_id(data)
        data = self._split_data(data)
        data = self._filter_min_trajectory_length(data)
        traj = data.groupby(by=["flag"], sort=False)["trajectory_id"].nunique()
        ci = data.groupby(by=["flag"], sort=False)["trajectory_id"].size()
        ele = data[["user_id", "category_name", "location_id"]].nunique()
        stat = pd.concat([traj, ci, ele]).to_frame().T
        stat.columns = [
            "#train_trajectory",
            "#validation_trajectory",
            "#test_trajectory",
            "#train_checkin",
            "#validation_checkin",
            "#test_checkin",
            "#user",
            "#category",
            "#location",
        ]
        log.info(f"\n{stat}")
        return data

    def _build_trajectory_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Builds trajectory IDs based on user ID and time differences.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added trajectory IDs.
        """
        df = df.sort_values(by=["user_id", "local_time"])
        df = df.reset_index(drop=True)
        df["time_diff"] = df["local_time"].diff()
        if df["user_id"].dtypes != "int64":
            log.warning("user_id is not int64")
            df["user_id"] = df["user_id"].astype("category").cat.codes
        df["diff_user"] = df["user_id"].diff().fillna(0) != 0  # type: ignore
        time_threshold = self.max_time_gap_in_one_trajectory_hours  # type: ignore
        df["is_long_time"] = df["time_diff"] > pd.Timedelta(time_threshold, unit="h")  # type: ignore
        df["trajectory_flag"] = df["is_long_time"] | df["diff_user"]
        df["trajectory_id"] = df["trajectory_flag"].cumsum()
        del df["time_diff"]
        del df["diff_user"]
        del df["is_long_time"]
        del df["trajectory_flag"]
        return df

    def _split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Splits the data into train, validation, and test sets based on global time ranges.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added flags indicating train, validation, or test set.
        """
        train_ratio = self.split_ratio_list[0]  # type: ignore
        validation_ratio = self.split_ratio_list[1]  # type: ignore
        df = df.sort_values(by=["local_time"])
        df = df.reset_index(drop=True)
        min_time = df["local_time"].min()
        max_time = df["local_time"].max()
        train_end = min_time + (max_time - min_time) * train_ratio
        validation_end = min_time + (max_time - min_time) * (
            train_ratio + validation_ratio
        )
        log.info(f"{min_time=}, {train_end=}, {validation_end=}, {max_time=}")
        df.loc[:, "flag"] = "validation"
        df.loc[df["local_time"] <= train_end, "flag"] = "train"
        df.loc[df["local_time"] > validation_end, "flag"] = "test"
        return df

    def _filter_min_trajectory_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out trajectories shorter than the specified minimum length.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing user check-in data.

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame with only trajectories meeting the minimum length requirement.
        """
        if self.iterative_filter_sparse_poi_user:  # type: ignore

            while True:
                # remove unseen locations and users
                train_locations = df[df["flag"] == "train"].location_id.unique()
                train_users = df[df["flag"] == "train"].user_id.unique()
                df = df[
                    df["location_id"].isin(train_locations)
                    & df["user_id"].isin(train_users)
                ]
                df = (
                    df.groupby(by=["flag", "trajectory_id"])
                    .filter(lambda x: len(x) > self.min_trajectory_length)  # type: ignore
                    .reset_index(drop=True)
                )
                before_max_trajectory_length_shape = df.shape
                df = (
                    df.groupby(by=["flag", "trajectory_id"])
                    .filter(lambda x: len(x) <= self.max_trajectory_length)  # type: ignore
                    .reset_index(drop=True)
                )
                after_max_trajectory_length_shape = df.shape
                log.info(
                    f"In While loop: {before_max_trajectory_length_shape=} {after_max_trajectory_length_shape=}"
                )

                filtered_train_locations = df[
                    df["flag"] == "train"
                ].location_id.unique()
                filtered_users = df[df["flag"] == "train"].user_id.unique()

                if len(train_locations) == len(filtered_train_locations) and len(
                    train_users
                ) == len(filtered_users):
                    log.info("No unseen locations or users")
                    break
                else:
                    num_droped_locations = len(train_locations) - len(
                        filtered_train_locations
                    )
                    num_droped_users = len(train_users) - len(filtered_users)
                    log.warning(
                        f"Some ({num_droped_locations=}) locations or ({num_droped_users=}) users in train data are removed while filter min trajectory length"
                    )
        else:
            df = (
                df.groupby(by=["flag", "trajectory_id"])
                .filter(lambda x: len(x) > self.min_trajectory_length)  # type: ignore
                .reset_index(drop=True)
            )
            before_max_trajectory_length_shape = df.shape
            df = (
                df.groupby(by=["flag", "trajectory_id"])
                .filter(lambda x: len(x) <= self.max_trajectory_length)  # type: ignore
                .reset_index(drop=True)
            )
            after_max_trajectory_length_shape = df.shape
            log.info(
                f"{before_max_trajectory_length_shape=} {after_max_trajectory_length_shape=}"
            )

        return df

class FileLoader:
    """A class to load, preprocess, and save processed data for location-based services.

    Parameters:
    -----------
    name : str, optional
        Name of the dataset (default is "foursquare_tsmc2014").
    state : str, optional
        State or region of the dataset (default is "NYC").
    raw_filepath : str, optional
        Filepath to the raw data file (default is an empty string).
    processed_filepath : str, optional
        Filepath to save the processed data file (default is an empty string).
    preprocessor : Preprocessor, optional
        Preprocessor instance for initial data cleaning (default is None).
    algorithm_preprocessor : AlgorithmProcessor, optional
        Algorithm-specific preprocessor instance for further processing (default is None).

    Methods:
    --------
    get_params(obj) -> str:
        Retrieves parameters of an object as a string.
    read_raw_data() -> pd.DataFrame:
        Reads raw data from a file.
    save_dataframe2file(df: pd.DataFrame) -> None:
        Saves a DataFrame to a file.
    preprocess_data() -> pd.DataFrame:
        Preprocesses the data using the provided preprocessors.
    get_data() -> tuple:
        Retrieves processed data, either from cache or by preprocessing.
    """

    def __init__(
        self,
        name: str = "foursquare_tsmc2014",
        state: str = "NYC",
        raw_filepath: str = "",
        processed_filepath: str = "",
        weights_only: bool = True,
        preprocessor: Preprocessor | None = None,
        algorithm_preprocessor: AlgorithmProcessor | None = None,
    ):
        self.name = name
        self.state = state
        self.raw_filepath = raw_filepath
        file_loader_params = self.get_params(self)
        preprocessor_params = self.get_params(preprocessor)
        algorithm_preprocessor_params = self.get_params(algorithm_preprocessor)
        params = f"{file_loader_params}_{preprocessor_params}_{algorithm_preprocessor_params}"
        log.info(f"FileLoader params: {params}")
        self.hash_suffix = hashlib.sha256(params.encode()).hexdigest()
        algorithm_name = algorithm_preprocessor.__class__.__name__.lower()
        pathlib.Path(processed_filepath).mkdir(parents=True, exist_ok=True)
        self.processed_filepath = (
            f"{processed_filepath}.{algorithm_name}.{self.hash_suffix}.gzip"
        )
        self.preprocessor = preprocessor
        self.algorithm_preprocessor = algorithm_preprocessor
        self.weights_only = weights_only

    @staticmethod
    def get_params(obj) -> str:
        """Retrieves parameters of an object as a string.

        Parameters:
        -----------
        obj : object
            Object whose parameters are to be retrieved.

        Returns:
        --------
        str
            String representation of the object's parameters.
        """
        if obj is None:
            return "None"
        params = [f"{k}={v}" for k, v in obj.__dict__.items() if not k.startswith("_")]
        return "_".join(params)

    def read_raw_data(self) -> pd.DataFrame:
        """Reads raw data from a file.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the raw data.

        Raises:
        -------
        FileNotFoundError
            If the file is not found.
        Exception
            If an error occurs while reading the file.
        """
        try:
            raw_data = pd.read_parquet(self.raw_filepath)
            return raw_data
        except FileNotFoundError:
            log.error(f"File not found: {self.raw_filepath}")
            raise
        except Exception as e:
            log.error(f"Error reading file: {e}")
            raise

    @contextmanager
    def _file_lock(self, path, mode="w+b", timeout=30):
        """
        Context manager for file locking.

        Read-only mode:
        - Simply opens and yields the file handle (no lock needed).

        Write mode:
        - Tries to create an exclusive lock file atomically.
        - If successful, yields True and ensures lock is cleaned up in finally block.
        - If lock file exists, waits up to `timeout` seconds for it to be released.
        - If timeout occurs or lock is stale (older than 5 min), removes it and tries again.
        - Yields False if another process holds a fresh lock.
        """
        lock_file_path = f"{path}.lock"

        # Read-only: no locking, just open and yield file handle
        if ("r" in mode) and ("w" not in mode and "+" not in mode):
            file_handle = None
            try:
                file_handle = open(path, mode)
                yield file_handle
            finally:
                if file_handle:
                    try:
                        file_handle.close()
                    except Exception:
                        pass
            return

        # Write-mode: attempt to acquire lock file atomically
        lock_acquired = False
        start_time = time.time()

        def is_lock_stale(lock_path, max_age_seconds=300):
            """Check if lock file is older than max_age (default 5 min)"""
            try:
                if os.path.exists(lock_path):
                    mtime = os.path.getmtime(lock_path)
                    age = time.time() - mtime
                    return age > max_age_seconds
            except Exception:
                pass
            return False

        try:
            # Try to acquire the lock by creating the lock file atomically
            while True:
                try:
                    fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    lock_acquired = True
                    break
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        # Lock file exists; check if it's stale
                        if is_lock_stale(lock_file_path):
                            log.debug(f"Stale lock file detected for {path}, removing...")
                            try:
                                os.remove(lock_file_path)
                            except Exception:
                                pass
                            continue  # Try to acquire lock again

                        # Lock file exists and is fresh; wait for it to be released
                        log.debug(f"Lock file exists for {path}, waiting for release...")
                        while os.path.exists(lock_file_path) and not is_lock_stale(lock_file_path):
                            if time.time() - start_time > timeout:
                                log.debug(
                                    f"Lock wait timeout ({timeout}s) for {path}, yielding False."
                                )
                                # Timeout: don't acquire lock, let caller retry
                                yield False
                                return
                            time.sleep(0.1)  # Check every 100ms

                        # Lock was released or became stale; try to acquire
                        if is_lock_stale(lock_file_path):
                            log.debug(f"Lock file became stale, removing...")
                            try:
                                os.remove(lock_file_path)
                            except Exception:
                                pass
                        continue
                    else:
                        raise

            # Successfully acquired the lock
            yield True

        finally:
            # Always clean up the lock file if we acquired it
            if lock_acquired:
                try:
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except Exception as e:
                    log.debug(f"Lock file cleanup warning: {e}")

    def _safe_write_file(self, write_func, target_path, *args, **kwargs):
        """Safely write file with atomic operations (write to temp, then rename).

        Ensures data is synced to disk before rename to prevent file not found errors.

        Parameters:
        -----------
        write_func : callable
            Function that performs the actual write operation
        target_path : str
            Target file path to write to
        *args, **kwargs :
            Arguments to pass to write_func
        """
        temp_path = None
        try:
            # Create temp file for atomic write
            temp_dir = pathlib.Path(target_path).parent
            temp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                dir=temp_dir, prefix=".", suffix=".tmp", delete=False
            ) as tmp_file:
                temp_path = tmp_file.name

            # Execute write operation to temp file
            write_func(temp_path, *args, **kwargs)

            # Ensure data is synced to disk
            try:
                with open(temp_path, "rb") as f:
                    os.fsync(f.fileno())
            except Exception as sync_err:
                log.debug(f"fsync warning: {sync_err}")

            # Atomically rename temp file to target (no lock needed for atomic rename)
            os.replace(temp_path, target_path)
            log.info(f"Written to: {target_path}")

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    log.debug(f"Failed to clean up temp file {temp_path}: {cleanup_err}")
            log.error(f"Error while writing: {e}")
            raise

    def save_dataframe2file(self, df: pd.DataFrame) -> None:
        """Saves a DataFrame to a file.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to be saved.

        Raises:
        -------
        Exception
            If an error occurs while saving the file.
        """
        try:
            output_dir = pathlib.Path(self.processed_filepath).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.processed_filepath, compression="gzip")
            log.info(f"Processed data {df.shape} saved to {self.processed_filepath}")
        except Exception as e:
            log.error(f"Error saving file: {e}")
            raise ValueError(f"Error saving file: {e}")

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocesses the data using the provided preprocessors.

        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame.

        Raises:
        -------
        Exception
            If an error occurs during preprocessing.
        """
        try:
            data = self.read_raw_data()
            if self.preprocessor is not None:
                data = self.preprocessor.process_data(data)
            if self.algorithm_preprocessor is not None:
                data = self.algorithm_preprocessor.process_data(data)
            self.save_dataframe2file(data)
            return data
        except Exception as e:
            log.error(f"Error during preprocessing: {e}")
            raise

    def get_data(self) -> tuple | None:
        """Retrieves processed data, either from cache or by preprocessing with multi-process
        coordination.

        For multi-process coordination:
        - If another process is preprocessing, wait for the cache file to appear (indefinite wait).
        - If this process acquires the lock, preprocess and save cache.

        Returns:
        --------
        tuple
            Tuple containing processed data elements.
        """
        if self.algorithm_preprocessor is None:
            log.warning("Algorithm preprocessor is None, returning None.")
            return None

        cache_file = pathlib.Path(self.processed_filepath).with_suffix(".cache")
        cache_path = str(cache_file)

        # First check if cache already exists (fast path)
        if cache_file.exists():
            try:
                log.info(f"Try load from: {cache_file}")
                with self._file_lock(cache_path, "rb", timeout=15) as f:
                    cache = torch.load(
                        f, weights_only=self.weights_only
                    )  # nosec B614
                log.info(f"Successfully loaded: {cache_file}")
                return cache
            except Exception as e:
                log.warning(
                    f"Error loading existing cache: {e}, will attempt preprocessing"
                )
                # Continue to preprocessing

        # Cache doesn't exist, try to acquire preprocessing lock
        log.debug(f"Cache file does not exist: {cache_file}")
        preprocess_lock_path = cache_path + ".preprocess"

        with self._file_lock(preprocess_lock_path, "w", timeout=30) as lock_acquired:
            # If we didn't acquire the lock, another process is preprocessing
            if lock_acquired is False:
                log.info(
                    "Another process is preprocessing, waiting for cache to appear..."
                )

                # Wait indefinitely for cache file to appear (another process is creating it)
                cache_wait_timeout = 7200  # 1 hour timeout
                cache_wait_start = time.time()
                while not cache_file.exists():
                    if time.time() - cache_wait_start > cache_wait_timeout:
                        raise TimeoutError(
                            f"Cache file did not appear within {cache_wait_timeout}s. "
                            f"Another process may have failed to create it."
                        )
                    time.sleep(1)  # Check every second

                # Cache file appeared, try to load it
                try:
                    log.info(f"Cache file appeared, loading from: {cache_file}")
                    with self._file_lock(cache_path, "rb", timeout=15) as f:
                        cache = torch.load(
                            f, weights_only=self.weights_only
                        )  # nosec B614
                    log.info(f"Successfully loaded cache from other process: {cache_file}")
                    return cache
                except Exception as e:
                    log.error(f"Failed to load cache file after waiting: {e}")
                    raise

            # We acquired the lock - double-check cache file in case another process
            # created it between the initial check and lock acquisition
            if cache_file.exists():
                try:
                    log.info("Cache was created by another process, loading...")
                    with open(cache_path, "rb") as f:
                        cache = torch.load(
                            f, weights_only=self.weights_only
                        )  # nosec B614
                    log.info(f"Successfully loaded: {cache_file}")
                    return cache
                except Exception as e:
                    log.warning(f"Failed to load cache file: {e}, will re-create")

            # Preprocess the data
            try:
                log.info("Starting data preprocessing...")
                self.preprocess_data()
                data = self.algorithm_preprocessor.to_cache()  # type: ignore

                # Save cache with atomic write
                log.info(f"Saving cache file to: {cache_path}...")
                def _write_cache(path, cache_data):
                    with open(path, "wb") as f:
                        torch.save(cache_data, f)

                self._safe_write_file(_write_cache, cache_path, data)

                # Verify by loading - retry if file not immediately available
                log.info("Verifying cached data...")
                max_verify_retries = 3
                for verify_attempt in range(max_verify_retries):
                    try:
                        if not cache_file.exists():
                            if verify_attempt < max_verify_retries - 1:
                                log.debug(f"Cache file not found yet, waiting... (attempt {verify_attempt + 1}/{max_verify_retries})")
                                time.sleep(0.5)
                                continue
                            else:
                                raise FileNotFoundError(f"Cache file not found at {cache_path}")

                        with open(cache_path, "rb") as f:
                            cache = torch.load(
                                f, weights_only=self.weights_only
                            )  # nosec B614
                        log.info(f"Cache successfully saved and verified: {cache_path}")
                        return cache
                    except (FileNotFoundError, IOError) as e:
                        if verify_attempt < max_verify_retries - 1:
                            log.debug(f"Verification attempt {verify_attempt + 1} failed: {e}, retrying...")
                            time.sleep(0.5)
                        else:
                            raise

            except Exception as e:
                log.error(f"Error during preprocessing: {e}")
                raise