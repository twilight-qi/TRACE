import ast
import datetime
import json
import logging
import pathlib
from copy import deepcopy

import numpy as np
import pandas as pd
import shapely
from pandarallel import pandarallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log", mode="w"),
        logging.StreamHandler(),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Prepare the location description texts
def load_gowalla_location_descriptions(
    location_filepath: str, city_polygons: dict
) -> pd.DataFrame:
    poi_dict = dict(filepath_or_buffer=location_filepath)
    poi_data = pd.read_csv(**poi_dict)  # type:ignore
    poi_data["category_name"] = poi_data["spot_categories"].map(
        lambda x: ast.literal_eval(x)[0]["name"]
    )
    location = poi_data[
        [
            "id",
            "lat",
            "lng",
            "category_name",
        ]
    ]
    location.columns = [
        "location_id",
        "latitude",
        "longitude",
        "category_name",
    ]
    location = location.copy()
    location.loc[:, "city_name"] = "Unknown"
    pandarallel.initialize(progress_bar=False, nb_workers=128)
    location.loc[:, "city_name"] = location.parallel_apply(
        lambda x: gps2city(x["longitude"], x["latitude"], city_polygons), axis=1
    )
    location = location[["location_id", "city_name", "category_name"]]
    return location


def build_city_polygon_map(polygon_json: dict) -> dict:
    polygons: dict = {}
    json_list: list = polygon_json["features"]
    for city in json_list:
        city_name: str = city["properties"]["name"]
        polygon_coordinates = city["geometry"]["coordinates"][0]
        polygon_coordinates = np.array(polygon_coordinates)
        polygon_coordinates = np.squeeze(polygon_coordinates)
        if polygon_coordinates.shape[1] != 2:
            raise ValueError(f"{city_name} has {polygon_coordinates.shape=}")
        if polygon_coordinates.ndim != 2:
            raise ValueError(f"{city_name} has {polygon_coordinates.ndim} dimensions")
        polygon_shape = shapely.polygons(polygon_coordinates)
        assert isinstance(polygon_shape, shapely.Polygon)
        polygons[city_name] = polygon_shape
        log.info(f"{city_name} Polygons generated.")
    return polygons


def gps2city(longitude: float, latitude: float, city_polygons: dict) -> str:
    point: shapely.geometry.Point = shapely.geometry.Point(longitude, latitude)
    for city, polygon in city_polygons.items():
        if polygon.contains(point):
            return city
    return "Unknown"


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Start dropna")
    df_shape = df.shape
    df = df.dropna(axis=0, how="any")
    log.info(f"{df_shape} -> {df.shape}")
    return df


def add_local_time(
    data: pd.DataFrame,
    datetime_column: str = "utc_time",
    timestamp_format: str = "%a %b %d %H:%M:%S %z %Y",
    timezone_column: str | None = "offset",
    local_time_column: str = "local_time",
) -> pd.DataFrame:
    log.info("Start add_local_time")
    if local_time_column in data.columns:
        log.info(f"{local_time_column} Existed")
        return data
    data = data.copy()
    data[datetime_column] = pd.to_datetime(
        data[datetime_column], utc=True, format=timestamp_format, errors="coerce"
    )
    data_shape = data.shape
    data = data.dropna(subset=[datetime_column])
    log.info(f"{data_shape} -> {data.shape}")
    if timezone_column is not None:
        data[local_time_column] = data[datetime_column] + data[timezone_column].map(
            lambda x: pd.Timedelta(minutes=x)
        )
    return data


def drop_duplicates(df: pd.DataFrame, subset=["user_id", "local_time"]) -> pd.DataFrame:
    log.info("Start drop_duplicates")
    df_shape = df.shape
    df = df.drop_duplicates(subset=subset, keep="first")
    log.info(f"{df_shape} -> {df.shape}")
    return df


def mean_location_gps(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Start mean_location_gps")
    poi_data = (
        df.groupby(by=["location_id", "latitude", "longitude"]).size().reset_index(name="count")
    )
    poi_gps = (
        poi_data.groupby(by=["location_id"])
        .agg({"latitude": "mean", "longitude": "mean"})
        .reset_index()
    )
    del df["latitude"]
    del df["longitude"]
    df_shape = df.shape
    df = pd.merge(df, poi_gps, on="location_id")
    log.info(f"{df_shape} -> {df.shape}")
    return df


def most_frequent_category_name_of_location(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Start most_frequent_category_name_of_location")
    poi_data = df.groupby(by=["location_id", "category_name"]).size().reset_index(name="count")
    if poi_data["location_id"].nunique() == len(poi_data):
        log.info("location_id and category_name are one-to-one, return directly")
        return df
    poi_data = poi_data.sort_values(by=["location_id", "count"], ascending=False)
    poi_category = (
        poi_data.groupby(by=["location_id"]).agg({"category_name": "first"}).reset_index()
    )
    original_len = df.shape
    df = df.drop(columns=["category_name"])
    df = pd.merge(df, poi_category, on="location_id")
    log.info(f"{original_len} -> {df.shape}")
    return df


def load_foursquare_tsmc2014(base_dir: str) -> list:
    log.info("Start load Foursquare TSMC2014 dataset")
    version = "dataset_tsmc2014/{city}./dataset_TSMC2014_{city}.txt"
    city_list = ["NYC", "TKY"]
    POI_names = [
        "user_id",
        "location_id",
        "location_category_id",
        "category_name",
        "latitude",
        "longitude",
        "offset",
        "utc_time",
    ]
    check_ins_list = []
    for city in city_list:
        data_dict = dict(
            filepath_or_buffer=base_dir + version.format(city=city),
            sep="\t",
            names=POI_names,
            encoding="ISO-8859-1",
        )
        city_check_ins = pd.read_csv(**data_dict)  # type:ignore
        original_len = len(city_check_ins)
        log.info(f"{city}: {original_len}")
        check_ins_list.append(city_check_ins)
    return check_ins_list


def load_foursquare_tist2015(base_dir: str, city_polygons: dict) -> pd.DataFrame:
    log.info("Start load Foursquare TIST2015 dataset")
    POI_names = [
        "location_id",
        "latitude",
        "longitude",
        "category_name",
        "country_abbr",
    ]
    poi_dict = dict(
        filepath_or_buffer=base_dir + "dataset_TIST2015./dataset_TIST2015_POIs.txt",
        sep="\t",
        names=POI_names,
        encoding="UTF-8",
    )
    poi_data = pd.read_csv(**poi_dict)  # type:ignore
    pandarallel.initialize(progress_bar=False, nb_workers=128)
    poi_data.loc[:, "city_name"] = poi_data.parallel_apply(
        lambda x: gps2city(x["longitude"], x["latitude"], city_polygons), axis=1
    )

    check_ins_names = ["user_id", "location_id", "utc_time", "offset"]
    check_ins_dict = dict(
        filepath_or_buffer=base_dir + "dataset_TIST2015./dataset_TIST2015_Checkins.txt",
        sep="\t",
        names=check_ins_names,
        encoding="UTF-8",
    )
    check_ins_data = pd.read_csv(**check_ins_dict)  # type:ignore
    raw_shape = check_ins_data.shape
    check_ins_data = pd.merge(check_ins_data, poi_data, on="location_id")
    log.info(f"TIST2015: {raw_shape} -> {check_ins_data.shape}")
    return check_ins_data


def load_foursquare_www2019(base_dir: str, city_polygons: dict) -> pd.DataFrame:
    log.info("Start load Foursquare WWW2019 dataset")
    POI_names = [
        "location_id",
        "latitude",
        "longitude",
        "category_name",
        "country_abbr",
    ]
    poi_dict = dict(
        filepath_or_buffer=base_dir + "dataset_WWW2019/raw_POIs.txt",
        sep="\t",
        names=POI_names,
        encoding="UTF-8",
    )
    poi_data = pd.read_csv(**poi_dict)  # type:ignore
    pandarallel.initialize(progress_bar=False, nb_workers=128)
    poi_data.loc[:, "city_name"] = poi_data.parallel_apply(
        lambda x: gps2city(x["longitude"], x["latitude"], city_polygons), axis=1
    )

    check_ins_names = ["user_id", "location_id", "utc_time", "offset"]
    check_ins_dict = dict(
        filepath_or_buffer=base_dir + "dataset_WWW2019/raw_Checkins_anonymized.txt",
        sep="\t",
        names=check_ins_names,
        encoding="UTF-8",
    )
    check_ins_data = pd.read_csv(**check_ins_dict)  # type:ignore
    raw_shape = check_ins_data.shape
    check_ins_data = pd.merge(check_ins_data, poi_data, on="location_id", how="inner")
    log.info(f"WWW2019: {raw_shape} -> {check_ins_data.shape}")
    return check_ins_data


def load_gowalla(base_dir: str, city_polygons: dict) -> pd.DataFrame:
    log.info("Start loading Gowalla dataset")
    data_dict = dict(
        filepath_or_buffer=base_dir + "Gowalla_totalCheckins.txt",
        sep="\t",
        names=["user_id", "local_time", "latitude", "longitude", "location_id"],
        parse_dates=["local_time"],
        encoding="UTF-8",
    )
    check_ins_data = pd.read_csv(**data_dict)  # type:ignore
    log.info(f"Gowalla: {check_ins_data.shape}")
    location_filepath = f"{base_dir}/gowalla_spots_subset1.csv"
    location = load_gowalla_location_descriptions(location_filepath, city_polygons)
    check_ins_data = pd.merge(check_ins_data, location, on="location_id", how="left")
    # mismatch on location ids
    mismatch = set(check_ins_data.location_id.unique()) - set(location.location_id.unique())
    log.info(f"mismatch: {len(mismatch)=}")
    columns = check_ins_data.columns.tolist()
    log.info(f"Gowalla {columns=}")
    return check_ins_data


def get_topk_cities(df: pd.DataFrame, k: int = 10) -> list:
    city_count = df.groupby("city_name").size().reset_index(name="count")
    city_count = city_count.sort_values(by="count", ascending=False)
    topk_city_count = city_count.head(k)
    city_list = topk_city_count["city_name"].tolist()
    if "Unknown" in city_list:
        topk_city_count = city_count.head(k + 1)
        city_list = topk_city_count["city_name"].tolist()
        city_list.remove("Unknown")
    log.info(f"Top {k} cities: {city_list}")
    return city_list

def rename2abbr():
    state2abbr = {
        "Texas": "tx",
        "California": "ca",
        "Florida": "fl",
        "New_York": "ny",
        "Washington": "wa",
        "Illinois": "il",
        "Oklahoma": "ok",
        "Georgia": "ga",
        "Ohio": "oh",
        "Pennsylvania": "pa",
        "New_Jersey": "nj",
        "Michigan": "mi",
        "TKY": "tky",
        "NYC": "nyc",
    }
    check_ins_data_dir = "./dataset/check_ins/basic_preprocessed/"
    gzip_files = list(pathlib.Path(check_ins_data_dir).glob("*.gzip"))
    for gzip_file in gzip_files:
        for state in state2abbr.keys():
            if state in gzip_file.name:
                new_file_name = gzip_file.name.replace(state, state2abbr[state])
                log.info(gzip_file.name, "->", new_file_name)
                gzip_file.rename(gzip_file.parent / new_file_name)

def main():
    base_dir_foursquare = "./dataset/foursquare/"
    base_dir_gowalla = "./dataset/gowalla/"
    polygons_filepath = "./dataset/us_state_polygon_json.json"
    cache_dir = "./dataset/cache/"
    check_ins_data_dir = "./dataset/check_ins/basic_preprocessed/"

    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(check_ins_data_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 100)
    log.info("Loading polygons data")
    with open(polygons_filepath) as f:
        us_state_polygons = json.load(f)
    city_polygons = build_city_polygon_map(us_state_polygons)

    log.info("=" * 100)
    log.info("Loading datasets")
    check_ins_list = []
    check_ins_list.extend(load_foursquare_tsmc2014(base_dir_foursquare))
    check_ins_list.append(load_foursquare_tist2015(base_dir_foursquare, city_polygons))
    check_ins_list.append(load_foursquare_www2019(base_dir_foursquare, city_polygons))
    check_ins_list.append(load_gowalla(base_dir_gowalla, city_polygons))

    log.info("=" * 100)
    log.info("Starting basic preprocess")
    basic_preprocess_pipeline = [
        dropna,
        add_local_time,
        drop_duplicates,
        mean_location_gps,
        most_frequent_category_name_of_location,
    ]
    descriptions = [
        "DropNa",
        "Add Local Time",
        "Drop Duplicates",
        "Mean Location GPS",
        "Most Frequent Category Name Of Location",
    ]

    basic_preprocessed_list = []

    for i, check_ins_data in enumerate(check_ins_list):
        original_shape = check_ins_data.shape
        data = deepcopy(check_ins_data)
        for func, desc in zip(basic_preprocess_pipeline, descriptions):
            data = func(data)
            columns = data.columns.tolist()
            log.info(f"[{i+1}] After {desc} {columns=} {data.shape}")

        basic_preprocessed_list.append(data)  # type: ignore
        log.info(f"[{i+1}] Check-in data: {original_shape} -> {data.shape}")
        log.info(f"[{i+1}] Preprocessing finished")

    log.info("=" * 100)
    log.info("Start saving basic preprocessed data to parquet file")
    names_list = [
        ("foursquare_tsmc2014", "NYC"),
        ("foursquare_tsmc2014", "TKY"),
        ("foursquare_tist2015",),
        ("foursquare_www2019",),
        ("gowalla",),
    ]

    for names, data in zip(names_list, basic_preprocessed_list):
        log.info(f"{'-'.join(names):-^100}")
        columns = data.columns.tolist()
        log.info(f"{columns=}")
        if "city_name" in columns:
            top10_cities = get_topk_cities(data)
            log.info(f"Top 10 cities: {top10_cities}")
            if names[0] == "gowalla":
                original_len = len(data)
                original_start = data["local_time"].min()
                original_stop = data["local_time"].max()
                data = data.loc[
                    data["local_time"]
                    >= datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc)
                ]
                current_len = len(data)
                current_start = data["local_time"].min()
                current_stop = data["local_time"].max()
                log.info(
                    f"Gowalla Time Range: "
                    f"{original_start} ~ {original_stop} "
                    f"({original_len} -> {current_len}), "
                    f"Current: {current_start} ~ {current_stop}"
                )
            else:
                for state in top10_cities:
                    d = data[data.city_name == state]
                    d = d.rename(columns={"city_name": "state"})
                    d = d.reset_index(drop=True)
                    name = names[0]
                    state_name = state.replace(" ", "_").replace("/", "_")
                    fp = f"{check_ins_data_dir}{name}_{state_name}.gzip"
                    d.to_parquet(fp, compression="gzip")
                    log.info(f"Save {d.shape} checkins to {fp}")
        else:
            data = data.reset_index(drop=True)
            name = names[0]
            state = names[1]
            fp = f"{check_ins_data_dir}{name}_{state}.gzip"
            data.to_parquet(fp, compression="gzip")
            log.info(f"Save {data.shape} checkins to {fp}")
        log.info(f"{'-'.join(names):-^100}")
    log.info("=" * 100)
    rename2abbr()
    log.info("Basic preprocessed Done.")


if __name__ == "__main__":
    main()
