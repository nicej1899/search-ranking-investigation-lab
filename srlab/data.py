"""Data preparation for MovieLens ml-latest-small.

Terminology: MovieLens rows are *rating interactions* (a member rated a title at a time).
They are not watch logs, search sessions, exposures or satisfaction labels. A title a member
has not rated is unobserved, not disliked.

Outputs (data/processed/):
  events.csv      event_id, member, item, ts, rating, split   (event_id stable: sorted by ts, userId, movieId)
  items.csv       item, movieId, title, year, genres, label, tmdbId, imdbId, runtime_min (always empty: no permitted source)
  split.json      cutoffs, counts, fingerprints
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common import PROC, RAW, dump_json, sha256_file

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western", "(no genres listed)"]

# Coarse single label per title used ONLY for motif node labels and UI colour. Priority order is a
# documented, deterministic rule over MovieLens genres (no external metadata).
LABEL_RULES = [
    ("Crime/Thriller", {"Crime", "Thriller", "Film-Noir"}),
    ("Horror", {"Horror"}),
    ("Mystery", {"Mystery"}),
    ("Kids/Animation", {"Children", "Animation"}),
    ("Comedy", {"Comedy"}),
    ("SciFi/Fantasy", {"Sci-Fi", "Fantasy"}),
    ("Action/Adventure", {"Action", "Adventure", "War", "Western"}),
    ("Romance", {"Romance"}),
    ("Documentary", {"Documentary"}),
    ("Drama", {"Drama"}),
]

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def coarse_label(genres: list[str]) -> str:
    g = set(genres)
    # A Mystery title that is also a Comedy and not Crime/Thriller is labelled Mystery (cozy-style).
    for name, keys in LABEL_RULES:
        if g & keys:
            return name
    return "Other"


def prepare() -> dict:
    ratings = pd.read_csv(RAW / "ratings.csv")
    movies = pd.read_csv(RAW / "movies.csv")
    links = pd.read_csv(RAW / "links.csv")

    ratings = ratings.sort_values(["timestamp", "userId", "movieId"], kind="mergesort").reset_index(drop=True)
    # stable integer ids
    members = np.sort(ratings.userId.unique())
    mid_sorted = np.sort(movies.movieId.unique())
    member_index = {int(u): i for i, u in enumerate(members)}
    item_index = {int(m): i for i, m in enumerate(mid_sorted)}

    ev = pd.DataFrame({
        "event_id": np.arange(len(ratings), dtype=np.int64),
        "member": ratings.userId.map(member_index).astype(np.int64),
        "user_id": ratings.userId.astype(np.int64),
        "item": ratings.movieId.map(item_index).astype(np.int64),
        "movie_id": ratings.movieId.astype(np.int64),
        "ts": ratings.timestamp.astype(np.int64),
        "rating": ratings.rating.astype(np.float32),
    })
    n = len(ev)
    t1 = int(ev.ts.iloc[int(n * TRAIN_FRAC)])
    t2 = int(ev.ts.iloc[int(n * (TRAIN_FRAC + VAL_FRAC))])
    ev["split"] = np.where(ev.ts < t1, "train", np.where(ev.ts < t2, "val", "test"))

    mv = movies.merge(links, on="movieId", how="left")
    mv["item"] = mv.movieId.map(item_index)
    mv = mv.sort_values("item").reset_index(drop=True)
    yr = mv.title.str.extract(r"\((\d{4})\)\s*$")[0]
    mv["year"] = pd.to_numeric(yr, errors="coerce")
    mv["clean_title"] = mv.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    mv["genre_list"] = mv.genres.str.split("|")
    mv["label"] = mv.genre_list.map(coarse_label)
    mv["runtime_min"] = np.nan  # no permitted runtime source available in this build (see docs/PROVENANCE.md)
    items = mv[["item", "movieId", "title", "clean_title", "year", "genres", "label", "tmdbId", "imdbId", "runtime_min"]]

    train_items = set(ev.loc[ev.split == "train", "item"].unique())
    items = items.assign(seen_in_train=items.item.isin(train_items))

    ev.to_csv(PROC / "events.csv", index=False)
    items.to_csv(PROC / "items.csv", index=False)

    info = {
        "source": "MovieLens ml-latest-small (generated 2018-09-26)",
        "raw_sha256": {f: sha256_file(RAW / f, 64) for f in ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]},
        "n_events": n, "n_members": len(members), "n_items": len(mid_sorted),
        "cutoffs": {"train_end_ts": t1, "val_end_ts": t2,
                    "train_end_utc": pd.to_datetime(t1, unit="s").isoformat(),
                    "val_end_utc": pd.to_datetime(t2, unit="s").isoformat()},
        "split_rule": "global chronological: train ts < train_end_ts (70% of events); val < val_end_ts (next 15%); test = rest",
        "counts": ev.split.value_counts().to_dict(),
        "members_per_split": ev.groupby("split").member.nunique().to_dict(),
        "items_seen_in_train": len(train_items),
        "cold_start": "Members are represented only through their history (no member id embedding), so new members are handled inductively. "
                      "Items unseen before train_end_ts share one learned UNK id embedding plus their genre features.",
        "events_sha": sha256_file(PROC / "events.csv"),
        "items_sha": sha256_file(PROC / "items.csv"),
    }
    dump_json(info, PROC / "split.json")
    return info


def load_events() -> pd.DataFrame:
    return pd.read_csv(PROC / "events.csv")


def load_items() -> pd.DataFrame:
    df = pd.read_csv(PROC / "items.csv")
    df["genre_list"] = df.genres.str.split("|")
    return df


def genre_matrix(items: pd.DataFrame) -> np.ndarray:
    m = np.zeros((len(items), len(GENRES)), dtype=np.float32)
    gi = {g: i for i, g in enumerate(GENRES)}
    for r, gl in enumerate(items.genre_list):
        for g in gl:
            if g in gi:
                m[r, gi[g]] = 1.0
    return m


def load_tags() -> pd.DataFrame:
    t = pd.read_csv(RAW / "tags.csv")
    t["tag_norm"] = t.tag.astype(str).str.lower().str.strip()
    return t


if __name__ == "__main__":
    print(prepare())
