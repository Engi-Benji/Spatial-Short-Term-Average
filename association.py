from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Number = int | float | np.integer | np.floating
type NumberArray = NDArray[np.integer | np.floating]


@dataclass(slots=True)
class Cluster:
    sample_start: Number
    sample_end: Number
    channel_start: Number
    channel_end: Number
    point_count: Number
    points: NumberArray

    def __getitem__(self, key: int) -> Number | NumberArray:
        match key:
            case 0:
                return self.sample_start
            case 1:
                return self.sample_end
            case 2:
                return self.channel_start
            case 3:
                return self.channel_end
            case 4:
                return self.point_count
            case 5:
                return self.points
            case _:
                raise ValueError(f"{key} should be an integer between 0 and 5")


def _intervals_overlap(
    a_start: Number, a_end: Number, b_start: Number, b_end: Number
) -> bool:
    # bool cast needed due to numpy bool not being python bool
    return bool(
        b_start <= a_start <= b_end <= a_end
        or a_start <= b_start <= a_end <= b_end
        or b_start <= a_start <= a_end <= b_end
        or a_start <= b_start <= b_end <= a_end
    )


def _channels_overlap(a: Cluster, b: Cluster, channel_offset: float) -> bool:
    midpoint = b.channel_start + int((b.channel_end - b.channel_start) / 2)
    start = midpoint - channel_offset
    end = midpoint + channel_offset
    extended_start = start if start < b.channel_start else b.channel_start
    extended_end = end if end > b.channel_end else b.channel_end
    return _intervals_overlap(
        a.channel_start, a.channel_end, extended_start, extended_end
    )


def _samples_overlap(a: Cluster, b: Cluster, sample_offset: float) -> bool:
    return _intervals_overlap(
        a.sample_start,
        a.sample_end,
        b.sample_start - sample_offset,
        b.sample_end + sample_offset,
    )


def _can_merge(
    a: Cluster, b: Cluster, channel_offset: float, sample_offset: float
) -> bool:
    return _channels_overlap(a, b, channel_offset) and _samples_overlap(
        a, b, sample_offset
    )


def _first_match(
    index: int,
    clusters: dict[int, Cluster],
    channel_offset: float,
    sample_offset: float,
) -> int | None:
    """Find this cluster's first matching neighbour"""
    cluster = clusters[index]
    for other_index, other in clusters.items():
        if other_index != index and _can_merge(
            cluster, other, channel_offset, sample_offset
        ):
            return other_index
    return None


def _merge_into(target: Cluster, other: Cluster) -> None:
    channel_start = min(target.channel_start, other.channel_start)
    channel_end = max(target.channel_end, other.channel_end)
    sample_start = min(target.sample_start, other.sample_start)
    sample_end = max(target.sample_end, other.sample_end)

    target.sample_start = target.sample_start if sample_start == 0 else sample_start
    target.sample_end = target.sample_end if sample_end == 0 else sample_end
    target.channel_start = target.channel_start if channel_start == 0 else channel_start
    target.channel_end = target.channel_end if channel_end == 0 else channel_end
    target.point_count = target.point_count + other.point_count
    target.points = np.array(
        [
            np.append(target.points[0], other.points[0]),
            np.append(target.points[1], other.points[1]),
        ]
    )


def _refresh_matches(
    clusters: dict[int, Cluster],
    first_matches: dict[int, int | None],
    changed: list[int],
    removed: int,
    channel_offset: float,
    sample_offset: float,
) -> None:
    """Update cached first matches after a merge"""
    changed_indices = set(changed)
    for index in clusters:
        previous_match = first_matches[index]

        # definitely stale entry that needs recalculating due to it or its previous match changing
        if (
            index in changed_indices
            or previous_match in changed_indices
            or previous_match == removed
        ):
            first_matches[index] = _first_match(
                index, clusters, channel_offset, sample_offset
            )
            continue

        # maybe changed, check if there are earlier matches available now
        for changed_index in changed:
            if previous_match is not None and changed_index >= previous_match:
                break
            if _can_merge(
                clusters[index], clusters[changed_index], channel_offset, sample_offset
            ):
                first_matches[index] = changed_index
                break


def cluster_association_points(
    cluster_data: Iterable[Cluster],
    sample_overlap: float,
    channel_overlap: float,
    fs: float,
    min_width: float,
    max_width: float,
    min_length: float,
    max_length: float,
) -> list[Cluster]:
    clusters = {
        index: cluster
        for index, cluster in enumerate(cluster_data)
        if min_width <= cluster.channel_end - cluster.channel_start < max_width
        and min_length < cluster.sample_end - cluster.sample_start < max_length
    }
    channel_offset = channel_overlap / 2
    sample_offset = (sample_overlap * fs) / 2
    first_matches = {
        index: _first_match(index, clusters, channel_offset, sample_offset)
        for index in clusters
    }

    while True:
        index = next(
            (index for index in clusters if first_matches[index] is not None), None
        )
        if index is None:
            return list(clusters.values())

        other_index = first_matches[index]
        assert other_index is not None
        target = clusters[index]
        _merge_into(target, clusters[other_index])
        del clusters[other_index]
        del first_matches[other_index]

        changed = [key for key, cluster in clusters.items() if cluster is target]
        _refresh_matches(
            clusters, first_matches, changed, other_index, channel_offset, sample_offset
        )
