import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from inspect import iscoroutinefunction

import dask
import pytest

from dask_labextension_gateway import DaskGatewayClusterManager

from .utils_test import temp_gateway


@pytest.fixture
async def gateway():
    async with temp_gateway() as g:
        with g.gateway_client(asynchronous=False) as gateway:
            yield gateway


async def _run_in_thread(f, *args, **kwargs):
    """Run a single function in a thread"""
    if iscoroutinefunction(f):
        args = (f(*args, **kwargs),)
        kwargs = {}
        f = asyncio.run
    with ThreadPoolExecutor(1) as pool:
        cf = pool.submit(f, *args, **kwargs)
        return await asyncio.wrap_future(cf)


def async_in_thread(f):
    """Decorator to run async tests in a thread

    avoids deadlock because some manager methods are sync
    This is fixable by making dask_labextension.manager fully async
    """

    @wraps(f)
    async def test_wrapped(*args, **kwargs):
        return await _run_in_thread(f, *args, **kwargs)

    return test_wrapped


@pytest.fixture
async def manager(gateway):
    manager = DaskGatewayClusterManager(gateway=gateway)
    # like 'async with' but make sure close runs in a thread
    # to avoid deadlock
    await manager.__aenter__()
    try:
        yield manager
    finally:
        await _run_in_thread(manager.__aexit__, *sys.exc_info())


async def wait_for_workers(manager, cluster_id, workers=2, timeout=10):
    model = manager.get_cluster(cluster_id)
    start = time.monotonic()
    while model["workers"] != workers:
        await asyncio.sleep(0.1)
        model = manager.get_cluster(cluster_id)
        assert time.monotonic() < start + timeout, model["workers"]


@async_in_thread
async def test_start(manager):
    # add cluster
    model = await manager.start_cluster()
    assert not model.get("adapt")

    # close cluster
    assert len(manager.list_clusters()) == 1
    await manager.close_cluster(model["id"])

    # add cluster with adaptive configuration
    model = await manager.start_cluster(
        configuration={"adapt": {"minimum": 1, "maximum": 3}}
    )
    model = manager.get_cluster(model["id"])
    assert model["workers"] == 1
    await manager.close_cluster(model["id"])


@async_in_thread
async def test_close(manager):
    # start a cluster
    model = await manager.start_cluster()

    # return None if a nonexistent cluster is closed
    assert await manager.close_cluster("fake") is None

    # close the cluster
    await manager.close_cluster(model["id"])
    assert manager.list_clusters() == []


@async_in_thread
async def test_get(manager):
    # start a cluster
    model = await manager.start_cluster()

    # return None if a nonexistent cluster is requested
    assert await manager.close_cluster("fake") is None

    # get the cluster by id
    assert model == manager.get_cluster(model["id"])

    # close the cluster
    await manager.close_cluster(model["id"])


@async_in_thread
async def test_list(manager):
    # start with an empty list
    assert manager.list_clusters() == []
    # start clusters
    model1 = await manager.start_cluster()
    model2 = await manager.start_cluster()

    models = manager.list_clusters()
    assert len(models) == 2
    assert model1 in models
    assert model2 in models


@async_in_thread
async def test_scale(manager):
    # add cluster with number of workers configuration
    model = await manager.start_cluster(configuration={"workers": 3})
    await wait_for_workers(manager, model["id"], 3)

    await asyncio.sleep(0.2)  # let workers settle # TODO: remove need for this

    # rescale the cluster
    model = await manager.scale_cluster(model["id"], 6)
    await wait_for_workers(manager, model["id"], 6)


@async_in_thread
async def test_adapt(manager):
    # add a new cluster
    model = await manager.start_cluster()
    model = manager.adapt_cluster(model["id"], 2, 4)
    await wait_for_workers(manager, model["id"], 2)


@async_in_thread
async def test_initial(gateway):
    with dask.config.set(
        {
            "labextension": {
                "initial": [{"name": "foo"}],
                "default": {"workers": 2},
            }
        }
    ):
        # Test asynchronous starting of clusters via a context
        async with DaskGatewayClusterManager(gateway=gateway) as manager:
            clusters = manager.list_clusters()
            assert len(clusters) == 1
            await wait_for_workers(manager, clusters[0]["id"], workers=2)

        # aexit closed clusters
        assert manager.gateway.list_clusters() == []

        # Test asynchronous starting of clusters outside of a context
        manager = DaskGatewayClusterManager(gateway=gateway)
        assert len(manager.list_clusters()) == 0
        await manager
        clusters = manager.list_clusters()
        assert len(clusters) == 1
        await wait_for_workers(manager, clusters[0]["id"], workers=2)
        await manager.close()
        # manager.close closed clusters
        assert manager.gateway.list_clusters() == []

        manager = await DaskGatewayClusterManager(gateway=gateway)
        clusters = manager.list_clusters()
        assert len(clusters) == 1
        await wait_for_workers(manager, clusters[0]["id"], workers=2)
        await manager.close()
        assert manager.gateway.list_clusters() == []


@async_in_thread
async def test_external_changes(gateway, manager):
    clusters = manager.list_clusters()
    assert clusters == []
    cluster = gateway.new_cluster()
    clusters = manager.list_clusters()
    assert len(clusters) == 1
    assert clusters[0]["name"] == cluster.name
    model1 = await manager.start_cluster()
    model2 = await manager.start_cluster()
    clusters = manager.list_clusters()
    assert len(clusters) == 3

    # stop a cluster from outside
    gateway.stop_cluster(model1["name"])
    clusters = manager.list_clusters()
    assert len(clusters) == 2
    assert model1["id"] not in [c["id"] for c in clusters]
    assert model2["id"] in [c["id"] for c in clusters]

    # close stops clusters started by the manager
    # but not those started outside
    await manager.close()
    clusters = gateway.list_clusters()
    assert len(clusters) == 1
    assert clusters[0].name == cluster.name
