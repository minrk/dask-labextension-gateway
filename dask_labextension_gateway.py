"""Apply dask gateway support to labextension"""

from __future__ import annotations

import re
import sys
import typing
from typing import Any, cast

import dask.config
from dask_labextension.manager import (
    DaskClusterManager,
    make_cluster_model,
)
from dask_gateway import Gateway

if typing.TYPE_CHECKING:
    import jupyter_server
    from dask_labextension.manager import ClusterModel

__version__ = "0.0.1.dev"

def _jupyter_server_extension_paths() -> list[dict[str,str]]:
    return [{"module": "dask_labextension_gateway"}]


def load_jupyter_server_extension(nb_server_app: "jupyter_server.serverapp.ServerApp") -> None:
    use_labextension = dask.config.get("labextension.use_gateway", False)
    if not use_labextension:
        nb_server_app.log.info("Not enabling Dask Gateway in dask jupyterlab extension")
        return
    nb_server_app.log.info("Enabling Dask Gateway in dask jupyterlab extension")
    from dask_labextension import manager
    manager.manager = DaskGatewayClusterManager()
    # already imported, need to patch module-level manager reference
    for submod in ('clusterhandler', 'dashboardhandler'):
        modname = f"dask_labextension.{submod}"
        if modname in sys.modules:
            nb_server_app.log.info(f"[dask_labextension_gateway] patching {modname}\n")
            sys.modules[modname].manager = manager.manager # type: ignore


def _cluster_id_from_name(cluster_id: str) -> str:
    """Make a cluster id from a cluster name (already an id itself)

    Only need this because of unnecessarily strict UUID regex in URL handler
    # Upstream fix https://github.com/dask/dask-labextension/pull/272
    """
    cluster_id = re.sub(r"[^\w]+", "", cluster_id)
    return f"u-u-i-d-{cluster_id}"


class DaskGatewayClusterManager(DaskClusterManager):
    gateway: Gateway

    def __init__(self) -> None:
        self.gateway = Gateway()
        super().__init__()

    def list_clusters(self) -> list[ClusterModel]:
        cluster_models = []
        self._cluster_names = {}
        for cluster_info in self.gateway.list_clusters():
            cluster_name = cluster_info.name
            cluster_id = _cluster_id_from_name(cluster_name)
            self._cluster_names[cluster_id] = cluster_name
            with self.gateway.connect(cluster_name) as cluster:
                cluster_model = make_cluster_model(
                    cluster_id, cluster_name, cluster, None
                )
            cluster_models.append(cluster_model)
        return cluster_models

    def get_cluster(self, cluster_id: str) -> ClusterModel | None:
        cluster_name = self._cluster_names.get(cluster_id)
        if cluster_name is None:
            return None
        with self.gateway.connect(cluster_name) as cluster:
            return make_cluster_model(cluster_id, cluster_name, cluster, None)

    async def start_cluster(self, cluster_id: str = "", configuration: dict[str, Any] | None = None) -> ClusterModel:
        # default cluster options come from gateway.cluster.options
        cluster = self.gateway.new_cluster(shutdown_on_close=False)
        cluster_id = _cluster_id_from_name(cluster.name)
        self._cluster_names[cluster_id] = cluster.name

        # apply dask.labextension default scale
        configuration = cast(dict, dask.config.merge(
            dask.config.get("labextension.default"),
            configuration or {},
        ))
        if configuration.get("adapt"):
            self.gateway.adapt_cluster(
                cluster.name, **configuration.get("adapt")
            )
        elif configuration.get("workers") is not None:
            self.gateway.scale_cluster(cluster.name, configuration["workers"])
        with cluster:
            model = make_cluster_model(cluster_id, cluster.name, cluster, None)
        return model

    async def close_cluster(self, cluster_id: str) -> ClusterModel | None:
        cluster_model = self.get_cluster(cluster_id)
        if cluster_model:
            self.gateway.stop_cluster(cluster_model["name"])
        return cluster_model

    async def scale_cluster(self, cluster_id: str, n: int) -> ClusterModel | None:
        if cluster_id not in self._cluster_names:
            return None
        self.gateway.scale_cluster(self._cluster_names[cluster_id], n)
        return self.get_cluster(cluster_id)

    def adapt_cluster(self, cluster_id: str, minimum: int, maximum: int) -> ClusterModel | None:
        cluster_model = self.get_cluster(cluster_id)
        if cluster_model is None:
            return None
        self.gateway.adapt_cluster(cluster_model["name"], minimum, maximum)
        return self.get_cluster(cluster_id)
