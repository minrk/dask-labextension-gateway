"""Apply dask gateway support to labextension"""

from __future__ import annotations

import asyncio
import typing
from typing import Any, cast

import dask.config
from dask_gateway import Gateway
from dask_gateway.client import ClusterStatus
from dask_labextension.manager import (
    DaskClusterManager,
    make_cluster_model,
)

if typing.TYPE_CHECKING:
    import jupyter_server
    from dask_gateway.client import ClusterReport
    from dask_labextension.manager import ClusterModel

__version__ = "0.2.0.dev"


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "dask_labextension_gateway"}]


def load_jupyter_server_extension(
    nb_server_app: jupyter_server.serverapp.ServerApp,
) -> None:
    use_labextension = dask.config.get("labextension.use_gateway", False)
    if not use_labextension:
        nb_server_app.log.info("Not enabling Dask Gateway in dask jupyterlab extension")
        return
    nb_server_app.log.info("Enabling Dask Gateway in dask jupyterlab extension")
    web_app = nb_server_app.web_app
    web_app.settings["dask_cluster_manager"] = DaskGatewayClusterManager()


def make_cluster_report_model(cluster_id: str, cluster: ClusterReport) -> ClusterModel:
    """make a cluster model from a ClusterReport instead of a connected Cluster

    e.g. for Pending clusters
    """
    return dict(
        id=cluster_id,
        name=f"{cluster.name} ({cluster.status.name})",
        scheduler_address=cluster.scheduler_address or "",
        dashboard_link=cluster.dashboard_link or "",
        workers=0,
        memory="0 B",
        cores=0,
    )


class DaskGatewayClusterManager(DaskClusterManager):
    gateway: Gateway
    _started_clusters: set[str]

    def __init__(self, *, gateway: Gateway | None = None) -> None:
        self._created_gateway = False
        if gateway is None:
            self._created_gateway = True
            gateway = Gateway(asynchronous=True)
        self.gateway = gateway
        self._started_clusters = set()
        super().__init__()

    async def close(self):
        if self._started_clusters:
            clusters = await self.gateway.list_clusters()
            cluster_names = {c.name for c in clusters}
        for cluster_name in self._started_clusters:
            if cluster_name in cluster_names:
                await self.gateway.stop_cluster(cluster_name)
        self._started_clusters = set()
        if self.gateway is not None and self._created_gateway:
            self.gateway.close()
            self.gateway = None

    async def list_clusters(self) -> list[ClusterModel]:
        cluster_models = []
        for cluster_info in await self.gateway.list_clusters():
            cluster_id = cluster_name = cluster_info.name
            if cluster_info.status == ClusterStatus.RUNNING:
                async with self.gateway.connect(cluster_name) as cluster:
                    cluster_model = make_cluster_model(
                        cluster_id, cluster_name, cluster, None
                    )
            else:
                cluster_model = make_cluster_report_model(cluster_id, cluster_info)
            cluster_models.append(cluster_model)
        return cluster_models

    async def get_cluster(self, cluster_id: str) -> ClusterModel | None:
        try:
            cluster_info = await self.gateway.get_cluster(cluster_id)
        except ValueError:
            return None
        if cluster_info.status == ClusterStatus.RUNNING:
            async with self.gateway.connect(cluster_id) as cluster:
                return make_cluster_model(cluster_id, cluster_id, cluster, None)
        else:
            return make_cluster_report_model(cluster_id, cluster_info)

    async def start_cluster(
        self, cluster_id: str = "", configuration: dict[str, Any] | None = None
    ) -> ClusterModel:
        # default cluster options come from gateway.cluster.options
        cluster_name = cluster_id = await self.gateway.submit()
        cluster_info = await self.gateway.get_cluster(cluster_name)
        self._started_clusters.add(cluster_name)

        # apply dask.labextension default scale
        configuration = cast(
            dict,
            dask.config.merge(
                dask.config.get("labextension.default"),
                configuration or {},
            ),
        )
        # wait for start; can't scale before cluster has started
        for _ in range(30):
            if cluster_info.status == ClusterStatus.PENDING:
                await asyncio.sleep(1)
            else:
                break
            cluster_info = await self.gateway.get_cluster(cluster_name)

        if cluster_info.status != ClusterStatus.RUNNING:
            return make_cluster_report_model(cluster_id, cluster_info)

        adapt = configuration.get("adapt")
        workers = configuration.get("workers")
        if adapt is None and workers is None:
            # default: adaptive, no limit
            await self.gateway.adapt_cluster(cluster_name)
        elif adapt is not None:
            await self.gateway.adapt_cluster(cluster_name, **adapt)
        elif workers is not None:
            await self.gateway.scale_cluster(cluster_name, workers)
        async with self.gateway.connect(cluster_name) as cluster:
            model = make_cluster_model(cluster_id, cluster_name, cluster, None)
        return model

    async def close_cluster(self, cluster_id: str) -> ClusterModel | None:
        cluster_model = await self.get_cluster(cluster_id)
        if cluster_model:
            self._started_clusters.discard(cluster_model["name"])
            await self.gateway.stop_cluster(cluster_model["name"])
        return cluster_model

    async def scale_cluster(self, cluster_id: str, n: int) -> ClusterModel | None:
        await self.gateway.scale_cluster(cluster_id, n)
        return await self.get_cluster(cluster_id)

    async def adapt_cluster(
        self, cluster_id: str, minimum: int, maximum: int
    ) -> ClusterModel | None:
        cluster_model = await self.get_cluster(cluster_id)
        if cluster_model is None:
            return None
        await self.gateway.adapt_cluster(cluster_id, minimum, maximum)
        return await self.get_cluster(cluster_id)
