"""Apply dask gateway support to labextension"""

from __future__ import annotations

import asyncio
import re
import sys
import typing
from functools import partial
from typing import Any, Callable, cast
from urllib.parse import urlparse

import dask.config
from dask_gateway import Gateway
from dask_gateway.client import ClusterStatus
from dask_labextension import manager
from dask_labextension.manager import (
    DaskClusterManager,
    make_cluster_model,
)
from jupyter_server.utils import url_path_join

if typing.TYPE_CHECKING:
    import jupyter_server
    from dask_gateway.client import ClusterReport
    from dask_labextension.manager import ClusterModel

__version__ = "0.2.0.dev"


def _jupyter_server_extension_points() -> list[dict[str, str]]:
    return [{"module": "dask_labextension_gateway"}]


def _normalize_dashboard_link(
    log, original_normalize: Callable[[str, Any], str], link: str, request: Any
) -> str:
    log.info(f"normalize {link}")
    link = original_normalize(link, request)
    log.info(f"normalize2 {link}")
    proxy_address = dask.config.get("labextension.gateway_proxy_address")
    if not proxy_address:
        log.info(f"returning {link}")
        return link
    public_address = dask.config.get("gateway.public_address")
    log.info(f"Replacing {public_address} with {proxy_address} in {link}")
    link = link.replace(public_address, proxy_address)
    log.info(f"Got {link}")
    return link


def load_jupyter_server_extension(
    nb_server_app: jupyter_server.serverapp.ServerApp,
) -> None:
    use_labextension = dask.config.get("labextension.use_gateway", False)
    if not use_labextension:
        nb_server_app.log.info("Not enabling Dask Gateway in dask jupyterlab extension")
        return
    nb_server_app.log.info("Enabling Dask Gateway in dask jupyterlab extension")
    web_app = nb_server_app.web_app
    base_url = web_app.settings["base_url"]

    manager.manager = DaskGatewayClusterManager()
    # already imported, need to patch module-level manager reference
    for submod in ("clusterhandler", "dashboardhandler"):
        modname = f"dask_labextension.{submod}"
        if modname in sys.modules:
            nb_server_app.log.info(f"[dask_labextension_gateway] patching {modname}\n")
            sys.modules[modname].manager = manager.manager  # type: ignore

    # patch dashboard handler to proxy via the gateway proxy
    # requires rewriting the proxy URL and allowing the gateway host as a proxy target
    # neither of which can be done via public APIs
    from dask_labextension import dashboardhandler
    from dask_labextension.dashboardhandler import DaskDashboardHandler

    # re-register dashboard proxy handler
    # with host_allowlist and absolute_url set
    dask_hosts = {"localhost", "127.0.0.1"}
    for key in ("gateway.public_address", "labextension.gateway_proxy_address"):
        address = dask.config.get(key)
        if address:
            dask_hosts.add(urlparse(address).netloc)
    nb_server_app.log.info("patching %s with %s", dashboardhandler, dask_hosts)

    # too late to apply settings,
    # have to patch into initialize
    # patching into initialize doesn't get called
    # HOW?????!!!!!!
    def initialize(original_initialize, *a, **kw):
        nb_server_app.log.info("initialize %s %s", a, kw)
        original_initialize(*a, **kw)
        # self.absolute_url = True
        # self.host_allowlist = dask_hosts

    DaskDashboardHandler.initialize = partial(
        DaskDashboardHandler.initialize, DaskDashboardHandler.initialize
    )
    # initialize doesn't work, monkeypatch private methods
    DaskDashboardHandler._check_host_allowlist = lambda self, host: host in dask_hosts
    DaskDashboardHandler.absolute_url = property(lambda self: True)
    DaskDashboardHandler.absolute_url.setter(lambda self, _ignored: None)

    # patch normalization
    # so proxied requests go directly to gateway endpoint
    dashboardhandler._normalize_dashboard_link = partial(
        _normalize_dashboard_link,
        nb_server_app.log,
        dashboardhandler._normalize_dashboard_link,
    )
    url = dask.config.get("gateway.public_address") + "something"
    new_url = dashboardhandler._normalize_dashboard_link(url, None)
    nb_server_app.log.info(f"Normalized {url} to {new_url}")


def _cluster_id_from_name(cluster_id: str) -> str:
    """Make a cluster id from a cluster name (already an id itself)

    Only need this because of unnecessarily strict UUID regex in URL handler
    # Upstream fix https://github.com/dask/dask-labextension/pull/272
    """
    cluster_id = re.sub(r"[^\w]+", "", cluster_id)
    return f"u-u-i-d-{cluster_id}"


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
            gateway = Gateway()
        self.gateway = gateway
        self._started_clusters = set()
        super().__init__()

    async def close(self):
        if self._started_clusters:
            clusters = self.gateway.list_clusters()
            cluster_names = {c.name for c in clusters}
        for cluster_name in self._started_clusters:
            if cluster_name in cluster_names:
                self.gateway.stop_cluster(cluster_name)
        self._started_clusters = set()
        if self.gateway is not None and self._created_gateway:
            self.gateway.close()
            self.gateway = None

    def list_clusters(self) -> list[ClusterModel]:
        cluster_models = []
        self._cluster_names = {}
        for cluster_info in self.gateway.list_clusters():
            cluster_name = cluster_info.name
            cluster_id = _cluster_id_from_name(cluster_name)
            self._cluster_names[cluster_id] = cluster_name
            if cluster_info.status == ClusterStatus.RUNNING:
                with self.gateway.connect(cluster_name) as cluster:
                    cluster_model = make_cluster_model(
                        cluster_id, cluster_name, cluster, None
                    )
            else:
                cluster_model = make_cluster_report_model(cluster_id, cluster_info)
            cluster_models.append(cluster_model)
        return cluster_models

    def get_cluster(self, cluster_id: str) -> ClusterModel | None:
        cluster_name = self._cluster_names.get(cluster_id)
        if cluster_name is None:
            return None
        try:
            cluster_info = self.gateway.get_cluster(cluster_name)
        except ValueError:
            return None
        if cluster_info.status == ClusterStatus.RUNNING:
            with self.gateway.connect(cluster_name) as cluster:
                return make_cluster_model(cluster_id, cluster_name, cluster, None)
        else:
            return make_cluster_report_model(cluster_id, cluster_info)

    async def start_cluster(
        self, cluster_id: str = "", configuration: dict[str, Any] | None = None
    ) -> ClusterModel:
        # default cluster options come from gateway.cluster.options
        cluster_name = self.gateway.submit()
        cluster_info = self.gateway.get_cluster(cluster_name)
        self._started_clusters.add(cluster_name)
        cluster_id = _cluster_id_from_name(cluster_name)
        self._cluster_names[cluster_id] = cluster_name

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
            cluster_info = self.gateway.get_cluster(cluster_name)

        if cluster_info.status != ClusterStatus.RUNNING:
            return make_cluster_report_model(cluster_id, cluster_info)

        adapt = configuration.get("adapt")
        workers = configuration.get("workers")
        if adapt is None and workers is None:
            # default: adaptive, no limit
            self.gateway.adapt_cluster(cluster_name)
        elif adapt is not None:
            self.gateway.adapt_cluster(cluster_name, **adapt)
        elif workers is not None:
            self.gateway.scale_cluster(cluster_name, workers)
        with self.gateway.connect(cluster_name) as cluster:
            model = make_cluster_model(cluster_id, cluster_name, cluster, None)
        return model

    async def close_cluster(self, cluster_id: str) -> ClusterModel | None:
        cluster_model = self.get_cluster(cluster_id)
        if cluster_model:
            self._started_clusters.discard(cluster_model["name"])
            self.gateway.stop_cluster(cluster_model["name"])
        return cluster_model

    async def scale_cluster(self, cluster_id: str, n: int) -> ClusterModel | None:
        if cluster_id not in self._cluster_names:
            return None
        self.gateway.scale_cluster(self._cluster_names[cluster_id], n)
        return self.get_cluster(cluster_id)

    def adapt_cluster(
        self, cluster_id: str, minimum: int, maximum: int
    ) -> ClusterModel | None:
        cluster_model = self.get_cluster(cluster_id)
        if cluster_model is None:
            return None
        self.gateway.adapt_cluster(cluster_model["name"], minimum, maximum)
        return self.get_cluster(cluster_id)
