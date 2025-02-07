# dask-labextension-gateway

Add Dask Gateway support to [dask-labextension](https://github.com/dask/dask-labextension).

This is a temporary package to address https://github.com/dask/dask-labextension/issues/204 .
once addressed, this package will not be maintained.

To use this package:

1. install it

```
pip install dask-labextension-gateway
```

2. set config `dask.labextension.use_gateway = true`, e.g. in `~/.config/dask/labextension.yaml`:

```yaml
labextension:
  use_gateway: true
```

3. configure your default gateway client, as you would normally (the Gateway used in the extension is constructed with `Gateway()`, using default options).
