from __future__ import annotations

import sys
import types
from importlib import import_module
from pathlib import Path


def ensure_verl_alias() -> None:
    # Provide a placeholder package so absolute imports like `import verl.utils`
    # don't crash during the initial import of the nested `alphaapollo.core.generation.verl`.
    verl_pkg_dir = Path(__file__).resolve().parent / "generation" / "verl"
    if "verl" not in sys.modules:
        placeholder = types.ModuleType("verl")
        placeholder.__path__ = [str(verl_pkg_dir)]
        sys.modules["verl"] = placeholder

    # Also provide a placeholder for the legacy import path `alphaapollo.core.verl.*`
    # so module execution like `python -m alphaapollo.core.verl.trainer.main_ppo` keeps working.
    legacy_name = "alphaapollo.core.verl"
    if legacy_name not in sys.modules:
        legacy = types.ModuleType(legacy_name)
        legacy.__path__ = [str(verl_pkg_dir)]
        sys.modules[legacy_name] = legacy

    # Now import the real package and overwrite the alias.
    real_name = "alphaapollo.core.generation.verl"
    real_pkg = import_module(real_name)
    sys.modules["verl"] = real_pkg
    sys.modules["alphaapollo.core.verl"] = real_pkg
    verl = real_pkg

    # Inject utils submodules before any code path that imports verl.utils.dataset
    # (e.g. multi_turn_rollout -> rollout_loop -> rl_dataset imports verl.utils.torch_functional).
    utils = verl.utils
    utils.hdfs_io = import_module("alphaapollo.core.generation.verl.utils.hdfs_io")
    utils.torch_functional = import_module(
        "alphaapollo.core.generation.verl.utils.torch_functional"
    )

    # Legacy alias: `alphaapollo.core.multi_turn_rollout.*` used by verl code.
    _ensure_multi_turn_rollout_alias()

    # Keep protocol module identity consistent across both import paths.
    # If one path was imported first in a long-lived process (e.g. Ray workers),
    # reuse that module object and bind both names to it.
    protocol = (
        sys.modules.get("verl.protocol")
        or sys.modules.get("alphaapollo.core.verl.protocol")
        or sys.modules.get("alphaapollo.core.generation.verl.protocol")
    )
    if protocol is None:
        protocol = import_module("alphaapollo.core.generation.verl.protocol")
    sys.modules["verl.protocol"] = protocol
    sys.modules["alphaapollo.core.verl.protocol"] = protocol
    sys.modules["alphaapollo.core.generation.verl.protocol"] = protocol

    # Patch strict type checks in mixed import-path environments.
    _patch_verl_type_guards()


def _patch_verl_type_guards() -> None:
    protocol = import_module("alphaapollo.core.generation.verl.protocol")
    decorator_main = import_module("alphaapollo.core.generation.verl.single_controller.base.decorator")
    decorator_alias = import_module("verl.single_controller.base.decorator")
    # Keep decorator module identity consistent across import paths.
    sys.modules["verl.single_controller.base.decorator"] = decorator_main
    sys.modules["alphaapollo.core.verl.single_controller.base.decorator"] = decorator_main
    sys.modules["alphaapollo.core.generation.verl.single_controller.base.decorator"] = decorator_main
    DataProto = protocol.DataProto

    def _normalize_dataproto_like(obj):
        if isinstance(obj, DataProto):
            return obj
        if all(hasattr(obj, attr) for attr in ("batch", "non_tensor_batch", "meta_info")):
            return DataProto(batch=obj.batch, non_tensor_batch=obj.non_tensor_batch, meta_info=obj.meta_info)
        return obj

    if not getattr(protocol.pad_dataproto_to_divisor, "_alphaapollo_patched", False):
        _orig_pad = protocol.pad_dataproto_to_divisor

        def _patched_pad(data, size_divisor):
            return _orig_pad(_normalize_dataproto_like(data), size_divisor)

        _patched_pad._alphaapollo_patched = True
        protocol.pad_dataproto_to_divisor = _patched_pad

    # Some modules import the function directly:
    # `from verl.protocol import pad_dataproto_to_divisor`.
    # Rebind their module-level symbol to the patched function when already loaded.
    for mod_name in (
        "alphaapollo.core.generation.multi_turn_rollout.rollout_loop",
        "alphaapollo.core.multi_turn_rollout.rollout_loop",
        "alphaapollo.core.generation.verl.trainer.ppo.ray_trainer",
        "verl.trainer.ppo.ray_trainer",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "pad_dataproto_to_divisor"):
            mod.pad_dataproto_to_divisor = protocol.pad_dataproto_to_divisor

    for decorator in (decorator_main, decorator_alias):
        if not getattr(decorator._split_args_kwargs_data_proto, "_alphaapollo_patched", False):
            _orig_split = decorator._split_args_kwargs_data_proto

            def _patched_split(chunks, *args, _orig_split=_orig_split, **kwargs):
                norm_args = tuple(_normalize_dataproto_like(arg) for arg in args)
                norm_kwargs = {k: _normalize_dataproto_like(v) for k, v in kwargs.items()}
                return _orig_split(chunks, *norm_args, **norm_kwargs)

            _patched_split._alphaapollo_patched = True
            decorator._split_args_kwargs_data_proto = _patched_split

        if not getattr(decorator._split_args_kwargs_data_proto_with_auto_padding, "_alphaapollo_patched", False):
            _orig_split_auto = decorator._split_args_kwargs_data_proto_with_auto_padding

            def _patched_split_auto(chunks, *args, _orig_split_auto=_orig_split_auto, **kwargs):
                norm_args = tuple(_normalize_dataproto_like(arg) for arg in args)
                norm_kwargs = {k: _normalize_dataproto_like(v) for k, v in kwargs.items()}
                return _orig_split_auto(chunks, *norm_args, **norm_kwargs)

            _patched_split_auto._alphaapollo_patched = True
            decorator._split_args_kwargs_data_proto_with_auto_padding = _patched_split_auto

        if not getattr(decorator.dispatch_dp_compute_data_proto, "_alphaapollo_patched", False):
            _orig_dispatch_proto = decorator.dispatch_dp_compute_data_proto

            def _patched_dispatch_proto(worker_group, *args, _decorator=decorator, **kwargs):
                # In mixed import-path setups, worker_group class identity can differ
                # even when behavior is compatible. Use capability checks instead.
                assert hasattr(worker_group, "world_size"), "worker_group must have world_size"
                splitted_args, splitted_kwargs = _decorator._split_args_kwargs_data_proto_with_auto_padding(
                    worker_group.world_size,
                    *args,
                    **kwargs,
                )
                return splitted_args, splitted_kwargs

            _patched_dispatch_proto._alphaapollo_patched = True
            decorator.dispatch_dp_compute_data_proto = _patched_dispatch_proto

            # Keep registry pointing to patched dispatch function.
            for _, mode_cfg in decorator.DISPATCH_MODE_FN_REGISTRY.items():
                dispatch_fn = mode_cfg.get("dispatch_fn")
                if dispatch_fn is _orig_dispatch_proto or getattr(dispatch_fn, "__name__", "") == "dispatch_dp_compute_data_proto":
                    mode_cfg["dispatch_fn"] = _patched_dispatch_proto

        if not getattr(decorator.collect_dp_compute_data_proto, "_alphaapollo_patched", False):
            _orig_collect_proto = decorator.collect_dp_compute_data_proto

            def _patched_collect_proto(worker_group, output, _decorator=decorator):
                import ray

                normalized_output = []
                for o in output:
                    if isinstance(o, ray.ObjectRef):
                        normalized_output.append(o)
                    else:
                        normalized_output.append(_normalize_dataproto_like(o))

                assert hasattr(worker_group, "world_size"), "worker_group must have world_size"
                assert len(normalized_output) == worker_group.world_size
                return _decorator._concat_data_proto_or_future(normalized_output)

            _patched_collect_proto._alphaapollo_patched = True
            decorator.collect_dp_compute_data_proto = _patched_collect_proto

            # Keep registry pointing to patched collect function.
            for _, mode_cfg in decorator.DISPATCH_MODE_FN_REGISTRY.items():
                collect_fn = mode_cfg.get("collect_fn")
                if collect_fn is _orig_collect_proto or getattr(collect_fn, "__name__", "") == "collect_dp_compute_data_proto":
                    mode_cfg["collect_fn"] = _patched_collect_proto

    _patch_worker_group_dispatch_compat()


def _patch_worker_group_dispatch_compat() -> None:
    wg_main = import_module("alphaapollo.core.generation.verl.single_controller.base.worker_group")
    wg_alias = import_module("verl.single_controller.base.worker_group")
    # Keep worker_group module identity consistent across import paths.
    sys.modules["verl.single_controller.base.worker_group"] = wg_main
    sys.modules["alphaapollo.core.verl.single_controller.base.worker_group"] = wg_main
    sys.modules["alphaapollo.core.generation.verl.single_controller.base.worker_group"] = wg_main

    if getattr(wg_main.WorkerGroup._bind_worker_method, "_alphaapollo_patched", False):
        return

    MAGIC_ATTR = wg_main.MAGIC_ATTR
    Dispatch = wg_main.Dispatch
    get_predefined_dispatch_fn = wg_main.get_predefined_dispatch_fn
    get_predefined_execute_fn = wg_main.get_predefined_execute_fn
    Execute = get_predefined_execute_fn.__globals__.get("Execute")

    def _patched_bind_worker_method(self, user_defined_cls, func_generator):
        method_names = []
        for method_name in dir(user_defined_cls):
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                continue

            if hasattr(method, MAGIC_ATTR):
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(attribute, dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]

                if not isinstance(dispatch_mode, Dispatch):
                    # Accept equivalent Dispatch enums from aliased modules.
                    if hasattr(dispatch_mode, "name") and dispatch_mode.name in Dispatch:
                        dispatch_mode = Dispatch[dispatch_mode.name]

                if isinstance(dispatch_mode, Dispatch):
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict), f"dispatch_mode must be Dispatch or dict, got {type(dispatch_mode)}"
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                if Execute is not None and not isinstance(execute_mode, Execute):
                    # Accept equivalent Execute enums from aliased modules.
                    if hasattr(execute_mode, "name") and execute_mode.name in Execute:
                        execute_mode = Execute[execute_mode.name]

                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]
                execute_fn = getattr(self, wg_execute_fn_name)
                assert callable(execute_fn), "execute_fn must be callable"

                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )
                setattr(self, method_name, func)
                method_names.append(method_name)

        return method_names

    _patched_bind_worker_method._alphaapollo_patched = True
    wg_main.WorkerGroup._bind_worker_method = _patched_bind_worker_method
    wg_alias.WorkerGroup._bind_worker_method = _patched_bind_worker_method


def _ensure_multi_turn_rollout_alias() -> None:
    """Expose `alphaapollo.core.multi_turn_rollout` as an alias to `alphaapollo.core.generation.multi_turn_rollout`."""
    pkg_dir = Path(__file__).resolve().parent / "generation" / "multi_turn_rollout"

    legacy_name = "alphaapollo.core.multi_turn_rollout"
    if legacy_name not in sys.modules:
        legacy = types.ModuleType(legacy_name)
        legacy.__path__ = [str(pkg_dir)]
        sys.modules[legacy_name] = legacy

    real_name = "alphaapollo.core.generation.multi_turn_rollout"
    real_pkg = import_module(real_name)
    sys.modules[legacy_name] = real_pkg


ensure_verl_alias()
