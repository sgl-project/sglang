"""Observe real EP resource calls without replacing communication operations.

Every wrapped method calls the original binding. Retaining Python references
also makes explicit destroy/reset necessary; garbage collection cannot satisfy
the resource-closure check by accident.
"""

from contextlib import ExitStack
from unittest.mock import patch

import torch


class EpAudit:
    def __init__(self):
        self.events = []
        self.groups = {}
        self.handles = {}
        self.graphs = []
        self.stack = ExitStack()
        self.last_work = -1
        self.last_wait = -1

    def record(self, operation, **fields):
        self.events.append({"operation": operation, **fields})

    def __enter__(self):
        import nccl.ep as ep

        create_group = ep.Group.create
        create_handle = ep.Group.create_handle
        update_handle = ep.Handle.update
        destroy_handle = ep.Handle.destroy
        destroy_group = ep.Group.destroy
        dispatch = ep.Handle.dispatch
        combine = ep.Handle.combine
        complete = ep.Handle.complete
        synchronize = torch.cuda.synchronize
        graph_class = torch.cuda.CUDAGraph
        audit = self

        def group_create(cls, comm, config):
            group = create_group(comm, config)
            key = id(group)
            audit.groups[key] = {
                "object": group,
                "closed": False,
                "capacity": config.max_dispatch_tokens_per_rank,
                "rdma_buffer_size": config.rdma_buffer_size,
            }
            audit.record(
                "group_create", group=key, capacity=config.max_dispatch_tokens_per_rank
            )
            return group

        def handle_create(group, *args, **kwargs):
            handle = create_handle(group, *args, **kwargs)
            audit.handles[id(handle)] = {
                "object": handle,
                "group": id(group),
                "closed": False,
            }
            audit.record("handle_create", group=id(group), handle=id(handle))
            return handle

        def handle_update(handle, *args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                raise AssertionError("Handle.update entered CUDA capture")
            result = update_handle(handle, *args, **kwargs)
            audit.record("handle_update", handle=id(handle))
            return result

        def handle_destroy(handle):
            if torch.cuda.is_current_stream_capturing():
                raise AssertionError("Handle.destroy entered CUDA capture")
            group_key = audit.handles[id(handle)]["group"]
            if audit.groups[group_key]["rdma_buffer_size"] == 0:
                live_graphs = [
                    id(graph)
                    for graph in audit.graphs
                    if group_key in graph.audit_groups and not graph.audit_reset
                ]
                if live_graphs:
                    raise AssertionError(
                        f"Persistent handle freed before graphs: {live_graphs}"
                    )
                if audit.last_wait < audit.last_work:
                    raise AssertionError(
                        "Persistent handle freed without waiting for GPU work"
                    )
            result = destroy_handle(handle)
            audit.handles[id(handle)]["closed"] = True
            audit.record("handle_destroy", handle=id(handle))
            return result

        def group_destroy(group):
            live = [
                key
                for key, value in audit.handles.items()
                if value["group"] == id(group) and not value["closed"]
            ]
            if live:
                raise AssertionError(f"Destroying group before its handles: {live}")
            if audit.last_wait < audit.last_work:
                raise AssertionError("EP group freed without waiting for GPU work")
            result = destroy_group(group)
            audit.groups[id(group)]["closed"] = True
            audit.record("group_destroy", group=id(group))
            return result

        def device_synchronize(*args, **kwargs):
            result = synchronize(*args, **kwargs)
            audit.record("device_synchronize")
            audit.last_wait = len(audit.events) - 1
            return result

        def observe_work(method, operation):
            def wrapped(handle, *args, **kwargs):
                result = method(handle, *args, **kwargs)
                audit.record(operation, handle=id(handle))
                audit.last_work = len(audit.events) - 1
                return result

            return wrapped

        class ObservedGraph(graph_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.audit_reset = False
                self.audit_groups = set()
                audit.graphs.append(self)
                audit.record("graph_create", graph=id(self))

            def capture_begin(self, *args, **kwargs):
                self.audit_groups = {
                    key
                    for key, value in audit.groups.items()
                    if not value["closed"] and value["rdma_buffer_size"] == 0
                }
                return super().capture_begin(*args, **kwargs)

            def capture_end(self):
                result = super().capture_end()
                audit.record("graph_capture_end", graph=id(self))
                audit.last_work = len(audit.events) - 1
                return result

            def replay(self):
                result = super().replay()
                audit.record("graph_replay", graph=id(self))
                audit.last_work = len(audit.events) - 1
                return result

            def reset(self):
                if audit.last_wait < audit.last_work:
                    raise AssertionError("Graph reset without waiting for GPU work")
                result = super().reset()
                self.audit_reset = True
                audit.record("graph_reset", graph=id(self))
                return result

        self.stack.enter_context(
            patch.object(ep.Group, "create", classmethod(group_create))
        )
        self.stack.enter_context(patch.object(ep.Group, "create_handle", handle_create))
        self.stack.enter_context(patch.object(ep.Handle, "update", handle_update))
        self.stack.enter_context(patch.object(ep.Handle, "destroy", handle_destroy))
        self.stack.enter_context(patch.object(ep.Group, "destroy", group_destroy))
        self.stack.enter_context(
            patch.object(ep.Handle, "dispatch", observe_work(dispatch, "dispatch"))
        )
        self.stack.enter_context(
            patch.object(ep.Handle, "combine", observe_work(combine, "combine"))
        )
        self.stack.enter_context(
            patch.object(ep.Handle, "complete", observe_work(complete, "complete"))
        )
        self.stack.enter_context(
            patch.object(torch.cuda, "synchronize", device_synchronize)
        )
        self.stack.enter_context(patch.object(torch.cuda, "CUDAGraph", ObservedGraph))
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def assert_closed(self):
        assert all(
            item["closed"] for item in self.groups.values()
        ), "Live EP groups remain"
        assert all(
            item["closed"] for item in self.handles.values()
        ), "Live EP handles remain"
        assert all(
            graph.audit_reset for graph in self.graphs
        ), "CUDA executables were not reset"
        return {
            "groups_created": len(self.groups),
            "handles_created": len(self.handles),
            "graphs_created": len(self.graphs),
            "all_explicitly_closed": True,
            "events": self.events,
        }
