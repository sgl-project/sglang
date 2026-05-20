import argparse
import bisect
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

parser = argparse.ArgumentParser(
    description="Convert SGLang OTEL trace files to Perfetto format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-i",
    "--input",
    dest="input_file",
    required=True,
    type=str,
    help="Path to the input OTEL trace file (JSON or JSONL format).",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output_file",
    type=str,
    default="sglang_trace_perfetto.json",
    help="Path to the output Perfetto JSON file.",
)
parser.add_argument(
    "-f", "--torch-file", dest="torch_file", help="specify torch profile file"
)

args = parser.parse_args()

perfetto_data = None
if args.torch_file:
    with open(args.torch_file, "r", encoding="utf-8") as file:
        perfetto_data = json.load(file)
        baseline = perfetto_data["baseTimeNanoseconds"]
else:
    baseline = 0


def id_generator():
    i = 0
    while True:
        yield i
        i += 1


relation_id_gen = id_generator()


class SpanLayoutContainer:
    def __init__(self):
        self.intervals = []

    def check_overlap(self, start, end):
        idx = bisect.bisect_left(self.intervals, (start, float("-inf")))

        if idx > 0:
            prev_start, prev_end = self.intervals[idx - 1]
            if prev_end > start:
                return True

        if idx < len(self.intervals):
            next_start, next_end = self.intervals[idx]
            if next_start < end:
                return True
        return False

    def insert_span(self, start, end):
        bisect.insort_left(self.intervals, (start, end))


def new_metadata_level1(name: str, pid):
    return {
        "name": "process_name",
        "ph": "M",
        "pid": pid,
        "args": {"name": name},
    }


def new_metadata_level2(name: str, pid, slot_seq):
    return {
        "name": "thread_name",
        "ph": "M",
        "pid": pid,
        "tid": slot_seq,
        "args": {"name": name},
    }


def __find_line(graph, trans_graph_status, slot_meta_data, pid, start, end):
    if pid in trans_graph_status:
        line = trans_graph_status[pid]
        if start == end:
            return line
        # check conflict
        if not graph[pid][line].check_overlap(start, end):
            return line

    if pid not in graph:
        line = 1
        graph[pid] = {line: SpanLayoutContainer()}
        trans_graph_status[pid] = line
        slot_meta_data.append(new_metadata_level2("slot", pid, line))
        return line

    for line in graph[pid]:
        if not graph[pid][line].check_overlap(start, end):
            trans_graph_status[pid] = line
            return line

    new_line = len(graph[pid]) + 1
    graph[pid][new_line] = SpanLayoutContainer()
    trans_graph_status[pid] = new_line
    slot_meta_data.append(new_metadata_level2("slot", pid, new_line))
    return new_line


OtelSpan = Dict[str, Any]

# ---------------------------------------------------------------------------
# HiCache module: virtual thread mapping for spans without parent hierarchy
# ---------------------------------------------------------------------------


def _is_hicache_span(span):
    """Check if a span belongs to the hicache module by its module attribute."""
    return span.get("attributes", {}).get("module") == "sglang::hicache"


def _build_hicache_span_tree(hicache_spans):
    """Build a virtual root→thread→slice tree for hicache spans.

    HiCache spans are top-level (no parent-child nesting), so they don't
    fit the root→thread→slice hierarchy expected by the perfetto converter.
    This function creates virtual root and thread spans to contain them:
      - Virtual root:  "hicache"
      - Virtual threads: one per unique pid (each hicache thread has its own pid)
      - Slices:        the actual hicache operation spans

    Since hicache_trace sets module/pid/host_id/tp_rank/dp_rank/pp_rank on
    every span, virtual thread attributes are inherited from the first child.
    Spans are grouped by their actual pid so that each background thread
    (scheduler, backup, prefetch, io_aux) lands on its own row.
    """
    if not hicache_spans:
        return []

    # Group by actual pid — each pid is one hicache thread
    pid_groups = defaultdict(list)
    for span in hicache_spans:
        pid = span.get("attributes", {}).get("pid", 0)
        pid_groups[pid].append(span)

    root_span = {
        "name": "hicache",
        "spanId": "hicache_virtual_root",
        "startTimeUnixNano": "0",
        "endTimeUnixNano": "0",
        "attributes": {"module": "sglang::hicache"},
        "child": [],
    }

    for pid, spans in pid_groups.items():
        # Inherit all thread-level attrs from the first child span
        first_attrs = spans[0].get("attributes", {})
        thread_attrs = {
            "pid": pid,
            "host_id": first_attrs.get("host_id", "hicache"),
            "thread_label": first_attrs.get("thread_label", "hicache"),
        }
        for rank_key in ("tp_rank", "dp_rank", "pp_rank"):
            if rank_key in first_attrs:
                thread_attrs[rank_key] = first_attrs[rank_key]

        thread_span = {
            "name": f"hicache::{thread_attrs['thread_label']}",
            "spanId": f"hicache_virtual_thread_{pid}",
            "parentSpanId": "hicache_virtual_root",
            "startTimeUnixNano": "0",
            "endTimeUnixNano": "0",
            "attributes": thread_attrs,
            "child": spans,
        }
        root_span["child"].append(thread_span)

    return [root_span]


def load_otel_data(path: str | Path):
    p = Path(path)
    with p.open("rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)  # JSON array
        else:
            data = [json.loads(line) for line in f if line.strip()]  # JSONL
    return data


def extract_all_otel_spans(otel_data):
    engine_otel_spans = []
    smg_otel_spans = []
    for line_data in otel_data:
        for resource_spans in line_data["resourceSpans"]:
            # filter: only keep spans which service.name is 'sglang' or 'smg'
            service_name = ""
            for attr in resource_spans["resource"]["attributes"]:
                if attr["key"] == "service.name":
                    service_name = attr["value"]["stringValue"]

            if service_name == "sglang":
                spans_ref = engine_otel_spans
            elif service_name == "smg":
                spans_ref = smg_otel_spans
            else:
                continue

            for scope_spans in resource_spans["scopeSpans"]:
                for span in scope_spans["spans"]:
                    if "attributes" in span:
                        attributes_dict = {
                            attr.get("key"): next(
                                iter(attr.get("value", {}).values()), None
                            )
                            for attr in span["attributes"]
                        }
                        span["attributes"] = attributes_dict
                    else:
                        span["attributes"] = {}
                    spans_ref.append(span)
    return engine_otel_spans, smg_otel_spans


def build_otel_span_tree(otel_spans):
    span_id_map = {span["spanId"]: span for span in otel_spans}
    for span in otel_spans:
        span["child"] = []

    root_spans = []
    hicache_spans = []

    for span in otel_spans:
        parent_span_id = span.get("parentSpanId", "")
        module_name = span.get("attributes", {}).get("module", "")
        if module_name == "sglang::request" or module_name == "sglang::mooncake":
            root_spans.append(span)
        elif _is_hicache_span(span):
            hicache_spans.append(span)
        elif parent_span_id in span_id_map:
            parent_span = span_id_map[parent_span_id]
            parent_span["child"].append(span)

        link_spans = []
        if "links" in span:
            for link in span["links"]:
                link_span = span_id_map.get(link["spanId"])
                if link_span:
                    link_spans.append(link_span)
            span["links"] = link_spans

    # Build hicache virtual tree and add to root_spans
    root_spans.extend(_build_hicache_span_tree(hicache_spans))

    return root_spans


def __convert_to_perfetto_span(span, rid, bootstrap_room, pid, host_id):
    if bootstrap_room:
        span["attributes"]["bootstrap_room"] = bootstrap_room
    if rid:
        span["attributes"]["rid"] = rid
    if host_id:
        span["host_id"] = host_id
    span["pid"] = pid

    span["startTimeUnixNano"] = int(span["startTimeUnixNano"])
    span["endTimeUnixNano"] = int(span["endTimeUnixNano"]) - 1000
    ts = span["startTimeUnixNano"]
    dur = span["endTimeUnixNano"] - ts
    if dur <= 0:
        dur = 1000  # minimum 1μs for visibility (e.g. hicache instant spans)
        span["endTimeUnixNano"] = ts + 1000

    perfetto_span = {
        "ph": "X",
        "name": span.get("name", "unknown"),
        "cat": "sglang",
        "ts": (ts - baseline) / 1000.0,
        "dur": dur / 1000.0,
        "pid": pid,
        "tid": 0,
        "args": span["attributes"],
    }

    span["perfetto_span"] = perfetto_span

    for child_span in span["child"]:
        __convert_to_perfetto_span(child_span, rid, bootstrap_room, pid, host_id)


def generate_perfetto_span(engine_root_spans, smg_otel_spans, thread_meta_data):
    for root_span in engine_root_spans:
        root_span["spans"] = []

        rid = root_span["attributes"].get("rid", "")
        bootstrap_room = root_span["attributes"].get("bootstrap_room", "")

        for thread_span in root_span["child"]:
            pid = int(thread_span["attributes"]["pid"])
            host_id = thread_span["attributes"]["host_id"]
            thread_name = f'{thread_span["attributes"]["host_id"][:8]}:{thread_span["attributes"]["thread_label"]}'
            if "pp_rank" in thread_span["attributes"]:
                thread_name += f"-PP{thread_span['attributes']['pp_rank']}"
            if "dp_rank" in thread_span["attributes"]:
                thread_name += f"-DP{thread_span['attributes']['dp_rank']}"
            if "tp_rank" in thread_span["attributes"]:
                thread_name += f"-TP{thread_span['attributes']['tp_rank']}"

            if pid not in thread_meta_data:
                thread_meta_data[pid] = new_metadata_level1(thread_name, pid)

            for span in thread_span["child"]:
                __convert_to_perfetto_span(span, rid, bootstrap_room, pid, host_id)
                root_span["spans"].append(span)

    smg_pid = "smg"
    thread_meta_data[smg_pid] = new_metadata_level1("smg", smg_pid)
    for span in smg_otel_spans:
        span["pid"] = smg_pid
        span["child"] = []
        __convert_to_perfetto_span(span, None, None, smg_pid, None)


def __set_span_tid(span, line):
    span["perfetto_span"]["tid"] = line

    for child_span in span["child"]:
        __set_span_tid(child_span, line)


def generate_perfetto_span_layout(engine_root_spans, smg_otel_spans, slot_meta_data):
    for root_span in engine_root_spans:
        root_span["spans"] = sorted(
            root_span["spans"], key=lambda x: int(x["startTimeUnixNano"])
        )

    engine_root_spans = sorted(
        engine_root_spans, key=lambda x: int(x["spans"][0]["startTimeUnixNano"])
    )
    graph = {}
    for root_span in engine_root_spans:
        req_thread_status = {}
        for span in root_span["spans"]:
            line = __find_line(
                graph,
                req_thread_status,
                slot_meta_data,
                span["perfetto_span"]["pid"],
                span["startTimeUnixNano"],
                span["endTimeUnixNano"],
            )
            graph[span["perfetto_span"]["pid"]][line].insert_span(
                span["startTimeUnixNano"], span["endTimeUnixNano"]
            )
            __set_span_tid(span, line)

    smg_otel_spans = sorted(smg_otel_spans, key=lambda x: int(x["startTimeUnixNano"]))
    req_thread_status = {}
    for span in smg_otel_spans:
        line = __find_line(
            graph,
            req_thread_status,
            slot_meta_data,
            span["perfetto_span"]["pid"],
            span["startTimeUnixNano"],
            span["endTimeUnixNano"],
        )
        graph[span["perfetto_span"]["pid"]][line].insert_span(
            span["startTimeUnixNano"], span["endTimeUnixNano"]
        )
        span["perfetto_span"]["tid"] = line


def __convert_to_perfetto_events(span):
    span["perfetto_events"] = []
    if "events" in span:
        for event in span["events"]:
            attributes_dict = {
                attr.get("key"): next(iter(attr.get("value", {}).values()), None)
                for attr in event["attributes"]
            }
            perfetto_event = {
                "ph": "i",
                "cat": "sglang",
                "ts": (int(event["timeUnixNano"]) - baseline) / 1000.0,
                "pid": span["perfetto_span"]["pid"],
                "tid": span["perfetto_span"]["tid"],
                "name": event.get("name", "unknown"),
                "args": attributes_dict,
            }

            span["perfetto_events"].append(perfetto_event)

    for child_span in span["child"]:
        __convert_to_perfetto_events(child_span)


def generate_perfetto_events(engine_root_spans, smg_otel_spans):
    spans = [span for root_span in engine_root_spans for span in root_span["spans"]]

    for span in spans:
        __convert_to_perfetto_events(span)

    for span in smg_otel_spans:
        __convert_to_perfetto_events(span)


def generate_perfetto_links(engine_root_spans, smg_otel_spans):
    # build link between engine span and smg span
    span_id_map = {span["spanId"]: span for span in smg_otel_spans}

    for root_span in engine_root_spans:
        if "parentSpanId" in root_span and root_span["parentSpanId"] in span_id_map:
            parent_span = span_id_map[root_span["parentSpanId"]]
            root_span["spans"][0]["links"] = [parent_span]

        for span in root_span["spans"]:
            span["perfetto_links"] = []

            if "links" in span:
                for link_span in span["links"]:
                    try:
                        link_perfetto_span = link_span["perfetto_span"]
                    except (KeyError, AttributeError):
                        continue

                    if "correlation" in link_perfetto_span["args"]:
                        id = link_perfetto_span["args"]["correlation"]
                    else:
                        id = next(relation_id_gen)
                        link_perfetto_span["args"]["correlation"] = id

                    perfetto_start_node = {
                        "ph": "s",
                        "id": id,
                        "pid": link_perfetto_span["pid"],
                        "tid": link_perfetto_span["tid"],
                        "ts": link_perfetto_span["ts"],
                        "cat": "ac2g",
                        "name": "ac2g",
                    }

                    perfetto_end_node = {
                        "ph": "f",
                        "id": id,
                        "pid": span["perfetto_span"]["pid"],
                        "tid": span["perfetto_span"]["tid"],
                        "ts": span["perfetto_span"]["ts"],
                        "cat": "ac2g",
                        "name": "ac2g",
                        "bp": "e",
                    }

                    span["perfetto_links"].append(perfetto_start_node)
                    span["perfetto_links"].append(perfetto_end_node)


def __gather_one_span(span):
    elems = []
    elems.append(span["perfetto_span"])
    if "perfetto_events" in span:
        elems.extend(span["perfetto_events"])
    if "perfetto_links" in span:
        elems.extend(span["perfetto_links"])

    for child_span in span["child"]:
        elems.extend(__gather_one_span(child_span))

    return elems


def gather_all_perfetto_elems(
    engine_root_spans, smg_otel_spans, thread_meta_data, slot_meta_data
):
    elems = []
    elems.extend(thread_meta_data.values())
    elems.extend(slot_meta_data)
    for root_span in engine_root_spans:
        for span in root_span["spans"]:
            elems.extend(__gather_one_span(span))

    for span in smg_otel_spans:
        elems.append(span["perfetto_span"])
        elems.extend(span["perfetto_events"])

    return elems


def write_json(perfetto_elems):
    global perfetto_data

    if args.torch_file:
        perfetto_data["traceEvents"].extend(perfetto_elems)
        filered_data = [
            item
            for item in perfetto_data["traceEvents"]
            if item.get("cat") != "gpu_user_annotation"
        ]
        perfetto_data["traceEvents"] = filered_data
    else:
        perfetto_data = perfetto_elems

    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(perfetto_data, file, ensure_ascii=False, indent=4)


def main():
    start_time = time.time()
    otel_data = load_otel_data(args.input_file)
    engine_otel_spans, smg_otel_spans = extract_all_otel_spans(otel_data)
    engine_root_spans = build_otel_span_tree(engine_otel_spans)
    thread_meta_data = {}
    generate_perfetto_span(engine_root_spans, smg_otel_spans, thread_meta_data)
    slot_meta_data = []
    generate_perfetto_span_layout(engine_root_spans, smg_otel_spans, slot_meta_data)
    generate_perfetto_events(engine_root_spans, smg_otel_spans)
    generate_perfetto_links(engine_root_spans, smg_otel_spans)
    perfetto_elems = gather_all_perfetto_elems(
        engine_root_spans, smg_otel_spans, thread_meta_data, slot_meta_data
    )
    write_json(perfetto_elems)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nConversion finished successfully!")
    print(f"Output written to: {args.output_file}")
    print(f"Execution time: {execution_time * 1000:.4f} ms")


if __name__ == "__main__":
    main()
