import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


# Helpers / parsing

def ensure_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def ensure_json_str(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return "{}"


def parse_wrapper(trace_cell: Any) -> Tuple[Dict[str, Any], bool]:
    was_string = isinstance(trace_cell, str)
    wrapper = ensure_dict(trace_cell)
    if "json_raw" in wrapper:
        wrapper["json_raw"] = ensure_json_str(wrapper["json_raw"])
    return wrapper, was_string


def parse_inner(wrapper: Dict[str, Any]) -> Dict[str, Any]:
    return ensure_dict(wrapper.get("json_raw", "{}"))


def get_session_uuid(wrapper: Dict[str, Any], inner: Dict[str, Any]) -> str:
    v = wrapper.get("session_uuid") or inner.get("session_uuid") or ""
    return str(v)


def get_timestamp_str(wrapper: Dict[str, Any], inner: Dict[str, Any]) -> str:
    v = wrapper.get("timestamp") or inner.get("timestamp") or ""
    return str(v)


def parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def remove_timestamps(wrapper: Dict[str, Any], inner: Dict[str, Any]) -> None:
    wrapper.pop("timestamp", None)
    inner.pop("timestamp", None)


def finalize_trace_cell(wrapper: Dict[str, Any], inner: Dict[str, Any], trace_was_str: bool) -> Any:
    wrapper["json_raw"] = json.dumps(inner, ensure_ascii=False)
    return json.dumps(wrapper, ensure_ascii=False) if trace_was_str else wrapper


# Internal records, sorting, output

def build_internal_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict("records")
    events: List[Dict[str, Any]] = []

    for idx, row in enumerate(records):
        wrapper, was_str = parse_wrapper(row.get("trace"))
        inner = parse_inner(wrapper)

        session_uuid = get_session_uuid(wrapper, inner)
        ts = parse_ts(get_timestamp_str(wrapper, inner))

        row_out = dict(row)
        row_out["_idx"] = idx
        row_out["_wrapper"] = wrapper
        row_out["_inner"] = inner
        row_out["_trace_was_str"] = was_str
        row_out["_session_uuid"] = session_uuid
        row_out["_ts"] = ts
        events.append(row_out)

    return events


def sort_events_by_timestamp_only(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        events,
        key=lambda e: (
            e["_ts"] if e["_ts"] is not None else datetime.min,
            e["_idx"],  # stable tiebreaker when timestamps collide or are missing
        ),
    )


def strip_internal_fields(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drop = {"_idx", "_wrapper", "_inner", "_trace_was_str", "_session_uuid", "_ts"}
    return [{k: v for k, v in e.items() if k not in drop} for e in events]


def remove_timestamps_and_rebuild_trace(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in events:
        wrapper = dict(e["_wrapper"])
        inner = dict(e["_inner"])
        remove_timestamps(wrapper, inner)

        rebuilt = dict(e)
        rebuilt["_wrapper"] = wrapper
        rebuilt["_inner"] = inner
        rebuilt["trace"] = finalize_trace_cell(wrapper, inner, e["_trace_was_str"])
        out.append(rebuilt)
    return out


def add_seq_column(df: pd.DataFrame) -> pd.DataFrame:
    def _sess(trace_cell: Any) -> str:
        w, _ = parse_wrapper(trace_cell)
        inner = parse_inner(w)
        return get_session_uuid(w, inner)

    sess = df["trace"].apply(_sess)
    df = df.copy()
    df.insert(0, "seq", sess.groupby(sess).cumcount() + 1)
    return df


# Main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing extracted parquet file")
    parser.add_argument("--output", required=True, help="Output directory for integrated parquet file")
    args = parser.parse_args()

    # Look for traces parquet first, then fall back to any parquet
    target = None
    
    # Priority 1: Look for traces file
    for root, _, files in os.walk(args.input):
        for file in sorted(files):
            if "traces" in file.lower() and file.endswith(".parquet"):
                target = os.path.join(root, file)
                break
        if target:
            break
    
    # Priority 2: Look for extracted_data.parquet
    if not target:
        fallback = os.path.join(args.input, "extracted_data.parquet")
        if os.path.exists(fallback):
            target = fallback
    
    # Priority 3: Find any parquet
    if not target:
        candidates = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith(".parquet"):
                    candidates.append(os.path.join(root, file))
        
        if not candidates:
            raise FileNotFoundError(f"No parquet file found in {args.input}")
        candidates.sort()
        target = candidates[0]
    
    df = pd.read_parquet(target)
    if "trace" not in df.columns:
        raise ValueError(f"Input parquet must contain a 'trace' column (found: {', '.join(df.columns)})")

    events = build_internal_events(df)
    events = sort_events_by_timestamp_only(events)
    events = remove_timestamps_and_rebuild_trace(events)

    out_df = pd.DataFrame(strip_internal_fields(events))

    if "timestamp" in out_df.columns:
        out_df = out_df.drop(columns=["timestamp"])

    out_df = add_seq_column(out_df)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "integrated_data.parquet")
    out_df.to_parquet(out_path, engine="pyarrow")


if __name__ == "__main__":
    main()
