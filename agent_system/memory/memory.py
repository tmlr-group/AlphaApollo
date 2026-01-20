# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import BaseMemory

# Precompiled patterns for efficient score extraction
SCORE_JSON_KEY_RE = re.compile(r"\"score\"\s*:\s*(-?\d+(?:\.\d+)?)")
SCORE_TEXT_RE = re.compile(r"\bScore:\s*(-?\d+(?:\.\d+)?)")

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Action {step_num}: '{act}', Observation {step_num}: '{obs}']"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    

class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

class OrderedRecordList(list):
    """
    List-like container that keeps records ordered by a specified key.
    """

    def __init__(
        self,
        sort_key: Optional[str],
        descending: bool = True,
        missing_value: Optional[float] = None,
    ):
        super().__init__()
        self.sort_key = sort_key
        self.descending = descending
        self.missing_value = missing_value
        self.ordered = sort_key is not None

    def _default_value(self) -> float:
        if self.missing_value is not None:
            return self.missing_value
        return float("-inf") if self.descending else float("inf")

    def _coerce(self, record: Dict[str, Any]) -> float:
        if not self.ordered:
            return 0.0
        value = None
        if isinstance(record, dict) and self.sort_key is not None:
            value = record.get(self.sort_key)
        if isinstance(value, (int, float)):
            return float(value)
        if value is not None:
            if not isinstance(value, (int, float)):
                return self._default_value()
            return float(value)
        return self._default_value()

    def append(self, record: Dict[str, Any]): 
        if not self.ordered:
            super().append(record)
            return

        sort_value = self._coerce(record)
        insert_idx = len(self)
        for idx, existing in enumerate(self):
            existing_value = self._coerce(existing)
            if self.descending:
                if sort_value > existing_value:
                    insert_idx = idx
                    break
            else:
                if sort_value < existing_value:
                    insert_idx = idx
                    break
        super().insert(insert_idx, record)

    def extend(self, records: Iterable[Dict[str, Any]]): 
        for record in records:
            self.append(record)


class NDimensionalSpaceList(list):
    """
    An n-dimensional grid for storing records.
    """
    def __init__(
        self,
        dims: List[str],
        performance_key: str = "score",
        complexity_key: str = "action",
    ):
        super().__init__()

        # Validate dimensions
        if not dims or not isinstance(dims, list):
            raise ValueError("dimensions must be a non-empty list")
        allowed_dims = {"performance", "complexity"}
        for d in dims:
            if d not in allowed_dims:
                raise ValueError(f"invalid dim '{d}', allowed: {allowed_dims}")

        # Initialize
        self._dims: List[str] = dims
        self._dim_ranks = {dim: [] for dim in allowed_dims}
        self._record_set = set()
        self._performance_key = performance_key
        self._complexity_key = complexity_key

    def _default_value(self, key: str) -> float:
        if key == self._performance_key:
            return 0.0
        elif key == self._complexity_key:
            return float("-inf")
        else:
            return float("-inf")

    def append(self, record: Dict[str, Any]) -> None:
        # Content-based deduplication
        content_key = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        if content_key in self._record_set:
            return
        self._record_set.add(content_key)
        for dim in self._dims:
            # Calculate the metric for each dimension.
            if dim == "performance":
                score = record.get(self._performance_key)
                if not isinstance(score, (int, float)):
                    return self._default_value(self._performance_key)
                value = float(score)
            elif dim == "complexity":
                action = record.get(self._complexity_key)
                # Use the negative length of the action to sort, since the longer the action, the worse it is.
                if not isinstance(action, str):
                    return self._default_value(self._complexity_key)
                value = -float(len(action))
            
            groups = self._dim_ranks[dim]
            inserted = False
            for idx, rank in enumerate(groups):
                key = next(iter(rank))
                if value == key:
                    rank[key].append(record)
                    inserted = True
                    break
                elif value > key:
                    groups.insert(idx, {value: [record]})
                    inserted = True
                    break
            if not inserted:
                groups.append({value: [record]})

    def extend(self, records: Iterable[Dict[str, Any]]):
        for record in records:
            self.append(record)

    def retrieve_records(self, history_length: int, strategy: str = "min_combined") -> List[Dict[str, Any]]:
        if strategy == "min_combined":
            return self.retrieve_records_min_combined(history_length)
        elif strategy == "random":
            return self.retrieve_records_random(history_length)
        else:
            raise ValueError(f"invalid strategy '{strategy}', allowed: 'min_combined', 'random'")

    def retrieve_records_min_combined(self, history_length: int) -> List[Dict[str, Any]]:
        if history_length <= 0:
            return []
        rank_sum_by_id: Dict[int, int] = {}
        record_by_id: Dict[int, Dict[str, Any]] = {}

        # Calculate the rank sum for each record
        for dim in self._dims:
            groups = self._dim_ranks[dim]
            for idx, rank in enumerate(groups):
                key = next(iter(rank))
                for rec in rank[key]:
                    content_key = json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str)
                    rid = hash(content_key)
                    rank_sum_by_id[rid] = rank_sum_by_id.get(rid, 0) + idx
                    if rid not in record_by_id:
                        record_by_id[rid] = rec
        if not rank_sum_by_id:
            return []
        
        # Sort the records by the rank sum, and return the top history_length records
        ordered = sorted(rank_sum_by_id.items(), key=lambda x: x[1])
        return [record_by_id[rid] for rid, _ in ordered[:history_length]]

    def retrieve_records_random(self, history_length: int) -> List[Dict[str, Any]]:
        if history_length <= 0 or not self._record_set:
            return []
        keys = list(self._record_set)
        random.shuffle(keys)
        keys = keys[:history_length]
        return [json.loads(k) for k in keys]

class EvolvingMemory(BaseMemory):
    """
    Memory manager for evolving tasks: stores records in a self-ordered list
    based on a configurable key (defaults to 'score').
    """

    def __init__(
        self,
        sort_key: Optional[str] = "score",
        descending: bool = True,
        missing_value: Optional[float] = None,
    ):
        self._data = None
        self.keys = None
        self.batch_size = 0
        self._sort_key = sort_key
        self._descending = descending
        self._missing_value = missing_value
        self._counter = 0

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [OrderedRecordList(
            self._sort_key, self._descending, self._missing_value
        ) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            # only track required per-step fields; ignore auxiliary like meta_info
            self.keys = [k for k in record.keys() if k != "meta_info"]
        else:
            missing = [k for k in self.keys if k not in record]
            assert not missing, f"EvolvingMemory.store() missing required keys: {missing}; got {list(record.keys())}"
        for env_idx in range(self.batch_size):
            entry: Dict[str, Any] = {}
            # copy non-meta_info fields per env
            for k in self.keys:
                entry[k] = record[k][env_idx]
            # extract score from top-level or meta_info
            entry[self._sort_key] = None
            if self._sort_key in record:
                entry[self._sort_key] = record[self._sort_key][env_idx]
            elif "meta_info" in record and isinstance(record["meta_info"], dict):
                meta = record["meta_info"]
                if self._sort_key in meta:
                    entry[self._sort_key] = meta[self._sort_key][env_idx]
            self._data[env_idx].append(entry)

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch the top-scoring interaction history across environment instances.
        Args:
            history_length (int):
                Maximum number of records to retrieve.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings, ordered by the sort key.
            valid_lengths : List[int]
                A list with the number of records represented in each context string.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            if not isinstance(self._data[env_idx], OrderedRecordList):
                memory_contexts.append("")
                valid_lengths.append(0)
                continue
            recent = self._data[env_idx][:history_length]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            contexts = []
            for j, rec in enumerate(recent):
                act = rec[action_key]
                obs = rec[obs_key]
                score = rec.get(self._sort_key) if self._sort_key else None
                contexts.append(
                    f"[Action: '{act}', Observation: '{obs}', Score: '{score}']"
                )

            memory_contexts.append("\n".join(contexts))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths


class NDimensionalMemory(BaseMemory):
    """
    Memory manager for evolving tasks: stores records in a n-dimensional memory space.
    """

    def __init__(
        self,
        dimensions: List[str] = [],
        performance_key: str = "score",
        complexity_key: str = "action",
    ):
        self._data = None
        self.keys = None
        self.batch_size = 0
        self._dimensions = dimensions
        self._performance_key = performance_key
        self._complexity_key = complexity_key
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [NDimensionalSpaceList(
            self._dimensions,
            self._performance_key,
            self._complexity_key,
        ) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            # only track required per-step fields; ignore auxiliary like meta_info
            self.keys = [k for k in record.keys() if k != "meta_info"]
        else:
            missing = [k for k in self.keys if k not in record]
            assert not missing, f"NDimensionalMemory.store() missing required keys: {missing}; got {list(record.keys())}"
        for env_idx in range(self.batch_size):
            entry: Dict[str, Any] = {}
            # copy non-meta_info fields per env
            for k in self.keys:
                entry[k] = record[k][env_idx]
            # extract score from top-level or meta_info
            for k in [self._performance_key, self._complexity_key]:
                entry[k] = None
                if k in record:
                    entry[k] = record[k][env_idx]
                elif "meta_info" in record and isinstance(record["meta_info"], dict):
                    meta = record["meta_info"]
                    if k in meta:
                        entry[k] = meta[k][env_idx]
            self._data[env_idx].append(entry)

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch the top-scoring interaction history across environment instances.
        Args:
            history_length (int):
                Maximum number of records to retrieve.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings, ordered by the sort key.
            valid_lengths : List[int]
                A list with the number of records represented in each context string.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            if not isinstance(self._data[env_idx], NDimensionalSpaceList):
                memory_contexts.append("")
                valid_lengths.append(0)
                continue
            
            sampled = self._data[env_idx].retrieve_records(history_length, strategy="min_combined")
            if not sampled:
                sampled = self._data[env_idx].retrieve_records(history_length, strategy="random")
            recent = sampled
            valid_len = len(recent)

            contexts = []
            for rec in recent:
                act = rec[action_key]
                obs = rec[obs_key]
                contexts.append(f"[Action: '{act}', Observation: '{obs}']")

            memory_contexts.append("\n".join(contexts))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
