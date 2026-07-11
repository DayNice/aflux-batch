import asyncio
import concurrent.futures
import threading
import time

import pytest

from aflux_batch import arun_batch, run_batch


@pytest.fixture
def anyio_backend():
    return "asyncio"


class MockWorkload:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def sync_task(self, index: int, sleep_time: float) -> int:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(sleep_time)
        with self.lock:
            self.active -= 1
        return index

    async def async_task(self, index: int, sleep_time: float) -> int:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(sleep_time)
        with self.lock:
            self.active -= 1
        return index


class TestSyncBatch:
    def test_with_list(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = [{"index": i, "sleep_time": 0.05 if i % 2 == 0 else 0.0} for i in range(20)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = run_batch(workload.sync_task, kwargs_iterable, executor=executor, batch_size=4)

        assert results == list(range(20))
        assert workload.max_active == 4

    def test_with_iterator(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = ({"index": i, "sleep_time": 0.05 if i % 2 == 0 else 0.0} for i in range(20))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = run_batch(workload.sync_task, kwargs_iterable, executor=executor, batch_size=4)

        assert results == list(range(20))
        assert workload.max_active == 4

    def test_default_executor(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = [{"index": i, "sleep_time": 0.0} for i in range(5)]
        results = run_batch(workload.sync_task, kwargs_iterable, batch_size=4)
        assert results == list(range(5))


class TestAsyncBatch:
    @pytest.mark.anyio
    async def test_with_list(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = [{"index": i, "sleep_time": 0.05 if i % 2 == 0 else 0.0} for i in range(20)]

        async with asyncio.TaskGroup() as task_group:
            results = await arun_batch(workload.async_task, kwargs_iterable, task_group=task_group, batch_size=4)

        assert results == list(range(20))
        assert workload.max_active == 4

    @pytest.mark.anyio
    async def test_with_iterator(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = ({"index": i, "sleep_time": 0.05 if i % 2 == 0 else 0.0} for i in range(20))

        async with asyncio.TaskGroup() as task_group:
            results = await arun_batch(workload.async_task, kwargs_iterable, task_group=task_group, batch_size=4)

        assert results == list(range(20))
        assert workload.max_active == 4

    @pytest.mark.anyio
    async def test_default_executor(self) -> None:
        workload = MockWorkload()
        kwargs_iterable = [{"index": i, "sleep_time": 0.0} for i in range(5)]
        results = await arun_batch(workload.async_task, kwargs_iterable, batch_size=4)
        assert results == list(range(5))
