import asyncio
import concurrent.futures
import time

import pytest

from aflux_batch import arun_batch, run_batch


@pytest.fixture
def anyio_backend():
    return "asyncio"


def sync_slow_task(index: int, sleep_time: float) -> int:
    time.sleep(sleep_time)
    return index


async def async_slow_task(index: int, sleep_time: float) -> int:
    await asyncio.sleep(sleep_time)
    return index


class TestSyncBatch:
    def test_order_preservation(self) -> None:
        kwargs_list = [
            {"index": 0, "sleep_time": 0.03},
            {"index": 1, "sleep_time": 0.02},
            {"index": 2, "sleep_time": 0.01},
            {"index": 3, "sleep_time": 0.00},
        ]

        batch_size = 4
        with concurrent.futures.ThreadPoolExecutor(batch_size) as executor:
            results = run_batch(executor, sync_slow_task, kwargs_list, batch_size)

        assert results == [0, 1, 2, 3]

    def test_iterator_handling(self) -> None:
        kwargs_iter = ({"index": i, "sleep_time": max(0.0, 0.03 - 0.01 * i)} for i in range(4))

        batch_size = 4
        with concurrent.futures.ThreadPoolExecutor(batch_size) as executor:
            results = run_batch(executor, sync_slow_task, kwargs_iter, batch_size)

        assert results == [0, 1, 2, 3]


class TestAsyncBatch:
    @pytest.mark.anyio
    async def test_order_preservation(self) -> None:
        kwargs_list = [
            {"index": 0, "sleep_time": 0.03},
            {"index": 1, "sleep_time": 0.02},
            {"index": 2, "sleep_time": 0.01},
            {"index": 3, "sleep_time": 0.00},
        ]

        async with asyncio.TaskGroup() as task_group:
            results = await arun_batch(task_group, async_slow_task, kwargs_list, batch_size=4)

        assert results == [0, 1, 2, 3]

    @pytest.mark.anyio
    async def test_iterator_handling(self) -> None:
        kwargs_iter = ({"index": i, "sleep_time": max(0.0, 0.03 - 0.01 * i)} for i in range(4))

        async with asyncio.TaskGroup() as task_group:
            results = await arun_batch(task_group, async_slow_task, kwargs_iter, batch_size=4)

        assert results == [0, 1, 2, 3]
