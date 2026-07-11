import asyncio
import concurrent.futures
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from concurrent.futures import (
    Executor,
    Future,
    ThreadPoolExecutor,
)
from contextlib import AsyncExitStack, ExitStack
from typing import Any, cast


class Payload:
    __slots__ = ("args", "kwargs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _sync_indexed_wrapper[T](
    index: int,
    func: Callable[..., T],
    payload: Payload,
) -> tuple[int, T]:
    return index, func(*payload.args, **payload.kwargs)


def iter_batch[T](
    func: Callable[..., T],
    payloads: Iterable[Payload | tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    executor: Executor,
    batch_size: int = 32,
) -> Iterator[tuple[int, T]]:
    batch_size = max(1, batch_size)

    pending: set[Future[tuple[int, T]]] = set()
    done: set[Future[tuple[int, T]]]

    for i, payload in enumerate(payloads):
        if not isinstance(payload, Payload):
            args, kwargs = payload
            payload = Payload(*args, **kwargs)

        while len(pending) >= batch_size:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                yield future.result()

        future = executor.submit(_sync_indexed_wrapper, i, func, payload)
        pending.add(future)  # ty: ignore[invalid-argument-type]

    for future in concurrent.futures.as_completed(pending):
        yield future.result()


def run_batch[T](
    func: Callable[..., T],
    payloads: Iterable[Payload | tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    executor: Executor | None = None,
    batch_size: int = 32,
) -> list[T]:
    batch_size = max(1, batch_size)

    with ExitStack() as stack:
        if executor is None:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=batch_size))

        results: list[T | None] = []
        for index, result in iter_batch(func, payloads, executor=executor, batch_size=batch_size):
            while len(results) <= index:
                results.append(None)
            results[index] = result

        return cast(list[T], results)


async def _async_indexed_wrapper[T](
    index: int,
    func: Callable[..., Awaitable[T]],
    payload: Payload,
) -> tuple[int, T]:
    return index, await func(*payload.args, **payload.kwargs)


async def aiter_batch[T](
    func: Callable[..., Awaitable[T]],
    payloads: Iterable[Payload | tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    task_group: asyncio.TaskGroup,
    batch_size: int = 32,
) -> AsyncIterator[tuple[int, T]]:
    batch_size = max(1, batch_size)

    pending: set[asyncio.Task[tuple[int, T]]] = set()
    done: set[asyncio.Task[tuple[int, T]]]

    for i, payload in enumerate(payloads):
        if not isinstance(payload, Payload):
            args, kwargs = payload
            payload = Payload(*args, **kwargs)

        while len(pending) >= batch_size:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                yield task.result()

        task = task_group.create_task(_async_indexed_wrapper(i, func, payload))
        pending.add(task)

    for coro in asyncio.as_completed(pending):
        yield await coro


async def arun_batch[T](
    func: Callable[..., Awaitable[T]],
    payloads: Iterable[Payload | tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    task_group: asyncio.TaskGroup | None = None,
    batch_size: int = 32,
) -> list[T]:
    batch_size = max(1, batch_size)

    async with AsyncExitStack() as stack:
        if task_group is None:
            task_group = await stack.enter_async_context(asyncio.TaskGroup())

        results: list[T | None] = []
        async for index, result in aiter_batch(func, payloads, task_group=task_group, batch_size=batch_size):
            while len(results) <= index:
                results.append(None)
            results[index] = result

        return cast(list[T], results)
