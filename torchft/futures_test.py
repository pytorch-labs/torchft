from datetime import timedelta
from unittest import TestCase

from torch.futures import Future

from torchft.futures import future_timeout, future_wait


class FuturesTest(TestCase):
    def test_future_wait(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            future_wait(fut, timeout=timedelta(seconds=0.01))

        # pyre-fixme[29]: Future is not a function
        fut = Future()
        fut.set_result(1)
        self.assertEqual(future_wait(fut, timeout=timedelta(seconds=1.0)), 1)

        # pyre-fixme[29]: Future is not a function
        fut = Future()
        fut.set_exception(RuntimeError("test"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            future_wait(fut, timeout=timedelta(seconds=1.0))

    def test_future_timeout(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(TimeoutError, "future did not complete within"):
            timed_fut.wait()

    def test_future_timeout_result(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=10))
        fut.set_result(1)
        self.assertEqual(timed_fut.wait(), 1)

    def test_future_timeout_exception(self) -> None:
        # pyre-fixme[29]: Future is not a function
        fut = Future()
        timed_fut = future_timeout(fut, timeout=timedelta(seconds=10))
        fut.set_exception(RuntimeError("test"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            timed_fut.wait()
