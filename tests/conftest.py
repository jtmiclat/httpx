import asyncio
import functools
import inspect

import pytest
import trustme
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    PrivateFormat,
)
from uvicorn.config import Config
from uvicorn.main import Server

from httpx import AsyncioBackend

backend_classes = [AsyncioBackend]

try:
    from httpx.concurrency.trio import TrioBackend
except ImportError:
    pass  # pragma: no cover
else:
    backend_classes.append(TrioBackend)


@pytest.fixture(
    # All parameters are marked with pytest.mark.asyncio because we need an
    # event loop set up in all cases (either for direct use, or to be able to run
    # things in the threadpool).
    params=[pytest.param(cls, marks=pytest.mark.asyncio) for cls in backend_classes]
)
def backend(request):
    backend_cls = request.param
    return backend_cls()


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """
    Test functions that use a concurrency backend other than asyncio must be run
    in a separate thread to avoid event loop clashes.
    """
    if "backend" not in pyfuncitem.fixturenames:
        return

    backend = pyfuncitem.funcargs["backend"]
    assert backend is not None

    if isinstance(backend, AsyncioBackend):
        return

    func = pyfuncitem.obj
    assert inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    async def wrapped(**kwargs):
        asyncio_backend = AsyncioBackend()
        await asyncio_backend.run_in_threadpool(backend.run, func, **kwargs)

    pyfuncitem.obj = wrapped


async def app(scope, receive, send):
    assert scope["type"] == "http"
    if scope["path"] == "/slow_response":
        await slow_response(scope, receive, send)
    elif scope["path"].startswith("/status"):
        await status_code(scope, receive, send)
    elif scope["path"].startswith("/echo_body"):
        await echo_body(scope, receive, send)
    else:
        await hello_world(scope, receive, send)


async def hello_world(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        }
    )
    await send({"type": "http.response.body", "body": b"Hello, world!"})


async def slow_response(scope, receive, send):
    await asyncio.sleep(0.1)
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        }
    )
    await send({"type": "http.response.body", "body": b"Hello, world!"})


async def status_code(scope, receive, send):
    status_code = int(scope["path"].replace("/status/", ""))
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [[b"content-type", b"text/plain"]],
        }
    )
    await send({"type": "http.response.body", "body": b"Hello, world!"})


async def echo_body(scope, receive, send):
    body = b""
    more_body = True

    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        }
    )
    await send({"type": "http.response.body", "body": body})


class CAWithPKEncryption(trustme.CA):
    """Implementation of trustme.CA() that can emit
    private keys that are encrypted with a password.
    """

    @property
    def encrypted_private_key_pem(self):
        return trustme.Blob(
            self._private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.TraditionalOpenSSL,
                BestAvailableEncryption(password=b"password"),
            )
        )


@pytest.fixture
def example_cert():
    ca = CAWithPKEncryption()
    ca.issue_cert("example.org")
    return ca


@pytest.fixture
def cert_pem_file(example_cert):
    with example_cert.cert_pem.tempfile() as tmp:
        yield tmp


@pytest.fixture
def cert_private_key_file(example_cert):
    with example_cert.private_key_pem.tempfile() as tmp:
        yield tmp


@pytest.fixture
def cert_encrypted_private_key_file(example_cert):
    with example_cert.encrypted_private_key_pem.tempfile() as tmp:
        yield tmp


@pytest.fixture
def restart_queue():
    return asyncio.Queue()


@pytest.fixture
def restart_server(restart_queue, backend):
    """Restart the running server from an async test function."""

    async def asyncio_restart():
        await restart_queue.put(None)
        await restart_queue.get()

    if isinstance(backend, AsyncioBackend):
        return asyncio_restart

    async def restart():
        await backend.run_in_threadpool(AsyncioBackend().run, asyncio_restart)

    return restart


async def watch_restarts(server, queue):
    # We can't let async tests run shutdown()/startup() directly because:
    # * They may be running in a different async environment (e.g. trio).
    # * When that is the case, that environment runs in a separate thread.
    # * But shutdown/startup() needs to run on the loop from which the server was
    # spawned, i.e. the one we're currently in.
    # * Even if we launched `AsyncioBackend().run` in the threadpool, that would run
    # on a *new event loop - in yet another thread!
    # So we use a queue as a thread-safe communication channel with async tests.

    while True:
        try:
            await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if server.should_exit:
                break
            continue

        await server.shutdown()
        await server.startup()

        await queue.put(None)


@pytest.fixture
async def server(restart_queue):
    config = Config(app=app, lifespan="off")
    server = Server(config=config)
    tasks = {
        asyncio.ensure_future(server.serve()),
        asyncio.ensure_future(watch_restarts(server, restart_queue)),
    }
    try:
        while not server.started:
            await asyncio.sleep(0.0001)
        yield server
    finally:
        server.should_exit = True
        await asyncio.wait(tasks)


@pytest.fixture
async def https_server(cert_pem_file, cert_private_key_file):
    config = Config(
        app=app,
        lifespan="off",
        ssl_certfile=cert_pem_file,
        ssl_keyfile=cert_private_key_file,
        port=8001,
    )
    server = Server(config=config)
    tasks = {
        asyncio.ensure_future(server.serve()),
        asyncio.ensure_future(watch_restarts(server, restart_queue)),
    }
    try:
        while not server.started:
            await asyncio.sleep(0.0001)
        yield server
    finally:
        server.should_exit = True
        await asyncio.wait(tasks)
