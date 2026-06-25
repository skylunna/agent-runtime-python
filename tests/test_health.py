import pytest
from httpx import AsyncClient, ASGITransport

from cdss_rag.main import app


@pytest.mark.asyncio
async def test_healthz():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://tset"
    ) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}