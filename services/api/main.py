from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from routers import alerts, assets, digital_twin, energy_flow, system

app = FastAPI(title="UrjaNet API", version="0.1.0")

app.include_router(assets.router)
app.include_router(alerts.router)
app.include_router(digital_twin.router)
app.include_router(energy_flow.router)
app.include_router(system.router)


@app.get("/ping")
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
