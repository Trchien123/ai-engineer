"""
Router for traffic sign metadata lookup.
Searches the sign metadata JSON by Descriptions field.
"""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config import settings

router = APIRouter(prefix="/api", tags=["sign-info"])

METADATA_PATH: Path = settings.DAMAGED_SIGN_DIR / settings.TRAFFIC_SIGN_RETRIEVER_METADATA


@lru_cache(maxsize=1)
def _load_metadata() -> list:
    """Load and cache the sign metadata JSON."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/sign-info")
async def get_sign_info(
    name: str = Query(..., description="Sign description text (Descriptions field)")
):
    """
    Look up traffic sign metadata by its Descriptions value.

    Returns the first JSON record whose Descriptions field matches (case-insensitive).
    Fields returned: category, Sign No, Descriptions, Standard sign?,
    Use by council, Legislative Reference, Primary Technical Reference,
    original_url, name.
    """
    try:
        metadata = _load_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Sign metadata file not found")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load sign metadata: {e}")

    name_lower = name.strip().lower()

    # Exact match first
    for record in metadata:
        if record.get("Descriptions", "").strip().lower() == name_lower:
            return _clean_record(record)

    # Partial match fallback
    for record in metadata:
        if name_lower in record.get("Descriptions", "").lower():
            return _clean_record(record)

    raise HTTPException(status_code=404, detail=f"No sign found matching: {name!r}")


def _clean_record(record: dict) -> dict:
    """Return a copy of the record without the local_image_path field."""
    return {k: v for k, v in record.items() if k != "local_image_path"}
