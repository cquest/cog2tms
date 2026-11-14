from functools import lru_cache
from io import BytesIO

from PIL import Image


@lru_cache(maxsize=16)
def empty_tile(fmt: str, size: int, bg: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    if fmt == "png":
        Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(buf, "PNG")
    elif fmt == "webp":
        Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(buf, "WEBP", exact=True)
    else:
        Image.new("RGB", (size, size), bg).save(buf, "JPEG", quality=80)
    return buf.getvalue()
