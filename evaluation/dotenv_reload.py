"""Nạp lại `.env` vào `os.environ` mỗi lần gọi (`override=True`)."""
from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def reload_dotenv() -> Path | None:
    """
    Tìm `.env` từ thư mục làm việc hiện tại trở lên, đọc lại file (ghi đè biến đã có).
    Trả về đường dẫn tuyệt đối tới file `.env` nếu tìm thấy.
    """
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        p = Path.cwd().resolve()
        for _ in range(32):
            cand = p / ".env"
            if cand.is_file():
                dotenv_path = str(cand.resolve())
                break
            if p.parent == p:
                break
            p = p.parent
    if not dotenv_path:
        return None
    path = Path(dotenv_path).resolve()
    load_dotenv(path, override=True, encoding="utf-8")
    return path
