"""Warstwa danych: SQLite + dataclass zadania."""
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "tasks.db"

STATUSES = ("todo", "in_progress", "done")
PRIORITIES = ("low", "normal", "high")


@dataclass
class Task:
    id: str
    description: str
    relation: Optional[str]
    status: str
    priority: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                relation    TEXT REFERENCES tasks(id) ON DELETE SET NULL,
                status      TEXT NOT NULL DEFAULT 'todo',
                priority    TEXT NOT NULL DEFAULT 'normal',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )"""
        )


def _row(r: sqlite3.Row) -> Task:
    return Task(**dict(r))


def list_tasks() -> list[Task]:
    with _conn() as c:
        return [_row(r) for r in c.execute("SELECT * FROM tasks ORDER BY created_at")]


def get_task(task_id: str) -> Optional[Task]:
    with _conn() as c:
        r = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row(r) if r else None


def add_task(description: str, priority: str = "normal", relation: Optional[str] = None) -> Task:
    if relation and not get_task(relation):
        raise ValueError(f"Zadanie nadrzędne `{relation}` nie istnieje.")
    task_id, now = str(uuid.uuid4()), _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?)",
            (task_id, description, relation, "todo", priority, now, now),
        )
    return get_task(task_id)


def update_task(task_id: str, description=None, status=None, priority=None, relation=None) -> Optional[Task]:
    task = get_task(task_id)
    if not task:
        return None
    if relation and (relation == task_id or not get_task(relation)):
        raise ValueError("Nieprawidłowe zadanie nadrzędne.")
    with _conn() as c:
        c.execute(
            "UPDATE tasks SET description=?, status=?, priority=?, relation=?, updated_at=? WHERE id=?",
            (
                description or task.description,
                status or task.status,
                priority or task.priority,
                relation or task.relation,
                _now(),
                task_id,
            ),
        )
    return get_task(task_id)


def delete_task(task_id: str) -> bool:
    with _conn() as c:
        c.execute("UPDATE tasks SET relation=NULL WHERE relation=?", (task_id,))
        return c.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount > 0
