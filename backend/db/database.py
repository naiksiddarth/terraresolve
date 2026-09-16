import sqlite3
from backend.config import DB_PATH

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT,
            progress_pct INTEGER DEFAULT 0,
            source_type TEXT NOT NULL,
            model_version TEXT,
            input_path TEXT,
            output_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()

def recover_interrupted_jobs(conn):
    conn.execute("""
        UPDATE jobs SET status='FAILED', error_message='interrupted by restart', updated_at=datetime('now')
        WHERE status IN ('RUNNING','PREPROCESSING','INFERENCE','POSTPROCESSING')
    """)
    conn.commit()
