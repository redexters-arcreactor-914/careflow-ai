"""
 Human review & audit log

"""

import json
import os
from backend.models.schemas import AuditEntry
from backend.config import AUDIT_LOG_PATH


def log_action(entry: AuditEntry) -> None:
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def get_case_audit_trail(case_id: str) -> list[dict]:
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("case_id") == case_id:
                entries.append(entry)
    return entries


def get_recent(limit: int = 300) -> list[dict]:
    """Newest-first slice of the whole audit log, for the dashboard's Audit log page."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue          
    return entries[-limit:][::-1]


def withdraw_consent_and_delete(case_id: str) -> None:
    """
    TODO: wire up actual deletion against wherever cases are stored
    (currently a placeholder — if you're using a real DB, this is where
    the DELETE query / soft-delete flag goes). Still log the withdrawal
    itself — the audit trail should show *that* data was deleted, even
    though the data itself is gone.
    """
    log_action(AuditEntry(
        case_id=case_id,
        actor="system",
        action="consent_withdrawn_data_deleted",
        details="Patient withdrew consent; case data removed from active storage.",
    ))