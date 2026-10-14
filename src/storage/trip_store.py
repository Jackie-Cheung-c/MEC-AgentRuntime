"""SQLite persistence for versioned trips and replan proposals."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from src.domain import ReplanProposal, TripState
from src.paths import DATA_DIR
from src.planning import ProposalConflictError, TripNotFoundError, TripVersionConflictError


class SQLiteTripRepository:
    def __init__(self, path: str | Path = DATA_DIR / "trips.db") -> None:
        self.path = Path(path)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS trips (
                    trip_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trips_user_updated
                    ON trips(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS replan_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    trip_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    base_version INTEGER NOT NULL,
                    proposal_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    confirmed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_proposals_trip
                    ON replan_proposals(user_id, trip_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_idempotency_keys_operation
                    ON idempotency_keys(operation, created_at DESC);
                """
            )

    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        """Serialize idempotent writers before they read the key table."""
        if not connection.in_transaction:
            connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _read_idempotent_result(
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT result_json FROM idempotency_keys
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()

    @staticmethod
    def _write_idempotent_result(
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        operation: str,
        result_json: str,
        created_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO idempotency_keys (
                idempotency_key, operation, result_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                idempotency_key,
                operation,
                result_json,
                created_at.isoformat(),
            ),
        )

    def get_idempotent_trip(self, idempotency_key: str) -> TripState | None:
        row = self._get_idempotent_result(idempotency_key)
        return TripState.model_validate_json(row) if row is not None else None

    def get_idempotent_proposal(
        self, idempotency_key: str
    ) -> ReplanProposal | None:
        row = self._get_idempotent_result(idempotency_key)
        return ReplanProposal.model_validate_json(row) if row is not None else None

    def _get_idempotent_result(self, idempotency_key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM idempotency_keys
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        return row["result_json"] if row is not None else None

    def create(
        self,
        state: TripState,
        *,
        idempotency_key: str | None = None,
    ) -> TripState:
        with self._connect() as connection:
            if idempotency_key:
                self._begin_immediate(connection)
                existing = self._read_idempotent_result(
                    connection, idempotency_key
                )
                if existing is not None:
                    return TripState.model_validate_json(existing["result_json"])
            connection.execute(
                "INSERT INTO trips VALUES (?, ?, ?, ?, ?)",
                (
                    str(state.trip_id), state.user_id, state.version,
                    state.model_dump_json(), state.updated_at.isoformat(),
                ),
            )
            if idempotency_key:
                self._write_idempotent_result(
                    connection,
                    idempotency_key=idempotency_key,
                    operation="create_trip_plan",
                    result_json=state.model_dump_json(),
                    created_at=datetime.now(timezone.utc),
                )
        return state

    def get(self, user_id: str, trip_id: UUID | str) -> TripState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM trips WHERE user_id = ? AND trip_id = ?",
                (user_id, str(trip_id)),
            ).fetchone()
        if row is None:
            raise TripNotFoundError(f"Trip {trip_id} was not found for this user.")
        return TripState.model_validate_json(row["state_json"])

    def save_proposal(
        self,
        proposal: ReplanProposal,
        *,
        idempotency_key: str | None = None,
    ) -> ReplanProposal:
        with self._connect() as connection:
            if idempotency_key:
                self._begin_immediate(connection)
                existing = self._read_idempotent_result(
                    connection, idempotency_key
                )
                if existing is not None:
                    return ReplanProposal.model_validate_json(
                        existing["result_json"]
                    )
            connection.execute(
                """INSERT INTO replan_proposals (
                    proposal_id, trip_id, user_id, base_version, proposal_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    str(proposal.proposal_id), str(proposal.trip_id), proposal.user_id,
                    proposal.base_version, proposal.model_dump_json(),
                    proposal.created_at.isoformat(),
                ),
            )
            if idempotency_key:
                self._write_idempotent_result(
                    connection,
                    idempotency_key=idempotency_key,
                    operation="propose_trip_replan",
                    result_json=proposal.model_dump_json(),
                    created_at=datetime.now(timezone.utc),
                )
        return proposal

    def get_proposal(
        self, user_id: str, trip_id: UUID | str, proposal_id: UUID | str
    ) -> ReplanProposal:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT proposal_json FROM replan_proposals
                   WHERE proposal_id = ? AND trip_id = ? AND user_id = ?""",
                (str(proposal_id), str(trip_id), user_id),
            ).fetchone()
        if row is None:
            raise ProposalConflictError("Replan proposal was not found.")
        return ReplanProposal.model_validate_json(row["proposal_json"])

    def confirm_proposal(
        self,
        user_id: str,
        trip_id: UUID | str,
        proposal_id: UUID | str,
        *,
        idempotency_key: str | None = None,
    ) -> TripState:
        confirmed_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            self._begin_immediate(connection)
            if idempotency_key:
                existing = self._read_idempotent_result(
                    connection, idempotency_key
                )
                if existing is not None:
                    return TripState.model_validate_json(existing["result_json"])
            row = connection.execute(
                """SELECT proposal_json, status FROM replan_proposals
                   WHERE proposal_id = ? AND trip_id = ? AND user_id = ?""",
                (str(proposal_id), str(trip_id), user_id),
            ).fetchone()
            if row is None:
                raise ProposalConflictError("Replan proposal was not found.")
            proposal = ReplanProposal.model_validate_json(row["proposal_json"])
            # A proposal ID is itself the business idempotency key for
            # confirmation. Replaying the same confirmation returns the same
            # committed state instead of turning into a conflict.
            if row["status"] == "confirmed":
                return proposal.proposed_state
            if row["status"] != "pending":
                raise ProposalConflictError("Replan proposal is no longer pending.")
            cursor = connection.execute(
                """UPDATE trips SET version = ?, state_json = ?, updated_at = ?
                   WHERE trip_id = ? AND user_id = ? AND version = ?""",
                (
                    proposal.proposed_state.version,
                    proposal.proposed_state.model_dump_json(),
                    proposal.proposed_state.updated_at.isoformat(),
                    str(trip_id), user_id, proposal.base_version,
                ),
            )
            if cursor.rowcount != 1:
                raise TripVersionConflictError(
                    "Trip changed after this proposal was created; create a new proposal."
                )
            confirmed_proposal = proposal.model_copy(
                update={"confirmed_at": confirmed_at}
            )
            connection.execute(
                """UPDATE replan_proposals SET status = 'confirmed', confirmed_at = ?,
                   proposal_json = ?
                   WHERE proposal_id = ?""",
                (
                    confirmed_at.isoformat(),
                    confirmed_proposal.model_dump_json(),
                    str(proposal_id),
                ),
            )
            if idempotency_key:
                self._write_idempotent_result(
                    connection,
                    idempotency_key=idempotency_key,
                    operation="confirm_trip_replan",
                    result_json=proposal.proposed_state.model_dump_json(),
                    created_at=confirmed_at,
                )
        return proposal.proposed_state
