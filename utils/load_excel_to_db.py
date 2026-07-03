"""
load_excel_to_db.py
===================
NBA regular-season Excel → SQLite ingestion pipeline.

Reads  regular_NBA.xlsx, validates every row with Pydantic,
and inserts into a normalised SQLite database with four tables:
  teams · players · player_stats · reports

Usage
-----
    python load_excel_to_db.py --excel regular_NBA.xlsx --db nba.db
    python load_excel_to_db.py --excel regular_NBA.xlsx --db nba.db --reset

Schema (also written to schema.sql on first run)
------
    teams        (team_id PK, code UNIQUE, full_name)
    players      (player_id PK, name UNIQUE, team_id FK, age)
    player_stats (stat_id PK, player_id FK, season, gp, w, l, min_per_game,
                  pts, fgm, fga, fg_pct, min_after_15, three_pa, three_p_pct,
                  ftm, fta, ft_pct, oreb, dreb, reb, ast, tov, stl, blk, pf,
                  fp, dd2, td3, plus_minus, off_rtg, def_rtg, net_rtg,
                  ast_pct, ast_to, ast_ratio, oreb_pct, dreb_pct, reb_pct,
                  to_ratio, efg_pct, ts_pct, usg_pct, pace, pie, poss)
    reports      (report_id PK, player_id FK, generated_at, content)

Dependencies
------------
    pip install pandas openpyxl pydantic
"""

from __future__ import annotations
import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Optional
from utils.db import DDL
from utils.models import TeamRow, PlayerStatsRow
from utils.config import DATABASE_FILE
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Create database sqlite
# ─────────────────────────────────────────────────────────────────────────────

def create_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()
    log.info("Schema created / verified.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Excel reading
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None

def read_excel(path: Path) -> tuple[list[TeamRow], list[PlayerStatsRow]]:
    """
    Parse the Excel file and return validated team and player-stats objects.

    The workbook structure:
      'Données NBA'          → player stats (header on row 1, data from row 2)
      'Equipe'               → team code ↔ full name mapping
      'Dictionnaire des données' → column definitions (documentation only)
    """
    xl = pd.ExcelFile(path)

    # ── Teams sheet ────────────────────────────────────────────────────────
    team_df = xl.parse("Equipe")
    team_df.columns = ["code", "full_name"]
    teams: list[TeamRow] = []
    for _, row in team_df.iterrows():
        try:
            teams.append(TeamRow(code=str(row["code"]), full_name=str(row["full_name"])))
        except Exception as exc:
            log.warning("Invalid team row %s: %s", row.to_dict(), exc)

    # ── Stats sheet ────────────────────────────────────────────────────────
    # Real column names are in row index 1 (row 0 is numeric placeholders)
    stats_df = xl.parse("Données NBA", header=1)
    stats_df = stats_df.dropna(axis=1, how="all")  # drop empty trailing cols

    # The min_after_15 column was mis-parsed as datetime.time(15, 0) by pandas
    # because of a cell format issue — rename it correctly
    rename_map: dict = {}
    for col in stats_df.columns:
        import datetime as dt
        if isinstance(col, dt.time):
            rename_map[col] = "min_after_15"
    stats_df = stats_df.rename(columns=rename_map)
    print(stats_df.columns.values)
    stats: list[PlayerStatsRow] = []
    errors = 0
    for _, row in stats_df.iterrows():
        raw = {
            "player":    str(row.get("Player", "")).strip(),
            "team":      str(row.get("Team", "")).strip(),
            "age":       _safe_int(row.get("Age")),
            "gp":        _safe_int(row.get("GP")),
            "w":         _safe_int(row.get("W")),
            "l":         _safe_int(row.get("L")),
            "min_pg":    _safe_float(row.get("Min")),
            "pts":       _safe_float(row.get("PTS")),
            "fgm":       _safe_float(row.get("FGM")),
            "fga":       _safe_float(row.get("FGA")),
            "fg_pct":    _safe_float(row.get("FG%")),
            "min_after_15":_safe_float(row.get("min_after_15")),
            "3pa":       _safe_float(row.get("3PA")),
            "3p_pct":    _safe_float(row.get("3P%")),
            "ftm":       _safe_float(row.get("FTM")),
            "fta":       _safe_float(row.get("FTA")),
            "ft_pct":    _safe_float(row.get("FT%")),
            "oreb":      _safe_float(row.get("OREB")),
            "dreb":      _safe_float(row.get("DREB")),
            "reb":       _safe_float(row.get("REB")),
            "ast":       _safe_float(row.get("AST")),
            "tov":       _safe_float(row.get("TOV")),
            "stl":       _safe_float(row.get("STL")),
            "blk":       _safe_float(row.get("BLK")),
            "pf":        _safe_float(row.get("PF")),
            "fp":        _safe_float(row.get("FP")),
            "dd2":       _safe_int(row.get("DD2")),
            "td3":       _safe_int(row.get("TD3")),
            "plus_minus": _safe_float(row.get("+/-")),
            "off_rtg":   _safe_float(row.get("OFFRTG")),
            "def_rtg":   _safe_float(row.get("DEFRTG")),
            "net_rtg":   _safe_float(row.get("NETRTG")),
            "ast_pct":   _safe_float(row.get("AST%")),
            "ast_to":    _safe_float(row.get("AST/TO")),
            "ast_ratio": _safe_float(row.get("AST RATIO")),
            "oreb_pct":  _safe_float(row.get("OREB%")),
            "dreb_pct":  _safe_float(row.get("DREB%")),
            "reb_pct":   _safe_float(row.get("REB%")),
            "to_ratio":  _safe_float(row.get("TO RATIO")),
            "efg_pct":   _safe_float(row.get("EFG%")),
            "ts_pct":    _safe_float(row.get("TS%")),
            "usg_pct":   _safe_float(row.get("USG%")),
            "pace":      _safe_float(row.get("PACE")),
            "pie":       _safe_float(row.get("PIE")),
            "poss":      _safe_int(row.get("POSS")),
        }
        # Skip rows with missing core identity fields
        if not raw["player"] or raw["age"] is None or raw["gp"] is None:
            continue
        try:
            stats.append(PlayerStatsRow(**raw))
        except Exception as exc:
            errors += 1
            log.warning("Row validation failed for '%s': %s", raw.get("player"), exc)

    log.info(
        "Parsed %d teams, %d valid player rows, %d validation errors.",
        len(teams), len(stats), errors,
    )
    return teams, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Database insertion
# ─────────────────────────────────────────────────────────────────────────────

SEASON = "2024-25"

def insert_teams(conn: sqlite3.Connection, teams: list[TeamRow]) -> dict[str, int]:
    """Insert teams and return code → team_id mapping."""
    try:
        cur = conn.cursor()
        for t in teams:
            cur.execute(
                "INSERT OR IGNORE INTO teams (code, full_name) VALUES (?, ?)",
                (t.code, t.full_name),
            )
        conn.commit()
        cur.execute("SELECT code, team_id FROM teams")
        mapping = {row[0]: row[1] for row in cur.fetchall()}
        log.info("Teams inserted/verified: %d", len(mapping))
        return mapping
    except Exception as e:
        log.error(f"Error happened while insertion in Team table: {e}")
        return {}

def insert_players_and_stats(
    conn: sqlite3.Connection,
    stats: list[PlayerStatsRow],
    team_map: dict[str, int],
) -> None:
    """Upsert players then insert/replace their season stats."""
    try:
        cur = conn.cursor()
        inserted_players = 0
        inserted_stats   = 0
        skipped_team     = 0

        for row in stats:
            #get team_id from Team table
            team_id = team_map.get(row.team)
            if team_id is None:
                log.warning(
                    "Unknown team code '%s' for player '%s' — inserting player without team.",
                    row.team, row.player,
                )
                skipped_team += 1

            # Upsert player (keep most recent age/team)
            cur.execute(
                """
                INSERT INTO players (name, team_id, age)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    team_id = excluded.team_id,
                    age     = excluded.age
                """,
                (row.player, team_id, row.age),
            )
            if cur.rowcount:
                inserted_players += 1

            # Get player_id
            cur.execute("SELECT player_id FROM players WHERE name = ?", (row.player,))
            player_id = cur.fetchone()[0]

            # Insert or replace season stats
            cur.execute(
                """
                INSERT OR REPLACE INTO player_stats (
                    player_id, season, gp, w, l, min_per_game,
                    pts, fgm, fga, fg_pct, min_after_15, three_pa, three_p_pct,
                    ftm, fta, ft_pct, oreb, dreb, reb, ast, tov, stl, blk, pf,
                    fp, dd2, td3, plus_minus, off_rtg, def_rtg, net_rtg,
                    ast_pct, ast_to, ast_ratio, oreb_pct, dreb_pct, reb_pct,
                    to_ratio, efg_pct, ts_pct, usg_pct, pace, pie, poss
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    player_id, SEASON, row.gp, row.w, row.l, row.min_pg,
                    row.pts, row.fgm, row.fga, row.fg_pct,
                    row.min_after_15, row.three_pa, row.three_pct,
                    row.ftm, row.fta, row.ft_pct,
                    row.oreb, row.dreb, row.reb,
                    row.ast, row.tov, row.stl, row.blk, row.pf,
                    row.fp, row.dd2, row.td3, row.plus_minus,
                    row.off_rtg, row.def_rtg, row.net_rtg,
                    row.ast_pct, row.ast_to, row.ast_ratio,
                    row.oreb_pct, row.dreb_pct, row.reb_pct,
                    row.to_ratio, row.efg_pct, row.ts_pct, row.usg_pct,
                    row.pace, row.pie, row.poss,
                ),
            )
            inserted_stats += 1

        conn.commit()
        log.info(
            "Players upserted: %d | Stats inserted: %d | Unknown teams: %d",
            inserted_players, inserted_stats, skipped_team,
        )
    except Exception as e:
        log.error(f"Error happened while inserting rows into Playes and stats tables: {e}")

def save_schema(db_path: Path) -> None:
    """Write the DDL to schema.sql next to the database."""
    schema_path = db_path.parent / "schema.sql"
    schema_path.write_text(DDL, encoding="utf-8")
    log.info("Schema written to %s", schema_path)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Load NBA Excel data into SQLite.")
    parser.add_argument("--excel", required=True, help="Path to the Excel file")
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop all tables and rebuild from scratch",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # delete all tables in db
    if args.reset:
        log.info("--reset: dropping all tables.")
        conn.executescript("""
            DROP TABLE IF EXISTS reports;
            DROP TABLE IF EXISTS player_stats;
            DROP TABLE IF EXISTS players;
            DROP TABLE IF EXISTS teams;
        """)

    create_db(conn)
    save_schema(DATABASE_FILE)

    teams, stats = read_excel(excel_path)
    team_map     = insert_teams(conn, teams)
    insert_players_and_stats(conn, stats, team_map)

    # Quick verification
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM teams")
    cur.execute("SELECT COUNT(*) FROM players")
    n_players = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM player_stats")
    n_stats = cur.fetchone()[0]
    log.info("Verification → players: %d | stats rows: %d", n_players, n_stats)

    conn.close()
    log.info("Done. Database saved to %s", DATABASE_FILE)


if __name__ == "__main__":
    main()
