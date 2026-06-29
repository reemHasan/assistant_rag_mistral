# ─────────────────────────────────────────────────────────────────────────────
#  Database creation
# ─────────────────────────────────────────────────────────────────────────────

DDL = """
-- ── teams ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    code      TEXT    NOT NULL UNIQUE,   -- e.g. "OKC", "MIN", "DEN"
    full_name TEXT    NOT NULL           -- e.g. "Oklahoma City Thunder"
);

-- ── players ─────────────────────────────────────────────────────────────────
-- One row per player (most recent team when traded mid-season)
CREATE TABLE IF NOT EXISTS players (
    player_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    team_id    INTEGER REFERENCES teams(team_id) ON DELETE SET NULL,
    age        INTEGER CHECK(age BETWEEN 18 AND 55)
);

-- ── player_stats ─────────────────────────────────────────────────────────────
-- Season-level aggregate statistics (one row per player × season)
CREATE TABLE IF NOT EXISTS player_stats (
    stat_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id    INTEGER NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
    season       TEXT    NOT NULL DEFAULT '2024-25',

    -- Games
    gp           INTEGER,   -- Games Played
    w            INTEGER,   -- Wins
    l            INTEGER,   -- Losses
    min_per_game REAL,      -- Minutes per game

    -- Scoring (season totals where PTS/FGM etc are totals, not per-game)
    pts          REAL,      -- Total points
    fgm          REAL,      -- Field goals made
    fga          REAL,      -- Field goals attempted
    fg_pct       REAL,      -- FG%
    min_after_15 REAL,      -- Minutes Played After 15 Minutes Elapsed
    three_pa     REAL,      -- 3-pointers attempted
    three_p_pct  REAL,      -- 3P%
    ftm          REAL,      -- Free throws made
    fta          REAL,      -- Free throws attempted
    ft_pct       REAL,      -- FT%

    -- Rebounds
    oreb         REAL,      -- Offensive rebounds
    dreb         REAL,      -- Defensive rebounds
    reb          REAL,      -- Total rebounds

    -- Playmaking / Defence
    ast          REAL,      -- Assists
    tov          REAL,      -- Turnovers
    stl          REAL,      -- Steals
    blk          REAL,      -- Blocks
    pf           REAL,      -- Personal fouls
    fp           REAL,      -- Fantasy Points
    dd2          INTEGER,   -- Double-doubles
    td3          INTEGER,   -- Triple-doubles
    plus_minus   REAL,      -- +/-

    -- Advanced ratings
    off_rtg      REAL,      -- Offensive Rating (pts per 100 possessions)
    def_rtg      REAL,      -- Defensive Rating (pts allowed per 100 possessions)
    net_rtg      REAL,      -- Net Rating = OFFRTG - DEFRTG
    ast_pct      REAL,      -- Assist %
    ast_to       REAL,      -- AST/TOV ratio
    ast_ratio    REAL,      -- Assist Ratio per 100 possessions
    oreb_pct     REAL,      -- Offensive rebound %
    dreb_pct     REAL,      -- Defensive rebound %
    reb_pct      REAL,      -- Total rebound %
    to_ratio     REAL,      -- Turnover Ratio per 100 possessions
    efg_pct      REAL,      -- Effective FG% (weights 3-pointers)
    ts_pct       REAL,      -- True Shooting % (includes FT)
    usg_pct      REAL,      -- Usage Rate
    pace         REAL,      -- Pace (possessions per 48 min)
    pie          REAL,      -- Player Impact Estimate
    poss         INTEGER,   -- Total possessions played

    UNIQUE(player_id, season)
);

-- ── reports ──────────────────────────────────────────────────────────────────
-- LLM-generated or analyst reports attached to a player
CREATE TABLE IF NOT EXISTS reports (
    report_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id    INTEGER NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
    generated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    content      TEXT    NOT NULL,          -- full report text
    report_type  TEXT    DEFAULT 'analysis' -- 'analysis'|'scout'|'summary'
);

-- ── Useful indexes ────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_stats_player  ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_stats_season  ON player_stats(season);
CREATE INDEX IF NOT EXISTS idx_reports_player ON reports(player_id);
CREATE INDEX IF NOT EXISTS idx_players_team  ON players(team_id);
"""
