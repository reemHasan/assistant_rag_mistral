
from langchain_core.prompts import PromptTemplate
# ─────────────────────────────────────────────────────────────────────────────
#  Few-shot examples  (10 patterns covering typical coach queries)
#     Schema is NOT hardcoded here — it comes from db.get_table_info()
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "q": "Who has the highest 3-point percentage this season (min 100 attempts)?",
        "sql": """\
SELECT p.name, t.code AS team,
       s.three_p_pct,
       CAST(s.min_after_15 AS INT) AS made,
       CAST(s.three_pa AS INT) AS att
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25' AND s.three_pa >= 100
ORDER BY s.three_p_pct DESC
LIMIT 10;""",
    },
    {
        "q": "Top 10 scorers per game this season",
        "sql": """\
SELECT p.name, t.code AS team,
       ROUND(s.pts * 1.0 / s.gp, 1) AS pts_per_game,
       s.gp
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25' AND s.gp >= 20
ORDER BY pts_per_game DESC
LIMIT 10;""",
    },
    {
        "q": "Compare teams by average net rating (offensive vs defensive)",
        "sql": """\
SELECT t.code, t.full_name,
       ROUND(AVG(s.off_rtg), 1) AS avg_off_rtg,
       ROUND(AVG(s.def_rtg), 1) AS avg_def_rtg,
       ROUND(AVG(s.net_rtg), 1) AS avg_net_rtg
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25'
GROUP BY t.team_id
ORDER BY avg_net_rtg DESC
LIMIT 15;""",
    },
    {
        "q": "Best true shooting percentage (min 500 points)",
        "sql": """\
SELECT p.name, t.code AS team,
       s.ts_pct,
       ROUND(s.pts * 1.0 / s.gp, 1) AS pts_pg,
       s.pts AS total_pts
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25' AND s.pts >= 500
ORDER BY s.ts_pct DESC
LIMIT 15;""",
    },
    {
        "q": "Which players score more than 25 points per game AND shoot above 50% from the field?",
        "sql": """\
SELECT p.name, t.code AS team,
       ROUND(s.pts * 1.0 / s.gp, 1) AS pts_pg,
       s.fg_pct
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25'
  AND (s.pts / s.gp) > 25
  AND s.fg_pct > 50
ORDER BY pts_pg DESC;""",
    },
    {
        "q": "Best defenders by combined steals + blocks per game (min 40 games)",
        "sql": """\
SELECT p.name, t.code AS team,
       ROUND((s.stl + s.blk) / s.gp, 2) AS stl_blk_pg,
       ROUND(s.stl / s.gp, 2) AS stl_pg,
       ROUND(s.blk / s.gp, 2) AS blk_pg
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25' AND s.gp >= 40
ORDER BY stl_blk_pg DESC
LIMIT 15;""",
    },
    {
        "q": "Show me Nikola Jokić complete stats this season",
        "sql": """\
SELECT p.name, t.full_name AS team, p.age,
       s.gp,
       ROUND(s.pts/s.gp,1) AS pts_pg,
       ROUND(s.reb/s.gp,1) AS reb_pg,
       ROUND(s.ast/s.gp,1) AS ast_pg,
       s.ts_pct, s.net_rtg, s.pie
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE p.name LIKE '%Joki%' AND s.season = '2024-25';""",
    },
    {
        "q": "Which players under 25 have the best Player Impact Estimate (PIE)?",
        "sql": """\
SELECT p.name, p.age, t.code AS team,
       s.pie,
       ROUND(s.pts/s.gp, 1) AS pts_pg
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25'
  AND p.age < 25
  AND s.gp >= 30
ORDER BY s.pie DESC
LIMIT 10;""",
    },
    {
        "q": "Best assist-to-turnover ratio (min 300 assists total)",
        "sql": """\
SELECT p.name, t.code AS team,
       s.ast_to AS ast_to_ratio,
       CAST(s.ast AS INT) AS total_ast,
       CAST(s.tov AS INT) AS total_tov
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25' AND s.ast >= 300
ORDER BY s.ast_to DESC
LIMIT 10;""",
    },
    {
        "q": "Which team has the highest average usage rate?",
        "sql": """\
SELECT t.code, t.full_name,
       ROUND(AVG(s.usg_pct), 1) AS avg_usg_pct,
       COUNT(p.player_id)        AS players_counted
FROM players p
JOIN teams t ON p.team_id = t.team_id
JOIN player_stats s ON p.player_id = s.player_id
WHERE s.season = '2024-25'
GROUP BY t.team_id
ORDER BY avg_usg_pct DESC
LIMIT 10;""",
    },
]

# ────────────────────────────────────────────────────────────────────────────
# 2.  SQL generation prompt
#     {schema} is filled from db.get_table_info() at construction time
#     {few_shots} is filled from _FEW_SHOT_BLOCK (built once at import)
#     {question} is filled per _run() call
# ─────────────────────────────────────────────────────────────────────────────

SQL_PROMPT = PromptTemplate.from_template("""\
You are an expert NBA database analyst. Generate a single valid SQLite query.
If the question asks for information that cannot exist in this database,
DO NOT generate SQL.
Instead return exactly: Information not available in the source.

SCHEMA (live from database):
{schema}

RULES:
1. Always filter WHERE s.season = '2024-25' unless another season is specified.
2. Divide totals by gp to get per-game averages: e.g. pts/gp, reb/gp.
3. Use LIKE '%name%' for fuzzy player name matching.
4. Always include player name and team code in SELECT.
5. Add LIMIT 10-20 unless a specific count is requested.
6. Return ONLY the raw SQL. No markdown. No backticks. No explanations.

FEW-SHOT EXAMPLES:
{few_shots}

Now generate SQL for:
Q: {question}
SQL:""")
