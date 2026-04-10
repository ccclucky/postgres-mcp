"""Natural language to SQL converter."""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Environment variable for API key
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass
class NlToSqlResult:
    """Result of NL to SQL conversion."""
    sql: str
    success: bool
    error: Optional[str] = None
    warning: Optional[str] = None


# Pattern templates for simple NL → SQL conversion
# Order matters: more specific patterns should come first
NL_PATTERNS = [
    # "查看表有多少条" / "统计表数量"
    (
        re.compile(r"^(?:查看|统计|查询)?(.+?)(?:有多少条|的总数|有多少|记录数|总行数)$", re.IGNORECASE),
        lambda m: f"SELECT COUNT(*) AS count FROM {m.group(1).strip()}"
    ),
    # "查看表的前N条" / "查询表的前10条"
    (
        re.compile(r"^(?:查看|查询)(?:.+?)?的前?(\d+)条$", re.IGNORECASE),
        lambda m: None  # Need table name, handled separately
    ),
    # "查看表最近N条" / "查询表最新10条"
    (
        re.compile(r"^(?:查看|查询|看看)(.+?)最近(\d+)条$", re.IGNORECASE),
        lambda m: None  # Need column name for ordering
    ),
]


def _match_template(query: str) -> Optional[str]:
    """Try to match query against simple templates.

    Returns SQL string if matched, None otherwise.
    """
    q = query.strip()

    # Helper to clean table name
    def clean_table(t: str) -> str:
        t = t.strip()
        t = re.sub(r"^(?:表|table)\s+", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+(?:表|table)$", "", t, flags=re.IGNORECASE)
        return t.strip()

    # Pattern: "查看/统计/查询 users (表) 有多少条"
    m = re.match(
        r"^(?:查看|统计|查询|看看)\s*(?:表\s+)?(.+?)(?:\s+表)?\s*(?:有多少条|的总数|有多少|记录数|总行数)\s*$",
        q,
        re.IGNORECASE
    )
    if m:
        table = clean_table(m.group(1))
        if table:
            return f"SELECT COUNT(*) AS count FROM {table}"

    # Pattern: "查看/查询/看看/显示/列出 users (表) 的前 10 条"
    m = re.match(
        r"^(?:查看|查询|看看|显示|列出)\s+(.+?)(?:\s+表)?\s+表\s*的前\s*(\d+)\s*条\s*$",
        q,
        re.IGNORECASE
    )
    if m:
        table = clean_table(m.group(1))
        limit = m.group(2)
        if table and limit.isdigit():
            return f"SELECT * FROM {table} LIMIT {int(limit)}"

    # Pattern: "查看/查询/看看 users (表) 最近 10 条"
    m = re.match(
        r"^(?:查看|查询|看看|显示|列出)\s+(.+?)(?:\s+表)?\s*表\s*最近\s*(\d+)\s*条\s*$",
        q,
        re.IGNORECASE
    )
    if m:
        table = clean_table(m.group(1))
        limit = m.group(2)
        if table and limit.isdigit():
            return f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT {int(limit)}"

    return None


def _validate_sql(sql: str) -> Optional[str]:
    """Validate generated SQL using SafeSqlDriver's validation logic.

    Returns warning string if suspicious, None if OK.
    """
    import pglast

    forbidden_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "CALL",
        "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"
    ]

    sql_upper = sql.upper()
    for kw in forbidden_keywords:
        if re.search(rf"\b{kw}\b", sql_upper):
            return f"Generated SQL contains forbidden keyword '{kw}'. Only SELECT is allowed."

    # Check for SELECT * without LIMIT
    if re.search(r"\bSELECT\s+\*\b", sql_upper) and "LIMIT" not in sql_upper:
        return "Query uses SELECT * without LIMIT. Consider specifying columns."

    return None


async def nl_to_sql(query: str, table_context: Optional[str] = None) -> NlToSqlResult:
    """Convert natural language query to SQL.

    Args:
        query: Natural language query (e.g., "查看用户表有多少条")
        table_context: Optional table schema context for LLM generation

    Returns:
        NlToSqlResult with generated SQL or error
    """
    # Step 1: Try template matching first (fast, no API call)
    template_sql = _match_template(query)
    if template_sql:
        warning = _validate_sql(template_sql)
        return NlToSqlResult(sql=template_sql, success=True, warning=warning)

    # Step 2: Use LLM for complex queries
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if not api_key:
        return NlToSqlResult(
            sql="",
            success=False,
            error=f"ANTHROPIC_API_KEY environment variable not set. "
                  f"Set it to enable natural language to SQL conversion."
        )

    try:
        import instructor
        from anthropic import Anthropic

        client = instructor.from_provider(Anthropic(api_key=api_key), model="claude-sonnet-4-20250514")

        # Build prompt with optional table context
        systemPrompt = (
            "You are a SQL expert. Convert natural language queries to PostgreSQL SELECT statements.\n"
            "Rules:\n"
            "- Only generate SELECT queries. No INSERT, UPDATE, DELETE, or any data modification.\n"
            "- Always add LIMIT unless the user explicitly asks for all rows.\n"
            "- Use proper JOIN syntax with explicit ON conditions.\n"
            "- Escape column/table names with double quotes if they contain special characters.\n"
        )
        if table_context:
            systemPrompt += f"\n\nAvailable table schema:\n{table_context}"

        userPrompt = f"Convert this natural language query to SQL:\n{query}"

        response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            response_model=SqlResponse
        )

        # Validate LLM output
        if not response.sql or not response.sql.strip().upper().startswith("SELECT"):
            return NlToSqlResult(
                sql="",
                success=False,
                error="LLM generated a non-SELECT query. Only SELECT is allowed."
            )

        warning = _validate_sql(response.sql)
        return NlToSqlResult(sql=response.sql, success=True, warning=warning)

    except Exception as e:
        logger.error(f"NL to SQL conversion failed: {e}")
        return NlToSqlResult(sql="", success=False, error=str(e))


# Pydantic model for instructor response
from pydantic import BaseModel, Field


class SqlResponse(BaseModel):
    """Structured SQL response from LLM."""
    sql: str = Field(description="The generated PostgreSQL SELECT statement")
    explanation: Optional[str] = Field(default=None, description="Brief explanation of the SQL")
