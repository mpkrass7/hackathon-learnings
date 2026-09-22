# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Agent Title
# MAGIC %md
# MAGIC # Genie Router Agent
# MAGIC
# MAGIC A tool-calling agent powered by Claude Sonnet that routes questions to the right Genie space:
# MAGIC * **Bakehouse Analytics** : sales, franchises, suppliers, and customer reviews
# MAGIC * **Detroit 911 Incidents** : emergency response data, call types, and geographic patterns

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install -q --upgrade openai databricks-sdk>=0.118.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration and Genie Space Lookup
import json
import re
import time

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# ---------------------------------------------------------------------------
# Configuration: change MODEL_ENDPOINT to match your serving endpoint name
# ---------------------------------------------------------------------------
MODEL = "il_model_service.default.il-claude-sonnet-5"

# Derive user prefix (same logic as the setup notebook)
user_info = w.current_user.me()
user_email = user_info.user_name
display_name = user_info.display_name or user_email.split("@")[0]
clean_name = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")
if clean_name and not clean_name[0].isalpha():
    clean_name = f"u-{clean_name}"


def find_space_id(client: WorkspaceClient, title_prefix: str) -> str:
    """Find a Genie space ID by title prefix."""
    resp = client.genie.list_spaces()
    while True:
        for space in resp.spaces or []:
            if space.title and space.title.startswith(title_prefix):
                return space.space_id
        if not resp.next_page_token:
            break
        resp = client.genie.list_spaces(page_token=resp.next_page_token)
    raise ValueError(f"No Genie space found starting with '{title_prefix}'")


BAKEHOUSE_SPACE_ID = find_space_id(w, f"{clean_name} - Bakehouse Analytics")
DETROIT_911_SPACE_ID = find_space_id(w, f"{clean_name} - Detroit 911 Incidents")

print(f"Model: {MODEL}")
print(f"Bakehouse space: {BAKEHOUSE_SPACE_ID}")
print(f"Detroit 911 space: {DETROIT_911_SPACE_ID}")

# COMMAND ----------

# DBTITLE 1,Tools Section Header
# MAGIC %md
# MAGIC ## Genie Space Routing Tools
# MAGIC
# MAGIC Each tool sends a natural-language question to its Genie space, waits for the
# MAGIC response, and returns the result (text + SQL + data) back to the agent.

# COMMAND ----------

# DBTITLE 1,Genie Query Helper and Tool Definitions
def _query_genie_space(space_id: str, question: str) -> str:
    """Send a question to a Genie space and return the response.

    Starts a new conversation, polls until the Genie message completes,
    then extracts text explanations, SQL, and query results.
    """
    msg = w.genie.start_conversation_and_wait(space_id=space_id, content=question)

    conversation_id = msg.conversation_id
    message_id = msg.id

    parts = []

    if msg.content:
        parts.append(msg.content)

    for att in msg.attachments or []:
        if att.text and hasattr(att.text, "content"):
            parts.append(att.text.content)

        if att.query:
            parts.append(f"SQL: {att.query.query}")
            try:
                result = w.genie.get_message_query_result(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=att.id,
                )
                if result.statement_response and result.statement_response.result:
                    columns = result.statement_response.result.columns or []
                    rows = result.statement_response.result.data_array or []
                    col_names = [c.name for c in columns]
                    parts.append(" | ".join(col_names))
                    for row in rows[:25]:  # cap at 25 rows for context window
                        parts.append(" | ".join(str(v) for v in row))
            except Exception as e:
                parts.append(f"(could not fetch query results: {e})")

    return "\n".join(parts) if parts else "No response from Genie space."


# Responses API tool schemas (name at top level, not nested under "function")
TOOLS = [
    {
        "type": "function",
        "name": "query_bakehouse",
        "description": (
            "Ask a question about bakehouse data: sales transactions, "
            "customer reviews, franchises, and suppliers. "
            "Use for any bakery or food business analytics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural-language question to ask the Bakehouse Genie space.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "query_detroit_911",
        "description": (
            "Ask a question about Detroit 911 incident data: call types, "
            "response times, priorities, neighborhoods, and geographic patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural-language question to ask the Detroit 911 Genie space.",
                }
            },
            "required": ["question"],
        },
    },
]

# Dispatcher maps tool names to callables
TOOL_DISPATCH = {
    "query_bakehouse": lambda q: _query_genie_space(BAKEHOUSE_SPACE_ID, q),
    "query_detroit_911": lambda q: _query_genie_space(DETROIT_911_SPACE_ID, q),
}

print(f"Tools defined: {list(TOOL_DISPATCH.keys())}")

# COMMAND ----------

# DBTITLE 1,Agent Section Header
# MAGIC %md
# MAGIC ## Agent Setup
# MAGIC
# MAGIC Uses the OpenAI SDK pointed at the Databricks serving endpoint.
# MAGIC A simple loop calls the model, dispatches any tool calls, feeds results back,
# MAGIC and repeats until the model returns a final text response.

# COMMAND ----------

# DBTITLE 1,Create the Agent
from openai import OpenAI

# Point at AI Gateway, not serving-endpoints
_auth = w.config.authenticate()
_api_key = _auth.get("Authorization", "").removeprefix("Bearer ")

client = OpenAI(
    base_url=f"{w.config.host}/ai-gateway/mlflow/v1",
    api_key=_api_key,
)

SYSTEM_PROMPT = (
    "You are a helpful data analyst assistant with access to two data sources:\n\n"
    "1. **Bakehouse Analytics**: sales transactions, customer reviews, franchises, "
    "and suppliers for a bakehouse business. Use `query_bakehouse` for any "
    "food/bakery business questions.\n\n"
    "2. **Detroit 911 Incidents**: emergency 911 call data for Detroit including "
    "call types, response times, priorities, neighborhoods, and geographic patterns. "
    "Use `query_detroit_911` for any emergency response or public safety questions.\n\n"
    "Route each question to the appropriate tool. If a question spans both domains, "
    "query both and synthesize the results. Summarize data clearly and concisely."
)


def run_agent(user_input: str, max_turns: int = 5) -> str:
    """Run a tool-calling agent loop using the OpenAI Responses API.

    Sends the user's question, dispatches any function_call outputs back
    to the model, and repeats until the model returns a final text response.

    Args:
        user_input: The user's natural-language question.
        max_turns: Safety cap on tool-call round-trips.

    Returns:
        The model's final text response.
    """
    input_items = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for _ in range(max_turns):
        response = client.responses.create(
            model=MODEL,
            input=input_items,
            tools=TOOLS,
        )

        # Collect tool calls from the response output
        tool_calls = [item for item in response.output if item.type == "function_call"]

        # No tool calls means the model is done
        if not tool_calls:
            return response.output_text or "(no response)"

        # Feed the model's output back as context, then append tool results
        input_items.extend(response.output)

        for tc in tool_calls:
            fn_args = json.loads(tc.arguments)
            print(f"  -> calling {tc.name}({fn_args['question'][:80]}...)")

            handler = TOOL_DISPATCH.get(tc.name)
            result = handler(fn_args["question"]) if handler else f"Unknown tool: {tc.name}"

            input_items.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": result,
            })

    return "(max tool-call turns reached)"


print("Agent ready! Use run_agent('your question') to chat.")

# COMMAND ----------

# DBTITLE 1,Test the Agent
print(run_agent("What are the top 5 neighborhoods by 911 call volume?"))