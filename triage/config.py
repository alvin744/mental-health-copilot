import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DATA_FILE = "triage_data.json"
POLICY_MODES = ["Conservative", "Balanced", "Lenient"]
CASE_STATUSES = ["OPEN", "IN REVIEW", "ESCALATED", "CLOSED"]
REVIEWER_DECISIONS = [
    "Approve system recommendation",
    "Escalate",
    "Modify response",
    "Defer",
]
