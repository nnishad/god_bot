import os
import time
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests

from datetime import datetime
import pytz

def get_current_time_with_tz():
    tz = pytz.timezone('Europe/London')  # UK Time (automatically handles BST/GMT)
    now = datetime.now(tz)
    return now.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")

# --- CONFIGURATION ---
CONTEXT_CACHE_FILE = "context_cache.json"
CHROMA_DIR = "chroma_mem"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CONTEXT_WINDOW = 15         # Number of recent messages to use for context
MEMORY_RETENTION = 7*24*3600   # Keep messages for 1 week
CHROMA_K = 5               # Top K memories from Chroma
LLM_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "qwen/qwen3-8b"
LLM_API_KEY = "lmstudio"   # For LM Studio
RA1_TAGS = ["/Ra1", "@Ra1", "Ra1"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RA1Bot")

# --- CONTEXT CACHE ---
class ContextCache:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.now = int(time.time())
        self._load()

    def _load(self):
        self.data = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    raw_data = json.load(f)
                # Prune expired messages on load
                for group_id, messages in raw_data.items():
                    self.data[group_id] = [
                        m for m in messages
                        if self.now - m.get('timestamp', 0) < MEMORY_RETENTION
                    ]
            except Exception as e:
                logger.error(f"Error loading context cache: {e}")
        else:
            self.data = {}

    def _save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f)

    def add_message(self, group_id: str, msg: Dict[str, Any]):
        now = int(time.time())
        msg['timestamp'] = now

        if group_id not in self.data:
            self.data[group_id] = []

        # Avoid duplicates
        msg_id = msg.get('msg_id')
        if msg_id and not any(m.get('msg_id') == msg_id for m in self.data[group_id]):
            self.data[group_id].append(msg)

        # Prune expired messages
        self.data[group_id] = [
            m for m in self.data[group_id]
            if now - m.get('timestamp', 0) < MEMORY_RETENTION
        ]

        self._save()

    def get_recent(self, group_id: str, window=CONTEXT_WINDOW) -> List[Dict[str, Any]]:
        return self.data.get(group_id, [])[-window:]

    def get_message_by_id(self, group_id: str, msg_id: str) -> Optional[Dict[str, Any]]:
        for m in self.data.get(group_id, []):
            if m.get('msg_id') == msg_id:
                return m
        return None

# --- CHROMADB MEMORY STORE ---
class MemoryStore:
    def __init__(self, chroma_dir: str, embedding_model: str):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.coll = self.client.get_or_create_collection("memories")
        self.model = SentenceTransformer(embedding_model, device='cuda')

    def add_fact(self, text: str, metadata: Dict[str, Any]):
        embedding = self.model.encode(text)
        fact_id = str(uuid.uuid4())  # Unique ID
        self.coll.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[fact_id]
        )

    def search(self, query: str, k=CHROMA_K):
        if not query.strip():
            return []
        embedding = self.model.encode(query)
        results = self.coll.query(
            query_embeddings=[embedding.tolist()],
            n_results=k
        )
        docs = results['documents'][0] if results.get('documents') else []
        metas = results['metadatas'][0] if results.get('metadatas') else []
        return list(zip(docs, metas))

# --- LLM CALLER WITH FUNCTION CALLING ---
def llm_function_call(
        context_msgs: List[Dict[str, Any]],
        quoted_msg: Optional[Dict[str, Any]],
        old_memories: List[Dict[str, Any]],
        current_msg: Dict[str, Any],
        bot_tags: List[str] = RA1_TAGS
) -> Dict[str, Any]:
    sys_prompt = (
        "You are RA1, a group chat member and intelligent assistant. "
        "Your role is to be helpful but precise: extract facts and answer questions, never guess, and never volunteer unprompted information.\n\n"

        "### CORE RULES ###\n"
        f"- Always respond ONLY when mentioned via {bot_tags} (e.g., /RA1 or @RA1)\n"
        "- Always prefer ABSOLUTE dates/times over relative ones (e.g., 'April 5, 2025' vs 'today')\n"
        "- Always prioritize QUOTED MESSAGES as high-priority context\n"
        "- Never add extra information unless explicitly asked\n"
        "- Never assume facts not provided in the message or memory\n\n"

        "### DATE/TIME CONTEXT ###\n"
        f"Current date/time: {get_current_time_with_tz()}. Interpret all time references relative to this.\n"
        "When saving facts involving time, convert expressions like 'tomorrow' or 'next week' to absolute dates.\n"
        "Example: 'Priya will arrive tomorrow' → 'Priya will arrive on April 6, 2025'.\n\n"

        "### FACT EXTRACTION GUIDELINES ###\n"
        "Common fact types include:\n"
        "- birthday (entity: person, detail: date)\n"
        "- location (entity: person/place, detail: location)\n"
        "- event (entity: event name, detail: time/place)\n"
        "- relationship (entity: person A, detail: relationship to person B)\n"
        "- reminder (entity: reminder ID, detail: action + datetime)\n"
        "Always match the most appropriate type."

        "### REPLY GENERATION ###\n"
        "If mentioned ({bot_tags}):\n"
        "- Answer ONLY the question asked\n"
        "- Use context + memory to generate concise, factual replies\n"
        "- Never mention yourself unless asked directly\n"
        "- Format replies naturally without markdown or special formatting\n"
        "- If no relevant info exists, say: 'I don't have that information yet.'\n\n"

        "### RESPONSE PRINCIPLES ###\n"
        "✅ DO:\n"
        "- Use exact names/dates from messages\n"
        "- Prioritize quoted message content\n"
        "- Be specific and concise\n\n"
        "❌ DO NOT:\n"
        "- Guess missing information\n"
        "- Add unprompted details\n"
        "- Mention yourself unless asked\n"
        "- Use relative time expressions\n\n"

        "### MEMORY USAGE ###\n"
        "Always use the most recent, relevant memory for replies. "
        "If multiple facts exist about the same topic, prioritize:\n"
        "1. Most recent timestamp\n"
        "2. Most specific detail\n"
        "3. Most direct mention\n\n"

        "Respond only by calling one or both functions: save_facts OR send_reply. Never reply directly."
    )

    context_str = "\n".join([f"{m['sender']}: {m['text']}" for m in context_msgs]) if context_msgs else ""
    memory_str = "\n".join([f"{meta.get('timestamp', '')} - {doc}" for doc, meta in old_memories]) if old_memories else ""
    quoted_str = f"QUOTED MESSAGE (HIGH PRIORITY): {quoted_msg['sender']}: {quoted_msg['text']}\n" if quoted_msg else ""
    msg_str = f"NEW MESSAGE from {current_msg['sender']}: {current_msg['text']}"

    full_prompt = (
        f"Context (last {CONTEXT_WINDOW} msgs):\n{context_str}\n\n"
        f"Old memories (top {CHROMA_K}):\n{memory_str}\n\n"
        f"{quoted_str}{msg_str}"
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": full_prompt}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_facts",
                "description": "Save extracted facts/entities from the group chat",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "entity": {"type": "string"},
                                    "detail": {"type": "string"},
                                    "mentioned_by": {"type": "string"},
                                    "timestamp": {"type": "integer"}
                                },
                                "required": ["type", "entity", "detail", "mentioned_by", "timestamp"]
                            }
                        }
                    },
                    "required": ["facts"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_reply",
                "description": "Send a reply message to the group",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                }
            }
        }
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(LLM_URL, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as e:
        logger.error(f"LLM API error: {e}")
        return {"facts": [], "reply": None}

    output = {"facts": [], "reply": None}

    try:
        tool_calls = result.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
        for tool in tool_calls:
            fn = tool.get('function', {}).get('name')
            args = tool.get('function', {}).get('arguments', {})
            if isinstance(args, str):
                args = json.loads(args)
            if fn == "save_facts":
                output["facts"].extend(args.get("facts", []))
            elif fn == "send_reply":
                output["reply"] = args.get("text")
    except (KeyError, json.JSONDecodeError, IndexError) as e:
        logger.error(f"Error parsing tool calls: {e}")

    return output

# --- MAIN BOT HANDLER ---
class RA1Bot:
    def __init__(self):
        self.context_cache = ContextCache(CONTEXT_CACHE_FILE)
        self.memory_store = MemoryStore(CHROMA_DIR, EMBEDDING_MODEL)

    def on_new_message(self, group_id: str, user_id: str, msg_id: str, msg_text: str, sender: str, quoted_msg_id: Optional[str] = None):
        logger.info(f"Processing message: {msg_id} from {sender} in group {group_id}")

        # 1. Gather context
        recent_msgs = self.context_cache.get_recent(group_id)

        # 2. Get quoted message if any
        quoted_msg = self.context_cache.get_message_by_id(group_id, quoted_msg_id) if quoted_msg_id else None

        # 3. Combine quoted message with current message for better memory retrieval
        search_query = msg_text
        if quoted_msg:
            search_query = quoted_msg['text'] + " " + msg_text

        # 4. Fetch relevant memories
        top_memories = self.memory_store.search(search_query, k=CHROMA_K)

        # 5. Prepare current message record
        current_msg = {
            "msg_id": msg_id,
            "sender": sender,
            "user_id": user_id,
            "text": msg_text
        }

        # 6. Call LLM
        output = llm_function_call(
            context_msgs=recent_msgs,
            quoted_msg=quoted_msg,
            old_memories=top_memories,
            current_msg=current_msg
        )

        # 7. Save extracted facts to memory
        for fact in output.get("facts", []):
            required_keys = ["type", "entity", "detail", "mentioned_by", "timestamp"]
            if all(k in fact for k in required_keys):
                text = f"{fact['entity']}: {fact['detail']}"
                self.memory_store.add_fact(text, fact)
            else:
                logger.warning("Invalid fact structure:", fact)

        # 8. Send reply if any
        if output.get("reply"):
            reply_msg_id = f"{msg_id}_ra1"
            reply_msg = {
                "msg_id": reply_msg_id,
                "sender": "RA1",
                "user_id": "RA1",
                "text": output["reply"]
            }
            self.context_cache.add_message(group_id, reply_msg)
            send_to_group(group_id, output["reply"])

        # 9. Add user message to context cache
        self.context_cache.add_message(group_id, current_msg)

# --- MOCK: Sending to group (replace with actual integration) ---
def send_to_group(group_id: str, text: str):
    print(f"[{group_id}] RA1: {text}")

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    bot = RA1Bot()
    bot.on_new_message("group123", "user456", "msg1", "Today is Priya's birthday!", "Amit")
    bot.on_new_message("group123", "user789", "msg2", "Ra1 whose birthday is today?", "Ravi")
    bot.on_new_message("group123", "user789", "msg3", "Happy birthday Priya!", "Ravi", quoted_msg_id="msg1")
    bot.on_new_message("user456", "user456", "msg4", "Ra1, remind me to pay rent.", "Amit")
    bot.on_new_message("user456", "user456", "msg4", "Ra1, when is priya birthday", "Amit")