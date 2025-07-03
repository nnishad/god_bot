import time
import os
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import your bot and helper functions
from bot import RA1Bot, get_current_time_with_tz

# --- UTILITY FUNCTIONS ---
import tempfile

def reset_test_environment():
    global CHROMA_DIR, CONTEXT_CACHE_FILE
    temp_dir = tempfile.mkdtemp()
    CHROMA_DIR = os.path.join(temp_dir, "chroma_mem")
    CONTEXT_CACHE_FILE = os.path.join(temp_dir, "context_cache.json")
    os.makedirs(CHROMA_DIR, exist_ok=True)

def load_context_cache():
    if not os.path.exists(CONTEXT_CACHE_FILE):
        return {}
    with open(CONTEXT_CACHE_FILE, "r") as f:
        return json.load(f)

def read_reminders_from_memory(bot):
    """Helper to fetch reminder facts from ChromaDB"""
    results = bot.memory_store.search("reminder")
    reminders = []
    for doc, meta in results:
        if meta.get("type") == "reminder":
            reminders.append({"text": doc, "meta": meta})
    return reminders

# --- TESTS ---

def test_fact_extraction_and_recall():
    reset_test_environment()
    bot = RA1Bot()

    # Step 1: User says something factual
    bot.on_new_message("group123", "user456", "msg1", "Priya loves chocolate cake.", "Amit")

    # Step 2: Someone asks about it
    bot.on_new_message("group123", "user789", "msg2", "/RA1 who loves chocolate cake?", "Ravi")

    # Check reply was generated
    context = load_context_cache()
    print(context)
    assert "Priya loves chocolate cake." in str(context)

def test_ambiguous_reference_resolution():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg3", "I'm going to meet Priya tomorrow at 3 PM.", "Amit")
    bot.on_new_message("group123", "user789", "msg4", "RA1, when is Priya meeting someone?", "Ravi")

    context = load_context_cache()
    print(context)
    assert any("Priya meeting" in m["text"] for m in context.get("group123", []))

def test_overwrite_fact():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg5", "Priya's birthday is on April 5th.", "Amit")
    bot.on_new_message("group123", "user789", "msg6", "No, Priya said her birthday is April 6th.", "Ravi")
    bot.on_new_message("group123", "user123", "msg7", "/RA1 when is Priya's birthday?", "Sam")

    context = load_context_cache()
    latest_reply = [m for m in context["group123"] if m["sender"] == "RA1"][-1]
    print(latest_reply)
    assert "April 6th" in latest_reply["text"]

def test_duplicate_message_handling():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg8", "Priya will be late today.", "Amit")
    bot.on_new_message("group123", "user456", "msg8", "Priya will be late today.", "Amit")  # Same ID

    context = load_context_cache()
    priya_messages = [m for m in context["group123"] if m["text"] == "Priya will be late today."]
    print(context["group123"])
    assert len(priya_messages) == 1

def test_multi_turn_conversation():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg9", "I bought a new phone yesterday.", "Amit")
    bot.on_new_message("group123", "user789", "msg10", "Was it an iPhone or Android?", "Ravi")
    bot.on_new_message("group123", "user456", "msg11", "It was an iPhone 15 Pro.", "Amit")
    bot.on_new_message("group123", "user123", "msg12", "/RA1 what phone did Amit buy?", "Sam")

    context = load_context_cache()
    latest_reply = [m for m in context["group123"] if m["sender"] == "RA1"][-1]
    print(latest_reply)
    assert "iPhone 15 Pro" in latest_reply["text"]

def test_timezone_aware_prompt():
    reset_test_environment()
    bot = RA1Bot()

    current_time_str = get_current_time_with_tz()
    assert "BST" in current_time_str or "GMT" in current_time_str

def test_long_context_chain():
    reset_test_environment()
    bot = RA1Bot()

    for i in range(1, 21):  # More than CONTEXT_WINDOW
        bot.on_new_message("group123", f"user{i}", f"msg{i+12}", f"Message {i} in long chain.", f"User{i}")

    context = load_context_cache()
    assert len(context["group123"]) <= 15  # CONTEXT_WINDOW limit

def test_quoted_message_priority():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg1", "Today is Priya's birthday!", "Amit")
    bot.on_new_message("group123", "user789", "msg2", "/RA1 whose birthday is today?", "Ravi", quoted_msg_id="msg1")

    context = load_context_cache()
    latest_reply = [m for m in context["group123"] if m["sender"] == "RA1"][-1]
    print(latest_reply)
    assert "Priya" in latest_reply["text"]

def test_no_response_when_not_tagged():
    reset_test_environment()
    bot = RA1Bot()

    bot.on_new_message("group123", "user456", "msg1", "Just finished my coffee.", "Amit")

    context = load_context_cache()
    assert all(m["sender"] != "RA1" for m in context.get("group123", []))

# Optional: Add more test cases here!