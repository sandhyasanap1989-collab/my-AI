import os
import re
import json
import uuid
import sqlite3
from datetime import datetime
from functools import wraps

import requests

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)


# ============================================================
# HELPAR
# Educational AI Tutor
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "change-this-secret-key-in-render",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "DATA_DIR",
    BASE_DIR,
)

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# CREATOR INFORMATION
# ============================================================
#
# CHANGE ONLY THESE VALUES.
#
# Do NOT put passwords, API keys, phone numbers,
# home address, or private information here.
#
# ============================================================

CREATOR_INFO = {
    "name": "Soham Chandrahas Sanap",
    "role": "Creator and developer of Helpar",
    "project": "Helpar",
    "about": (
        "Helpar is an educational AI project created "
        "to help students learn and solve questions."
    ),
    "extra": "",
}


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================
#
# Set these in Render Environment Variables:
#
# HF_TOKEN = your Hugging Face token
#
# Optional:
# HF_MODEL = openai/gpt-oss-120b
#
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

HF_MODEL = os.environ.get(
    "HF_MODEL",
    "openai/gpt-oss-120b",
).strip()

HF_TIMEOUT = int(
    os.environ.get(
        "HF_TIMEOUT",
        "120",
    )
)


# ============================================================
# OPTIONAL LOCAL OLLAMA
# ============================================================
#
# This is OPTIONAL.
#
# Render cannot normally reach Ollama running on your
# personal Windows computer through 127.0.0.1.
#
# We therefore DO NOT depend on Ollama.
#
# If you later run Ollama on the same server, you can
# configure OLLAMA_URL.
#
# ============================================================

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "",
).strip().rstrip("/")

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2:3b",
).strip()

OLLAMA_TIMEOUT = int(
    os.environ.get(
        "OLLAMA_TIMEOUT",
        "120",
    )
)


# ============================================================
# DATA FILES
# ============================================================

USERS_FILE = os.path.join(
    DATA_DIR,
    "users.json",
)

DATABASE_FILE = os.path.join(
    DATA_DIR,
    "helpar.db",
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(filename, default):
    try:
        if not os.path.exists(filename):
            return default

        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception as error:
        print(
            "JSON LOAD ERROR:",
            repr(error),
        )

        return default


def save_json(filename, data):
    temporary_file = filename + ".tmp"

    try:
        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        os.replace(
            temporary_file,
            filename,
        )

        return True

    except Exception as error:
        print(
            "JSON SAVE ERROR:",
            repr(error),
        )

        try:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
        except Exception:
            pass

        return False


# ============================================================
# USERS FILE
# ============================================================

if not os.path.exists(USERS_FILE):
    save_json(
        USERS_FILE,
        {},
    )


# ============================================================
# DATABASE
# ============================================================

def get_db():
    connection = sqlite3.connect(
        DATABASE_FILE,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    connection = get_db()

    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(chat_id)
                REFERENCES chats(id)
                ON DELETE CASCADE
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chats_username
            ON chats(username)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_chat
            ON messages(chat_id)
            """
        )

        connection.commit()

    except Exception as error:
        connection.rollback()

        print(
            "DATABASE INITIALIZATION ERROR:",
            repr(error),
        )

    finally:
        connection.close()


initialize_database()


# ============================================================
# TIME
# ============================================================

def now_iso():
    return datetime.utcnow().isoformat()


# ============================================================
# AUTHENTICATION
# ============================================================

def current_username():
    return session.get("username")


def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if not session.get("username"):
            return jsonify({
                "success": False,
                "message": "Please login first.",
            }), 401

        return function(
            *args,
            **kwargs,
        )

    return wrapper


# ============================================================
# CREATOR QUESTIONS
# ============================================================

def is_creator_question(text):

    text = str(
        text or ""
    ).lower().strip()

    creator_phrases = [
        "who created you",
        "who is your creator",
        "who created helpar",
        "who made you",
        "who made helpar",
        "who developed you",
        "who developed helpar",
        "who built you",
        "who built helpar",
        "who programmed you",
        "who programmed helpar",
        "who designed you",
        "who designed helpar",
        "who is behind helpar",
        "who is behind you",
        "who is your developer",
        "who is helpar's developer",
        "who is helpar developer",
        "who is your maker",
        "who is helpar's maker",
        "who is helpar maker",
        "tell me about your creator",
        "tell me about helpar creator",
        "tell me about your developer",
        "tell me about helpar developer",
        "who is your father",
        "who is helpar's father",
        "who is helpar father",
        "who is your dad",
        "who is helpar's dad",
        "who is helpar dad",
    ]

    for phrase in creator_phrases:

        if phrase in text:
            return True

    creator_words = (
        "creator",
        "developer",
        "maker",
        "father",
        "dad",
        "created",
        "developed",
        "built",
        "made",
    )

    helpar_words = (
        "you",
        "helpar",
        "your",
    )

    has_creator_word = any(
        word in text
        for word in creator_words
    )

    has_helpar_reference = any(
        word in text
        for word in helpar_words
    )

    return (
        has_creator_word
        and has_helpar_reference
    )


def creator_response():

    name = CREATOR_INFO.get(
        "name",
        "the creator",
    )

    role = CREATOR_INFO.get(
        "role",
        "the developer",
    )

    project = CREATOR_INFO.get(
        "project",
        "Helpar",
    )

    about = CREATOR_INFO.get(
        "about",
        "",
    )

    extra = CREATOR_INFO.get(
        "extra",
        "",
    )

    response = (
        f"I was created and developed by {name}. "
        f"They are the {role} of {project}.\n\n"
        f"{about}"
    )

    if extra:
        response += (
            f"\n\n{extra}"
        )

    return response


# ============================================================
# BASIC BUILT-IN ANSWERS
# ============================================================
#
# These work WITHOUT Ollama or Hugging Face.
#
# ============================================================

BASIC_ANSWERS = {

    # ----------------------------
    # GREETINGS
    # ----------------------------

    "hi":
        "Hi! 👋 I'm Helpar. How can I help you today?",

    "hii":
        "Hii! 👋 I'm Helpar. What would you like to learn?",

    "hello":
        "Hello! 👋 I'm Helpar, your educational AI tutor. What would you like to learn?",

    "hey":
        "Hey! 👋 What would you like to learn today?",

    "good morning":
        "Good morning! ☀️ How can I help you learn today?",

    "good afternoon":
        "Good afternoon! 😊 What would you like to study?",

    "good evening":
        "Good evening! 🌆 How can I help you?",

    "how are you":
        "I'm doing great! 🤖 Ready to help you learn.",

    "thanks":
        "You're welcome! 😊",

    "thank you":
        "You're welcome! 😊",

    "thanks helpar":
        "You're welcome! 😊",

    "bye":
        "Goodbye! 👋 Keep learning!",

    # ----------------------------
    # HELPAR
    # ----------------------------

    "what is your name":
        "My name is Helpar. 🤖 I'm an educational AI tutor.",

    "who are you":
        (
            "I'm Helpar, an educational AI tutor. "
            "I can help with Mathematics, Physics, Chemistry, "
            "Biology, Computer Science, and general learning."
        ),

    "what can you do":
        (
            "I can help you with:\n\n"
            "• Mathematics\n"
            "• Physics\n"
            "• Chemistry\n"
            "• Biology\n"
            "• Basic Computer Science\n"
            "• General educational questions\n"
            "• Step-by-step problem solving"
        ),

    # ----------------------------
    # COMPUTER
    # ----------------------------

    "what is a computer":
        (
            "A computer is an electronic device that accepts data, "
            "processes it according to instructions, stores information, "
            "and produces output."
        ),

    "define computer":
        (
            "A computer is an electronic device that accepts data, "
            "processes it according to instructions, stores information, "
            "and produces output."
        ),

    "what is cpu":
        (
            "CPU stands for Central Processing Unit. "
            "It executes instructions and performs calculations "
            "in a computer."
        ),

    "what is a cpu":
        (
            "CPU stands for Central Processing Unit. "
            "It executes instructions and performs calculations "
            "in a computer."
        ),

    "define cpu":
        (
            "CPU stands for Central Processing Unit. "
            "It executes instructions and performs calculations "
            "in a computer."
        ),

    "what is ram":
        (
            "RAM stands for Random Access Memory. "
            "It temporarily stores data and programs that the "
            "computer is currently using."
        ),

    "what is a ram":
        (
            "RAM stands for Random Access Memory. "
            "It temporarily stores data and programs that the "
            "computer is currently using."
        ),

    "define ram":
        (
            "RAM stands for Random Access Memory. "
            "It temporarily stores data and programs that the "
            "computer is currently using."
        ),

    "what is rom":
        (
            "ROM stands for Read-Only Memory. "
            "It stores information that is generally retained "
            "when the computer is turned off."
        ),

    "define rom":
        (
            "ROM stands for Read-Only Memory. "
            "It stores information that is generally retained "
            "when the computer is turned off."
        ),

    "what is hardware":
        (
            "Hardware refers to the physical components of a computer, "
            "such as the CPU, RAM, keyboard, storage, and display."
        ),

    "define hardware":
        (
            "Hardware refers to the physical components of a computer, "
            "such as the CPU, RAM, keyboard, storage, and display."
        ),

    "what is software":
        (
            "Software is a collection of programs and instructions "
            "that tell a computer what to do."
        ),

    "define software":
        (
            "Software is a collection of programs and instructions "
            "that tell a computer what to do."
        ),

    "what is an operating system":
        (
            "An operating system is system software that manages "
            "computer hardware and provides services for applications. "
            "Examples include Windows, Linux, macOS, Android, and iOS."
        ),

    "what is os":
        (
            "An operating system is system software that manages "
            "computer hardware and provides services for applications."
        ),

    "what is keyboard":
        (
            "A keyboard is an input device used to enter letters, "
            "numbers, symbols, and commands into a computer."
        ),

    "what is a keyboard":
        (
            "A keyboard is an input device used to enter letters, "
            "numbers, symbols, and commands into a computer."
        ),

    "what is mouse":
        (
            "A computer mouse is a pointing input device used to "
            "move a pointer and interact with items on a screen."
        ),

    "what is a mouse":
        (
            "A computer mouse is a pointing input device used to "
            "move a pointer and interact with items on a screen."
        ),

    "what is internet":
        (
            "The Internet is a worldwide network of connected "
            "computer networks that communicate using standard "
            "networking protocols."
        ),

    "what is the internet":
        (
            "The Internet is a worldwide network of connected "
            "computer networks that communicate using standard "
            "networking protocols."
        ),

    "what is wifi":
        (
            "Wi-Fi is a wireless networking technology that allows "
            "devices to connect to a network using radio signals."
        ),

    "what is wi fi":
        (
            "Wi-Fi is a wireless networking technology that allows "
            "devices to connect to a network using radio signals."
        ),

    # ----------------------------
    # PHYSICS
    # ----------------------------

    "what is force":
        (
            "Force is a push or pull that can change the motion or "
            "shape of an object.\n\n"
            "SI unit: newton (N)"
        ),

    "define force":
        (
            "Force is a push or pull that can change the motion or "
            "shape of an object.\n\n"
            "SI unit: newton (N)"
        ),

    "what is speed":
        (
            "Speed is the distance travelled per unit time.\n\n"
            "Formula:\n"
            "Speed = Distance ÷ Time\n\n"
            "SI unit: m/s"
        ),

    "define speed":
        (
            "Speed is the distance travelled per unit time.\n\n"
            "Formula:\n"
            "Speed = Distance ÷ Time\n\n"
            "SI unit: m/s"
        ),

    "what is velocity":
        (
            "Velocity is the rate of change of displacement with time. "
            "It is a vector quantity, so it has magnitude and direction."
        ),

    "define velocity":
        (
            "Velocity is the rate of change of displacement with time. "
            "It is a vector quantity, so it has magnitude and direction."
        ),

    "what is acceleration":
        (
            "Acceleration is the rate of change of velocity with time.\n\n"
            "Formula:\n"
            "a = (v - u) / t\n\n"
            "SI unit: m/s²"
        ),

    "define acceleration":
        (
            "Acceleration is the rate of change of velocity with time.\n\n"
            "Formula:\n"
            "a = (v - u) / t\n\n"
            "SI unit: m/s²"
        ),

    "what is gravity":
        (
            "Gravity is the attractive force between masses. "
            "Near Earth's surface, gravitational acceleration is "
            "approximately 9.8 m/s²."
        ),

    "define gravity":
        (
            "Gravity is the attractive force between masses. "
            "Near Earth's surface, gravitational acceleration is "
            "approximately 9.8 m/s²."
        ),

    "what is newton's first law":
        (
            "Newton's First Law states that an object remains at rest "
            "or continues moving with uniform velocity unless acted "
            "upon by an external unbalanced force."
        ),

    "newtons first law":
        (
            "Newton's First Law states that an object remains at rest "
            "or continues moving with uniform velocity unless acted "
            "upon by an external unbalanced force."
        ),

    # ----------------------------
    # CHEMISTRY
    # ----------------------------

    "what is atom":
        (
            "An atom is the smallest unit of an element that retains "
            "the chemical properties of that element. It contains "
            "protons, neutrons, and electrons."
        ),

    "what is an atom":
        (
            "An atom is the smallest unit of an element that retains "
            "the chemical properties of that element. It contains "
            "protons, neutrons, and electrons."
        ),

    "define atom":
        (
            "An atom is the smallest unit of an element that retains "
            "the chemical properties of that element."
        ),

    "what is molecule":
        (
            "A molecule is a group of two or more atoms chemically "
            "bonded together."
        ),

    "what is a molecule":
        (
            "A molecule is a group of two or more atoms chemically "
            "bonded together."
        ),

    "define molecule":
        (
            "A molecule is a group of two or more atoms chemically "
            "bonded together."
        ),

    "what is element":
        (
            "An element is a pure substance made of atoms that all "
            "have the same number of protons."
        ),

    "what is an element":
        (
            "An element is a pure substance made of atoms that all "
            "have the same number of protons."
        ),

    "what is compound":
        (
            "A compound is a pure substance formed when two or more "
            "different elements combine chemically in fixed proportions."
        ),

    "what is a compound":
        (
            "A compound is a pure substance formed when two or more "
            "different elements combine chemically in fixed proportions."
        ),

    "what is ph":
        (
            "pH is a measure related to the acidity or basicity of an "
            "aqueous solution. At about 25°C, pH 7 is neutral, values "
            "below 7 are acidic, and values above 7 are basic."
        ),

    # ----------------------------
    # BIOLOGY
    # ----------------------------

    "what is cell":
        (
            "A cell is the basic structural and functional unit of life."
        ),

    "what is a cell":
        (
            "A cell is the basic structural and functional unit of life."
        ),

    "define cell":
        (
            "A cell is the basic structural and functional unit of life."
        ),

    "what is photosynthesis":
        (
            "Photosynthesis is the process by which green plants and "
            "some other organisms use light energy to make food from "
            "carbon dioxide and water, releasing oxygen."
        ),

    "define photosynthesis":
        (
            "Photosynthesis is the process by which green plants and "
            "some other organisms use light energy to make food from "
            "carbon dioxide and water, releasing oxygen."
        ),

    "what is dna":
        (
            "DNA stands for deoxyribonucleic acid. "
            "It stores genetic information used in the development "
            "and functioning of organisms."
        ),

    "what is respiration":
        (
            "Cellular respiration is a set of metabolic reactions "
            "through which cells release usable energy from nutrients "
            "such as glucose."
        ),

    "define respiration":
        (
            "Cellular respiration is a set of metabolic reactions "
            "through which cells release usable energy from nutrients "
            "such as glucose."
        ),

    # ----------------------------
    # MATHS
    # ----------------------------

    "what is pi":
        (
            "π (pi) is the ratio of a circle's circumference to its "
            "diameter. Its approximate value is 3.14159."
        ),

    "define pi":
        (
            "π (pi) is the ratio of a circle's circumference to its "
            "diameter. Its approximate value is 3.14159."
        ),

    "what is a prime number":
        (
            "A prime number is a whole number greater than 1 that has "
            "exactly two positive factors: 1 and itself."
        ),

    "define prime number":
        (
            "A prime number is a whole number greater than 1 that has "
            "exactly two positive factors: 1 and itself."
        ),

    "what is percentage":
        (
            "A percentage expresses a quantity as a fraction of 100.\n\n"
            "Percentage = (Part ÷ Whole) × 100"
        ),

    "define percentage":
        (
            "A percentage expresses a quantity as a fraction of 100.\n\n"
            "Percentage = (Part ÷ Whole) × 100"
        ),
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(
        text or ""
    ).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[!?.,]+$",
        "",
        text,
    )

    return text


# ============================================================
# BASIC MATH CALCULATOR
# ============================================================
#
# Supports simple arithmetic.
#
# Examples:
# 2+2
# 10-5
# 5*6
# 20/4
# 2^3
#
# It does NOT execute arbitrary Python code.
#
# ============================================================

def safe_calculate(expression):

    expression = expression.strip()

    if len(expression) > 100:
        return None

    if not re.fullmatch(
        r"[0-9+\-*/().%\s^]+",
        expression,
    ):
        return None

    expression = expression.replace(
        "^",
        "**",
    )

    if "**" in expression:
        parts = expression.split("**")

        if len(parts) != 2:
            return None

        try:
            base = float(parts[0].strip())
            exponent = float(parts[1].strip())

            if abs(exponent) > 20:
                return None

        except Exception:
            return None

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {},
        )

        if not isinstance(
            result,
            (int, float),
        ):
            return None

        if abs(result) > 10**100:
            return None

        if isinstance(
            result,
            float,
        ) and result.is_integer():
            result = int(result)

        return result

    except Exception:
        return None


def get_math_answer(text):

    cleaned = normalize_text(text)

    # Direct expression
    if re.fullmatch(
        r"[0-9+\-*/().%\s^]+",
        cleaned,
    ):

        result = safe_calculate(
            cleaned
        )

        if result is not None:
            return (
                f"Let's calculate it step by step.\n\n"
                f"{cleaned} = {result}"
            )

    # "calculate ..."
    prefixes = [
        "calculate ",
        "solve ",
        "what is ",
    ]

    for prefix in prefixes:

        if cleaned.startswith(prefix):

            expression = cleaned[
                len(prefix):
            ].strip()

            result = safe_calculate(
                expression
            )

            if result is not None:

                return (
                    f"Let's calculate it.\n\n"
                    f"{expression} = {result}"
                )

    return None


# ============================================================
# GET BASIC ANSWER
# ============================================================

def get_basic_answer(text):

    cleaned = normalize_text(
        text
    )

    if cleaned in BASIC_ANSWERS:
        return BASIC_ANSWERS[
            cleaned
        ]

    math_answer = get_math_answer(
        text
    )

    if math_answer:
        return math_answer

    # Additional simple patterns

    if cleaned.startswith(
        "hi "
    ):
        return (
            "Hi! 👋 I'm Helpar. "
            "How can I help you today?"
        )

    if cleaned.startswith(
        "hello "
    ):
        return (
            "Hello! 👋 I'm Helpar. "
            "What would you like to learn?"
        )

    return None


# ============================================================
# CHAT FUNCTIONS
# ============================================================

def create_chat(username):

    chat_id = str(
        uuid.uuid4()
    )

    timestamp = now_iso()

    connection = get_db()

    try:

        connection.execute(
            """
            INSERT INTO chats
            (id, username, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                username,
                "New Chat",
                timestamp,
                timestamp,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    return chat_id


def chat_exists(
    chat_id,
    username,
):

    connection = get_db()

    try:

        row = connection.execute(
            """
            SELECT id
            FROM chats
            WHERE id = ? AND username = ?
            """,
            (
                chat_id,
                username,
            ),
        ).fetchone()

        return row is not None

    finally:
        connection.close()


def get_latest_chat(username):

    connection = get_db()

    try:

        row = connection.execute(
            """
            SELECT id
            FROM chats
            WHERE username = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (
                username,
            ),
        ).fetchone()

        if row:
            return row["id"]

        return None

    finally:
        connection.close()


def get_current_chat():

    username = current_username()

    if not username:
        return None

    saved_chat = session.get(
        "chat_id"
    )

    if saved_chat:

        if chat_exists(
            saved_chat,
            username,
        ):
            return saved_chat

    latest = get_latest_chat(
        username
    )

    if latest:

        session["chat_id"] = latest

        return latest

    new_chat = create_chat(
        username
    )

    session["chat_id"] = new_chat

    return new_chat


def save_message(
    chat_id,
    role,
    content,
):

    connection = get_db()

    try:

        timestamp = now_iso()

        connection.execute(
            """
            INSERT INTO messages
            (chat_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                chat_id,
                role,
                content,
                timestamp,
            ),
        )

        connection.execute(
            """
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                timestamp,
                chat_id,
            ),
        )

        connection.commit()

        return True

    except Exception as error:

        connection.rollback()

        print(
            "SAVE MESSAGE ERROR:",
            repr(error),
        )

        return False

    finally:
        connection.close()


def set_chat_title(
    chat_id,
    title,
):

    connection = get_db()

    try:

        connection.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                now_iso(),
                chat_id,
            ),
        )

        connection.commit()

    finally:
        connection.close()


def get_chat(
    chat_id,
    username,
):

    connection = get_db()

    try:

        chat = connection.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM chats
            WHERE id = ? AND username = ?
            """,
            (
                chat_id,
                username,
            ),
        ).fetchone()

        if not chat:
            return None

        messages = connection.execute(
            """
            SELECT
                role,
                content,
                timestamp
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (
                chat_id,
            ),
        ).fetchall()

        return {
            "id": chat["id"],
            "title": chat["title"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"],
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                }
                for row in messages
            ],
        }

    finally:
        connection.close()


# ============================================================
# HUGGING FACE AI
# ============================================================

def call_huggingface(
    prompt,
    conversation=None,
):

    if conversation is None:
        conversation = []

    if not HF_TOKEN:
        return (
            False,
            "Online AI is not configured yet. "
            "Add HF_TOKEN to your Render environment variables."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Helpar, a helpful educational AI tutor. "
                "Answer clearly and accurately. "
                "Use simple language when appropriate. "
                "For mathematics, show useful steps. "
                "For physics, show formulas and units. "
                "For chemistry, check equations carefully. "
                "For biology, use correct scientific terms. "
                "For computer science, explain concepts clearly. "
                "Do not invent facts. "
                "If you do not know something, say so."
            ),
        }
    ]

    for item in conversation[-12:]:

        role = item.get(
            "role"
        )

        content = item.get(
            "content",
            "",
        )

        if role not in (
            "user",
            "assistant",
        ):
            continue

        if not content:
            continue

        messages.append({
            "role": role,
            "content": str(content),
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 1500,
    }

    try:

        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=HF_TIMEOUT,
        )

        if response.status_code != 200:

            print(
                "HUGGING FACE ERROR:",
                response.status_code,
                response.text,
            )

            return (
                False,
                "Online AI returned HTTP "
                + str(response.status_code),
            )

        data = response.json()

        choices = data.get(
            "choices"
        )

        if not choices:
            return (
                False,
                "Online AI returned no answer.",
            )

        message = choices[0].get(
            "message",
            {},
        )

        answer = message.get(
            "content",
            "",
        )

        answer = str(
            answer
        ).strip()

        if not answer:
            return (
                False,
                "Online AI returned an empty answer.",
            )

        return (
            True,
            answer,
        )

    except requests.exceptions.Timeout:

        return (
            False,
            "Online AI took too long to respond.",
        )

    except requests.exceptions.ConnectionError:

        return (
            False,
            "Could not connect to the online AI service.",
        )

    except Exception as error:

        print(
            "HUGGING FACE EXCEPTION:",
            repr(error),
        )

        return (
            False,
            "Online AI error: "
            + str(error),
        )


# ============================================================
# OPTIONAL OLLAMA
# ============================================================

def call_ollama(
    prompt,
    conversation=None,
):

    if not OLLAMA_URL:
        return (
            False,
            "Ollama is not configured.",
        )

    if conversation is None:
        conversation = []

    messages = [
        {
            "role": "system",
            "content": (
                "You are Helpar, an educational AI tutor. "
                "Give clear, accurate educational answers."
            ),
        }
    ]

    for item in conversation[-12:]:

        role = item.get(
            "role"
        )

        content = item.get(
            "content",
            "",
        )

        if role in (
            "user",
            "assistant",
        ) and content:

            messages.append({
                "role": role,
                "content": str(content),
            })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1500,
        },
    }

    try:

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        if response.status_code != 200:

            return (
                False,
                "Ollama returned HTTP "
                + str(response.status_code),
            )

        data = response.json()

        message = data.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            return (
                False,
                "Ollama returned no message.",
            )

        answer = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not answer:
            return (
                False,
                "Ollama returned an empty answer.",
            )

        return (
            True,
            answer,
        )

    except requests.exceptions.ConnectionError:

        return (
            False,
            "Cannot connect to Ollama.",
        )

    except requests.exceptions.Timeout:

        return (
            False,
            "Ollama took too long to respond.",
        )

    except Exception as error:

        return (
            False,
            str(error),
        )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["POST"],
)
def login():

    data = request.get_json(
        silent=True
    ) or {}

    login_value = str(
        data.get(
            "login",
            "",
        )
    ).strip()

    password = str(
        data.get(
            "password",
            "",
        )
    )

    if not login_value or not password:

        return jsonify({
            "success": False,
            "message": (
                "Enter username/email and password."
            ),
        }), 400

    users = load_json(
        USERS_FILE,
        {},
    )

    username_found = None
    user_found = None

    for username, user in users.items():

        if not isinstance(
            user,
            dict,
        ):
            continue

        email = str(
            user.get(
                "email",
                "",
            )
        ).lower()

        phone = str(
            user.get(
                "phone",
                "",
            )
        )

        if (
            username.lower()
            == login_value.lower()
            or email
            == login_value.lower()
            or phone
            == login_value
        ):

            username_found = username
            user_found = user

            break

    if not user_found:

        return jsonify({
            "success": False,
            "message": (
                "Invalid username, email or phone."
            ),
        }), 401

    try:

        password_correct = check_password_hash(
            user_found.get(
                "password",
                "",
            ),
            password,
        )

    except Exception:

        password_correct = False

    if not password_correct:

        return jsonify({
            "success": False,
            "message": "Incorrect password.",
        }), 401

    session["username"] = username_found

    latest = get_latest_chat(
        username_found
    )

    if latest:

        session["chat_id"] = latest

    else:

        session["chat_id"] = create_chat(
            username_found
        )

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "username": username_found,
    })


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["POST"],
)
def register():

    data = request.get_json(
        silent=True
    ) or {}

    username = str(
        data.get(
            "username",
            "",
        )
    ).strip()

    email = str(
        data.get(
            "email",
            "",
        )
    ).strip()

    phone = str(
        data.get(
            "phone",
            "",
        )
    ).strip()

    password = str(
        data.get(
            "password",
            "",
        )
    )

    if len(username) < 3:

        return jsonify({
            "success": False,
            "message": (
                "Username must contain at least 3 characters."
            ),
        }), 400

    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+",
        username,
    ):

        return jsonify({
            "success": False,
            "message": (
                "Username contains invalid characters."
            ),
        }), 400

    if not email and not phone:

        return jsonify({
            "success": False,
            "message": (
                "Enter an email or phone number."
            ),
        }), 400

    if email:

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email,
        ):

            return jsonify({
                "success": False,
                "message": (
                    "Enter a valid email address."
                ),
            }), 400

    if len(password) < 6:

        return jsonify({
            "success": False,
            "message": (
                "Password must contain at least 6 characters."
            ),
        }), 400

    users = load_json(
        USERS_FILE,
        {},
    )

    for existing_username, user in users.items():

        if not isinstance(
            user,
            dict,
        ):
            continue

        if (
            existing_username.lower()
            == username.lower()
        ):

            return jsonify({
                "success": False,
                "message": (
                    "Username already exists."
                ),
            }), 409

        existing_email = str(
            user.get(
                "email",
                "",
            )
        ).lower()

        if (
            email
            and existing_email
            == email.lower()
        ):

            return jsonify({
                "success": False,
                "message": (
                    "Email already exists."
                ),
            }), 409

        existing_phone = str(
            user.get(
                "phone",
                "",
            )
        )

        if (
            phone
            and existing_phone
            == phone
        ):

            return jsonify({
                "success": False,
                "message": (
                    "Phone number already exists."
                ),
            }), 409

    users[username] = {
        "email": email,
        "phone": phone,
        "password": generate_password_hash(
            password
        ),
        "created_at": now_iso(),
    }

    if not save_json(
        USERS_FILE,
        users,
    ):

        return jsonify({
            "success": False,
            "message": (
                "Could not save the account."
            ),
        }), 500

    session["username"] = username

    session["chat_id"] = create_chat(
        username
    )

    return jsonify({
        "success": True,
        "message": (
            "Account created successfully."
        ),
        "username": username,
    })


# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# NEW CHAT
# ============================================================

@app.route(
    "/new-chat",
    methods=["POST"],
)
@login_required
def new_chat():

    username = current_username()

    chat_id = create_chat(
        username
    )

    session["chat_id"] = chat_id

    return jsonify({
        "success": True,
        "chat_id": chat_id,
        "title": "New Chat",
    })


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history",
    methods=["GET"],
)
@login_required
def history():

    username = current_username()

    connection = get_db()

    try:

        rows = connection.execute(
            """
            SELECT
                id,
                title,
                created_at,
                updated_at
            FROM chats
            WHERE username = ?
            ORDER BY updated_at DESC
            """,
            (
                username,
            ),
        ).fetchall()

        chats = []

        for row in rows:

            chats.append({
                "id": row["id"],
                "title": (
                    row["title"]
                    or "New Chat"
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })

        return jsonify({
            "success": True,
            "chats": chats,
        })

    except Exception as error:

        print(
            "HISTORY ERROR:",
            repr(error),
        )

        return jsonify({
            "success": False,
            "message": (
                "Could not load history."
            ),
        }), 500

    finally:
        connection.close()


# ============================================================
# OPEN CHAT
# ============================================================

@app.route(
    "/chat/<chat_id>",
    methods=["GET"],
)
@login_required
def open_chat(chat_id):

    username = current_username()

    chat = get_chat(
        chat_id,
        username,
    )

    if not chat:

        return jsonify({
            "success": False,
            "message": "Chat not found.",
        }), 404

    session["chat_id"] = chat_id

    return jsonify({
        "success": True,
        "chat": chat,
    })


# ============================================================
# MAIN CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"],
)
@login_required
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    user_message = str(
        data.get(
            "message",
            "",
        )
    ).strip()

    if not user_message:

        return jsonify({
            "success": False,
            "message": (
                "Please enter a message."
            ),
        }), 400

    username = current_username()

    chat_id = get_current_chat()

    if not chat_id:

        return jsonify({
            "success": False,
            "message": (
                "Could not create chat."
            ),
        }), 500

    # ========================================================
    # CREATOR QUESTION
    # ========================================================

    if is_creator_question(
        user_message
    ):

        answer = creator_response()

        save_message(
            chat_id,
            "user",
            user_message,
        )

        save_message(
            chat_id,
            "assistant",
            answer,
        )

        current_chat = get_chat(
            chat_id,
            username,
        )

        if (
            current_chat
            and current_chat.get(
                "title"
            ) == "New Chat"
        ):

            set_chat_title(
                chat_id,
                user_message[:50],
            )

        return jsonify({
            "success": True,
            "answer": answer,
            "response": answer,
            "reply": answer,
            "message": answer,
            "chat_id": chat_id,
            "source": "built-in",
        })

    # ========================================================
    # BASIC BUILT-IN ANSWER
    # ========================================================

    basic_answer = get_basic_answer(
        user_message
    )

    if basic_answer:

        save_message(
            chat_id,
            "user",
            user_message,
        )

        save_message(
            chat_id,
            "assistant",
            basic_answer,
        )

        current_chat = get_chat(
            chat_id,
            username,
        )

        if (
            current_chat
            and current_chat.get(
                "title"
            ) == "New Chat"
        ):

            set_chat_title(
                chat_id,
                user_message[:50],
            )

        return jsonify({
            "success": True,
            "answer": basic_answer,
            "response": basic_answer,
            "reply": basic_answer,
            "message": basic_answer,
            "chat_id": chat_id,
            "source": "built-in",
        })

    # ========================================================
    # GET OLD CONVERSATION
    # ========================================================

    old_chat = get_chat(
        chat_id,
        username,
    )

    conversation = []

    if old_chat:

        conversation = old_chat.get(
            "messages",
            [],
        )

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    if not save_message(
        chat_id,
        "user",
        user_message,
    ):

        return jsonify({
            "success": False,
            "message": (
                "Could not save your message."
            ),
        }), 500

    # ========================================================
    # CHAT TITLE
    # ========================================================

    if (
        old_chat
        and old_chat.get(
            "title"
        ) == "New Chat"
    ):

        title = user_message[:50]

        if len(user_message) > 50:
            title += "..."

        set_chat_title(
            chat_id,
            title,
        )

    # ========================================================
    # ONLINE AI
    # ========================================================

    success, answer = call_huggingface(
        user_message,
        conversation,
    )

    # ========================================================
    # OPTIONAL OLLAMA FALLBACK
    # ========================================================

    if not success and OLLAMA_URL:

        print(
            "Hugging Face failed:",
            answer,
        )

        ollama_success, ollama_answer = call_ollama(
            user_message,
            conversation,
        )

        if ollama_success:

            success = True
            answer = ollama_answer

    # ========================================================
    # AI FAILURE
    # ========================================================

    if not success:

        print(
            "AI ERROR:",
            answer,
        )

        # Keep the error user-friendly.
        # IMPORTANT: no "Start Ollama" message because
        # Ollama is not required for the Render version.

        return jsonify({
            "success": False,
            "answer": (
                "⚠️ "
                + answer
            ),
            "error": answer,
            "chat_id": chat_id,
        }), 502

    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    save_message(
        chat_id,
        "assistant",
        answer,
    )

    return jsonify({
        "success": True,
        "answer": answer,
        "response": answer,
        "reply": answer,
        "message": answer,
        "chat_id": chat_id,
        "source": "online-ai",
    })


# ============================================================
# IMPROVE
# ============================================================

@app.route(
    "/improve",
    methods=["POST"],
)
@login_required
def improve():

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get(
            "question",
            "",
        )
    ).strip()

    answer = str(
        data.get(
            "answer",
            "",
        )
    ).strip()

    action = str(
        data.get(
            "action",
            "improve",
        )
    ).strip().lower()

    instructions = {

        "improve":
            "Improve the answer while keeping it correct.",

        "check":
            "Check the answer carefully and correct any errors.",

        "explain":
            "Explain the answer step by step in simple language.",

        "short":
            "Make the answer shorter while keeping the important information.",
    }

    instruction = instructions.get(
        action,
        instructions["improve"],
    )

    prompt = f"""
Question:
{question}

Existing answer:
{answer}

Task:
{instruction}

Give the corrected and useful result.
"""

    success, result = call_huggingface(
        prompt
    )

    if not success and OLLAMA_URL:

        success, result = call_ollama(
            prompt
        )

    if not success:

        return jsonify({
            "success": False,
            "answer": (
                "⚠️ "
                + result
            ),
            "error": result,
        }), 502

    return jsonify({
        "success": True,
        "answer": result,
        "response": result,
        "reply": result,
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return jsonify({

        "status": "ok",

        "application": "Helpar",

        "ai_provider": (
            "Hugging Face"
            if HF_TOKEN
            else "Built-in only"
        ),

        "online_ai_configured": bool(
            HF_TOKEN
        ),

        "online_ai_model": HF_MODEL,

        "ollama_configured": bool(
            OLLAMA_URL
        ),

        "ollama_model": OLLAMA_MODEL,

        "creator_configured": bool(
            CREATOR_INFO.get("name")
            and CREATOR_INFO.get("name")
            != "YOUR NAME"
        ),

        "routes": {
            "home": "/",
            "login": "/login",
            "register": "/register",
            "chat": "/chat",
            "history": "/history",
            "new_chat": "/new-chat",
            "improve": "/improve",
            "health": "/health",
        },
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        logged_in=(
            "username" in session
        ),
        username=session.get(
            "username",
            "",
        ),
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "success": False,
        "message": "Route not found.",
        "path": request.path,
    }), 404


@app.errorhandler(500)
def internal_error(error):

    print(
        "INTERNAL SERVER ERROR:",
        repr(error),
    )

    return jsonify({
        "success": False,
        "message": (
            "Internal server error."
        ),
    }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000",
        )
    )

    print()
    print("=" * 55)
    print(" HELPAR")
    print("=" * 55)
    print(
        "AI Provider :",
        "Hugging Face"
        if HF_TOKEN
        else "Built-in answers",
    )
    print(
        "Model       :",
        HF_MODEL,
    )
    print(
        "Ollama      :",
        "Configured"
        if OLLAMA_URL
        else "Disabled",
    )
    print(
        "Creator     :",
        CREATOR_INFO["name"],
    )
    print(
        "Port        :",
        port,
    )
    print("=" * 55)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
