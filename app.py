import os
import re
import json
import uuid
import sqlite3
import ast
import operator
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
# HALPER
# Local basic answers + Ollama
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "halper-secret-key-change-this"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "DATA_DIR",
    BASE_DIR
)

os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# CREATOR INFORMATION
# ============================================================
# Change ONLY these public values.
#
# Do NOT put passwords, API keys, phone numbers,
# home address, or other private information here.
# ============================================================

CREATOR_INFO = {
    "name": "Soham Chandrahas Sanap",

    "role": "Creator and developer of Halper",

    "about": (
        "Halper is an educational AI project created "
        "to help students learn and solve questions."
    ),

    "extra": (
        "Halper can help with mathematics, physics, "
        "chemistry, biology, computer basics, and general "
        "educational questions."
    ),
}


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

OLLAMA_TIMEOUT = int(
    os.environ.get(
        "OLLAMA_TIMEOUT",
        "180"
    )
)


# ============================================================
# DATA FILES
# ============================================================

USERS_FILE = os.path.join(
    DATA_DIR,
    "users.json"
)

DATABASE_FILE = os.path.join(
    DATA_DIR,
    "halper.db"
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
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception as error:
        print(
            "JSON LOAD ERROR:",
            repr(error)
        )

        return default


def save_json(filename, data):
    temporary_file = filename + ".tmp"

    try:
        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        os.replace(
            temporary_file,
            filename
        )

        return True

    except Exception as error:
        print(
            "JSON SAVE ERROR:",
            repr(error)
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
        {}
    )


# ============================================================
# DATABASE
# ============================================================

def get_db():
    connection = sqlite3.connect(
        DATABASE_FILE,
        timeout=30
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
            CREATE INDEX IF NOT EXISTS
            idx_chats_username
            ON chats(username)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_messages_chat
            ON messages(chat_id)
            """
        )

        connection.commit()

    except Exception as error:

        connection.rollback()

        print(
            "DATABASE INITIALIZATION ERROR:",
            repr(error)
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
# CREATOR QUESTION DETECTION
# ============================================================

def is_creator_question(text):

    text = str(
        text or ""
    ).lower().strip()

    creator_phrases = [

        "who created you",
        "who is your creator",
        "who created halper",

        "who made you",
        "who made halper",

        "who developed you",
        "who developed halper",

        "who built you",
        "who built halper",

        "who programmed you",
        "who programmed halper",

        "who designed you",
        "who designed halper",

        "who is behind halper",
        "who is behind you",

        "who is your developer",
        "who is halper's developer",
        "who is halper developer",

        "who is your maker",
        "who is halper's maker",
        "who is halper maker",

        "tell me about your creator",
        "tell me about halper creator",

        "tell me about your developer",
        "tell me about halper developer",

        "who is your father",
        "who is halper's father",
        "who is halper father",

        "who is your dad",
        "who is halper's dad",
        "who is halper dad",
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

    halper_words = (
        "you",
        "halper",
        "your",
    )

    has_creator_word = any(
        word in text
        for word in creator_words
    )

    has_halper_reference = any(
        word in text
        for word in halper_words
    )

    return (
        has_creator_word
        and has_halper_reference
    )


def creator_response():

    name = CREATOR_INFO.get(
        "name",
        "the creator"
    )

    role = CREATOR_INFO.get(
        "role",
        "the developer"
    )

    project = CREATOR_INFO.get(
        "project",
        "Halper"
    )

    about = CREATOR_INFO.get(
        "about",
        ""
    )

    extra = CREATOR_INFO.get(
        "extra",
        ""
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
                "message": "Please login first."
            }), 401

        return function(
            *args,
            **kwargs
        )

    return wrapper


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
            (
                id,
                username,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                username,
                "New Chat",
                timestamp,
                timestamp
            )
        )

        connection.commit()

    finally:

        connection.close()

    return chat_id


def chat_exists(chat_id, username):

    connection = get_db()

    try:

        row = connection.execute(
            """
            SELECT id
            FROM chats
            WHERE id = ?
            AND username = ?
            """,
            (
                chat_id,
                username
            )
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
            (username,)
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
            username
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
    content
):

    connection = get_db()

    try:

        timestamp = now_iso()

        connection.execute(
            """
            INSERT INTO messages
            (
                chat_id,
                role,
                content,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                chat_id,
                role,
                content,
                timestamp
            )
        )

        connection.execute(
            """
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
            """,
            (
                timestamp,
                chat_id
            )
        )

        connection.commit()

        return True

    except Exception as error:

        connection.rollback()

        print(
            "SAVE MESSAGE ERROR:",
            repr(error)
        )

        return False

    finally:

        connection.close()


def set_chat_title(
    chat_id,
    title
):

    connection = get_db()

    try:

        connection.execute(
            """
            UPDATE chats
            SET title = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                now_iso(),
                chat_id
            )
        )

        connection.commit()

    finally:

        connection.close()


def get_chat(
    chat_id,
    username
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
            WHERE id = ?
            AND username = ?
            """,
            (
                chat_id,
                username
            )
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
            (chat_id,)
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
                    "timestamp": row["timestamp"]
                }

                for row in messages
            ]
        }

    finally:

        connection.close()


# ============================================================
# LOCAL MATHEMATICS
# ============================================================

# Only safe mathematical operators are allowed.
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_math_eval(expression):

    expression = expression.strip()

    if len(expression) > 100:
        raise ValueError(
            "Expression is too long."
        )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    def calculate(node):

        if isinstance(
            node,
            ast.Expression
        ):
            return calculate(
                node.body
            )

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):
                return node.value

            raise ValueError(
                "Invalid number."
            )

        if isinstance(
            node,
            ast.BinOp
        ):

            left = calculate(
                node.left
            )

            right = calculate(
                node.right
            )

            operation = SAFE_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Operator not allowed."
                )

            return operation(
                left,
                right
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            value = calculate(
                node.operand
            )

            operation = SAFE_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Operator not allowed."
                )

            return operation(
                value
            )

        raise ValueError(
            "Invalid expression."
        )

    result = calculate(tree)

    if isinstance(
        result,
        float
    ):

        if result.is_integer():
            return str(
                int(result)
            )

        return str(
            round(result, 10)
        )

    return str(result)


def local_math_answer(text):

    original = str(
        text or ""
    ).strip()

    if not original:
        return None

    expression = original.lower()

    # Remove common calculation wording.
    replacements = [
        "what is",
        "calculate",
        "solve",
        "find",
        "answer",
        "equals",
        "=",
        "?",
    ]

    for word in replacements:

        expression = expression.replace(
            word,
            ""
        )

    expression = expression.strip()

    # Convert common multiplication symbols.
    expression = expression.replace(
        "×",
        "*"
    )

    expression = expression.replace(
        "x",
        "*"
    )

    expression = expression.replace(
        "÷",
        "/"
    )

    # Remove commas from large numbers.
    expression = expression.replace(
        ",",
        ""
    )

    # Only allow a pure arithmetic expression.
    if not re.fullmatch(
        r"[0-9\s+\-*/%.()]+",
        expression
    ):
        return None

    if not re.search(
        r"[+\-*/%]",
        expression
    ):
        return None

    try:

        result = safe_math_eval(
            expression
        )

        return (
            f"The answer is **{result}**."
        )

    except Exception:
        return None


# ============================================================
# BASIC QUESTION ANSWERS
# ============================================================

BASIC_RESPONSES = {

    "hi": (
        "Hi! 👋 I'm Halper. "
        "How can I help you learn today?"
    ),

    "hello": (
        "Hello! 👋 I'm Halper. "
        "Ask me a question about Maths, Physics, "
        "Chemistry, Biology, computers, or another subject."
    ),

    "hey": (
        "Hey! 👋 What would you like to learn today?"
    ),

    "good morning": (
        "Good morning! ☀️ "
        "What would you like to study today?"
    ),

    "good afternoon": (
        "Good afternoon! "
        "What can I help you learn?"
    ),

    "good evening": (
        "Good evening! "
        "What would you like to study?"
    ),

    "how are you": (
        "I'm doing well! 😊 "
        "I'm ready to help you learn."
    ),

    "thanks": (
        "You're welcome! 😊"
    ),

    "thank you": (
        "You're welcome! 😊"
    ),

    "bye": (
        "Goodbye! 👋 Keep learning!"
    ),
}


def basic_question_answer(text):

    cleaned = str(
        text or ""
    ).strip().lower()

    cleaned = re.sub(
        r"[!?.,]+$",
        "",
        cleaned
    ).strip()

    if cleaned in BASIC_RESPONSES:

        return BASIC_RESPONSES[
            cleaned
        ]

    # Computer basics.
    if cleaned in (
        "what is a computer",
        "what is computer",
        "define computer",
    ):

        return (
            "**Computer:** A computer is an electronic "
            "device that accepts data, processes it, "
            "stores information, and produces output."
        )

    if cleaned in (
        "what is cpu",
        "what is a cpu",
        "define cpu",
    ):

        return (
            "**CPU:** CPU stands for Central Processing Unit. "
            "It executes instructions and performs calculations "
            "needed by a computer."
        )

    if cleaned in (
        "what is ram",
        "what is a ram",
        "define ram",
    ):

        return (
            "**RAM:** RAM stands for Random Access Memory. "
            "It temporarily stores data and instructions "
            "that the computer is actively using."
        )

    if cleaned in (
        "what is rom",
        "what is a rom",
        "define rom",
    ):

        return (
            "**ROM:** ROM stands for Read-Only Memory. "
            "It stores data that generally remains available "
            "when the device is turned off."
        )

    if cleaned in (
        "what is an operating system",
        "what is operating system",
        "define operating system",
    ):

        return (
            "An **operating system (OS)** is software that "
            "manages computer hardware and provides services "
            "for applications. Examples include Windows, "
            "Linux, Android, and macOS."
        )

    if cleaned in (
        "what is software",
        "define software",
    ):

        return (
            "**Software** is a collection of programs and "
            "instructions that tell a computer what to do."
        )

    if cleaned in (
        "what is hardware",
        "define hardware",
    ):

        return (
            "**Hardware** refers to the physical parts of a "
            "computer, such as the CPU, RAM, keyboard, mouse, "
            "storage drive, and monitor."
        )

    # Physics basics.
    if cleaned in (
        "what is force",
        "define force",
    ):

        return (
            "**Force** is a push or pull that can change the "
            "motion of an object. Its SI unit is the newton (N)."
        )

    if cleaned in (
        "what is speed",
        "define speed",
    ):

        return (
            "**Speed** is the distance travelled per unit time.\n\n"
            "Formula:\n"
            "Speed = Distance / Time\n\n"
            "SI unit: m/s"
        )

    if cleaned in (
        "what is velocity",
        "define velocity",
    ):

        return (
            "**Velocity** is displacement per unit time and "
            "has both magnitude and direction.\n\n"
            "SI unit: m/s"
        )

    if cleaned in (
        "what is acceleration",
        "define acceleration",
    ):

        return (
            "**Acceleration** is the rate of change of velocity "
            "with respect to time.\n\n"
            "a = Change in velocity / Time\n\n"
            "SI unit: m/s²"
        )

    if cleaned in (
        "what is newtons first law",
        "what is newton's first law",
    ):

        return (
            "**Newton's First Law:** An object remains at rest "
            "or continues moving with constant velocity unless "
            "acted upon by a net external force."
        )

    if cleaned in (
        "what is newtons second law",
        "what is newton's second law",
    ):

        return (
            "**Newton's Second Law:** The net force on an object "
            "equals its mass multiplied by its acceleration.\n\n"
            "F = ma"
        )

    if cleaned in (
        "what is newtons third law",
        "what is newton's third law",
    ):

        return (
            "**Newton's Third Law:** When one object exerts a "
            "force on another object, the second object exerts "
            "an equal and opposite force on the first."
        )

    # Chemistry basics.
    if cleaned in (
        "what is an atom",
        "what is atom",
        "define atom",
    ):

        return (
            "An **atom** is the basic unit of an element that "
            "retains the chemical properties of that element."
        )

    if cleaned in (
        "what is a molecule",
        "what is molecule",
        "define molecule",
    ):

        return (
            "A **molecule** is a group of two or more atoms "
            "chemically bonded together."
        )

    if cleaned in (
        "what is an element",
        "what is element",
        "define element",
    ):

        return (
            "A **chemical element** is a pure substance made "
            "of atoms with the same number of protons."
        )

    if cleaned in (
        "what is a compound",
        "what is compound",
        "define compound",
    ):

        return (
            "A **compound** is a substance formed when atoms "
            "of two or more different elements chemically "
            "combine in fixed proportions."
        )

    if cleaned in (
        "what is ph",
        "what is p h",
    ):

        return (
            "**pH** is a scale used to describe how acidic or "
            "basic an aqueous solution is. Lower values are "
            "generally more acidic, while higher values are "
            "generally more basic."
        )

    # Biology basics.
    if cleaned in (
        "what is a cell",
        "what is cell",
        "define cell",
    ):

        return (
            "A **cell** is the basic structural and functional "
            "unit of living organisms."
        )

    if cleaned in (
        "what is photosynthesis",
        "define photosynthesis",
    ):

        return (
            "**Photosynthesis** is the process by which green "
            "plants use light energy to make glucose from "
            "carbon dioxide and water, releasing oxygen."
        )

    if cleaned in (
        "what is respiration",
        "define respiration",
    ):

        return (
            "**Cellular respiration** is a set of chemical "
            "reactions through which cells release usable "
            "energy from nutrients such as glucose."
        )

    if cleaned in (
        "what is dna",
        "define dna",
    ):

        return (
            "**DNA** stands for deoxyribonucleic acid. "
            "It stores genetic information in living organisms "
            "and many viruses."
        )

    if cleaned in (
        "what is a gene",
        "what is gene",
        "define gene",
    ):

        return (
            "A **gene** is a segment of genetic material that "
            "contains information used to produce a functional "
            "product, such as a protein or functional RNA."
        )

    return None


# ============================================================
# LOCAL ANSWER ENGINE
# ============================================================

def get_local_answer(text):

    # Creator questions always handled locally.
    if is_creator_question(text):
        return creator_response()

    # Mathematics.
    math_answer = local_math_answer(text)

    if math_answer:
        return math_answer

    # Basic educational questions.
    basic_answer = basic_question_answer(text)

    if basic_answer:
        return basic_answer

    return None


# ============================================================
# OLLAMA CONNECTION CHECK
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )

        if response.status_code != 200:

            return (
                False,
                f"Ollama returned HTTP "
                f"{response.status_code}"
            )

        data = response.json()

        models = data.get(
            "models",
            []
        )

        model_names = []

        for model in models:

            if isinstance(
                model,
                dict
            ):

                name = model.get(
                    "name"
                )

                if name:
                    model_names.append(
                        name
                    )

        return (
            True,
            {
                "models": model_names,
                "selected_model": OLLAMA_MODEL,
                "model_available":
                    OLLAMA_MODEL in model_names,
            }
        )

    except requests.exceptions.ConnectionError:

        return (
            False,
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except Exception as error:

        return (
            False,
            str(error)
        )


# ============================================================
# OLLAMA AI
# ============================================================

def call_ollama(
    prompt,
    conversation=None
):

    if conversation is None:
        conversation = []

    messages = [

        {
            "role": "system",

            "content": (
                "You are Halper, a helpful educational AI tutor. "

                "Give accurate, clear and friendly answers. "

                "For mathematics, show useful steps and "
                "check calculations carefully. "

                "For physics, show formulas and units. "

                "For chemistry, check equations carefully. "

                "For biology, use correct scientific terms. "

                "For computer questions, explain concepts "
                "clearly for students. "

                "Do not invent facts. "

                "If you are unsure about something, say so. "

                "If asked about your creator, developer, "
                "maker, father, or who made you, the application "
                "handles that separately."
            )
        }
    ]

    # Add recent conversation.
    for item in conversation[-12:]:

        role = item.get(
            "role"
        )

        content = item.get(
            "content",
            ""
        )

        if role not in (
            "user",
            "assistant"
        ):
            continue

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "content": str(content)
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    payload = {

        "model": OLLAMA_MODEL,

        "messages": messages,

        "stream": False,

        "options": {
            "temperature": 0.2,
            "num_predict": 1500
        }
    }

    try:

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        if response.status_code != 200:

            print(
                "OLLAMA HTTP ERROR:",
                response.status_code,
                response.text
            )

            return (
                False,
                "Ollama returned HTTP "
                + str(response.status_code)
            )

        data = response.json()

        message = data.get(
            "message"
        )

        if not isinstance(
            message,
            dict
        ):

            return (
                False,
                "Ollama returned no message."
            )

        answer = message.get(
            "content",
            ""
        )

        answer = str(
            answer
        ).strip()

        if not answer:

            return (
                False,
                "Ollama returned an empty answer."
            )

        return (
            True,
            answer
        )

    except requests.exceptions.ConnectionError:

        return (
            False,
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            False,
            "Ollama took too long to respond."
        )

    except Exception as error:

        print(
            "OLLAMA ERROR:",
            repr(error)
        )

        return (
            False,
            str(error)
        )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["POST"]
)
def login():

    data = request.get_json(
        silent=True
    ) or {}

    login_value = str(
        data.get(
            "login",
            ""
        )
    ).strip()

    password = str(
        data.get(
            "password",
            ""
        )
    )

    if not login_value or not password:

        return jsonify({
            "success": False,
            "message":
                "Enter username/email and password."
        }), 400

    users = load_json(
        USERS_FILE,
        {}
    )

    username_found = None
    user_found = None

    for username, user in users.items():

        if not isinstance(
            user,
            dict
        ):
            continue

        email = str(
            user.get(
                "email",
                ""
            )
        ).lower()

        phone = str(
            user.get(
                "phone",
                ""
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
            "message":
                "Invalid username, email or phone."
        }), 401

    try:

        password_correct = check_password_hash(
            user_found.get(
                "password",
                ""
            ),
            password
        )

    except Exception:

        password_correct = False

    if not password_correct:

        return jsonify({
            "success": False,
            "message":
                "Incorrect password."
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
        "username": username_found
    })


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["POST"]
)
def register():

    data = request.get_json(
        silent=True
    ) or {}

    username = str(
        data.get(
            "username",
            ""
        )
    ).strip()

    email = str(
        data.get(
            "email",
            ""
        )
    ).strip()

    phone = str(
        data.get(
            "phone",
            ""
        )
    ).strip()

    password = str(
        data.get(
            "password",
            ""
        )
    )

    if len(username) < 3:

        return jsonify({
            "success": False,
            "message":
                "Username must contain at least 3 characters."
        }), 400

    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+",
        username
    ):

        return jsonify({
            "success": False,
            "message":
                "Username contains invalid characters."
        }), 400

    if not email and not phone:

        return jsonify({
            "success": False,
            "message":
                "Enter an email or phone number."
        }), 400

    if email:

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email
        ):

            return jsonify({
                "success": False,
                "message":
                    "Enter a valid email address."
            }), 400

    if len(password) < 6:

        return jsonify({
            "success": False,
            "message":
                "Password must contain at least 6 characters."
        }), 400

    users = load_json(
        USERS_FILE,
        {}
    )

    for existing_username, user in users.items():

        if not isinstance(
            user,
            dict
        ):
            continue

        if (
            existing_username.lower()
            == username.lower()
        ):

            return jsonify({
                "success": False,
                "message":
                    "Username already exists."
            }), 409

        existing_email = str(
            user.get(
                "email",
                ""
            )
        ).lower()

        if (
            email
            and existing_email
            == email.lower()
        ):

            return jsonify({
                "success": False,
                "message":
                    "Email already exists."
            }), 409

        existing_phone = str(
            user.get(
                "phone",
                ""
            )
        )

        if (
            phone
            and existing_phone
            == phone
        ):

            return jsonify({
                "success": False,
                "message":
                    "Phone number already exists."
            }), 409

    users[username] = {

        "email": email,

        "phone": phone,

        "password":
            generate_password_hash(
                password
            ),

        "created_at":
            now_iso()
    }

    if not save_json(
        USERS_FILE,
        users
    ):

        return jsonify({
            "success": False,
            "message":
                "Could not save the account."
        }), 500

    session["username"] = username

    session["chat_id"] = create_chat(
        username
    )

    return jsonify({
        "success": True,
        "message":
            "Account created successfully.",
        "username": username
    })


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
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
    methods=["POST"]
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
        "title": "New Chat"
    })


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
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
            (username,)
        ).fetchall()

        chats = []

        for row in rows:

            chats.append({
                "id": row["id"],
                "title":
                    row["title"]
                    or "New Chat",
                "created_at":
                    row["created_at"],
                "updated_at":
                    row["updated_at"]
            })

        return jsonify({
            "success": True,
            "chats": chats
        })

    except Exception as error:

        print(
            "HISTORY ERROR:",
            repr(error)
        )

        return jsonify({
            "success": False,
            "message":
                "Could not load history."
        }), 500

    finally:

        connection.close()


# ============================================================
# OPEN CHAT
# ============================================================

@app.route(
    "/chat/<chat_id>",
    methods=["GET"]
)
@login_required
def open_chat(chat_id):

    username = current_username()

    chat = get_chat(
        chat_id,
        username
    )

    if not chat:

        return jsonify({
            "success": False,
            "message":
                "Chat not found."
        }), 404

    session["chat_id"] = chat_id

    return jsonify({
        "success": True,
        "chat": chat
    })


# ============================================================
# MAIN CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
@login_required
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    user_message = str(
        data.get(
            "message",
            ""
        )
    ).strip()

    if not user_message:

        return jsonify({
            "success": False,
            "message":
                "Please enter a message."
        }), 400

    username = current_username()

    chat_id = get_current_chat()

    if not chat_id:

        return jsonify({
            "success": False,
            "message":
                "Could not create chat."
        }), 500

    # --------------------------------------------------------
    # LOCAL ANSWER FIRST
    # --------------------------------------------------------

    local_answer = get_local_answer(
        user_message
    )

    if local_answer:

        save_message(
            chat_id,
            "user",
            user_message
        )

        save_message(
            chat_id,
            "assistant",
            local_answer
        )

        current_chat = get_chat(
            chat_id,
            username
        )

        if (
            current_chat
            and current_chat.get("title")
            == "New Chat"
        ):

            set_chat_title(
                chat_id,
                user_message[:50]
            )

        return jsonify({
            "success": True,
            "answer": local_answer,
            "response": local_answer,
            "reply": local_answer,
            "message": local_answer,
            "chat_id": chat_id,
            "source": "local"
        })

    # --------------------------------------------------------
    # GET PREVIOUS CONVERSATION
    # --------------------------------------------------------

    old_chat = get_chat(
        chat_id,
        username
    )

    conversation = []

    if old_chat:

        conversation = old_chat.get(
            "messages",
            []
        )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    if not save_message(
        chat_id,
        "user",
        user_message
    ):

        return jsonify({
            "success": False,
            "message":
                "Could not save your message."
        }), 500

    # --------------------------------------------------------
    # CHAT TITLE
    # --------------------------------------------------------

    if (
        old_chat
        and old_chat.get("title")
        == "New Chat"
    ):

        title = user_message[:50]

        if len(user_message) > 50:
            title += "..."

        set_chat_title(
            chat_id,
            title
        )

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    success, answer = call_ollama(
        user_message,
        conversation
    )

    if not success:

        print(
            "AI ERROR:",
            answer
        )

        return jsonify({
            "success": False,

            "answer":
                "⚠️ " + answer,

            "error": answer,

            "chat_id": chat_id,

            "source": "ollama"
        }), 502

    # --------------------------------------------------------
    # SAVE AI RESPONSE
    # --------------------------------------------------------

    save_message(
        chat_id,
        "assistant",
        answer
    )

    return jsonify({
        "success": True,
        "answer": answer,
        "response": answer,
        "reply": answer,
        "message": answer,
        "chat_id": chat_id,
        "source": "ollama"
    })


# ============================================================
# IMPROVE
# ============================================================

@app.route(
    "/improve",
    methods=["POST"]
)
@login_required
def improve():

    data = request.get_json(
        silent=True
    ) or {}

    question = str(
        data.get(
            "question",
            ""
        )
    ).strip()

    answer = str(
        data.get(
            "answer",
            ""
        )
    ).strip()

    action = str(
        data.get(
            "action",
            "improve"
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
            "Make the answer shorter while keeping the important information."
    }

    instruction = instructions.get(
        action,
        instructions["improve"]
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

    # First try local answer for simple questions.
    local_result = get_local_answer(
        question
    )

    if local_result:

        return jsonify({
            "success": True,
            "answer": local_result,
            "response": local_result,
            "reply": local_result,
            "source": "local"
        })

    success, result = call_ollama(
        prompt
    )

    if not success:

        return jsonify({
            "success": False,
            "answer":
                "⚠️ " + result,
            "error": result
        }), 502

    return jsonify({
        "success": True,
        "answer": result,
        "response": result,
        "reply": result,
        "source": "ollama"
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    ollama_ok, ollama_info = check_ollama()

    return jsonify({

        "status": "ok",

        "application": "Halper",

        "ai_provider": "Ollama",

        "ollama_url":
            OLLAMA_URL,

        "ollama_model":
            OLLAMA_MODEL,

        "ollama_connected":
            ollama_ok,

        "ollama_info":
            ollama_info,

        "local_answers":
            True,

        "local_math":
            True,

        "creator_configured":
            bool(
                CREATOR_INFO.get("name")
                and
                CREATOR_INFO.get("name")
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

            "health": "/health"
        }
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
            ""
        )
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "success": False,
        "message":
            "Route not found.",
        "path":
            request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):

    print(
        "INTERNAL SERVER ERROR:",
        repr(error)
    )

    return jsonify({
        "success": False,
        "message":
            "Internal server error."
    }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    print()
    print("=" * 55)
    print(" HALPER")
    print("=" * 55)
    print(
        "AI Provider :",
        "Ollama"
    )
    print(
        "Model       :",
        OLLAMA_MODEL
    )
    print(
        "Ollama URL  :",
        OLLAMA_URL
    )
    print(
        "Creator     :",
        CREATOR_INFO["name"]
    )
    print(
        "Local Math  :",
        "Enabled"
    )
    print(
        "Basic PCMB  :",
        "Enabled"
    )
    print(
        "Port        :",
        port
    )
    print("=" * 55)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
