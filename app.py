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
    url_for
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# HALPER
# Local answers + Ollama
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

CREATOR_INFO = {
    "name": "YOUR NAME",
    "role": "Creator and developer of Halper",
    "project": "Halper",

    "about": (
        "Halper is an educational AI tutor designed "
        "to help students learn mathematics, physics, "
        "chemistry, biology and other subjects."
    ),

    "extra": (
        "Halper combines local educational answers "
        "with optional AI assistance."
    )
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
        "120"
    )
)


# ============================================================
# FILES
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

            if os.path.exists(
                temporary_file
            ):
                os.remove(
                    temporary_file
                )

        except Exception:
            pass

        return False


# ============================================================
# USERS
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
# AUTHENTICATION
# ============================================================

def current_username():

    return session.get(
        "username"
    )


def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if not session.get(
            "username"
        ):

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
# CHAT DATABASE FUNCTIONS
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
# CREATOR QUESTION
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
        "who is halper dad"
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
        "made"
    )

    halper_words = (
        "you",
        "halper",
        "your"
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
        and
        has_halper_reference
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
# SAFE LOCAL MATH CALCULATOR
# ============================================================

_ALLOWED_OPERATORS = {

    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,

    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def safe_math_eval(node):

    if isinstance(
        node,
        ast.Expression
    ):

        return safe_math_eval(
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
            "Invalid number"
        )

    if isinstance(
        node,
        ast.BinOp
    ):

        operator_type = type(
            node.op
        )

        if operator_type not in _ALLOWED_OPERATORS:

            raise ValueError(
                "Operator not allowed"
            )

        left = safe_math_eval(
            node.left
        )

        right = safe_math_eval(
            node.right
        )

        if operator_type is ast.Pow:

            if abs(right) > 10:

                raise ValueError(
                    "Power too large"
                )

        return _ALLOWED_OPERATORS[
            operator_type
        ](
            left,
            right
        )

    if isinstance(
        node,
        ast.UnaryOp
    ):

        operator_type = type(
            node.op
        )

        if operator_type not in _ALLOWED_OPERATORS:

            raise ValueError(
                "Operator not allowed"
            )

        return _ALLOWED_OPERATORS[
            operator_type
        ](
            safe_math_eval(
                node.operand
            )
        )

    raise ValueError(
        "Invalid expression"
    )


def calculate_math(text):

    expression = str(
        text or ""
    ).strip()

    expression = expression.replace(
        "×",
        "*"
    )

    expression = expression.replace(
        "÷",
        "/"
    )

    expression = expression.replace(
        "^",
        "**"
    )

    expression = expression.replace(
        ",",
        ""
    )

    patterns = [

        r"^\s*(?:what\s+is\s+)?"
        r"([0-9+\-*/().%\s*]+)"
        r"\s*\??\s*$",

        r"^\s*calculate\s+"
        r"([0-9+\-*/().%\s*]+)"
        r"\s*\??\s*$",

        r"^\s*solve\s+"
        r"([0-9+\-*/().%\s*]+)"
        r"\s*\??\s*$"
    ]

    matched_expression = None

    for pattern in patterns:

        match = re.fullmatch(
            pattern,
            expression,
            re.IGNORECASE
        )

        if match:

            matched_expression = (
                match.group(1)
            )

            break

    if not matched_expression:

        return None

    if len(
        matched_expression
    ) > 100:

        return None

    try:

        tree = ast.parse(
            matched_expression,
            mode="eval"
        )

        result = safe_math_eval(
            tree
        )

        if isinstance(
            result,
            float
        ):

            if result.is_integer():

                result = int(result)

            else:

                result = round(
                    result,
                    10
                )

        return (
            f"The answer is **{result}**."
        )

    except Exception:

        return None


# ============================================================
# BASIC ANSWERS
# ============================================================

def local_answer(text):

    original = str(
        text or ""
    ).strip()

    lower = original.lower()

    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if lower in greetings:

        return (
            "Hello! 👋\n\n"
            "I'm Halper, your educational AI tutor. "
            "You can ask me questions about mathematics, "
            "physics, chemistry, biology, computers, "
            "or other subjects."
        )

    # --------------------------------------------------------
    # HOW ARE YOU
    # --------------------------------------------------------

    if lower in {
        "how are you",
        "how are you?",
        "how r u"
    }:

        return (
            "I'm doing great! 😊 "
            "What would you like to learn today?"
        )

    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    if lower in {
        "thanks",
        "thank you",
        "thx",
        "thank u"
    }:

        return (
            "You're welcome! 😊 "
            "Ask me another question whenever you want."
        )

    # --------------------------------------------------------
    # WHO ARE YOU
    # --------------------------------------------------------

    if lower in {
        "who are you",
        "what are you",
        "what is halper",
        "what is halper?"
    }:

        return (
            "I'm Halper, an educational AI tutor. "
            "I can help with mathematics, physics, "
            "chemistry, biology, computer basics, "
            "and many other learning topics."
        )

    # --------------------------------------------------------
    # CREATOR
    # --------------------------------------------------------

    if is_creator_question(
        original
    ):

        return creator_response()

    # --------------------------------------------------------
    # LOCAL MATH
    # --------------------------------------------------------

    math_result = calculate_math(
        original
    )

    if math_result:

        return math_result

    # --------------------------------------------------------
    # MATHEMATICS
    # --------------------------------------------------------

    if (
        "pythagoras theorem" in lower
        or "pythagorean theorem" in lower
    ):

        return (
            "The Pythagorean theorem applies to a "
            "right-angled triangle:\n\n"
            "a² + b² = c²\n\n"
            "where c is the hypotenuse."
        )

    if (
        "quadratic formula" in lower
    ):

        return (
            "For ax² + bx + c = 0, the quadratic "
            "formula is:\n\n"
            "x = (-b ± √(b² - 4ac)) / 2a"
        )

    if (
        "newton's second law" in lower
        or "newtons second law" in lower
    ):

        return (
            "Newton's second law states that the net "
            "force on an object equals its mass multiplied "
            "by its acceleration.\n\n"
            "F = ma\n\n"
            "F = force in newtons (N)\n"
            "m = mass in kilograms (kg)\n"
            "a = acceleration in m/s²"
        )

    if (
        "newton's first law" in lower
        or "newtons first law" in lower
    ):

        return (
            "Newton's first law is the law of inertia. "
            "An object remains at rest or continues in "
            "uniform straight-line motion unless acted "
            "upon by a net external force."
        )

    if (
        "newton's third law" in lower
        or "newtons third law" in lower
    ):

        return (
            "Newton's third law states that when one object "
            "exerts a force on another object, the second "
            "object exerts an equal and opposite force "
            "on the first."
        )

    # --------------------------------------------------------
    # PHYSICS
    # --------------------------------------------------------

    if (
        "speed" in lower
        and "velocity" in lower
    ):

        return (
            "Speed is the distance travelled per unit time "
            "and is a scalar quantity.\n\n"
            "Velocity is displacement per unit time and "
            "is a vector quantity.\n\n"
            "Speed = Distance / Time\n"
            "Velocity = Displacement / Time"
        )

    if (
        "kinetic energy" in lower
    ):

        return (
            "Kinetic energy is the energy possessed by an "
            "object because of its motion.\n\n"
            "KE = ½mv²\n\n"
            "m = mass\n"
            "v = velocity"
        )

    if (
        "potential energy" in lower
    ):

        return (
            "Gravitational potential energy near Earth's "
            "surface is:\n\n"
            "PE = mgh\n\n"
            "m = mass\n"
            "g = acceleration due to gravity\n"
            "h = height"
        )

    if (
        "work done" in lower
        and "force" in lower
    ):

        return (
            "When a constant force acts in the direction "
            "of displacement:\n\n"
            "W = F × s\n\n"
            "More generally:\n"
            "W = Fs cos θ"
        )

    if (
        "ohm's law" in lower
        or "ohms law" in lower
    ):

        return (
            "Ohm's law states:\n\n"
            "V = IR\n\n"
            "V = voltage\n"
            "I = current\n"
            "R = resistance"
        )

    if (
        "gravity" in lower
        and (
            "earth" in lower
            or "acceleration" in lower
        )
    ):

        return (
            "Near Earth's surface, the acceleration due "
            "to gravity is approximately 9.8 m/s² downward."
        )

    # --------------------------------------------------------
    # CHEMISTRY
    # --------------------------------------------------------

    if (
        "photosynthesis" in lower
    ):

        return (
            "Photosynthesis is the process by which green "
            "plants use light energy to make glucose from "
            "carbon dioxide and water.\n\n"
            "Simplified equation:\n"
            "6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂\n\n"
            "Chlorophyll captures the light energy needed "
            "for the process."
        )

    if (
        "atom" in lower
        and (
            "what is" in lower
            or "define" in lower
        )
    ):

        return (
            "An atom is the basic unit of an element that "
            "retains the chemical properties of that element. "
            "It contains a nucleus with protons and neutrons, "
            "surrounded by electrons."
        )

    if (
        "mole" in lower
        and (
            "chemistry" in lower
            or "what is" in lower
            or "define" in lower
        )
    ):

        return (
            "A mole is the SI unit used to measure amount "
            "of substance. One mole contains approximately "
            "6.022 × 10²³ elementary entities."
        )

    if (
        "ph" in lower
        and "scale" in lower
    ):

        return (
            "The pH scale indicates how acidic or basic "
            "an aqueous solution is. At about 25°C, pH 7 "
            "is neutral, values below 7 are acidic, and "
            "values above 7 are basic."
        )

    if (
        "periodic table" in lower
    ):

        return (
            "The periodic table organizes chemical elements "
            "according to their atomic number and recurring "
            "chemical properties."
        )

    # --------------------------------------------------------
    # BIOLOGY
    # --------------------------------------------------------

    if (
        "cell" in lower
        and (
            "what is" in lower
            or "define" in lower
        )
    ):

        return (
            "A cell is the basic structural and functional "
            "unit of life. Living organisms may consist of "
            "one cell or many cells."
        )

    if (
        "mitochondria" in lower
    ):

        return (
            "Mitochondria are organelles involved in cellular "
            "energy production. They are often called the "
            "powerhouses of the cell because they produce "
            "much of the cell's ATP."
        )

    if (
        "dna" in lower
    ):

        return (
            "DNA, or deoxyribonucleic acid, stores genetic "
            "information used in the development and functioning "
            "of living organisms."
        )

    if (
        "respiration" in lower
        and "cellular" in lower
    ):

        return (
            "Cellular respiration is the set of metabolic "
            "processes through which cells release usable "
            "energy from nutrients such as glucose."
        )

    if (
        "human heart" in lower
    ):

        return (
            "The human heart is a muscular organ that pumps "
            "blood through the circulatory system. It has "
            "four chambers: two atria and two ventricles."
        )

    # --------------------------------------------------------
    # COMPUTER BASICS
    # --------------------------------------------------------

    if (
        "what is cpu" in lower
        or "what is a cpu" in lower
    ):

        return (
            "CPU stands for Central Processing Unit. "
            "It executes instructions and performs the "
            "main processing tasks of a computer."
        )

    if (
        "what is ram" in lower
        or "what is a ram" in lower
    ):

        return (
            "RAM stands for Random Access Memory. "
            "It temporarily stores data and programs "
            "that the computer is actively using."
        )

    if (
        "what is rom" in lower
        or "what is a rom" in lower
    ):

        return (
            "ROM stands for Read-Only Memory. It is "
            "non-volatile memory used to store information "
            "that generally needs to remain available "
            "when power is removed."
        )

    if (
        "what is operating system" in lower
        or "what is an operating system" in lower
    ):

        return (
            "An operating system is system software that "
            "manages computer hardware and provides services "
            "for applications. Examples include Windows, "
            "Linux and macOS."
        )

    if (
        "what is gpu" in lower
        or "what is a gpu" in lower
    ):

        return (
            "GPU stands for Graphics Processing Unit. "
            "It is designed to perform graphics and many "
            "parallel computation tasks efficiently."
        )

    if (
        "what is python" in lower
        or "what is python programming" in lower
    ):

        return (
            "Python is a high-level programming language "
            "known for its readable syntax. It is widely "
            "used for education, web development, automation, "
            "data science and artificial intelligence."
        )

    # --------------------------------------------------------
    # NO LOCAL ANSWER
    # --------------------------------------------------------

    return None


# ============================================================
# OLLAMA CHECK
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=5
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
                    OLLAMA_MODEL in model_names
            }
        )

    except requests.exceptions.ConnectionError:

        return (
            False,
            "Ollama is not available."
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
                "You are Halper, an educational AI tutor. "
                "Give accurate, clear and friendly answers. "
                "For mathematics, show useful steps. "
                "For physics, show formulas and units. "
                "For chemistry, check equations carefully. "
                "For biology, use correct scientific terms. "
                "Do not invent facts. "
                "Do not claim to be human. "
                "If asked about your creator, use the "
                "creator information supplied by the application."
            )
        }

    ]

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
            "temperature": 0.3,
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
                response.status_code
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

        answer = str(
            message.get(
                "content",
                ""
            )
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
            "Ollama is not available."
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
# FINAL ANSWER ENGINE
# ============================================================

def get_answer(
    user_message,
    conversation=None
):

    # --------------------------------------------------------
    # FIRST: LOCAL ANSWERS
    # --------------------------------------------------------

    local = local_answer(
        user_message
    )

    if local:

        return (
            local,
            "local"
        )

    # --------------------------------------------------------
    # SECOND: OLLAMA
    # --------------------------------------------------------

    ollama_success, ollama_answer = call_ollama(
        user_message,
        conversation
    )

    if ollama_success:

        return (
            ollama_answer,
            "ollama"
        )

    # --------------------------------------------------------
    # THIRD: SAFE FALLBACK
    # --------------------------------------------------------

    return (
        "I can still help with basic questions, "
        "math calculations, and PCMB topics even when "
        "the AI service is unavailable.\n\n"
        "Try asking something like:\n"
        "• What is Newton's second law?\n"
        "• Explain photosynthesis\n"
        "• What is kinetic energy?\n"
        "• What is DNA?\n"
        "• What is CPU?\n"
        "• 21122 + 3947829",
        "fallback"
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
            or
            email
            == login_value.lower()
            or
            phone
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
            and
            existing_email
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
            and
            existing_phone
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
        "username":
            username
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
                    or
                    "New Chat",

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

# IMPORTANT:
# This fixes the earlier route problem.
# It MUST contain <chat_id>.

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
    # OLD CONVERSATION
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
    # TITLE
    # --------------------------------------------------------

    if (
        old_chat
        and
        old_chat.get(
            "title"
        )
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
    # ANSWER
    # --------------------------------------------------------

    answer, source = get_answer(
        user_message,
        conversation
    )

    # --------------------------------------------------------
    # SAVE ANSWER
    # --------------------------------------------------------

    save_message(
        chat_id,
        "assistant",
        answer
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "answer": answer,

        "response": answer,

        "reply": answer,

        "message": answer,

        "chat_id": chat_id,

        "source": source
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
            "Check the answer carefully and correct errors.",

        "explain":
            "Explain the answer step by step in simple language.",

        "short":
            "Make the answer shorter while keeping important information."
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

    # Try local answer first.

    local_result = local_answer(
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

    # Try Ollama.

    success, result = call_ollama(
        prompt
    )

    if success:

        return jsonify({
            "success": True,
            "answer": result,
            "response": result,
            "reply": result,
            "source": "ollama"
        })

    # Safe fallback.

    return jsonify({
        "success": True,
        "answer": answer,
        "response": answer,
        "reply": answer,
        "source": "fallback"
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

        "creator_configured":
            bool(
                CREATOR_INFO.get(
                    "name"
                )
                and
                CREATOR_INFO.get(
                    "name"
                ) != "YOUR NAME"
            ),

        "local_answers":
            True,

        "routes": {

            "home": "/",

            "login": "/login",

            "register": "/register",

            "chat": "/chat",

            "open_chat":
                "/chat/<chat_id>",

            "history": "/history",

            "new_chat":
                "/new-chat",

            "improve":
                "/improve",

            "health":
                "/health"
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
            "username"
            in session
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
        "Ollama + Local Answers"
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
        "Port        :",
        port
    )
    print(
        "Local Math  :",
        "Enabled"
    )
    print(
        "Basic PCMB  :",
        "Enabled"
    )
    print("=" * 55)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
