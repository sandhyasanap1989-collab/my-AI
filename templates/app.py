from flask import Flask, render_template, request, Response, session, redirect, url_for
import os
import re
import json
import threading

from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-key",
)

USERS_FILE = "users.json"
QUESTION_SETS_FILE = "question_sets.json"

FILE_LOCK = threading.Lock()


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# CREATOR
# ============================================================

CREATOR_NAME = "Soham Chandrahas Sanap"

CREATOR_RESPONSE = f"""
My creator is **{CREATOR_NAME}**.

He built My AI as a study assistant for Mathematics,
Physics, Chemistry and Biology.
""".strip()


# IMPORTANT:
# Every phrase that means "who made you" must reach this
# function instead of the normal AI.

CREATOR_PATTERNS = [
    r"\bwho\s+created\s+you\b",
    r"\bwho\s+made\s+you\b",
    r"\bwho\s+developed\s+you\b",
    r"\bwho\s+built\s+you\b",
    r"\bwho\s+is\s+your\s+creator\b",
    r"\bwho\s+is\s+your\s+maker\b",
    r"\bwho\s+is\s+your\s+father\b",
    r"\bwho\s+is\s+your\s+dad\b",
    r"\bwho\s+is\s+your\s+parent\b",
    r"\bwho\s+made\s+this\s+ai\b",
    r"\bwho\s+built\s+this\s+ai\b",
    r"\bwho\s+created\s+this\s+ai\b",
    r"\bwho\s+developed\s+this\s+ai\b",
    r"\btell\s+me\s+your\s+creator\b",
    r"\btell\s+me\s+who\s+made\s+you\b",
    r"\bmeans\s+who\s+created\s+you\b",
]


def is_creator_question(question):
    q = " ".join(
        question.lower().strip().split()
    )

    return any(
        re.search(pattern, q)
        for pattern in CREATOR_PATTERNS
    )


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return default


def save_json(path, data):
    temp = path + ".tmp"

    with open(
        temp,
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
        temp,
        path,
    )


# ============================================================
# USERS
# ============================================================

def load_users():
    data = load_json(
        USERS_FILE,
        {},
    )

    return (
        data
        if isinstance(data, dict)
        else {}
    )


def save_users(data):
    with FILE_LOCK:
        save_json(
            USERS_FILE,
            data,
        )


# ============================================================
# QUESTION SET STORAGE
# ============================================================

def get_username():
    return session.get("username")


def load_question_sets():
    data = load_json(
        QUESTION_SETS_FILE,
        {},
    )

    return (
        data
        if isinstance(data, dict)
        else {}
    )


def get_question_set():
    username = get_username()

    if not username:
        return None

    sets = load_question_sets()

    return sets.get(username)


def save_question_set(
    request_text,
    answer,
    subject,
    difficulty,
):
    username = get_username()

    if not username:
        return

    sets = load_question_sets()

    sets[username] = {
        "request": request_text,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty,
    }

    with FILE_LOCK:
        save_json(
            QUESTION_SETS_FILE,
            sets,
        )


def clear_question_set():
    username = get_username()

    if not username:
        return

    sets = load_question_sets()

    if username in sets:
        del sets[username]

        with FILE_LOCK:
            save_json(
                QUESTION_SETS_FILE,
                sets,
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def get_history():
    return session.get(
        "chat_history",
        [],
    )


def save_history(
    question,
    answer,
    category="normal",
):
    history = get_history()

    history.append({
        "question": question,
        "answer": answer[:12000],
        "category": category,
    })

    # Keep session small.
    session["chat_history"] = history[-8:]
    session.modified = True


# ============================================================
# FORMATTING
# ============================================================

PLAIN_MATH_RULE = """
Never use LaTeX.

Use readable notation:
1/2
√2
√3
sin θ
cos θ
tan θ
sec θ
cosec θ
cot θ
π
θ
α
β
Ω
μ
λ
Δ
×
÷
≤
≥
≠
≈
x²
x³

Do not output:
\\frac
\\sqrt
\\sin
\\cos
\\tan
\\left
\\right
\\[
\\]
\\(
\\)
$$
"""


def clean_output(text):
    if not text:
        return ""

    text = str(text)

    replacements = {
        "\\[": "",
        "\\]": "",
        "\\(": "",
        "\\)": "",
        "$$": "",

        "\\times": "×",
        "\\cdot": "·",
        "\\div": "÷",
        "\\pm": "±",

        "\\leq": "≤",
        "\\le": "≤",
        "\\geq": "≥",
        "\\ge": "≥",
        "\\neq": "≠",
        "\\ne": "≠",
        "\\approx": "≈",

        "\\rightarrow": "→",
        "\\to": "→",

        "\\Omega": "Ω",
        "\\mu": "μ",
        "\\lambda": "λ",
        "\\rho": "ρ",
        "\\sigma": "σ",
        "\\epsilon": "ε",
        "\\theta": "θ",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\Delta": "Δ",
        "\\omega": "ω",
        "\\pi": "π",
        "\\infty": "∞",
        "\\circ": "°",

        "\\sin": "sin",
        "\\cos": "cos",
        "\\tan": "tan",
        "\\sec": "sec",
        "\\csc": "cosec",
        "\\cot": "cot",
        "\\log": "log",
        "\\ln": "ln",

        "\\left": "",
        "\\right": "",

        "\\quad": " ",
        "\\qquad": " ",
        "\\;": " ",
        "\\,": " ",
        "\\!": "",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"\\(?:d)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"(\1/\2)",
        text,
    )

    text = re.sub(
        r"\\sqrt\s*\{([^{}]*)\}",
        r"√(\1)",
        text,
    )

    text = re.sub(
        r"\\(?:text|mathrm|mathbf|mathit|boxed)\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\[A-Za-z]+",
        lambda m: m.group(0)[1:],
        text,
    )

    text = text.replace(
        "√(2)",
        "√2",
    )
    text = text.replace(
        "√(3)",
        "√3",
    )
    text = text.replace(
        "√(5)",
        "√5",
    )

    text = text.replace(
        "^2",
        "²",
    )
    text = text.replace(
        "^3",
        "³",
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# SUBJECT DETECTION
# ============================================================

def detect_subject(question):
    q = question.lower()

    groups = {
        "math": [
            "math",
            "mathematics",
            "algebra",
            "equation",
            "quadratic",
            "polynomial",
            "trigonometry",
            "trigonometric",
            "sin",
            "cos",
            "tan",
            "sec",
            "cosec",
            "cot",
            "geometry",
            "identity",
            "surds",
            "probability",
            "permutation",
            "combination",
            "sequence",
            "series",
            "calculus",
            "integral",
            "derivative",
            "matrix",
            "determinant",
            "vector",
            "complex",
            "number theory",
            "divisibility",
            "function",
            "proof",
        ],

        "physics": [
            "physics",
            "force",
            "velocity",
            "acceleration",
            "newton",
            "momentum",
            "work",
            "energy",
            "power",
            "friction",
            "gravitation",
            "gravity",
            "current",
            "voltage",
            "resistance",
            "circuit",
            "electric",
            "magnetic",
            "lens",
            "mirror",
            "refraction",
            "heat",
            "temperature",
            "pressure",
            "density",
            "rotation",
            "torque",
            "oscillation",
            "shm",
            "wave",
            "capacitor",
            "induction",
        ],

        "chemistry": [
            "chemistry",
            "mole",
            "molarity",
            "stoichiometry",
            "redox",
            "acid",
            "base",
            "equilibrium",
            "enthalpy",
            "electrochemistry",
            "organic",
            "alkane",
            "alkene",
            "benzene",
            "reaction",
            "compound",
            "atom",
            "electron",
        ],

        "biology": [
            "biology",
            "cell",
            "mitosis",
            "meiosis",
            "chromosome",
            "gene",
            "genetics",
            "allele",
            "dna",
            "rna",
            "protein",
            "enzyme",
            "photosynthesis",
            "respiration",
            "plant",
            "animal",
            "reproduction",
            "reproduction",
            "evolution",
            "ecology",
            "hormone",
            "neuron",
        ],
    }

    scores = {
        key: sum(
            word in q
            for word in words
        )
        for key, words in groups.items()
    }

    best = max(
        scores,
        key=scores.get,
    )

    return (
        best
        if scores[best] > 0
        else "general"
    )


# ============================================================
# DIFFICULTY
# ============================================================

def detect_difficulty(question):
    q = question.lower()

    hard_words = [
        "jee advanced",
        "jee adv",
        "hardest",
        "hard",
        "difficult",
        "challenging",
        "olympiad",
        "prove",
        "derive",
        "derivation",
        "multi-step",
        "constraint",
        "optimization",
        "small oscillation",
        "multiple correct",
        "integer answer",
    ]

    score = sum(
        q.count(word) * 2
        for word in hard_words
    )

    if len(q) > 350:
        score += 2

    if score >= 6:
        return "advanced"

    if score >= 2:
        return "intermediate"

    return "basic"


# ============================================================
# AI PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a careful study assistant.

Answer the user's actual question.

Never switch subjects without a reason.
Never invent personal information.
Never use unrelated memory.

Read the complete question.
Check important calculations.
Answer every requested part.
"""

MATH_PROMPT = GENERAL_PROMPT + """
You are an expert Mathematics solver.

For difficult Mathematics:
- identify the target
- identify the given information
- choose the correct method
- solve step by step
- check algebra
- check arithmetic
- verify the final result
- verify MCQ options
- check domains and conditions
- actually prove identities when asked
"""

PHYSICS_PROMPT = GENERAL_PROMPT + """
You are an expert Physics solver.

For difficult Physics:
- understand the physical setup
- identify forces and constraints
- choose coordinates
- find equilibrium when necessary
- write the governing equations
- derive the result
- check signs
- check units
- check dimensions
- check limiting cases when useful
"""

CHEMISTRY_PROMPT = GENERAL_PROMPT + """
You are an expert Chemistry solver.
Check reactions, stoichiometry, units and numerical calculations.
"""

BIOLOGY_PROMPT = GENERAL_PROMPT + """
You are an expert Biology solver.
Use correct biological concepts and answer the exact question.
"""


def subject_prompt(subject):
    return {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }.get(
        subject,
        GENERAL_PROMPT,
    ) + "\n\n" + PLAIN_MATH_RULE


# ============================================================
# TEXT AI
# ============================================================

def text_ai(
    instructions,
    user_input,
    reasoning="medium",
    max_tokens=7000,
):
    if not hf_client:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    instructions += (
        "\n\n"
        + PLAIN_MATH_RULE
    )

    try:
        response = hf_client.responses.create(
            model=TEXT_MODEL,
            instructions=instructions,
            input=user_input,
            reasoning={
                "effort": reasoning,
            },
            max_output_tokens=max_tokens,
        )

        return clean_output(
            response.output_text or ""
        )

    except Exception as error:

        print(
            "TEXT RESPONSES ERROR:",
            repr(error),
        )

        response = hf_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            temperature=0.05,
            max_tokens=max_tokens,
        )

        if not response.choices:
            return ""

        return clean_output(
            response.choices[0]
            .message
            .content
            or ""
        )


# ============================================================
# VISION / CAMERA
# ============================================================

VISION_PROMPT = """
You are My AI's image-question solver.

Read the COMPLETE image.

Determine whether it contains Mathematics, Physics,
Chemistry, Biology or another academic topic.

Actually solve the question.

Do not invent text that is not visible.

For Mathematics:
check algebra and arithmetic.

For Physics:
check equations, units and dimensions.

For Chemistry:
check reactions and numerical values.

For Biology:
read diagrams and labels carefully.
"""


def analyze_image(
    image_data,
    question,
):
    if not hf_client:
        return (
            "❌ HF_TOKEN is missing."
        )

    if not image_data:
        return (
            "❌ No image was received."
        )

    if not image_data.startswith(
        "data:image/"
    ):
        return (
            "❌ Invalid image data."
        )

    if len(image_data) > 12_000_000:
        return (
            "❌ Image is too large."
        )

    prompt = (
        question.strip()
        if question.strip()
        else
        "Read this entire image and solve the academic question."
    )

    try:
        response = hf_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        VISION_PROMPT
                        + "\n\n"
                        + PLAIN_MATH_RULE
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                            },
                        },
                    ],
                },
            ],
            temperature=0.05,
            max_tokens=7000,
        )

        if not response.choices:
            return (
                "❌ Vision AI returned no answer."
            )

        answer = (
            response.choices[0]
            .message
            .content
            or ""
        )

        return clean_output(
            answer
        )

    except Exception as error:

        print(
            "VISION ERROR:",
            repr(error),
        )

        return (
            "❌ I could not analyze that image right now.\n\n"
            + str(error)
        )


# ============================================================
# FOLLOW-UP INTENT
# ============================================================

def normalize(text):
    return " ".join(
        text.lower().strip().split()
    )


def is_answer_key_request(q):
    x = normalize(q)

    exact = {
        "ans",
        "answer",
        "answers",
        "answer key",
        "show answers",
        "give answers",
        "give ans",
        "ans of all",
        "answers of all",
        "answer all",
        "give ans of all",
        "solutions",
    }

    if x in exact:
        return True

    return bool(
        re.search(
            r"\b(answers?|ans)\b.*\ball\b",
            x,
        )
    )


def is_mcq_conversion_request(q):
    x = normalize(q)

    phrases = [
        "multiple choice",
        "multiple-choice",
        "mcq",
        "mcqs",
        "mcq questions",
        "multiple choice questions",
    ]

    conversion_words = [
        "adapt",
        "convert",
        "turn",
        "change",
        "make",
        "transform",
    ]

    has_format = any(
        phrase in x
        for phrase in phrases
    )

    has_conversion = any(
        word in x
        for word in conversion_words
    )

    return (
        has_format
        and has_conversion
    )


def is_option_request(q):
    x = normalize(q)

    return x in {
        "with options",
        "add options",
        "give options",
        "make mcq",
        "make mcqs",
        "make them mcq",
        "make these mcq",
    }


def is_explain_all_request(q):
    x = normalize(q)

    return x in {
        "explain all",
        "explain answers",
        "explain all questions",
        "solve all",
        "solve all questions",
    }


def continuation_count(q):
    x = normalize(q)

    patterns = [
        r"^(other|another|next|more|remaining)\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?(?:the\s+)?other\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?another\s+(\d+)\s+questions?$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            x,
        )

        if match:
            number_group = (
                2
                if match.lastindex == 2
                else 1
            )

            return min(
                max(
                    int(
                        match.group(
                            number_group
                        )
                    ),
                    1,
                ),
                50,
            )

    return None


# ============================================================
# QUESTION GENERATION
# ============================================================

def requested_count(q):
    match = re.search(
        r"\b(\d+)\s+(?:questions?|mcqs?)\b",
        q.lower(),
    )

    if not match:
        return None

    return min(
        max(
            int(
                match.group(1)
            ),
            1,
        ),
        50,
    )


def extract_question_numbers(text):
    return sorted({
        int(number)
        for number in re.findall(
            r"(?:^|\n)\s*(\d+)\.\s+",
            text,
        )
    })


def last_question_number(text):
    numbers = extract_question_numbers(
        text
    )

    return (
        max(numbers)
        if numbers
        else 0
    )


def validate_mcq_batch(
    text,
    start,
    end,
):
    expected = set(
        range(
            start,
            end + 1,
        )
    )

    actual = set(
        extract_question_numbers(
            text
        )
    )

    missing = sorted(
        expected - actual
    )

    if missing:
        return False

    # Table detector.
    if text.count("|") > 2:
        return False

    # Require A-D somewhere for each block.
    blocks = re.split(
        r"(?:^|\n)\s*\d+\.\s+",
        text,
    )[1:]

    wanted = (
        end - start + 1
    )

    if len(blocks) < wanted:
        return False

    for block in blocks[:wanted]:
        options = set(
            re.findall(
                r"(?:^|\n)\s*([ABCD])\.\s+",
                block,
                flags=re.MULTILINE,
            )
        )

        if options != {
            "A",
            "B",
            "C",
            "D",
        }:
            return False

    return True


def make_mcq_batch(
    subject,
    difficulty,
    start,
    end,
    previous_context="",
):
    prompt = f"""
Generate exactly {end - start + 1}
multiple-choice questions.

Number them {start} through {end}.

STRICT FORMAT:

{start}. Question
A. Option
B. Option
C. Option
D. Option

STRICT RULES:
- No Markdown table.
- No | characters.
- No LaTeX.
- Every question complete.
- Every question has A, B, C and D.
- No answer key.
- No introduction.
- No conclusion.
- Keep the exact topic.
- Keep the exact subject.
- Make questions different.
- Check numbering before finishing.

Subject: {subject}
Difficulty: {difficulty}

Previous context:
{previous_context}
"""

    return text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="medium",
        max_tokens=7500,
    )


def generate_questions(
    subject,
    difficulty,
    count,
    start=1,
    previous_context="",
):
    pieces = []

    current = start
    final_number = (
        start
        + count
        - 1
    )

    while current <= final_number:
        end = min(
            current + 4,
            final_number,
        )

        batch = make_mcq_batch(
            subject,
            difficulty,
            current,
            end,
            previous_context,
        )

        attempts = 0

        while (
            not validate_mcq_batch(
                batch,
                current,
                end,
            )
            and attempts < 2
        ):
            attempts += 1

            batch = make_mcq_batch(
                subject,
                difficulty,
                current,
                end,
                previous_context,
            )

        if not validate_mcq_batch(
            batch,
            current,
            end,
        ):
            raise RuntimeError(
                f"Could not validate questions "
                f"{current}-{end}."
            )

        pieces.append(
            batch
        )

        current = end + 1

    result = clean_output(
        "\n\n".join(pieces)
    )

    return result


# ============================================================
# QUESTION-SET CONVERTER
# ============================================================

def convert_existing_set_to_mcq(
    active_set,
):
    subject = active_set[
        "subject"
    ]

    original = active_set[
        "answer"
    ]

    prompt = f"""
Convert the EXISTING question set below into
multiple-choice questions.

THIS IS A CONVERSION TASK, NOT A NEW QUESTION TASK.

CRITICAL RULES:

1. Use ONLY the supplied questions.
2. Keep EXACTLY the same subject.
3. Keep EXACTLY the same topic.
4. Keep the same question meaning.
5. Do NOT switch to Chemistry, Physics, Mathematics,
   Biology or another subject.
6. Do NOT invent unrelated questions.
7. Add exactly four options:
   A. ...
   B. ...
   C. ...
   D. ...
8. Keep the original numbering.
9. Return the COMPLETE set.
10. Do not provide the answer key.
11. Do not use a Markdown table.
12. Do not use the | character.
13. Do not use LaTeX.

SUBJECT:
{subject}

EXISTING QUESTIONS:
{original}
"""

    result = text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="medium",
        max_tokens=18000,
    )

    return result


# ============================================================
# ANSWER EXISTING SET
# ============================================================

def answer_existing_set(
    active_set,
):
    subject = active_set[
        "subject"
    ]

    original = active_set[
        "answer"
    ]

    prompt = f"""
Answer EVERY question in the existing question set.

This is an ANSWER-THE-EXISTING-SET task.

Rules:
- Use ONLY these questions.
- Keep original numbering.
- Do not create new questions.
- Do not switch subject.
- Give the correct option when options exist.
- Give a concise explanation.
- Do not discuss the creator.
- Check every answer.

Subject:
{subject}

Question set:
{original}
"""

    return text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="high",
        max_tokens=18000,
    )


# ============================================================
# ADVANCED MATH / PHYSICS VERIFICATION
# ============================================================

def verify_solution(
    question,
    answer,
    subject,
):
    if subject not in {
        "math",
        "physics",
    }:
        return answer

    verifier = f"""
You are the verification stage for a difficult
{subject} problem.

Check the proposed solution.

Check:
- equations
- algebra
- arithmetic
- signs
- units
- dimensions
- conditions
- final answer
- MCQ option

If anything is wrong, correct it.

Return only the corrected educational solution.
Do not reveal hidden chain-of-thought.
Do not use LaTeX.

QUESTION:
{question}

PROPOSED ANSWER:
{answer}
"""

    try:
        return text_ai(
            subject_prompt(subject),
            verifier,
            reasoning="high",
            max_tokens=10000,
        )

    except Exception:
        return answer


# ============================================================
# NORMAL CONTEXT
# ============================================================

def build_normal_context(question):
    history = [
        item
        for item in get_history()
        if item.get(
            "category"
        ) == "normal"
    ]

    if not history:
        return question

    # Long/new questions stand on their own.
    if len(
        question.split()
    ) > 7:
        return question

    recent = history[-4:]

    parts = [
        "Relevant recent normal conversation:"
    ]

    for item in recent:
        parts.append(
            "USER: "
            + item["question"]
        )
        parts.append(
            "MY AI: "
            + item["answer"]
        )

    parts.append(
        "CURRENT QUESTION: "
        + question
    )

    parts.append(
        "Use previous context only when it is clearly relevant."
    )

    return "\n".join(parts)


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        logged_in=(
            "username" in session
        ),
        username=session.get(
            "username"
        ),
    )


@app.route(
    "/register",
    methods=["POST"],
)
def register():
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    username = data.get(
        "username",
        "",
    ).strip()

    email = data.get(
        "email",
        "",
    ).strip().lower()

    phone = data.get(
        "phone",
        "",
    ).strip()

    password = data.get(
        "password",
        "",
    )

    if not username or not password:
        return {
            "success": False,
            "message":
                "Username and password are required.",
        }, 400

    if not email and not phone:
        return {
            "success": False,
            "message":
                "Enter either an email or phone number.",
        }, 400

    if len(password) < 6:
        return {
            "success": False,
            "message":
                "Password must be at least 6 characters.",
        }, 400

    all_users = load_users()

    for name, user in all_users.items():

        if name.lower() == username.lower():
            return {
                "success": False,
                "message":
                    "Username already exists.",
            }, 400

        if (
            email
            and user.get(
                "email",
                "",
            ).lower() == email
        ):
            return {
                "success": False,
                "message":
                    "Email already registered.",
            }, 400

        if (
            phone
            and user.get(
                "phone",
                "",
            ) == phone
        ):
            return {
                "success": False,
                "message":
                    "Phone already registered.",
            }, 400

    all_users[username] = {
        "email": email,
        "phone": phone,
        "password":
            generate_password_hash(
                password
            ),
    }

    save_users(
        all_users
    )

    session["username"] = username
    session["chat_history"] = []

    clear_question_set()

    return {
        "success": True,
        "message":
            "Account created successfully.",
    }


@app.route(
    "/login",
    methods=["POST"],
)
def login():
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    login_value = data.get(
        "login",
        "",
    ).strip()

    password = data.get(
        "password",
        "",
    )

    all_users = load_users()

    for username, user in all_users.items():

        matches = (
            username.lower()
            == login_value.lower()
            or
            user.get(
                "email",
                "",
            ).lower()
            == login_value.lower()
            or
            user.get(
                "phone",
                "",
            )
            == login_value
        )

        if matches:
            if check_password_hash(
                user.get(
                    "password",
                    "",
                ),
                password,
            ):
                session["username"] = username
                session["chat_history"] = []

                clear_question_set()

                return {
                    "success": True,
                    "message":
                        "Login successful.",
                }

            break

    return {
        "success": False,
        "message":
            "Invalid username/email/phone or password.",
    }, 401


@app.route("/logout")
def logout():
    session.clear()

    return redirect(
        url_for("home")
    )


@app.route(
    "/new-chat",
    methods=["POST"],
)
def new_chat():
    session["chat_history"] = []

    clear_question_set()

    return {
        "success": True,
    }


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"],
)
def chat():

    if "username" not in session:
        return Response(
            "Please login first.",
            status=401,
            mimetype="text/plain",
        )

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    question = data.get(
        "message",
        "",
    ).strip()

    image_data = data.get(
        "image",
        "",
    )

    # ========================================================
    # IMAGE / CAMERA FIRST
    # ========================================================

    if image_data:

        answer = analyze_image(
            image_data,
            question,
        )

        save_history(
            question or "Image question",
            answer,
            "normal",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    if not question:
        return Response(
            "Please type a question or capture an image.",
            mimetype="text/plain",
        )

    # ========================================================
    # CREATOR MUST BE CHECKED BEFORE NORMAL AI
    # ========================================================

    if is_creator_question(question):

        answer = clean_output(
            CREATOR_RESPONSE
        )

        save_history(
            question,
            answer,
            "creator",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # ========================================================
    # BASIC CHAT
    # ========================================================

    normalized = normalize(question)

    if normalized in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
    }:

        answer = (
            "Hello! 👋 I'm My AI.\n\n"
            "Ask Mathematics, Physics, Chemistry, "
            "Biology or general questions."
        )

        save_history(
            question,
            answer,
            "normal",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # ========================================================
    # GET ACTIVE QUESTION SET
    # ========================================================

    active_set = get_question_set()

    # ========================================================
    # EXACT QUESTION-SET FOLLOW-UP ROUTING
    #
    # THIS MUST HAPPEN BEFORE NORMAL AI.
    # ========================================================

    if active_set:

        # ----------------------------------------------------
        # CONVERT TO MCQ
        # ----------------------------------------------------

        if is_mcq_conversion_request(
            question
        ) or is_option_request(
            question
        ):

            answer = convert_existing_set_to_mcq(
                active_set
            )

            answer = clean_output(
                answer
            )

            # Replace active set with converted set.
            save_question_set(
                active_set["request"],
                answer,
                active_set["subject"],
                active_set["difficulty"],
            )

            save_history(
                question,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        # ----------------------------------------------------
        # ANSWERS OF ALL
        # ----------------------------------------------------

        if is_answer_key_request(
            question
        ):

            answer = answer_existing_set(
                active_set
            )

            answer = clean_output(
                answer
            )

            save_history(
                question,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        # ----------------------------------------------------
        # EXPLAIN ALL
        # ----------------------------------------------------

        if is_explain_all_request(
            question
        ):

            answer = text_ai(
                subject_prompt(
                    active_set["subject"]
                )
                + """
Solve every question in the existing question set.

Keep the original numbering.
Do not skip questions.
Do not create unrelated questions.
""",
                active_set["answer"],
                reasoning="high",
                max_tokens=18000,
            )

            answer = clean_output(
                answer
            )

            save_history(
                question,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        # ----------------------------------------------------
        # CONTINUE WITH A NUMBER
        # ----------------------------------------------------

        amount = continuation_count(
            question
        )

        if amount is not None:

            start = (
                last_question_number(
                    active_set["answer"]
                )
                + 1
            )

            answer = generate_questions(
                active_set["subject"],
                active_set["difficulty"],
                amount,
                start,
                active_set["answer"],
            )

            combined = clean_output(
                active_set["answer"]
                + "\n\n"
                + answer
            )

            save_question_set(
                active_set["request"],
                combined,
                active_set["subject"],
                active_set["difficulty"],
            )

            save_history(
                question,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        # ----------------------------------------------------
        # "CONTINUE" / "MORE"
        # ----------------------------------------------------

        if normalized in {
            "continue",
            "more",
            "another",
        }:

            start = (
                last_question_number(
                    active_set["answer"]
                )
                + 1
            )

            answer = generate_questions(
                active_set["subject"],
                active_set["difficulty"],
                5,
                start,
                active_set["answer"],
            )

            combined = clean_output(
                active_set["answer"]
                + "\n\n"
                + answer
            )

            save_question_set(
                active_set["request"],
                combined,
                active_set["subject"],
                active_set["difficulty"],
            )

            save_history(
                question,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

    # ========================================================
    # NEW QUESTION SET
    # ========================================================

    count = requested_count(
        question
    )

    if count is not None:

        detected_subject = detect_subject(
            question
        )

        detected_difficulty = detect_difficulty(
            question
        )

        answer = generate_questions(
            detected_subject,
            detected_difficulty,
            count,
        )

        answer = clean_output(
            answer
        )

        save_question_set(
            question,
            answer,
            detected_subject,
            detected_difficulty,
        )

        save_history(
            question,
            answer,
            "question_generation",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # ========================================================
    # NORMAL / HARD MATH / PHYSICS
    # ========================================================

    detected_subject = detect_subject(
        question
    )

    detected_difficulty = detect_difficulty(
        question
    )

    context = build_normal_context(
        question
    )

    if detected_difficulty == "advanced":

        answer = text_ai(
            subject_prompt(
                detected_subject
            )
            + """
ADVANCED SOLVING MODE

Solve very carefully.

Recheck:
- equations
- algebra
- arithmetic
- signs
- units
- dimensions
- conditions
- final result
- MCQ options

Never guess.
""",
            context,
            reasoning="high",
            max_tokens=10000,
        )

        if detected_subject in {
            "math",
            "physics",
        }:

            answer = verify_solution(
                question,
                answer,
                detected_subject,
            )

    elif detected_difficulty == "intermediate":

        answer = text_ai(
            subject_prompt(
                detected_subject
            ),
            context,
            reasoning="medium",
            max_tokens=6000,
        )

    else:

        answer = text_ai(
            subject_prompt(
                detected_subject
            ),
            context,
            reasoning="low",
            max_tokens=3500,
        )

    answer = clean_output(
        answer
    )

    save_history(
        question,
        answer,
        "normal",
    )

    return Response(
        answer,
        mimetype="text/plain",
    )


# ============================================================
# IMPROVE / CHECK / EXPLAIN / SHORTEN
# ============================================================

@app.route(
    "/improve",
    methods=["POST"],
)
def improve():

    if "username" not in session:
        return Response(
            "Please login first.",
            status=401,
            mimetype="text/plain",
        )

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    question = data.get(
        "question",
        "",
    )

    old_answer = data.get(
        "answer",
        "",
    )

    action = data.get(
        "action",
        "improve",
    )

    tasks = {
        "improve":
            "Improve the answer and make it clearer.",
        "check":
            "Check the answer for mistakes and correct them.",
        "explain":
            "Explain the answer in more detail.",
        "short":
            "Make the answer shorter without losing important information.",
    }

    task = tasks.get(
        action,
        tasks["improve"],
    )

    detected_subject = detect_subject(
        question
    )

    answer = text_ai(
        subject_prompt(
            detected_subject
        )
        + "\n\n"
        + task,
        "QUESTION:\n"
        + question
        + "\n\nANSWER:\n"
        + old_answer,
        reasoning=(
            "high"
            if detected_subject in {
                "math",
                "physics",
            }
            else "medium"
        ),
        max_tokens=9000,
    )

    return Response(
        clean_output(answer),
        mimetype="text/plain",
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():
    return {
        "status": "ok",
        "hf_configured": bool(HF_TOKEN),
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "creator_routing": True,
        "question_set_routing": True,
        "math_verification": True,
        "physics_verification": True,
        "vision": True,
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MY AI")
    print("=" * 60)
    print("HF configured       :", bool(HF_TOKEN))
    print("Creator routing     : ON")
    print("Father/dad routing  : ON")
    print("Question-set memory : ON")
    print("MCQ conversion      : ON")
    print("Answer-key routing  : ON")
    print("Follow-ups          : ON")
    print("Hard Mathematics    : ON")
    print("Hard Physics        : ON")
    print("Image analysis      : ON")
    print("Plain math          : ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
