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
    "change-this-secret-key"
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
        api_key=HF_TOKEN
    )


# ============================================================
# CREATOR
# ============================================================

CREATOR_NAME = "Soham Chandrahas Sanap"

CREATOR_RESPONSE = f"""
My creator is **{CREATOR_NAME}**.

He built Halper as a study assistant for
Mathematics, Physics, Chemistry and Biology.
""".strip()


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
]


def is_creator_question(question):
    q = " ".join(question.lower().strip().split())

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
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError):
        return default


def save_json(path, data):
    temp = path + ".tmp"

    with open(temp, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    os.replace(temp, path)


# ============================================================
# USERS
# ============================================================

def load_users():
    data = load_json(USERS_FILE, {})

    if isinstance(data, dict):
        return data

    return {}


def save_users(data):
    with FILE_LOCK:
        save_json(USERS_FILE, data)


# ============================================================
# QUESTION SET STORAGE
# ============================================================

def get_username():
    return session.get("username")


def load_question_sets():
    data = load_json(QUESTION_SETS_FILE, {})

    if isinstance(data, dict):
        return data

    return {}


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
    difficulty
):
    username = get_username()

    if not username:
        return

    sets = load_question_sets()

    sets[username] = {
        "request": request_text,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty
    }

    with FILE_LOCK:
        save_json(
            QUESTION_SETS_FILE,
            sets
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
                sets
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def get_history():
    return session.get("chat_history", [])


def save_history(
    question,
    answer,
    category="normal"
):
    history = get_history()

    history.append({
        "question": question,
        "answer": str(answer)[:12000],
        "category": category
    })

    session["chat_history"] = history[-8:]
    session.modified = True


# ============================================================
# MATH FORMATTING
# ============================================================

PLAIN_MATH_RULE = """
IMPORTANT MATHEMATICS FORMATTING RULES:

Never use LaTeX.

Use readable notation such as:

1/2
√2
√(25+x)
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
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\approx": "≈",
        r"\rightarrow": "→",
        r"\to": "→",
        r"\Omega": "Ω",
        r"\mu": "μ",
        r"\lambda": "λ",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\epsilon": "ε",
        r"\theta": "θ",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\Delta": "Δ",
        r"\omega": "ω",
        r"\pi": "π",
        r"\infty": "∞",
        r"\circ": "°",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\sec": "sec",
        r"\csc": "cosec",
        r"\cot": "cot",
        r"\log": "log",
        r"\ln": "ln",
        r"\left": "",
        r"\right": "",
        r"\quad": " ",
        r"\qquad": " "
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Basic frac conversion
    text = re.sub(
        r"\\(?:d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1/\2)",
        text
    )

    # Basic sqrt conversion
    text = re.sub(
        r"\\sqrt\s*\{([^{}]+)\}",
        r"√(\1)",
        text
    )

    # Remove math wrappers
    text = text.replace(r"\[", "")
    text = text.replace(r"\]", "")
    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")
    text = text.replace("$$", "")

    # Powers
    text = text.replace("^2", "²")
    text = text.replace("^3", "³")

    # Cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

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
            "square root",
            "sqrt"
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
            "induction"
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
            "electron"
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
            "evolution",
            "ecology",
            "hormone",
            "neuron"
        ]
    }

    scores = {}

    for subject, words in groups.items():
        score = 0

        for word in words:
            if word in q:
                score += 1

        scores[subject] = score

    best = max(scores, key=scores.get)

    if scores[best] > 0:
        return best

    # Mathematical symbols/numbers often indicate math
    if re.search(
        r"[0-9]\s*[\+\-\*\/=√²³]",
        q
    ):
        return "math"

    return "general"


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
        "multiple correct",
        "integer answer"
    ]

    score = 0

    for word in hard_words:
        if word in q:
            score += 2

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
You are Halper, a careful educational study assistant.

Answer the user's actual question.

Read the COMPLETE question before answering.

Never change the meaning of the question.

Never invent missing information.

Never assume a different equation from the one the user wrote.

If the question contains Mathematics, preserve every:
- bracket
- parenthesis
- square root
- fraction
- exponent
- denominator
- equality sign

Do not guess.

If the question is ambiguous, explicitly say what interpretation you are using.
"""


MATH_PROMPT = GENERAL_PROMPT + """

You are an expert Mathematics solver.

MATHEMATICS ACCURACY PROTOCOL:

1. Copy the original mathematical expression mentally exactly.
2. Identify the numerator and denominator separately.
3. Respect order of operations.
4. Treat √(a+b) differently from √a+b.
5. Treat a/(√b) differently from a/√(b+x).
6. Never cancel terms unless cancellation is mathematically valid.
7. Show the algebra step-by-step.
8. Check arithmetic independently.
9. Substitute the final answer back into the ORIGINAL equation.
10. If substitution does not reproduce the original equation, correct the answer.
11. Check for extraneous solutions.
12. If there are multiple possible interpretations, state them.

For equations, always perform a final verification.

Do NOT simply rely on intuition.
"""


PHYSICS_PROMPT = GENERAL_PROMPT + """

You are an expert Physics solver.

For Physics:

- identify known quantities
- identify the unknown
- choose the correct formula
- substitute carefully
- preserve units
- check signs
- check dimensions
- check the final result
- do not invent values
"""


CHEMISTRY_PROMPT = GENERAL_PROMPT + """

You are an expert Chemistry solver.

Check:
- chemical equations
- balancing
- stoichiometry
- units
- molar quantities
- signs
- numerical calculations
"""


BIOLOGY_PROMPT = GENERAL_PROMPT + """

You are an expert Biology solver.

Use accurate biological terminology.

Answer exactly what was asked.

Do not invent facts or labels that are not present.
"""


def subject_prompt(subject):
    prompts = {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT
    }

    return prompts.get(
        subject,
        GENERAL_PROMPT
    ) + "\n\n" + PLAIN_MATH_RULE


# ============================================================
# TEXT AI
# ============================================================

def text_ai(
    instructions,
    user_input,
    reasoning="medium",
    max_tokens=7000
):
    if not hf_client:
        raise RuntimeError(
            "HF_TOKEN is missing."
        )

    full_instructions = (
        instructions
        + "\n\n"
        + PLAIN_MATH_RULE
    )

    try:
        # Preferred Responses API
        response = hf_client.responses.create(
            model=TEXT_MODEL,
            instructions=full_instructions,
            input=user_input,
            reasoning={
                "effort": reasoning
            },
            max_output_tokens=max_tokens
        )

        answer = getattr(
            response,
            "output_text",
            ""
        )

        return clean_output(answer)

    except Exception as first_error:

        print(
            "RESPONSES API ERROR:",
            repr(first_error)
        )

        try:
            # Fallback Chat Completions API
            response = hf_client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": full_instructions
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=0.05,
                max_tokens=max_tokens
            )

            if not response.choices:
                return ""

            answer = (
                response.choices[0]
                .message
                .content
                or ""
            )

            return clean_output(answer)

        except Exception as second_error:
            print(
                "CHAT API ERROR:",
                repr(second_error)
            )

            raise RuntimeError(
                "Hugging Face AI request failed. "
                + str(second_error)
            )


# ============================================================
# MATH / PHYSICS VERIFICATION
# ============================================================

def verify_solution(
    question,
    answer,
    subject
):
    if subject not in {
        "math",
        "physics"
    }:
        return answer

    verifier_prompt = f"""
You are the FINAL ACCURACY CHECKER for Halper.

The subject is {subject}.

The user asked:

{question}

The first solver produced:

{answer}

IMPORTANT:

Check the ORIGINAL QUESTION, not just the proposed answer.

For Mathematics:

1. Re-read every symbol.
2. Preserve brackets and square roots.
3. Check numerator and denominator.
4. Check order of operations.
5. Check algebra.
6. Check arithmetic.
7. Solve again independently when necessary.
8. Substitute the final answer into the ORIGINAL equation.
9. If the proposed answer is wrong, replace it.
10. Do not blindly agree with the first solver.

For Physics:

1. Recheck formulas.
2. Recheck units.
3. Recheck signs.
4. Recheck numerical substitution.
5. Check dimensions.
6. Correct the result if necessary.

Return the corrected educational solution.

Do not mention this verification process.

Do not use LaTeX.
"""

    try:
        checked = text_ai(
            subject_prompt(subject)
            + "\n\n"
            + verifier_prompt,
            question
            + "\n\nPROPOSED SOLUTION:\n"
            + answer,
            reasoning="high",
            max_tokens=10000
        )

        if checked:
            return checked

    except Exception as error:
        print(
            "VERIFICATION ERROR:",
            repr(error)
        )

    return answer


# ============================================================
# VISION
# ============================================================

VISION_PROMPT = """
You are Halper's image-question solver.

Read the COMPLETE image carefully.

Determine whether it contains:
Mathematics, Physics, Chemistry, Biology or another academic topic.

Actually solve the question.

Do not invent text that is not visible.

For Mathematics:
- read every number
- read every symbol
- preserve brackets
- preserve square roots
- check arithmetic
- verify the final answer

For Physics:
- check formulas
- check units
- check numerical values

For Chemistry:
- check reactions
- check equations
- check numerical values

For Biology:
- read diagrams and labels carefully.

Do not use LaTeX.
"""


def analyze_image(
    image_data,
    question
):
    if not hf_client:
        return "❌ HF_TOKEN is missing."

    if not image_data:
        return "❌ No image was received."

    if not image_data.startswith("data:image/"):
        return "❌ Invalid image data."

    if len(image_data) > 12_000_000:
        return "❌ Image is too large."

    prompt = (
        question.strip()
        if question.strip()
        else
        "Read the entire image and solve the academic question."
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
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            temperature=0.05,
            max_tokens=7000
        )

        if not response.choices:
            return "❌ Vision AI returned no answer."

        answer = (
            response.choices[0]
            .message
            .content
            or ""
        )

        return clean_output(answer)

    except Exception as error:
        print(
            "VISION ERROR:",
            repr(error)
        )

        return (
            "❌ I could not analyze that image right now.\n\n"
            + str(error)
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    return " ".join(
        text.lower().strip().split()
    )


# ============================================================
# FOLLOW-UP INTENTS
# ============================================================

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
        "solutions"
    }

    if x in exact:
        return True

    return bool(
        re.search(
            r"\b(answers?|ans)\b.*\ball\b",
            x
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
        "multiple choice questions"
    ]

    conversion_words = [
        "adapt",
        "convert",
        "turn",
        "change",
        "make",
        "transform"
    ]

    has_format = any(
        phrase in x
        for phrase in phrases
    )

    has_conversion = any(
        word in x
        for word in conversion_words
    )

    return has_format and has_conversion


def is_option_request(q):
    x = normalize(q)

    return x in {
        "with options",
        "add options",
        "give options",
        "make mcq",
        "make mcqs",
        "make them mcq",
        "make these mcq"
    }


def is_explain_all_request(q):
    x = normalize(q)

    return x in {
        "explain all",
        "explain answers",
        "explain all questions",
        "solve all",
        "solve all questions"
    }


def continuation_count(q):
    x = normalize(q)

    patterns = [
        r"^(other|another|next|more|remaining)\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?(?:the\s+)?other\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?another\s+(\d+)\s+questions?$"
    ]

    for pattern in patterns:
        match = re.match(pattern, x)

        if match:
            groups = match.groups()

            number = int(groups[-1])

            return min(
                max(number, 1),
                50
            )

    return None


# ============================================================
# QUESTION COUNT
# ============================================================

def requested_count(q):
    match = re.search(
        r"\b(\d+)\s+(?:questions?|mcqs?)\b",
        q.lower()
    )

    if not match:
        return None

    return min(
        max(int(match.group(1)), 1),
        50
    )


# ============================================================
# QUESTION NUMBERING
# ============================================================

def extract_question_numbers(text):
    numbers = re.findall(
        r"(?:^|\n)\s*(\d+)\.\s+",
        text
    )

    return sorted({
        int(number)
        for number in numbers
    })


def last_question_number(text):
    numbers = extract_question_numbers(text)

    if numbers:
        return max(numbers)

    return 0


# ============================================================
# MCQ VALIDATION
# ============================================================

def validate_mcq_batch(
    text,
    start,
    end
):
    expected = set(
        range(start, end + 1)
    )

    actual = set(
        extract_question_numbers(text)
    )

    if not expected.issubset(actual):
        return False

    if "|" in text:
        return False

    blocks = re.split(
        r"(?:^|\n)\s*\d+\.\s+",
        text
    )[1:]

    wanted = end - start + 1

    if len(blocks) < wanted:
        return False

    for block in blocks[:wanted]:
        options = set(
            re.findall(
                r"(?:^|\n)\s*([ABCD])\.\s+",
                block
            )
        )

        if options != {
            "A",
            "B",
            "C",
            "D"
        }:
            return False

    return True


# ============================================================
# MCQ GENERATION
# ============================================================

def make_mcq_batch(
    subject,
    difficulty,
    start,
    end,
    previous_context=""
):
    amount = end - start + 1

    prompt = f"""
Generate exactly {amount} multiple-choice questions.

Number them from {start} through {end}.

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
- Every question must be complete.
- Every question must have A, B, C and D.
- No answer key.
- No introduction.
- No conclusion.
- Keep the exact subject.
- Do not switch subjects.
- Make questions different.
- Check the numbering before finishing.

Subject: {subject}

Difficulty: {difficulty}

Previous context:
{previous_context}
"""

    return text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="medium",
        max_tokens=7500
    )


def generate_questions(
    subject,
    difficulty,
    count,
    start=1,
    previous_context=""
):
    pieces = []

    current = start
    final_number = start + count - 1

    while current <= final_number:

        end = min(
            current + 4,
            final_number
        )

        batch = make_mcq_batch(
            subject,
            difficulty,
            current,
            end,
            previous_context
        )

        attempts = 0

        while (
            not validate_mcq_batch(
                batch,
                current,
                end
            )
            and attempts < 2
        ):
            attempts += 1

            batch = make_mcq_batch(
                subject,
                difficulty,
                current,
                end,
                previous_context
            )

        if not validate_mcq_batch(
            batch,
            current,
            end
        ):
            raise RuntimeError(
                f"Could not validate questions "
                f"{current}-{end}."
            )

        pieces.append(batch)

        current = end + 1

    return clean_output(
        "\n\n".join(pieces)
    )


# ============================================================
# QUESTION-SET CONVERTER
# ============================================================

def convert_existing_set_to_mcq(
    active_set
):
    subject = active_set["subject"]
    original = active_set["answer"]

    prompt = f"""
Convert the EXISTING question set below into
multiple-choice questions.

THIS IS A CONVERSION TASK.

Rules:

1. Use ONLY the supplied questions.
2. Keep exactly the same subject.
3. Keep exactly the same topic.
4. Keep the same meaning.
5. Do NOT create unrelated questions.
6. Add exactly four options.
7. Use A, B, C and D.
8. Keep original numbering.
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

    return text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="medium",
        max_tokens=18000
    )


# ============================================================
# ANSWER EXISTING SET
# ============================================================

def answer_existing_set(active_set):
    subject = active_set["subject"]
    original = active_set["answer"]

    prompt = f"""
Answer EVERY question in the existing question set.

Rules:

- Use ONLY these questions.
- Keep original numbering.
- Do not create new questions.
- Do not switch subject.
- Give the correct option when options exist.
- Give a concise explanation.
- Check every answer.
- For Mathematics, verify calculations by substitution when applicable.

Subject:

{subject}

Question set:

{original}
"""

    return text_ai(
        subject_prompt(subject),
        prompt,
        reasoning="high",
        max_tokens=18000
    )


# ============================================================
# NORMAL CONTEXT
# ============================================================

def build_normal_context(question):
    history = [
        item
        for item in get_history()
        if item.get("category") == "normal"
    ]

    if not history:
        return question

    if len(question.split()) > 7:
        return question

    recent = history[-4:]

    parts = [
        "Relevant recent conversation:"
    ]

    for item in recent:
        parts.append(
            "USER: " + item["question"]
        )

        parts.append(
            "HALPER: " + item["answer"]
        )

    parts.append(
        "CURRENT QUESTION: " + question
    )

    parts.append(
        "Use previous context only when clearly relevant."
    )

    return "\n".join(parts)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        logged_in="username" in session,
        username=session.get("username")
    )


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

    username = data.get(
        "username",
        ""
    ).strip()

    email = data.get(
        "email",
        ""
    ).strip().lower()

    phone = data.get(
        "phone",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    if not username or not password:
        return {
            "success": False,
            "message": "Username and password are required."
        }, 400

    if not email and not phone:
        return {
            "success": False,
            "message": "Enter either an email or phone number."
        }, 400

    if len(password) < 6:
        return {
            "success": False,
            "message": "Password must be at least 6 characters."
        }, 400

    all_users = load_users()

    for name, user in all_users.items():

        if name.lower() == username.lower():
            return {
                "success": False,
                "message": "Username already exists."
            }, 400

        if (
            email
            and user.get("email", "").lower() == email
        ):
            return {
                "success": False,
                "message": "Email already registered."
            }, 400

        if (
            phone
            and user.get("phone", "") == phone
        ):
            return {
                "success": False,
                "message": "Phone already registered."
            }, 400

    all_users[username] = {
        "email": email,
        "phone": phone,
        "password": generate_password_hash(password)
    }

    save_users(all_users)

    session["username"] = username
    session["chat_history"] = []

    clear_question_set()

    return {
        "success": True,
        "message": "Account created successfully."
    }


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

    login_value = data.get(
        "login",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    all_users = load_users()

    for username, user in all_users.items():

        matches = (
            username.lower() == login_value.lower()
            or user.get("email", "").lower() == login_value.lower()
            or user.get("phone", "") == login_value
        )

        if matches:

            if check_password_hash(
                user.get("password", ""),
                password
            ):
                session["username"] = username
                session["chat_history"] = []

                clear_question_set()

                return {
                    "success": True,
                    "message": "Login successful."
                }

            break

    return {
        "success": False,
        "message": "Invalid username/email/phone or password."
    }, 401


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
def new_chat():
    session["chat_history"] = []

    clear_question_set()

    return {
        "success": True
    }


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    if "username" not in session:
        return Response(
            "Please login first.",
            status=401,
            mimetype="text/plain"
        )

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "message",
        ""
    ).strip()

    image_data = data.get(
        "image",
        ""
    )

    # ========================================================
    # IMAGE FIRST
    # ========================================================

    if image_data:

        answer = analyze_image(
            image_data,
            question
        )

        save_history(
            question or "Image question",
            answer,
            "normal"
        )

        return Response(
            answer,
            mimetype="text/plain"
        )

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question:
        return Response(
            "Please type a question or capture an image.",
            mimetype="text/plain"
        )

    # ========================================================
    # CREATOR
    # ========================================================

    if is_creator_question(question):

        answer = clean_output(
            CREATOR_RESPONSE
        )

        save_history(
            question,
            answer,
            "creator"
        )

        return Response(
            answer,
            mimetype="text/plain"
        )

    # ========================================================
    # BASIC GREETING
    # ========================================================

    normalized = normalize(question)

    if normalized in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo"
    }:

        answer = (
            "Hello! 👋 I'm Halper.\n\n"
            "Ask me Mathematics, Physics, Chemistry, "
            "Biology or general questions."
        )

        save_history(
            question,
            answer,
            "normal"
        )

        return Response(
            answer,
            mimetype="text/plain"
        )

    # ========================================================
    # ACTIVE QUESTION SET
    # ========================================================

    active_set = get_question_set()

    if active_set:

        # ----------------------------------------------------
        # MCQ CONVERSION
        # ----------------------------------------------------

        if (
            is_mcq_conversion_request(question)
            or is_option_request(question)
        ):

            answer = convert_existing_set_to_mcq(
                active_set
            )

            answer = clean_output(answer)

            save_question_set(
                active_set["request"],
                answer,
                active_set["subject"],
                active_set["difficulty"]
            )

            save_history(
                question,
                answer,
                "question_generation"
            )

            return Response(
                answer,
                mimetype="text/plain"
            )

        # ----------------------------------------------------
        # ANSWER ALL
        # ----------------------------------------------------

        if is_answer_key_request(question):

            answer = answer_existing_set(
                active_set
            )

            answer = clean_output(answer)

            save_history(
                question,
                answer,
                "question_generation"
            )

            return Response(
                answer,
                mimetype="text/plain"
            )

        # ----------------------------------------------------
        # EXPLAIN ALL
        # ----------------------------------------------------

        if is_explain_all_request(question):

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
                max_tokens=18000
            )

            answer = clean_output(answer)

            save_history(
                question,
                answer,
                "question_generation"
            )

            return Response(
                answer,
                mimetype="text/plain"
            )

        # ----------------------------------------------------
        # CONTINUE WITH NUMBER
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
                active_set["answer"]
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
                active_set["difficulty"]
            )

            save_history(
                question,
                answer,
                "question_generation"
            )

            return Response(
                answer,
                mimetype="text/plain"
            )

        # ----------------------------------------------------
        # MORE / CONTINUE
        # ----------------------------------------------------

        if normalized in {
            "continue",
            "more",
            "another"
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
                active_set["answer"]
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
                active_set["difficulty"]
            )

            save_history(
                question,
                answer,
                "question_generation"
            )

            return Response(
                answer,
                mimetype="text/plain"
            )

    # ========================================================
    # NEW QUESTION SET
    # ========================================================

    count = requested_count(question)

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
            count
        )

        answer = clean_output(answer)

        save_question_set(
            question,
            answer,
            detected_subject,
            detected_difficulty
        )

        save_history(
            question,
            answer,
            "question_generation"
        )

        return Response(
            answer,
            mimetype="text/plain"
        )

    # ========================================================
    # NORMAL QUESTION
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

    # ========================================================
    # ADVANCED
    # ========================================================

    if detected_difficulty == "advanced":

        answer = text_ai(
            subject_prompt(
                detected_subject
            )
            + """

ADVANCED SOLVING MODE

Solve very carefully.

Before giving the final answer:

- re-read the original question
- preserve all brackets
- preserve square roots
- check numerator and denominator
- check equations
- check algebra
- check arithmetic
- check signs
- check units
- check dimensions
- check conditions
- verify the final result
- substitute the result back when possible

Never guess.
""",
            context,
            reasoning="high",
            max_tokens=10000
        )

        if detected_subject in {
            "math",
            "physics"
        }:
            answer = verify_solution(
                question,
                answer,
                detected_subject
            )

    # ========================================================
    # INTERMEDIATE
    # ========================================================

    elif detected_difficulty == "intermediate":

        answer = text_ai(
            subject_prompt(
                detected_subject
            ),
            context,
            reasoning="medium",
            max_tokens=6000
        )

        if detected_subject in {
            "math",
            "physics"
        }:
            answer = verify_solution(
                question,
                answer,
                detected_subject
            )

    # ========================================================
    # BASIC
    # ========================================================

    else:

        answer = text_ai(
            subject_prompt(
                detected_subject
            ),
            context,
            reasoning="low",
            max_tokens=3500
        )

        # Even basic mathematical questions get checked.
        if detected_subject in {
            "math",
            "physics"
        }:
            answer = verify_solution(
                question,
                answer,
                detected_subject
            )

    answer = clean_output(answer)

    save_history(
        question,
        answer,
        "normal"
    )

    return Response(
        answer,
        mimetype="text/plain"
    )


# ============================================================
# IMPROVE / CHECK / EXPLAIN / SHORTEN
# ============================================================

@app.route(
    "/improve",
    methods=["POST"]
)
def improve():

    if "username" not in session:
        return Response(
            "Please login first.",
            status=401,
            mimetype="text/plain"
        )

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "question",
        ""
    )

    old_answer = data.get(
        "answer",
        ""
    )

    action = data.get(
        "action",
        "improve"
    )

    tasks = {
        "improve":
            "Improve the answer and make it clearer.",

        "check":
            "Check the answer for mistakes and correct them.",

        "explain":
            "Explain the answer in more detail.",

        "short":
            "Make the answer shorter without losing important information."
    }

    task = tasks.get(
        action,
        tasks["improve"]
    )

    detected_subject = detect_subject(
        question
    )

    answer = text_ai(
        subject_prompt(
            detected_subject
        )
        + "\n\n"
        + task
        + """

For Mathematics and Physics:

Check the original question carefully.

Do not silently change the equation.

Preserve brackets, roots, fractions and signs.

If the existing answer is mathematically wrong,
correct it before returning the improved answer.
""",
        "QUESTION:\n"
        + question
        + "\n\nANSWER:\n"
        + old_answer,
        reasoning=(
            "high"
            if detected_subject in {
                "math",
                "physics"
            }
            else "medium"
        ),
        max_tokens=9000
    )

    if detected_subject in {
        "math",
        "physics"
    }:
        answer = verify_solution(
            question,
            answer,
            detected_subject
        )

    return Response(
        clean_output(answer),
        mimetype="text/plain"
    )


# ============================================================
# HEALTH CHECK
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
        "plain_math": True
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HALPER")
    print("=" * 60)

    print(
        "HF configured       :",
        bool(HF_TOKEN)
    )

    print(
        "Creator routing     : ON"
    )

    print(
        "Question-set memory : ON"
    )

    print(
        "MCQ conversion      : ON"
    )

    print(
        "Answer-key routing  : ON"
    )

    print(
        "Follow-ups          : ON"
    )

    print(
        "Hard Mathematics    : ON"
    )

    print(
        "Hard Physics        : ON"
    )

    print(
        "Math verification   : ON"
    )

    print(
        "Image analysis      : ON"
    )

    print(
        "Plain math          : ON"
    )

    print("=" * 60)

    # Render provides PORT.
    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )