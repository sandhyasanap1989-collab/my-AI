from flask import (
    Flask,
    render_template,
    request,
    Response,
    session,
    redirect,
    url_for,
)

import os
import re
import json
import ast
import math
import operator
import threading

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

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
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# CREATOR
# ============================================================

CREATOR_RESPONSE = """
My creator is Soham Chandrahas Sanap.

He is 15 years old and is studying in Class 10 in 2026
at Nimbark English School in Beed district, Maharashtra, India.

His main interests are Mathematics and web development.

He built My AI as an AI study assistant to help students
with Mathematics, Physics, Chemistry and Biology.
""".strip()


CREATOR_QUESTIONS = {
    "who created you",
    "who created you?",
    "who made you",
    "who made you?",
    "who is your creator",
    "who is your creator?",
    "who developed you",
    "who developed you?",
    "who built you",
    "who built you?",
    "who is your developer",
    "who is your developer?",
    "who made this ai",
    "who made this ai?",
    "who built this ai",
    "who built this ai?",
    "who is behind you",
    "who is behind this ai",
    "who is your maker",
    "who is your maker?",
    "tell me about your creator",
    "tell me about your creator?",
    "who is soham chandrahas sanap",
    "who is soham chandrahas sanap?",
}


def is_creator_question(question):
    return question.strip().lower() in CREATOR_QUESTIONS


# ============================================================
# USER FILE
# ============================================================

def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return default


def save_json_file(path, data):
    temp_path = path + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    os.replace(
        temp_path,
        path,
    )


def load_users():
    data = load_json_file(
        USERS_FILE,
        {},
    )

    return data if isinstance(data, dict) else {}


def save_users(users):
    with FILE_LOCK:
        save_json_file(
            USERS_FILE,
            users,
        )


# ============================================================
# QUESTION-SET STORAGE
#
# Stored on server instead of Flask's cookie session.
# ============================================================

def load_question_sets():
    data = load_json_file(
        QUESTION_SETS_FILE,
        {},
    )

    return data if isinstance(data, dict) else {}


def get_current_username():
    return session.get("username")


def get_question_set():
    username = get_current_username()

    if not username:
        return None

    all_sets = load_question_sets()

    return all_sets.get(username)


def save_question_set(
    request_text,
    answer,
    subject,
    difficulty,
):
    username = get_current_username()

    if not username:
        return

    all_sets = load_question_sets()

    all_sets[username] = {
        "request": request_text,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty,
    }

    with FILE_LOCK:
        save_json_file(
            QUESTION_SETS_FILE,
            all_sets,
        )


def clear_question_set():
    username = get_current_username()

    if not username:
        return

    all_sets = load_question_sets()

    if username in all_sets:
        del all_sets[username]

        with FILE_LOCK:
            save_json_file(
                QUESTION_SETS_FILE,
                all_sets,
            )


# ============================================================
# PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a high-quality student study assistant.

You help with:
Mathematics, Physics, Chemistry, Biology and academic questions.

Rules:
1. Read the entire question.
2. Identify exactly what is being asked.
3. Never guess.
4. Answer every requested part.
5. Show useful calculations.
6. Check arithmetic.
7. Check units when relevant.
8. Verify important results.
9. Use only relevant conversation context.
10. Never invent user information.
11. Finish the answer completely.

Never output raw LaTeX commands.

Use readable notation such as:
x²
x³
√2
a/b
F = ma
ω² = k/m
"""


MATH_PROMPT = """
You are My AI's expert Mathematics solver.

Handle basic through very difficult mathematics.

Topics:
- Algebra
- Trigonometry
- Trigonometric identities
- Equations
- Quadratics
- Polynomials
- Surds
- Sequences and series
- Binomial theorem
- Permutations and combinations
- Probability
- Functions
- Logarithms
- Inequalities
- Geometry
- Coordinate geometry
- Vectors
- Matrices
- Determinants
- Complex numbers
- Number theory
- Divisibility
- Limits
- Differentiation
- Integration
- Calculus
- JEE Main
- JEE Advanced
- Olympiad mathematics

For difficult problems:
1. Understand the exact target.
2. Identify useful information.
3. Choose the correct method.
4. Derive carefully.
5. Check algebra.
6. Check arithmetic.
7. Check the final result.
8. Verify MCQ options.
9. For identities, actually prove them.
10. Never guess.

Never output raw LaTeX.
"""


PHYSICS_PROMPT = """
You are My AI's expert Physics solver.

Handle school, JEE Main, JEE Advanced and very difficult
multi-step Physics problems.

Topics:
- Kinematics
- Newton's laws
- Friction
- Work, energy and power
- Momentum
- Centre of mass
- Circular motion
- Rotation
- Rolling
- Gravitation
- SHM
- Oscillations
- Waves
- Electrostatics
- Capacitors
- Current electricity
- Magnetism
- Electromagnetic induction
- AC
- Ray optics
- Wave optics
- Thermodynamics
- Kinetic theory
- Modern physics

For difficult problems:
1. Understand the physical setup.
2. Identify all bodies.
3. Identify forces and constraints.
4. Choose coordinates.
5. Find equilibrium when required.
6. Write governing equations.
7. Use approximations only when justified.
8. Derive the requested quantity.
9. Check signs.
10. Check dimensions.
11. Check limiting cases when useful.
12. Verify the final result.

For small oscillations:
- Find equilibrium first.
- Define displacement.
- Find restoring force.
- Apply the correct approximation.
- Compare with SHM.

Never guess.
Never output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are My AI's expert Chemistry solver.

Handle school, JEE Main and JEE Advanced.

Topics:
- Mole concept
- Stoichiometry
- Atomic structure
- Chemical bonding
- Thermodynamics
- Equilibrium
- Ionic equilibrium
- Electrochemistry
- Chemical kinetics
- Solutions
- Redox
- Inorganic chemistry
- Organic chemistry
- Isomerism
- Reaction mechanisms

For numericals:
Given → formula/reaction → substitution → calculation
→ verification → final answer.

Check units, molar masses, ratios and limiting reagent.

Never guess.
"""


BIOLOGY_PROMPT = """
You are My AI's expert Biology solver.

Handle school and competitive Biology.

Topics:
- Cell biology
- Genetics
- Mendelian inheritance
- DNA
- RNA
- Molecular biology
- Biotechnology
- Human physiology
- Plant physiology
- Reproduction
- Evolution
- Ecology

For difficult questions:
- identify the exact concept
- explain the process
- compare alternatives
- calculate genetics probabilities carefully

Do not invent facts.
"""


VISION_PROMPT = """
You are My AI's academic image-question solver.

Read the entire uploaded image.

Identify:
- Mathematics
- Physics
- Chemistry
- Biology
- General academic content

Then solve the question.

Mathematics:
Read all visible numbers and symbols and solve carefully.

Physics:
Identify the physical system, values, laws, equations,
and derive the result with unit checks.

Chemistry:
Read reactions and numerical information and solve carefully.

Biology:
Read diagrams and labels carefully.

Do not invent anything that is not visible.
If something is unclear, say exactly what is unclear.
Actually solve the question.
Do not merely describe the image.

Never output raw LaTeX.
"""


def get_prompt(subject):
    return {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }.get(
        subject,
        GENERAL_PROMPT,
    )


# ============================================================
# OUTPUT CLEANING
# ============================================================

def clean_ai_output(text):
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
        "\\mp": "∓",
        "\\leq": "≤",
        "\\le": "≤",
        "\\geq": "≥",
        "\\ge": "≥",
        "\\neq": "≠",
        "\\ne": "≠",
        "\\approx": "≈",
        "\\infty": "∞",
        "\\rightarrow": "→",
        "\\to": "→",
        "\\Rightarrow": "⇒",
        "\\Leftarrow": "⇐",
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
        "\\circ": "°",
        "\\quad": " ",
        "\\qquad": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"(\1/\2)",
        text,
    )

    text = re.sub(
        r"\\dfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"(\1/\2)",
        text,
    )

    text = re.sub(
        r"\\text\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\mathrm\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\mathbf\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\boxed\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = text.replace(
        "\\left",
        "",
    )

    text = text.replace(
        "\\right",
        "",
    )

    text = re.sub(
        r"\\([a-zA-Z]+)",
        r"\1",
        text,
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# HISTORY
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

    history.append(
        {
            "question": question,
            "answer": answer[:12000],
            "category": category,
        }
    )

    # Avoid huge cookie/session growth.
    session["chat_history"] = history[-8:]
    session.modified = True


# ============================================================
# CREATOR / BASIC
# ============================================================

def basic_response(question):
    q = question.strip().lower()

    if is_creator_question(question):
        return CREATOR_RESPONSE

    if q in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
    }:
        return (
            "Hello! 👋 I'm My AI.\n\n"
            "Ask me Mathematics, Physics, Chemistry, "
            "Biology or any academic question."
        )

    if q in {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty",
    }:
        return "You're welcome! 😊"

    if q in {
        "who are you",
        "who are you?",
        "what are you",
        "what are you?",
    }:
        return (
            "I'm My AI 🤖, a study assistant for "
            "Mathematics, Physics, Chemistry and Biology."
        )

    return None


# ============================================================
# SUBJECT
# ============================================================

def detect_subject(question):
    q = question.lower()

    groups = {
        "math": [
            "equation",
            "algebra",
            "quadratic",
            "polynomial",
            "trigonometry",
            "trigonometric",
            "sin",
            "cos",
            "tan",
            "sec",
            "cosec",
            "geometry",
            "identity",
            "surds",
            "probability",
            "sequence",
            "series",
            "binomial",
            "permutation",
            "combination",
            "matrix",
            "determinant",
            "calculus",
            "integral",
            "derivative",
            "logarithm",
            "vector",
            "complex",
            "inequality",
            "number theory",
            "divisibility",
            "function",
            "proof",
        ],

        "physics": [
            "force",
            "velocity",
            "acceleration",
            "momentum",
            "newton",
            "work",
            "energy",
            "power",
            "friction",
            "gravitation",
            "projectile",
            "current",
            "voltage",
            "resistance",
            "resistor",
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
            "rotational",
            "torque",
            "angular momentum",
            "oscillation",
            "oscillations",
            "shm",
            "wave",
            "capacitor",
            "induction",
            "electromagnetic",
        ],

        "chemistry": [
            "mole",
            "molarity",
            "molality",
            "stoichiometry",
            "limiting reagent",
            "oxidation",
            "reduction",
            "redox",
            "acid",
            "base",
            "ph",
            "salt",
            "equilibrium",
            "enthalpy",
            "electrochemistry",
            "organic",
            "alkane",
            "alkene",
            "alkyne",
            "benzene",
            "alcohol",
            "aldehyde",
            "ketone",
            "reaction",
            "atom",
            "electron",
            "compound",
            "kinetics",
        ],

        "biology": [
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
            "tissue",
            "organ",
            "ecosystem",
            "ecology",
            "evolution",
            "hormone",
            "neuron",
            "digestion",
            "reproduction",
            "heredity",
            "blood",
            "heart",
            "kidney",
            "lung",
            "brain",
        ],
    }

    scores = {
        subject: sum(
            word in q
            for word in words
        )
        for subject, words in groups.items()
    }

    subject = max(
        scores,
        key=scores.get,
    )

    if scores[subject] == 0:
        return "general"

    return subject


# ============================================================
# DIFFICULTY
# ============================================================

def detect_difficulty(question):
    q = question.lower()

    score = 0

    hard_terms = [
        "jee advanced",
        "jee adv",
        "olympiad",
        "hard",
        "hardest",
        "difficult",
        "challenging",
        "derive",
        "derivation",
        "prove",
        "show that",
        "maximum",
        "minimum",
        "equilibrium",
        "small oscillation",
        "small oscillations",
        "multi-step",
        "multiple correct",
        "integer answer",
        "match the following",
        "constraint",
        "optimization",
        "differential equation",
        "complex number",
        "number theory",
        "collision",
        "rotation",
        "angular momentum",
        "electromagnetic induction",
    ]

    for term in hard_terms:
        if term in q:
            score += 2

    if len(q) > 300:
        score += 2

    if len(q) > 700:
        score += 2

    if score >= 6:
        return "advanced"

    if score >= 2:
        return "intermediate"

    return "basic"


# ============================================================
# QUESTION COMMANDS
# ============================================================

def continuation_count(question):
    q = question.strip().lower()

    patterns = [
        r"^other\s+(\d+)\s+questions?$",
        r"^another\s+(\d+)\s+questions?$",
        r"^next\s+(\d+)\s+questions?$",
        r"^more\s+(\d+)\s+questions?$",
        r"^remaining\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?(?:the\s+)?other\s+(\d+)\s+questions?$",
        r"^give\s+(?:me\s+)?another\s+(\d+)\s+questions?$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            q,
        )

        if match:
            return min(
                max(
                    int(match.group(1)),
                    1,
                ),
                50,
            )

    return None


def answer_key_command(question):
    q = question.strip().lower()

    return q in {
        "answers",
        "answer",
        "answer key",
        "show answers",
        "ans",
        "ans of all",
        "answers of all",
        "answer all",
        "give ans",
        "give ans of all",
        "give answers",
    }


def options_command(question):
    q = question.strip().lower()

    return q in {
        "with options",
        "add options",
        "give options",
    }


def explanation_command(question):
    q = question.strip().lower()

    return q in {
        "explain all",
        "explain answers",
        "explain all questions",
        "solutions",
        "solve all",
    }


def is_question_generation(question):
    q = question.lower()

    if continuation_count(question) is not None:
        return True

    if answer_key_command(question):
        return True

    if options_command(question):
        return True

    if explanation_command(question):
        return True

    phrases = [
        "give me questions",
        "give me question",
        "generate questions",
        "generate question",
        "make questions",
        "make question",
        "create questions",
        "create question",
        "practice questions",
        "mcq",
        "mcqs",
        "multiple choice",
        "with options",
        "question paper",
        "test paper",
        "sample paper",
    ]

    return any(
        phrase in q
        for phrase in phrases
    )


def requested_question_count(question):
    q = question.lower()

    patterns = [
        r"\b(\d+)\s+questions?\b",
        r"\b(\d+)\s+mcqs?\b",
        r"\btop\s+(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            q,
        )

        if match:
            return min(
                max(
                    int(match.group(1)),
                    1,
                ),
                50,
            )

    return None


# ============================================================
# QUESTION NUMBERING
# ============================================================

def extract_question_numbers(text):
    numbers = re.findall(
        r"(?:^|\n)\s*(?:\*\*)?(\d+)\.",
        text,
    )

    return sorted(
        set(
            int(number)
            for number in numbers
        )
    )


def last_question_number(text):
    numbers = extract_question_numbers(
        text
    )

    return (
        max(numbers)
        if numbers
        else 0
    )


# ============================================================
# NORMAL CONTEXT
#
# CREATOR INFORMATION IS NEVER INCLUDED.
# QUESTION SET IS NEVER INCLUDED HERE.
# ============================================================

def build_normal_context(question):
    history = get_history()

    useful = [
        item
        for item in history
        if item.get("category") == "normal"
    ]

    # A long/independent question should not inherit random
    # older context.
    if len(question.split()) > 6:
        return question

    if not useful:
        return question

    recent = useful[-4:]

    parts = [
        "RELEVANT RECENT CONVERSATION:",
        "",
    ]

    for item in recent:
        parts.append(
            "USER:\n" + item["question"]
        )

        parts.append(
            "MY AI:\n" + item["answer"]
        )

        parts.append("")

    parts.append(
        "NEW USER QUESTION:\n"
        + question
    )

    parts.append(
        "\nAnswer only using relevant context."
    )

    return "\n".join(parts)


# ============================================================
# TEXT AI
# ============================================================

def text_call(
    instructions,
    user_input,
    reasoning="medium",
    max_tokens=6000,
):
    if not hf_client:
        raise RuntimeError(
            "HF_TOKEN is missing."
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

        return clean_ai_output(
            response.output_text or ""
        )

    except Exception as response_error:
        print(
            "RESPONSES API ERROR:",
            repr(response_error),
        )

        # Fallback for providers that don't expose Responses API.
        completion = hf_client.chat.completions.create(
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

        if not completion.choices:
            return ""

        return clean_ai_output(
            completion.choices[0]
            .message
            .content
            or ""
        )


# ============================================================
# QUESTION BATCH GENERATOR
# ============================================================

def generate_question_batch(
    subject,
    difficulty,
    start_number,
    end_number,
    previous_context="",
):
    count = (
        end_number
        - start_number
        + 1
    )

    prompt = get_prompt(
        subject
    )

    prompt += f"""

QUESTION GENERATION MODE

Generate exactly {count} MCQ questions.

Number:
{start_number} through {end_number}

Requirements:
- Same subject: {subject}
- Difficulty: {difficulty}
- Every question must be complete.
- Give A, B, C and D options.
- Make each question different.
- Do not give answers.
- Do not restart numbering.
- Do not add unrelated commentary.
- Use readable mathematical notation.
- Before finishing, check every number exists.

Previous context:
{previous_context}
"""

    # Five/ten-question batches stay safely below one giant output.
    return text_call(
        instructions=prompt,
        user_input=prompt,
        reasoning="medium",
        max_tokens=min(
            max(
                5000,
                count * 1100,
            ),
            10000,
        ),
    )


def generate_questions_in_batches(
    subject,
    difficulty,
    total,
    start_number=1,
    previous_context="",
):
    parts = []

    current = start_number

    final_number = (
        start_number
        + total
        - 1
    )

    while current <= final_number:

        batch_end = min(
            current + 4,
            final_number,
        )

        batch = generate_question_batch(
            subject=subject,
            difficulty=difficulty,
            start_number=current,
            end_number=batch_end,
            previous_context=previous_context,
        )

        if not batch.strip():
            raise RuntimeError(
                f"Empty batch {current}-{batch_end}"
            )

        expected = set(
            range(
                current,
                batch_end + 1,
            )
        )

        found = set(
            extract_question_numbers(
                batch
            )
        )

        missing = sorted(
            expected - found
        )

        # Repair missing question numbers.
        attempts = 0

        while missing and attempts < 2:

            attempts += 1

            repair = generate_question_batch(
                subject=subject,
                difficulty=difficulty,
                start_number=missing[0],
                end_number=missing[-1],
                previous_context=batch,
            )

            batch += (
                "\n\n"
                + repair
            )

            found = set(
                extract_question_numbers(
                    batch
                )
            )

            missing = sorted(
                expected - found
            )

        if missing:
            raise RuntimeError(
                "Missing question numbers: "
                + ", ".join(
                    map(str, missing)
                )
            )

        parts.append(
            batch
        )

        current = (
            batch_end
            + 1
        )

    final_text = "\n\n".join(
        parts
    )

    # Final number check.
    actual = set(
        extract_question_numbers(
            final_text
        )
    )

    expected_final = set(
        range(
            start_number,
            final_number + 1,
        )
    )

    missing_final = sorted(
        expected_final - actual
    )

    if missing_final:
        raise RuntimeError(
            "Final question-set validation failed: "
            + ", ".join(
                map(str, missing_final)
            )
        )

    return clean_ai_output(
        final_text
    )


# ============================================================
# QUESTION-SET COMMANDS
# ============================================================

def modify_question_set(
    question_set,
    command,
):
    q = command.strip().lower()

    subject = question_set["subject"]
    old_set = question_set["answer"]

    # --------------------------------------------------------
    # ANSWERS OF ALL
    # --------------------------------------------------------

    if answer_key_command(command):

        prompt = get_prompt(
            subject
        )

        prompt += """

The user wants the ANSWERS OF ALL QUESTIONS in the
previous question set.

Rules:
- Use only the previous question set below.
- Do not talk about the creator.
- Do not generate new questions.
- Answer every question.
- Keep the same numbering.
- Give the correct option.
- Give a short explanation for each.
- Check calculations carefully.
- Do not stop early.
"""

        result = text_call(
            instructions=prompt,
            user_input=(
                "Previous question set:\n\n"
                + old_set
                + "\n\nGive the answers of all questions."
            ),
            reasoning="high",
            max_tokens=16000,
        )

        return result

    # --------------------------------------------------------
    # ADD OPTIONS
    # --------------------------------------------------------

    if options_command(command):

        prompt = get_prompt(
            subject
        )

        prompt += """

Add exactly four options A, B, C and D to every question
in the existing question set.

Rules:
- Keep the same questions.
- Keep the same numbering.
- Keep the same subject.
- Do not talk about the creator.
- Do not answer the questions.
- Return the COMPLETE question set.
"""

        return text_call(
            instructions=prompt,
            user_input=(
                "Existing question set:\n\n"
                + old_set
                + "\n\nUser request:\n"
                + command
            ),
            reasoning="medium",
            max_tokens=16000,
        )

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    if explanation_command(command):

        prompt = get_prompt(
            subject
        )

        prompt += """

Solve every question in the existing question set.

Rules:
- Keep the original numbering.
- Solve every question.
- Show useful steps.
- Do not skip any question.
- Do not talk about the creator.
"""

        return text_call(
            instructions=prompt,
            user_input=(
                "Existing question set:\n\n"
                + old_set
                + "\n\nUser request:\n"
                + command
            ),
            reasoning="high",
            max_tokens=18000,
        )

    return None


# ============================================================
# ADVANCED VERIFIER
# ============================================================

def verify_advanced_answer(
    question,
    answer,
    subject,
):
    if subject not in {
        "math",
        "physics",
    }:
        return answer

    verifier_prompt = """
You are a strict verifier for Mathematics and Physics.

Check the proposed answer.

Check:
- equation correctness
- algebra
- arithmetic
- signs
- dimensions
- units
- limiting cases where useful
- MCQ option

If wrong, correct it.

Return only the corrected educational answer.
Do not mention private reasoning.
Do not output raw LaTeX.
"""

    try:

        checked = text_call(
            instructions=verifier_prompt,
            user_input=(
                "QUESTION:\n"
                + question
                + "\n\nPROPOSED ANSWER:\n"
                + answer
            ),
            reasoning="high",
            max_tokens=10000,
        )

        return (
            checked
            if checked.strip()
            else answer
        )

    except Exception as error:

        print(
            "VERIFIER ERROR:",
            repr(error),
        )

        return answer


# ============================================================
# CAMERA / IMAGE
# ============================================================

def analyze_image(
    image_data,
    question,
):
    if not hf_client:
        return (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment → HF_TOKEN."
        )

    if not image_data.startswith(
        "data:image/"
    ):
        return "❌ Invalid image data."

    if len(image_data) > 12_000_000:
        return (
            "❌ Image is too large.\n\n"
            "Please use a smaller or clearer image."
        )

    prompt = (
        question.strip()
        if question.strip()
        else
        "Read this image carefully and solve the academic question step by step."
    )

    try:

        response = hf_client.responses.create(
            model=VISION_MODEL,
            instructions=VISION_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data,
                        },
                    ],
                }
            ],
            max_output_tokens=7000,
        )

        answer = clean_ai_output(
            response.output_text or ""
        )

        if answer:
            return answer

    except Exception as error:

        print(
            "VISION RESPONSES ERROR:",
            repr(error),
        )

    # Fallback to Chat Completions
    try:

        response = hf_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": VISION_PROMPT,
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

        if response.choices:

            answer = clean_ai_output(
                response.choices[0]
                .message
                .content
                or ""
            )

            if answer:
                return answer

    except Exception as error:

        print(
            "VISION FALLBACK ERROR:",
            repr(error),
        )

        return (
            "❌ Vision AI error:\n\n"
            + str(error)
        )

    return (
        "❌ Vision AI could not analyze the image."
    )


# ============================================================
# LOCAL CALCULATOR
# ============================================================

ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression):

    try:

        tree = ast.parse(
            expression,
            mode="eval",
        )

    except (
        SyntaxError,
        ValueError,
    ):

        return None

    def calculate(node):

        if isinstance(
            node,
            ast.Constant,
        ):

            if isinstance(
                node.value,
                (int, float),
            ):
                return node.value

            raise ValueError()

        if isinstance(
            node,
            ast.UnaryOp,
        ):

            operation = (
                ALLOWED_UNARY_OPERATORS.get(
                    type(node.op)
                )
            )

            if operation is None:
                raise ValueError()

            return operation(
                calculate(
                    node.operand
                )
            )

        if isinstance(
            node,
            ast.BinOp,
        ):

            operation = (
                ALLOWED_BINARY_OPERATORS.get(
                    type(node.op)
                )
            )

            if operation is None:
                raise ValueError()

            left = calculate(
                node.left
            )

            right = calculate(
                node.right
            )

            if (
                isinstance(
                    node.op,
                    ast.Pow,
                )
                and abs(right) > 10000
            ):
                raise ValueError()

            return operation(
                left,
                right,
            )

        raise ValueError()

    try:

        return calculate(
            tree.body
        )

    except (
        ArithmeticError,
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ):

        return None


def solve_calculation(question):

    q = question.strip()

    q = q.replace(
        "×",
        "*",
    )

    q = q.replace(
        "÷",
        "/",
    )

    q = q.replace(
        "−",
        "-",
    )

    q = q.replace(
        "^",
        "**",
    )

    q = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        q,
    )

    q = re.sub(
        r"^(what is|calculate|find|solve|evaluate)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )

    if not re.fullmatch(
        r"[0-9\s\.\+\-\*\/%\(\)]+",
        q,
    ):
        return None

    if not re.search(
        r"[\+\-\*\/%]",
        q,
    ):
        return None

    result = safe_calculate(
        q
    )

    if result is None:
        return None

    if isinstance(
        result,
        int,
    ):

        answer = f"{result:,}"

    else:

        if not math.isfinite(
            result
        ):
            return None

        answer = (
            str(int(result))
            if result.is_integer()
            else f"{result:.12g}"
        )

    return (
        "### Calculation\n\n"
        f"**{q} = {answer} ✅**"
    )


# ============================================================
# x + 1/x
# ============================================================

def solve_power_recurrence(question):

    q = question.lower()

    given = re.search(
        r"x\s*\+\s*1\s*/\s*x\s*=\s*"
        r"(-?\d+(?:\.\d+)?)",
        q,
    )

    if not given:
        return None

    a = float(
        given.group(1)
    )

    target = re.search(
        r"x\s*(?:\^|\*\*)?(\d+)"
        r"\s*\+\s*1\s*/\s*x\s*(?:\^|\*\*)?\1",
        q,
    )

    if not target:
        return None

    n = int(
        target.group(1)
    )

    if not 1 <= n <= 50:
        return None

    values = {
        0: 2,
        1: a,
    }

    for k in range(
        1,
        n,
    ):

        values[k + 1] = (
            a * values[k]
            - values[k - 1]
        )

    lines = [
        "### Solution",
        "",
        f"Given: x + 1/x = {a:g}",
        "",
        "Let Sₙ = xⁿ + 1/xⁿ",
        "",
        "S₀ = 2",
        f"S₁ = {a:g}",
        "",
    ]

    for k in range(
        2,
        n + 1,
    ):

        lines.extend(
            [
                f"**Step {k - 1}:**",
                "",
                f"S{k} = {a:g}S{k-1} − S{k-2}",
                "",
                (
                    f"S{k} = {a:g}"
                    f"({values[k - 1]:g})"
                    f" − {values[k - 2]:g}"
                ),
                "",
                f"**S{k} = {values[k]:g}**",
                "",
            ]
        )

    lines.extend(
        [
            "### Final Answer",
            "",
            (
                f"**x^{n} + 1/x^{n} = "
                f"{values[n]:g} ✅**"
            ),
        ]
    )

    return "\n".join(lines)


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


# ============================================================
# REGISTER
# ============================================================

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
                "Enter either email or phone number.",
        }, 400

    if len(password) < 6:

        return {
            "success": False,
            "message":
                "Password must be at least 6 characters.",
        }, 400

    users = load_users()

    for existing_username, user in users.items():

        if (
            existing_username.lower()
            == username.lower()
        ):

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
            ).lower()
            == email
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
            )
            == phone
        ):

            return {
                "success": False,
                "message":
                    "Phone number already registered.",
            }, 400

    users[username] = {
        "email": email,
        "phone": phone,
        "password":
            generate_password_hash(
                password
            ),
    }

    save_users(
        users
    )

    session["username"] = username
    session["chat_history"] = []

    clear_question_set()

    return {
        "success": True,
        "message":
            "Account created successfully.",
    }


# ============================================================
# LOGIN
# ============================================================

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

    users = load_users()

    for username, user in users.items():

        username_match = (
            username.lower()
            == login_value.lower()
        )

        email_match = (
            bool(
                user.get(
                    "email"
                )
            )
            and
            user.get(
                "email",
                "",
            ).lower()
            == login_value.lower()
        )

        phone_match = (
            bool(
                user.get(
                    "phone"
                )
            )
            and
            user.get(
                "phone",
                "",
            )
            == login_value
        )

        if (
            username_match
            or email_match
            or phone_match
        ):

            if check_password_hash(
                user.get(
                    "password",
                    "",
                ),
                password,
            ):

                session["username"] = username
                session["chat_history"] = []

                # A new login starts a fresh question set.
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
            mimetype="text/plain",
            status=401,
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
    # IMAGE / CAMERA
    # ========================================================

    if image_data:

        answer = analyze_image(
            image_data,
            question,
        )

        save_history(
            question or "Image question",
            answer,
            category="normal",
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
    # EXPLICIT CREATOR / BASIC
    # ========================================================

    simple = basic_response(
        question
    )

    if simple:

        category = (
            "creator"
            if is_creator_question(
                question
            )
            else "normal"
        )

        save_history(
            question,
            simple,
            category=category,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # ========================================================
    # ACTIVE QUESTION SET
    # ========================================================

    question_set = get_question_set()

    # --------------------------------------------------------
    # "ans of all"
    # --------------------------------------------------------

    if answer_key_command(
        question
    ):

        if question_set:

            answer = modify_question_set(
                question_set,
                question,
            )

            save_history(
                question,
                answer,
                category="question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        # If there is no question set, DO NOT let normal
        # memory turn it into creator information.
        return Response(
            "There is no active question set. "
            "First ask me to generate some questions.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # "with options"
    # --------------------------------------------------------

    if options_command(
        question
    ):

        if question_set:

            updated = modify_question_set(
                question_set,
                question,
            )

            save_question_set(
                question_set["request"],
                updated,
                question_set["subject"],
                question_set["difficulty"],
            )

            save_history(
                question,
                updated,
                category="question_generation",
            )

            return Response(
                updated,
                mimetype="text/plain",
            )

        return Response(
            "There is no active question set. "
            "Generate questions first.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # "explain all"
    # --------------------------------------------------------

    if explanation_command(
        question
    ):

        if question_set:

            answer = modify_question_set(
                question_set,
                question,
            )

            save_history(
                question,
                answer,
                category="question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        return Response(
            "There is no active question set. "
            "Generate questions first.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # "other 14 questions"
    # --------------------------------------------------------

    continuation = continuation_count(
        question
    )

    if continuation is not None:

        if question_set:

            start_number = (
                last_question_number(
                    question_set["answer"]
                )
                + 1
            )

            generated = generate_questions_in_batches(
                subject=question_set["subject"],
                difficulty=question_set["difficulty"],
                total=continuation,
                start_number=start_number,
                previous_context=question_set["answer"],
            )

            combined = (
                question_set["answer"]
                + "\n\n"
                + generated
            )

            save_question_set(
                question_set["request"],
                combined,
                question_set["subject"],
                question_set["difficulty"],
            )

            save_history(
                question,
                generated,
                category="question_generation",
            )

            return Response(
                generated,
                mimetype="text/plain",
            )

        return Response(
            "There is no active question set to continue.",
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL CALCULATOR
    # ========================================================

    result = solve_calculation(
        question
    )

    if result:

        save_history(
            question,
            result,
            category="normal",
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL POWER SOLVER
    # ========================================================

    result = solve_power_recurrence(
        question
    )

    if result:

        save_history(
            question,
            result,
            category="normal",
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # NEW QUESTION SET
    # ========================================================

    if is_question_generation(
        question
    ):

        count = requested_question_count(
            question
        )

        if count:

            subject = detect_subject(
                question
            )

            difficulty = detect_difficulty(
                question
            )

            generated = generate_questions_in_batches(
                subject=subject,
                difficulty=difficulty,
                total=count,
                start_number=1,
            )

            save_question_set(
                question,
                generated,
                subject,
                difficulty,
            )

            save_history(
                question,
                generated,
                category="question_generation",
            )

            return Response(
                generated,
                mimetype="text/plain",
            )

    # ========================================================
    # NORMAL / HARD AI
    # ========================================================

    subject = detect_subject(
        question
    )

    difficulty = detect_difficulty(
        question
    )

    context = build_normal_context(
        question
    )

    prompt = get_prompt(
        subject
    )

    if difficulty == "advanced":

        prompt += """

ADVANCED REASONING MODE.

Solve carefully.

Recheck:
- equations
- algebra
- arithmetic
- units
- signs
- limiting cases
- MCQ options

Never guess.

Do not reveal private chain-of-thought.
Show the useful derivation and verification only.
"""

        reasoning = "high"
        max_tokens = 9000

    elif difficulty == "intermediate":

        prompt += """

INTERMEDIATE MODE.

Show important steps and verify the result.
"""

        reasoning = "medium"
        max_tokens = 5000

    else:

        prompt += """

BASIC MODE.

Answer clearly and concisely.
"""

        reasoning = "low"
        max_tokens = 3000

    # ========================================================
    # HARD MATH / PHYSICS VERIFY
    # ========================================================

    if (
        difficulty == "advanced"
        and subject in {
            "math",
            "physics",
        }
    ):

        try:

            draft = text_call(
                instructions=prompt,
                user_input=context,
                reasoning=reasoning,
                max_tokens=max_tokens,
            )

            answer = verify_advanced_answer(
                question,
                draft,
                subject,
            )

            save_history(
                question,
                answer,
                category="normal",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

        except Exception as error:

            print(
                "ADVANCED ERROR:",
                repr(error),
            )

    # ========================================================
    # NORMAL AI
    # ========================================================

    try:

        answer = text_call(
            instructions=prompt,
            user_input=context,
            reasoning=reasoning,
            max_tokens=max_tokens,
        )

        if not answer:
            answer = (
                "❌ The AI returned an empty answer."
            )

        save_history(
            question,
            answer,
            category="normal",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    except Exception as error:

        print(
            "TEXT AI ERROR:",
            repr(error),
        )

        return Response(
            "❌ Hugging Face error:\n\n"
            + str(error),
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
            mimetype="text/plain",
            status=401,
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
    ).strip()

    old_answer = data.get(
        "answer",
        "",
    ).strip()

    action = data.get(
        "action",
        "improve",
    ).strip().lower()

    actions = {
        "improve":
            "Improve the answer and make it clearer and more complete.",

        "check":
            "Check the answer for errors and correct them.",

        "explain":
            "Explain the answer in more detail.",

        "short":
            "Make the answer shorter while keeping important information.",
    }

    instruction = actions.get(
        action,
        actions["improve"],
    )

    subject = detect_subject(
        question
    )

    prompt = f"""
Improve an educational answer.

QUESTION:
{question}

CURRENT ANSWER:
{old_answer}

TASK:
{instruction}

Check calculations, algebra, equations,
units and final result.

Return only the improved answer.
Do not output raw LaTeX.
"""

    try:

        result = text_call(
            instructions=prompt,
            user_input=prompt,
            reasoning=(
                "high"
                if subject in {"math", "physics"}
                else "medium"
            ),
            max_tokens=8000,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    except Exception as error:

        return Response(
            "❌ AI error:\n\n"
            + str(error),
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
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MY AI")
    print("=" * 70)

    print(
        "HF configured :",
        bool(HF_TOKEN),
    )

    print(
        "Normal AI     : ON"
    )

    print(
        "Hard Math     : ON"
    )

    print(
        "Hard Physics  : ON"
    )

    print(
        "Chemistry     : ON"
    )

    print(
        "Biology       : ON"
    )

    print(
        "Calculator    : ON"
    )

    print(
        "Question Sets : ON"
    )

    print(
        "Batch Repair  : ON"
    )

    print(
        "Safe Memory   : ON"
    )

    print(
        "Vision        : ON"
    )

    print(
        "Camera API    : ON"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
