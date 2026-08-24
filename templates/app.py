from flask import Flask, render_template, request, Response, session, redirect, url_for
import os
import re
import json
import ast
import math
import operator

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


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face says omitting the provider suffix selects
# the fastest available provider automatically.
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
My creator is **Soham Chandrahas Sanap**.

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
# USER STORAGE
# ============================================================

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


# ============================================================
# PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a student study assistant.

You help with:
- Mathematics
- Physics
- Chemistry
- Biology
- General science
- Academic questions

Handle basic through very difficult questions.

Rules:
1. Read the complete question.
2. Identify exactly what is being asked.
3. Never guess.
4. Answer every requested part.
5. Show useful calculations.
6. Check important arithmetic.
7. Check units when relevant.
8. Verify important results.
9. Understand relevant follow-up context.
10. Keep simple answers concise.
11. Give useful derivations for difficult questions.
12. Finish completely.

Never output raw LaTeX commands such as:
\\(
\\)
\\[
\\]
\\frac
\\boxed
\\text{}
\\begin{}
\\end{}
$$

Use readable notation such as:
x²
x³
x⁵
√2
a/b
F = ma
ω² = k/m
"""


MATH_PROMPT = """
You are an expert Mathematics solver.

Handle:
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
- Olympiad-style mathematics

For difficult problems:
1. Understand the exact target.
2. Identify relevant information.
3. Choose a correct method.
4. Derive step by step.
5. Simplify carefully.
6. Check algebra.
7. Check the final result.
8. Verify MCQ options.
9. For identities, actually prove the identity.
10. Never guess.

For equations, check roots.
For inequalities, check equality cases.
For probability, check validity.
For geometry, respect all constraints.

Never output raw LaTeX.
"""


PHYSICS_PROMPT = """
You are an expert Physics solver for school, JEE Main and JEE Advanced.

Handle:
- Kinematics
- Newton's laws
- Friction
- Work, energy and power
- Momentum
- Centre of mass
- Circular motion
- Rotation
- Rolling motion
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
2. Identify all relevant bodies.
3. Identify forces and constraints.
4. Choose coordinates.
5. Find equilibrium if needed.
6. Write governing equations.
7. Use approximations only when justified.
8. Derive the requested quantity.
9. Check signs.
10. Check dimensions.
11. Check limiting cases when useful.
12. Verify the final result.

For small-oscillation problems:
- Find equilibrium first.
- Define a small displacement.
- Find the restoring force.
- Apply the correct approximation.
- Compare with the SHM relation.

Never guess.
Do not output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are an expert Chemistry solver.

Handle school, JEE Main and JEE Advanced questions.

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

For numerical problems:
Given
Formula/reaction
Substitution
Calculation
Verification
Final Answer

Check units, molar masses, ratios and limiting reagent.
Never guess.
Do not output raw LaTeX.
"""


BIOLOGY_PROMPT = """
You are an expert Biology solver.

Handle school and competitive Biology questions.

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
1. Identify the exact concept.
2. State relevant facts.
3. Explain the process in order.
4. Compare alternatives when needed.
5. For genetics, calculate probabilities carefully.
6. For MCQs, evaluate the options.

Do not invent facts.
"""


VISION_PROMPT = """
You are My AI's academic image-question solver.

Read the entire uploaded image.

Identify whether it is:
- Mathematics
- Physics
- Chemistry
- Biology
- General academic content

Then solve it.

For Mathematics:
- Read all visible numbers and symbols.
- Reconstruct the question.
- Solve step by step.
- Check algebra and arithmetic.

For Physics:
- Identify the physical system.
- Extract given values.
- Choose the correct laws.
- Derive the result.
- Check dimensions and units.

For Chemistry:
- Read values and equations.
- Balance reactions when necessary.
- Calculate carefully.

For Biology:
- Read labels and diagrams.
- Answer all requested parts.

Do not invent text that is not visible.
If something is unclear, say what is unclear.
Do not merely describe the image.
Actually solve the question.
Do not output raw LaTeX.
"""


def get_prompt(subject):
    return {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }.get(subject, GENERAL_PROMPT)


# ============================================================
# BASIC ANSWERS
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
        "hi!",
        "hello!",
        "hey!",
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
# MEMORY
#
# IMPORTANT:
# creator information, normal chat and question sets are
# completely separated.
# ============================================================

def save_history(question, answer, category="normal"):
    history = session.get("chat_history", [])

    history.append({
        "question": question,
        "answer": answer[:12000],
        "category": category,
    })

    session["chat_history"] = history[-15:]
    session.modified = True


def save_question_set(
    request_text,
    answer,
    subject,
    difficulty,
):
    session["question_set"] = {
        "request": request_text,
        "answer": answer[:60000],
        "subject": subject,
        "difficulty": difficulty,
    }

    session.modified = True


def get_question_set():
    return session.get("question_set")


def clear_question_set():
    session.pop("question_set", None)
    session.modified = True


# ============================================================
# FOLLOW-UP / CONTINUATION
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
        match = re.match(pattern, q)

        if match:
            return min(
                max(int(match.group(1)), 1),
                50,
            )

    return None


def is_question_set_modification(question):
    q = question.strip().lower()

    return q in {
        "with options",
        "add options",
        "give options",
        "with answers",
        "give answers",
        "give answer key",
        "answer key",
        "show answers",
    }


def is_followup(question):
    q = question.strip().lower()

    if continuation_count(question) is not None:
        return True

    if is_question_set_modification(question):
        return True

    short = {
        "other",
        "others",
        "another",
        "another one",
        "more",
        "more examples",
        "also",
        "top three",
        "top 3",
        "top five",
        "top 5",
        "which one",
        "which one is easiest",
        "which is easiest",
        "which is better",
        "why",
        "why?",
        "how",
        "how?",
        "explain more",
        "continue",
    }

    if q in short:
        return True

    prefixes = (
        "and ",
        "also ",
        "what about ",
        "how about ",
        "then ",
        "compare ",
        "difference ",
        "differences ",
        "which one ",
        "which is ",
        "why is ",
        "why are ",
        "how is ",
        "how are ",
    )

    if q.startswith(prefixes):
        return True

    return len(q.split()) <= 3


# ============================================================
# QUESTION NUMBER HELPERS
# ============================================================

def extract_question_numbers(text):
    numbers = re.findall(
        r"(?:^|\n)\s*(?:\*\*)?(\d+)\.",
        text,
    )

    return sorted({
        int(n)
        for n in numbers
    })


def last_question_number(text):
    numbers = extract_question_numbers(text)

    return max(numbers) if numbers else 0


# ============================================================
# NORMAL CONTEXT
# ============================================================

def build_normal_context(question):
    history = session.get(
        "chat_history",
        [],
    )

    # NEVER include creator category automatically.
    useful = [
        item
        for item in history
        if item.get("category") == "normal"
    ]

    if not useful:
        return question

    if not is_followup(question):
        return question

    recent = useful[-5:]

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
        "NEW USER QUESTION:\n" + question
    )

    parts.append(
        "\nAnswer using only relevant previous context."
    )

    return "\n".join(parts)


# ============================================================
# SUBJECT DETECTION
# ============================================================

def detect_subject(question):
    q = question.lower()

    scores = {
        "math": 0,
        "physics": 0,
        "chemistry": 0,
        "biology": 0,
    }

    math_words = [
        "equation", "algebra", "quadratic", "polynomial",
        "trigonometry", "trigonometric", "sin", "cos",
        "tan", "sec", "cosec", "geometry", "identity",
        "surds", "probability", "permutation",
        "combination", "sequence", "series", "matrix",
        "determinant", "calculus", "integral", "derivative",
        "logarithm", "vector", "complex", "binomial",
        "inequality", "number theory", "divisibility",
        "function", "proof",
    ]

    physics_words = [
        "force", "velocity", "acceleration", "momentum",
        "newton", "work", "energy", "power", "friction",
        "gravitation", "projectile", "current", "voltage",
        "resistance", "resistor", "circuit", "electric",
        "magnetic", "lens", "mirror", "refraction",
        "heat", "temperature", "pressure", "density",
        "rotation", "rotational", "torque", "angular momentum",
        "oscillation", "oscillations", "shm", "wave",
        "capacitor", "induction", "electromagnetic",
    ]

    chemistry_words = [
        "mole", "molarity", "molality", "stoichiometry",
        "limiting reagent", "oxidation", "reduction",
        "redox", "acid", "base", "ph", "salt",
        "equilibrium", "enthalpy", "electrochemistry",
        "organic", "alkane", "alkene", "alkyne", "benzene",
        "alcohol", "aldehyde", "ketone", "reaction",
        "atom", "electron", "compound", "kinetics",
    ]

    biology_words = [
        "cell", "mitosis", "meiosis", "chromosome", "gene",
        "genetics", "allele", "dna", "rna", "protein",
        "enzyme", "photosynthesis", "respiration", "plant",
        "animal", "tissue", "organ", "ecosystem", "ecology",
        "evolution", "hormone", "neuron", "digestion",
        "reproduction", "heredity", "blood", "heart",
        "kidney", "lung", "brain",
    ]

    for word in math_words:
        if word in q:
            scores["math"] += 1

    for word in physics_words:
        if word in q:
            scores["physics"] += 1

    for word in chemistry_words:
        if word in q:
            scores["chemistry"] += 1

    for word in biology_words:
        if word in q:
            scores["biology"] += 1

    subject = max(
        scores,
        key=scores.get,
    )

    return (
        subject
        if scores[subject] > 0
        else "general"
    )


# ============================================================
# DIFFICULTY
# ============================================================

def detect_difficulty(question):
    q = question.lower()

    score = 0

    hard_words = [
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

    for word in hard_words:
        if word in q:
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
# QUESTION GENERATION
# ============================================================

def is_question_generation(question):
    q = question.lower()

    if continuation_count(question) is not None:
        return True

    if is_question_set_modification(question):
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
        match = re.search(pattern, q)

        if match:
            return min(
                max(int(match.group(1)), 1),
                50,
            )

    return None


# ============================================================
# QUESTION SET: "WITH OPTIONS"
# ============================================================

def modify_question_set(
    question_set,
    user_request,
):
    prompt = get_prompt(
        question_set["subject"]
    )

    prompt += f"""

QUESTION SET MODIFICATION

Previous request:
{question_set["request"]}

Previous questions:
{question_set["answer"]}

New user request:
{user_request}

The user wants to modify the PREVIOUS QUESTION SET.

Rules:
- Stay on exactly the same subject.
- Do not talk about the creator.
- Do not use unrelated conversation.
- Keep all existing questions.
- Add or change only what the user requested.
- If the user says "with options", add A, B, C and D
  to every question.
- Return the complete question set.
"""

    return text_call(
        instructions=prompt,
        user_input=prompt,
        reasoning="medium",
        max_tokens=14000,
    )


# ============================================================
# SAFE CALCULATOR
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
        if isinstance(node, ast.Constant):
            if isinstance(
                node.value,
                (int, float),
            ):
                return node.value
            raise ValueError()

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_UNARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError()

            return operation(
                calculate(node.operand)
            )

        if isinstance(node, ast.BinOp):
            operation = ALLOWED_BINARY_OPERATORS.get(
                type(node.op)
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
                isinstance(node.op, ast.Pow)
                and abs(right) > 10000
            ):
                raise ValueError()

            return operation(
                left,
                right,
            )

        raise ValueError()

    try:
        return calculate(tree.body)

    except (
        ArithmeticError,
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ):
        return None


def solve_calculation(question):
    q = question.strip()

    q = q.replace("×", "*")
    q = q.replace("÷", "/")
    q = q.replace("−", "-")
    q = q.replace("^", "**")

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

    result = safe_calculate(q)

    if result is None:
        return None

    if isinstance(result, int):
        answer = f"{result:,}"

    else:
        if not math.isfinite(result):
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
# x + 1/x SOLVER
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

    for k in range(1, n):
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

    for k in range(2, n + 1):
        lines.extend([
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
        ])

    lines.extend([
        "### Final Answer",
        "",
        (
            f"**x^{n} + 1/x^{n} = "
            f"{values[n]:g} ✅**"
        ),
    ])

    return "\n".join(lines)


# ============================================================
# HUGGING FACE TEXT CALL
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

        return (
            response.output_text
            or ""
        )

    except Exception as response_error:
        print(
            "RESPONSES API ERROR:",
            repr(response_error),
        )

        # Fallback to Chat Completions.
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

        if response.choices:
            return (
                response.choices[0]
                .message
                .content
                or ""
            )

        return ""


# ============================================================
# QUESTION BATCH GENERATION
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

    prompt = get_prompt(subject)

    prompt += f"""

QUESTION GENERATION MODE

Generate exactly {count} multiple-choice questions.

Start at Question {start_number}.
End at Question {end_number}.

Requirements:
- Same subject: {subject}
- Difficulty: {difficulty}
- Every question must be complete.
- Give A, B, C and D options.
- Make each question different.
- Do not provide answers unless explicitly requested.
- Do not restart numbering.
- Do not add unrelated commentary.
- Before finishing, verify every number exists.

Previous context:
{previous_context}
"""

    return text_call(
        instructions=prompt,
        user_input=prompt,
        reasoning="medium",
        max_tokens=min(
            max(
                5000,
                count * 1200,
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
    pieces = []

    current = start_number
    final_number = (
        start_number + total - 1
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
                f"Empty batch {current}-{batch_end}."
            )

        expected = set(
            range(
                current,
                batch_end + 1,
            )
        )

        found = set(
            extract_question_numbers(batch)
        )

        missing = sorted(
            expected - found
        )

        # Repair only the missing questions.
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

            if repair.strip():
                batch += "\n\n" + repair

            found = set(
                extract_question_numbers(batch)
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

        pieces.append(batch)
        current = batch_end + 1

    final_text = "\n\n".join(pieces)

    final_numbers = set(
        extract_question_numbers(final_text)
    )

    expected_final = set(
        range(
            start_number,
            final_number + 1,
        )
    )

    missing_final = sorted(
        expected_final - final_numbers
    )

    if missing_final:
        raise RuntimeError(
            "Final check failed. Missing: "
            + ", ".join(
                map(str, missing_final)
            )
        )

    return final_text


# ============================================================
# ADVANCED ANSWER VERIFICATION
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

    verifier = """
You are a strict Mathematics and Physics answer verifier.

Check:
- equations
- algebra
- arithmetic
- units
- signs
- final result
- MCQ option if present

If wrong, correct it.
If correct, keep it.
Return only the final educational answer.

Do not reveal private reasoning.
Do not output raw LaTeX.
"""

    try:
        checked = text_call(
            instructions=verifier,
            user_input=(
                "QUESTION:\n"
                + question
                + "\n\nANSWER:\n"
                + answer
            ),
            reasoning="high",
            max_tokens=9000,
        )

        return checked.strip() or answer

    except Exception as error:
        print(
            "VERIFICATION ERROR:",
            repr(error),
        )
        return answer


# ============================================================
# IMAGE / CAMERA
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

    if not image_data.startswith("data:image/"):
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

        answer = (
            response.output_text
            or ""
        )

        if answer.strip():
            return answer

    except Exception as error:
        print(
            "VISION RESPONSES ERROR:",
            repr(error),
        )

    # Fallback
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
            return (
                response.choices[0]
                .message
                .content
                or
                "❌ Vision AI returned no answer."
            )

    except Exception as error:
        print(
            "VISION FALLBACK ERROR:",
            repr(error),
        )

        return (
            "❌ Vision AI error:\n\n"
            + str(error)
        )

    return "❌ Vision AI could not analyze this image."


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        logged_in=("username" in session),
        username=session.get("username"),
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
        request.get_json(silent=True)
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
            and user.get("email", "").lower()
            == email
        ):
            return {
                "success": False,
                "message":
                    "Email already registered.",
            }, 400

        if (
            phone
            and user.get("phone", "")
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
            generate_password_hash(password),
    }

    save_users(users)

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
        request.get_json(silent=True)
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
            bool(user.get("email"))
            and
            user.get(
                "email",
                "",
            ).lower()
            == login_value.lower()
        )

        phone_match = (
            bool(user.get("phone"))
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
        "success": True
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
        request.get_json(silent=True)
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
    # CREATOR / BASIC
    # ========================================================

    simple = basic_response(
        question
    )

    if simple:

        category = (
            "creator"
            if is_creator_question(question)
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
    # QUESTION SET MODIFICATION
    # Example: "with options"
    # ========================================================

    if is_question_set_modification(question):

        question_set = get_question_set()

        if question_set:

            updated = modify_question_set(
                question_set,
                question,
            )

            if not updated:
                return Response(
                    "❌ Could not update the question set.",
                    mimetype="text/plain",
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

    # ========================================================
    # QUESTION SET CONTINUATION
    # Example: "other 14 questions"
    # ========================================================

    continuation = continuation_count(
        question
    )

    if continuation is not None:

        question_set = get_question_set()

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
    # LOCAL x + 1/x
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

    if is_question_generation(question):

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

            # Store ONLY as question-set memory.
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
    # NORMAL AI / HARD AI
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

ADVANCED MODE

Solve very carefully.

Before finalizing:
- Recheck equations.
- Recheck algebra.
- Recheck arithmetic.
- Recheck units.
- Recheck signs.
- Check limiting cases when useful.
- Verify MCQ options.

Never guess.

Do not reveal private chain-of-thought.
Show the useful derivation and verification only.
"""

        reasoning = "high"
        max_tokens = 9000

    elif difficulty == "intermediate":

        prompt += """

INTERMEDIATE MODE

Show the important steps and verify the result.
"""

        reasoning = "medium"
        max_tokens = 5000

    else:

        prompt += """

BASIC MODE

Answer clearly and concisely.
"""

        reasoning = "low"
        max_tokens = 3000

    # ========================================================
    # ADVANCED MATH / PHYSICS = VERIFY
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
        request.get_json(silent=True)
        or {}
    )

    question = data.get(
        "question",
        "",
    ).strip()

    answer = data.get(
        "answer",
        "",
    ).strip()

    action = data.get(
        "action",
        "improve",
    ).strip().lower()

    instructions = {
        "improve":
            "Improve the answer and make it clearer and more complete.",
        "check":
            "Check the answer for mistakes and correct them.",
        "explain":
            "Explain the answer in more detail.",
        "short":
            "Make the answer shorter while keeping important steps.",
    }

    task = instructions.get(
        action,
        instructions["improve"],
    )

    subject = detect_subject(
        question
    )

    prompt = f"""
Improve an educational answer.

QUESTION:
{question}

CURRENT ANSWER:
{answer}

TASK:
{task}

Check calculations, algebra, units and the final answer.
Do not output raw LaTeX.
Return the final answer.
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
# HEALTH CHECK
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

    print("HF configured :", bool(HF_TOKEN))
    print("Hard Maths    : ON")
    print("Hard Physics  : ON")
    print("PCMB          : ON")
    print("Question sets : ON")
    print("Batch repair  : ON")
    print("Safe memory   : ON")
    print("Vision        : ON")
    print("Camera API    : ON")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
