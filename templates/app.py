from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context,
    session,
    redirect,
    url_for,
)

import os
import re
import math
import ast
import operator
import json

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
    "change-this-secret-in-render",
)

USERS_FILE = "users.json"


# ============================================================
# CREATOR INFORMATION
# ============================================================

CREATOR_RESPONSE = """
🤖 My creator is **Soham Chandrahas Sanap**.

He is 15 years old and is studying in Class 10 in 2026
at Nimbark English School in Beed district, Maharashtra, India.

His main interests are Mathematics and web development.

He built My AI as an AI study assistant to help students
with Mathematics, Physics, Chemistry and Biology.
""".strip()


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "openai/gpt-oss-120b:cheapest"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct:cheapest"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# USER STORAGE
# ============================================================

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(
            USERS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_users(users):
    with open(
        USERS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            users,
            file,
            indent=2,
        )


# ============================================================
# AI PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are My AI, a helpful PCMB study assistant.

You help with:
- Mathematics
- Physics
- Chemistry
- Biology
- General science
- General questions

IMPORTANT:
- Answer every requested part.
- Never stop halfway through a sentence.
- Never leave an equation unfinished.
- Show important calculations.
- Use simple student-friendly language.
- Do not invent missing information.
- Understand short follow-up questions using the previous conversation.
- Always finish with a clear Final Answer.

FORMATTING:
Never output raw LaTeX.

Do not use:
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

Use normal readable text instead.
"""


MATH_SYSTEM_PROMPT = """
You are My AI's Mathematics specialist.

Rules:
- Solve step by step.
- Use simple textbook-style mathematics.
- Use x², x³, x⁵ and √ when useful.
- Do not use raw LaTeX.
- Do not use \\frac, \\boxed, \\text{}, \\begin{} or \\end{}.
- Answer every part.
- Check the final result.
"""


PHYSICS_SYSTEM_PROMPT = """
You are My AI's Physics specialist.

For numerical questions use:

Given
Formula
Substitution
Calculation
Verification
Final Answer

Use correct units.
Answer every requested part.
Do not output raw LaTeX.
"""


CHEMISTRY_SYSTEM_PROMPT = """
You are My AI's Chemistry specialist.

For numerical questions:
- Balance the equation when needed.
- Calculate molar mass carefully.
- Convert units correctly.
- Calculate moles and molarity.
- Use stoichiometric ratios.
- Identify limiting reagent.
- Include units.
- Complete every part.
- Do not output raw LaTeX.
"""


BIOLOGY_SYSTEM_PROMPT = """
You are My AI's Biology specialist.

Explain biology clearly and accurately.

For processes:
Explain steps in order.

For comparisons:
Compare every requested point.

For genetics:
Show genotype, phenotype and probability clearly.

Do not invent biological facts.
Do not output raw LaTeX.
"""


VISION_SYSTEM_PROMPT = """
You are My AI's image-question specialist.

Read the uploaded image carefully.

If it contains Mathematics, Physics, Chemistry or Biology,
solve or explain the question step by step.

Transcribe important visible text when useful.

Do not invent information that is not visible.

Do not output raw LaTeX.
"""


# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

    # --------------------------------------------------------
    # CREATOR
    # --------------------------------------------------------

    creator_questions = {
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
        "who created",
        "who created?",
        "you are created by",
        "you were created by",
        "you were made by",
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

    if q in creator_questions:
        return CREATOR_RESPONSE

    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

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
            "Biology or general questions."
        )

    if q == "good morning":
        return "Good morning! ☀️ How can I help you?"

    if q == "good afternoon":
        return "Good afternoon! 👋 How can I help you?"

    if q == "good evening":
        return "Good evening! 🌙 How can I help you?"

    # --------------------------------------------------------
    # COMMON
    # --------------------------------------------------------

    if q in {
        "how are you",
        "how are you?",
    }:
        return "I'm doing great! 🤖 Ask me a difficult question."

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

    if q in {
        "what can you do",
        "what can you do?",
        "help",
        "help me",
    }:
        return (
            "I can help with:\n\n"
            "🧮 Mathematics\n"
            "⚡ Physics\n"
            "🧪 Chemistry\n"
            "🧬 Biology\n"
            "🔢 Large calculations\n"
            "📷 Questions from images\n"
            "🎤 Voice input"
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
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
    }:
        return "👍 Great! Ask your next question."

    if q in {
        "bye",
        "goodbye",
        "see you",
        "see ya",
    }:
        return "Goodbye! 👋"

    return None


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def is_followup(question):
    q = question.strip().lower()

    # Very common short follow-ups
    short_followups = {
        "other",
        "others",
        "another",
        "another one",
        "more",
        "more examples",
        "next",
        "also",
        "and another",
        "what about",
        "what about it",
        "same",
        "same question",
        "explain more",
        "more detail",
        "top three",
        "top 3",
        "top five",
        "top 5",
        "which one",
        "which is easiest",
        "which one is easiest",
        "which is better",
        "why",
        "why?",
        "how",
        "how?",
    }

    if q in short_followups:
        return True

    # Prefix-based follow-ups
    prefixes = (
        "and ",
        "also ",
        "what about ",
        "how about ",
        "then ",
        "another ",
        "other ",
        "more ",
        "compare ",
        "difference ",
        "differences ",
        "give another ",
        "give me another ",
        "tell me another ",
        "what is the other ",
        "what are the other ",
        "which one ",
        "which is ",
        "why is ",
        "why are ",
        "how is ",
        "how are ",
    )

    if q.startswith(prefixes):
        return True

    # Very short questions usually depend on
    # the previous message.
    if len(q.split()) <= 3:
        return True

    return False


def build_ai_context(question):
    history = session.get(
        "chat_history",
        [],
    )

    if not history:
        return question

    if not is_followup(question):
        return question

    # Include several recent turns so that
    # follow-up questions have useful context.
    recent = history[-4:]

    parts = [
        "The following is the recent conversation.",
        "Use it to understand the user's new question.",
        "",
    ]

    for item in recent:
        parts.append(
            "User: " + item["question"]
        )

        parts.append(
            "My AI: " + item["answer"]
        )

        parts.append("")

    parts.append(
        "New user question: " + question
    )

    parts.append("")
    parts.append(
        "Answer the new question using the conversation context."
    )

    return "\n".join(parts)


def save_history(question, answer):
    history = session.get(
        "chat_history",
        [],
    )

    history.append(
        {
            "question": question,
            "answer": answer[:8000],
        }
    )

    # Keep the latest 10 turns.
    session["chat_history"] = history[-10:]

    session.modified = True


# ============================================================
# SUBJECT DETECTION
# ============================================================

def detect_subject(question):
    q = question.lower()

    math_words = [
        "equation",
        "quadratic",
        "algebra",
        "surds",
        "sqrt",
        "√",
        "svg",
        "trigonometry",
        "sin",
        "cos",
        "tan",
        "geometry",
        "probability",
        "permutation",
        "combination",
        "sequence",
        "series",
        "polynomial",
        "coordinate",
        "logarithm",
        "matrix",
        "determinant",
        "calculus",
        "integral",
        "differentiate",
        "limit",
    ]

    physics_words = [
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
        "magnetic",
        "electric field",
        "lens",
        "mirror",
        "refraction",
        "heat",
        "temperature",
        "pressure",
        "density",
        "fluid",
        "motion",
    ]

    chemistry_words = [
        "mole",
        "moles",
        "molar",
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
        "ester",
        "ion",
        "atom",
        "electron",
        "compound",
        "reaction",
        "chemical",
    ]

    biology_words = [
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
    ]

    scores = {
        "math": sum(word in q for word in math_words),
        "physics": sum(word in q for word in physics_words),
        "chemistry": sum(word in q for word in chemistry_words),
        "biology": sum(word in q for word in biology_words),
    }

    subject = max(
        scores,
        key=scores.get,
    )

    if scores[subject] == 0:
        return "general"

    return subject


def get_prompt(subject):
    if subject == "math":
        return MATH_SYSTEM_PROMPT

    if subject == "physics":
        return PHYSICS_SYSTEM_PROMPT

    if subject == "chemistry":
        return CHEMISTRY_SYSTEM_PROMPT

    if subject == "biology":
        return BIOLOGY_SYSTEM_PROMPT

    return SYSTEM_PROMPT


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
    expression = expression.strip()

    if not expression or len(expression) > 1000:
        return None

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )
    except (SyntaxError, ValueError):
        return None

    def calculate(node):

        if isinstance(node, ast.Constant):
            if isinstance(
                node.value,
                (int, float),
            ):
                return node.value

            raise ValueError()

        if isinstance(node, ast.BinOp):

            operation = ALLOWED_BINARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError()

            left = calculate(node.left)
            right = calculate(node.right)

            if (
                isinstance(node.op, ast.Pow)
                and abs(right) > 10000
            ):
                raise ValueError()

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):

            operation = ALLOWED_UNARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError()

            return operation(
                calculate(node.operand)
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


def clean_calculator_expression(text):
    expression = text.strip()

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("−", "-")
    expression = expression.replace("–", "-")
    expression = expression.replace("—", "-")
    expression = expression.replace("^", "**")

    expression = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        expression,
    )

    expression = re.sub(
        r"^(what is|calculate|find|solve|evaluate)\s+",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    return expression.strip()


def looks_like_calculation(text):
    expression = clean_calculator_expression(text)

    if not re.fullmatch(
        r"[0-9\s\.\+\-\*\/%\(\)]+",
        expression,
    ):
        return False

    return bool(
        re.search(
            r"[\+\-\*\/%]",
            expression,
        )
    )


def solve_calculation(question):
    if not looks_like_calculation(question):
        return None

    expression = clean_calculator_expression(
        question
    )

    result = safe_calculate(
        expression
    )

    if result is None:
        return None

    if isinstance(result, int):
        answer = f"{result:,}"

    elif isinstance(result, float):

        if not math.isfinite(result):
            return None

        if result.is_integer():
            answer = f"{int(result):,}"
        else:
            answer = f"{result:.12g}"

    else:
        return None

    return f"""
### Calculation

**Expression**

{expression}

**Answer**

**{answer} ✅**
""".strip()


# ============================================================
# MATH
# ============================================================

def normalize_math_text(text):
    text = text.replace("svg", "√")
    text = text.replace("SVG", "√")

    superscripts = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )

    return text.translate(
        superscripts
    )


def fmt(value):
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))

    return (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
    )


def solve_power_recurrence(question):
    q = normalize_math_text(
        question.lower()
    )

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
        r"x\s*(?:(?:\^|\*\*)\s*)?(\d+)"
        r"\s*\+\s*1\s*/\s*x\s*"
        r"(?:(?:\^|\*\*)\s*)?\1",
        q,
    )

    if not target:
        return None

    n = int(
        target.group(1)
    )

    if n < 1 or n > 50:
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
        "Given:",
        "",
        f"x + 1/x = {fmt(a)}",
        "",
        "Let:",
        "",
        "Sₙ = xⁿ + 1/xⁿ",
        "",
        "S₀ = 2",
        "",
        f"S₁ = {fmt(a)}",
        "",
    ]

    for k in range(2, n + 1):
        lines.extend(
            [
                f"**Step {k - 1}: Find S{k}**",
                "",
                f"S{k} = {fmt(a)}S{k - 1} − S{k - 2}",
                "",
                (
                    f"S{k} = {fmt(a)}"
                    f"({fmt(values[k - 1])})"
                    f" − {fmt(values[k - 2])}"
                ),
                "",
                f"**S{k} = {fmt(values[k])}**",
                "",
            ]
        )

    lines.extend(
        [
            "### Final Answer",
            "",
            f"**x^{n} + 1/x^{n} = {fmt(values[n])} ✅**",
        ]
    )

    return "\n".join(lines)


# ============================================================
# PHYSICS CIRCUIT
# ============================================================

def solve_circuit(question):
    q = question.lower()

    if "series" not in q or "parallel" not in q:
        return None

    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:v|volt|volts)\b",
        q,
    )

    resistor_values = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*"
            r"(?:Ω|ohm|ohms)",
            question,
            re.IGNORECASE,
        )
    ]

    if not voltage_match or len(resistor_values) < 3:
        return None

    voltage = float(
        voltage_match.group(1)
    )

    match = re.search(
        r"parallel\s+combination\s+of\s+"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:Ω|ohm|ohms)"
        r"\s*(?:and|&)\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:Ω|ohm|ohms)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return None

    r1 = float(match.group(1))
    r2 = float(match.group(2))

    others = resistor_values.copy()

    try:
        others.remove(r1)
        others.remove(r2)
    except ValueError:
        return None

    if not others:
        return None

    rp = (
        r1 * r2
        / (r1 + r2)
    )

    rt = (
        sum(others)
        + rp
    )

    total_i = (
        voltage
        / rt
    )

    series_voltages = [
        total_i * r
        for r in others
    ]

    vp = (
        voltage
        - sum(series_voltages)
    )

    i1 = vp / r1
    i2 = vp / r2

    series_powers = [
        total_i ** 2 * r
        for r in others
    ]

    p1 = vp ** 2 / r1
    p2 = vp ** 2 / r2

    total_power = (
        voltage * total_i
    )

    component_power = (
        sum(series_powers)
        + p1
        + p2
    )

    lines = [
        "### Solution",
        "",
        "**1. Parallel resistance**",
        "",
        "Rₚ = (R₁ × R₂)/(R₁ + R₂)",
        "",
        f"Rₚ = ({fmt(r1)} × {fmt(r2)})/"
        f"({fmt(r1)} + {fmt(r2)})",
        "",
        f"**Rₚ = {fmt(rp)} Ω**",
        "",
        "**2. Total resistance**",
        "",
        f"Rₜ = {' + '.join(fmt(x) for x in others)} + {fmt(rp)}",
        "",
        f"**Rₜ = {fmt(rt)} Ω**",
        "",
        "**3. Total current**",
        "",
        "I = V/Rₜ",
        "",
        f"**I = {fmt(total_i)} A**",
        "",
        "**4. Parallel-section voltage**",
        "",
        f"**Vₚ = {fmt(vp)} V**",
        "",
        f"**5. Current through {fmt(r1)} Ω**",
        "",
        f"**I₁ = {fmt(i1)} A**",
        "",
        f"**6. Current through {fmt(r2)} Ω**",
        "",
        f"**I₂ = {fmt(i2)} A**",
        "",
        "**7. Power**",
        "",
    ]

    for resistor, power in zip(
        others,
        series_powers,
    ):
        lines.extend(
            [
                f"Power in {fmt(resistor)} Ω = **{fmt(power)} W**",
                "",
            ]
        )

    lines.extend(
        [
            f"Power in {fmt(r1)} Ω = **{fmt(p1)} W**",
            "",
            f"Power in {fmt(r2)} Ω = **{fmt(p2)} W**",
            "",
            f"Total power = **{fmt(total_power)} W**",
            "",
            "### Verification",
            "",
            f"Sum of resistor powers = {fmt(component_power)} W",
            "",
            f"Battery power = {fmt(total_power)} W",
            "",
            "**Power check ✅**",
        ]
    )

    return "\n".join(lines)


# ============================================================
# CHEMISTRY
# ============================================================

def solve_iron_hcl(question):
    q = question.lower()

    if "fe" not in q or "hcl" not in q:
        return None

    if not (
        "limiting" in q
        or "hydrogen" in q
        or "h2" in q
        or "fecl2" in q
    ):
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*?(?:iron|fe)",
        question,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mL.*?(?:hcl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M.*?(?:hcl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    if not (
        mass_match
        and volume_match
        and molarity_match
    ):
        return None

    mass_fe = float(
        mass_match.group(1)
    )

    volume_hcl = (
        float(volume_match.group(1))
        / 1000
    )

    molarity_hcl = float(
        molarity_match.group(1)
    )

    fe_moles = mass_fe / 56

    hcl_moles = (
        molarity_hcl
        * volume_hcl
    )

    hcl_required = (
        2 * fe_moles
    )

    if hcl_moles < hcl_required:
        limiting = "HCl"
        h2_moles = (
            hcl_moles / 2
        )
        fe_used = h2_moles
    else:
        limiting = "Fe"
        h2_moles = fe_moles
        fe_used = fe_moles

    h2_volume = (
        h2_moles * 22.4
    )

    fecl2_mass = (
        fe_used * 127
    )

    return f"""
### Solution

Fe + 2HCl → FeCl₂ + H₂

**1. Moles of Fe**

n = m/M

n = {fmt(mass_fe)}/56

**n(Fe) = {fmt(fe_moles)} mol**

**2. Moles of HCl**

n = M × V

V = {fmt(volume_hcl)} L

**n(HCl) = {fmt(hcl_moles)} mol**

**3. Limiting reagent**

HCl required = {fmt(hcl_required)} mol

Available HCl = {fmt(hcl_moles)} mol

**Limiting reagent = {limiting}**

**4. Moles of H₂**

**{fmt(h2_moles)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M = 127 g mol⁻¹

**Mass = {fmt(fecl2_mass)} g**

### Final Answer

- **Fe = {fmt(fe_moles)} mol**
- **HCl = {fmt(hcl_moles)} mol**
- **Limiting reagent = {limiting}**
- **H₂ = {fmt(h2_moles)} mol**
- **H₂ volume = {fmt(h2_volume)} L**
- **FeCl₂ = {fmt(fecl2_mass)} g ✅**
""".strip()


def solve_nacl(question):
    q = question.lower()

    if "nacl" not in q:
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*nacl",
        q,
    )

    if not mass_match:
        return None

    mass = float(
        mass_match.group(1)
    )

    moles = (
        mass / 58.5
    )

    return f"""
### Solution

Molar mass of NaCl:

M = 23 + 35.5

M = 58.5 g mol⁻¹

n = m/M

n = {fmt(mass)}/58.5

**n = {fmt(moles)} mol**

### Final Answer

**{fmt(moles)} mol NaCl ✅**
""".strip()


# ============================================================
# HUGGING FACE AI
# IMPORTANT MEMORY FIX IS HERE
# ============================================================

def stream_text_ai(
    ai_question,
    subject,
    original_question,
):
    """
    Sends the context-aware prompt to Hugging Face.

    When the stream finishes, save the ORIGINAL user
    question and the COMPLETE final answer to session history.

    This fixes:
        User: suggest a book...
        AI: ...
        User: give top three
        AI: should answer from the previous book answer.
    """

    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ Online AI is unavailable because HF_TOKEN is missing.\n\n"
            "Local calculator and local solvers are still available."
        )
        return

    full_answer = ""

    try:

        stream = hf_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": get_prompt(subject),
                },
                {
                    "role": "user",
                    "content": ai_question,
                },
            ],
            stream=True,
            temperature=0.15,
            max_tokens=2500,
        )

        for chunk in stream:

            if not chunk.choices:
                continue

            text = (
                chunk.choices[0]
                .delta
                .content
            )

            if text:
                full_answer += text
                yield text

        # ====================================================
        # IMPORTANT:
        # SAVE THE FINISHED ONLINE AI ANSWER
        # ====================================================

        if full_answer.strip():

            save_history(
                original_question,
                full_answer,
            )

        except Exception as error:

        print(
            "Hugging Face error:",
            repr(error),
        )

        yield (
            "❌ Hugging Face error:\n\n"
            + str(error)
        )


# ============================================================
# VISION AI
# ============================================================

def analyze_image(
    image_data,
    prompt,
):
    if not HF_TOKEN or hf_client is None:
        return (
            "❌ Image AI is unavailable because HF_TOKEN is missing."
        )

    if not image_data.startswith(
        "data:image/"
    ):
        return "❌ Invalid image."

    if len(image_data) > 8_000_000:
        return (
            "❌ Image is too large. "
            "Please select a smaller image."
        )

    try:

        response = hf_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                prompt.strip()
                                if prompt.strip()
                                else
                                "Read this image and solve the question."
                            ),
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
            temperature=0.1,
            max_tokens=2500,
        )

        return (
            response.choices[0]
            .message
            .content
        )

    except Exception as error:

        print(
            "Vision error:",
            repr(error),
        )

        return (
            "❌ I could not analyze that image right now."
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if "username" not in session:

        return render_template(
            "index.html",
            logged_in=False,
        )

    return render_template(
        "index.html",
        logged_in=True,
        username=session["username"],
    )


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
            "message": (
                "Username and password are required."
            ),
        }, 400

    # Email OR phone
    if not email and not phone:

        return {
            "success": False,
            "message": (
                "Enter either email or phone number."
            ),
        }, 400

    if len(password) < 6:

        return {
            "success": False,
            "message": (
                "Password must be at least 6 characters."
            ),
        }, 400

    users = load_users()

    for existing_username, user in users.items():

        if (
            existing_username.lower()
            == username.lower()
        ):

            return {
                "success": False,
                "message": "Username already exists.",
            }, 400

        if (
            email
            and user.get("email", "").lower()
            == email
        ):

            return {
                "success": False,
                "message": "Email already registered.",
            }, 400

        if (
            phone
            and user.get("phone", "")
            == phone
        ):

            return {
                "success": False,
                "message": (
                    "Phone number already registered."
                ),
            }, 400

    users[username] = {
        "email": email,
        "phone": phone,
        "password": generate_password_hash(
            password
        ),
    }

    save_users(users)

    session["username"] = username

    return {
        "success": True,
        "message": "Account created successfully.",
    }


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
            user.get("email", "").lower()
            == login_value.lower()
        )

        phone_match = (
            bool(user.get("phone"))
            and
            user.get("phone", "")
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

                return {
                    "success": True,
                    "message": "Login successful.",
                }

            break

    return {
        "success": False,
        "message": (
            "Invalid username/email/phone or password."
        ),
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

    data = request.get_json(
        silent=True
    ) or {}

    original_question = data.get(
        "message",
        "",
    ).strip()

    image_data = data.get(
        "image",
        "",
    )

    # ========================================================
    # IMAGE
    # ========================================================

    if image_data:

        prompt = (
            original_question
            if original_question
            else
            "Read this image and solve the question."
        )

        answer = analyze_image(
            image_data,
            prompt,
        )

        save_history(
            original_question or "Image question",
            answer,
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # ========================================================
    # EMPTY
    # ========================================================

    if not original_question:

        return Response(
            "Please type a message or select an image.",
            mimetype="text/plain",
        )

    # ========================================================
    # BASIC RESPONSE
    # ========================================================

    simple = basic_response(
        original_question
    )

    if simple:

        # Save the basic response too, because
        # the next user message may depend on it.
        save_history(
            original_question,
            simple,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL CALCULATOR
    # ========================================================

    result = solve_calculation(
        original_question
    )

    if result:

        save_history(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL SOLVERS
    #
    # Use only the NEW question here.
    # This prevents:
    #
    # User: x^5...
    # User: then x^6...
    #
    # from re-solving x^5.
    # ========================================================

    normalized = normalize_math_text(
        original_question
    )

    result = solve_power_recurrence(
        normalized
    )

    if result:

        save_history(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_circuit(
        normalized
    )

    if result:

        save_history(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_iron_hcl(
        normalized
    )

    if result:

        save_history(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_nacl(
        normalized
    )

    if result:

        save_history(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # ONLINE AI
    #
    # IMPORTANT:
    # 1. build context from previous saved turns
    # 2. send context to HF
    # 3. after streaming ends, save ORIGINAL question
    #    + COMPLETE answer
    # ========================================================

    ai_question = build_ai_context(
        normalized
    )

    subject = detect_subject(
        normalized
    )

    return Response(
        stream_with_context(
            stream_text_ai(
                ai_question,
                subject,
                original_question,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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

    data = request.get_json(
        silent=True
    ) or {}

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

    if not question or not old_answer:

        return Response(
            "Missing question or answer.",
            mimetype="text/plain",
        )

    instructions = {
        "improve":
            "Improve the answer and make it clearer and more complete.",

        "check":
            "Check the answer carefully for mistakes and correct them.",

        "explain":
            "Explain the answer in more detail while keeping it student-friendly.",

        "short":
            "Make the answer shorter while keeping important steps.",
    }

    instruction = instructions.get(
        action,
        instructions["improve"],
    )

    subject = detect_subject(
        question
    )

    prompt = f"""
Original question:

{question}

Previous answer:

{old_answer}

Task:

{instruction}

Rules:
- Check calculations.
- Answer every requested part.
- Keep mathematics readable.
- Do not output raw LaTeX.
- Do not invent information.
- Finish completely.
"""

    # Save the improved result after streaming
    # through the same mechanism.
    return Response(
        stream_with_context(
            stream_text_ai(
                prompt,
                subject,
                question,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# START SERVER
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
        "Login         : ON"
    )

    print(
        "Calculator    : ON"
    )

    print(
        "Math solver   : ON"
    )

    print(
        "Physics       : ON"
    )

    print(
        "Chemistry     : ON"
    )

    print(
        "Biology AI    : ON"
    )

    print(
        "Camera        : ON"
    )

    print(
        "Voice         : ON"
    )

    print(
        "Creator info  : ON"
    )

    print(
        "Follow-ups    : ON"
    )

    print(
        "Persistent conversation memory in session: ON"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
