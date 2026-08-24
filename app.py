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
# APP CONFIG
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

CREATOR_NAME = "Soham Chandrahas Sanap"

CREATOR_RESPONSE = """
🤖 My creator is **Soham Chandrahas Sanap**.

He is 15 years old and is studying in Class 10 in 2026
at Nimbark English School in Beed district, Maharashtra, India.

His main interests are Mathematics and web development.

He built My AI as an AI study assistant to help students
with their studies, including Mathematics, Physics, Chemistry
and Biology.
""".strip()


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "openai/gpt-oss-120b:cheapest"

# Vision model used for image questions.
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

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        pass

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
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are My AI, a helpful PCMB study assistant.

Subjects:
- Mathematics
- Physics
- Chemistry
- Biology
- General science
- General questions

Rules:
- Read the complete question.
- Answer every requested part.
- Never stop halfway through a sentence.
- Never leave an equation unfinished.
- Show important calculations.
- Use simple student-friendly language.
- Do not invent missing information.
- For follow-up questions, use previous context.
- For current facts, use approximate wording when appropriate.
- Always finish with a clear Final Answer.
"""


MATH_SYSTEM_PROMPT = r"""
You are My AI's Mathematics specialist.

Formatting rules:
- Do not use raw LaTeX commands.
- Do not output \frac, \boxed, \begin, \end or similar raw commands.
- Do not put equations inside [ ... ].
- Use readable notation such as x², x³, x⁵ and √.
- Never replace √ with x.
- In this project, "svg" means √.
- Show important algebraic steps.
- Answer every numbered part.
- Check calculations.
- Never leave an equation incomplete.

Always finish with:

### Final Answer

**answer ✅**
"""


PHYSICS_SYSTEM_PROMPT = """
You are My AI's Physics specialist.

For numerical questions use:

### Given
Known values and units.

### Formula
Relevant equation.

### Solution
Substitute values and calculate.

### Verification
Check units or the physical relation.

### Final Answer
Give the result with units.

Complete every requested part.
"""


CHEMISTRY_SYSTEM_PROMPT = """
You are My AI's Chemistry specialist.

For numerical questions:
- Write and balance the reaction when necessary.
- Calculate molar mass carefully.
- Convert mL to L.
- Calculate moles and molarity.
- Use stoichiometric ratios.
- Identify limiting reagent.
- Calculate products.
- Include units.
- Answer every requested part.

Finish with a clear Final Answer.
"""


BIOLOGY_SYSTEM_PROMPT = """
You are My AI's Biology specialist.

Explain biology accurately and clearly.

For processes:
- Give events in correct order.

For comparisons:
- Compare every requested point.

For genetics:
- Show genotype and phenotype ratios.
- Show probabilities clearly.

Do not invent biological facts.
Complete every requested part.
"""


VISION_SYSTEM_PROMPT = """
You are My AI's image-question specialist.

The user has uploaded an image.

Read the image carefully.

If it contains:
- Mathematics: solve it step by step.
- Physics: solve it with formulas and units.
- Chemistry: solve it carefully with equations and calculations.
- Biology: identify structures or answer the question accurately.

Transcribe important text from the image when needed.

Do not invent text that is not visible.

End with a clear Final Answer.
"""


# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

    # --------------------------------------------------------
    # CREATOR
    # --------------------------------------------------------

    if q in {
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
        "who is soham chandrahas sanap",
        "who is soham chandrahas sanap?",
        "tell me about your creator",
        "tell me about your creator?",
        "who is your developer",
        "who is your developer?",
    }:
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
    # COMMON QUESTIONS
    # --------------------------------------------------------

    if q in {
        "how are you",
        "how are you?",
        "how r u",
        "how r u?",
    }:
        return (
            "I'm doing great! 🤖\n\n"
            "Give me a difficult question."
        )

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
            "🎤 Voice input\n"
            "📚 General questions"
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
        "potential",
        "lens",
        "mirror",
        "refraction",
        "wavelength",
        "frequency",
        "heat",
        "specific heat",
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
        "thermochemistry",
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
        "proton",
        "periodic table",
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
        "nervous",
        "digestion",
        "reproduction",
        "fertilisation",
        "fertilization",
        "heredity",
        "blood",
        "heart",
        "kidney",
        "lung",
        "brain",
    ]

    scores = {
        "math": sum(
            word in q
            for word in math_words
        ),
        "physics": sum(
            word in q
            for word in physics_words
        ),
        "chemistry": sum(
            word in q
            for word in chemistry_words
        ),
        "biology": sum(
            word in q
            for word in biology_words
        ),
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
# FOLLOW-UP MEMORY
# ============================================================

def is_followup(question):
    q = question.strip().lower()

    # Very short follow-up phrases
    short_followups = {
        "other",
        "others",
        "another",
        "another one",
        "more",
        "more examples",
        "next",
        "and another",
        "also",
        "also tell me",
        "what about",
        "what about it",
        "then",
        "same",
        "same question",
        "explain more",
        "more detail",
        "another one?",
        "other?",
    }

    if q in short_followups:
        return True

    # Follow-up phrases that normally depend on
    # the previous question.
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
        "what is the other ",
        "what are the other ",
        "give another ",
        "give me another ",
        "tell me another ",
    )

    if q.startswith(prefixes):
        return True

    # Very short questions are often follow-ups.
    words = q.split()

    if len(words) <= 3:
        return True

    return False


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

    if not expression:
        return None

    if len(expression) > 1000:
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

            # Avoid extremely large powers.
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
                calculate(node.operand)
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


def clean_calculator_expression(text):
    expression = text.strip()

    expression = expression.replace(
        "×",
        "*",
    )

    expression = expression.replace(
        "÷",
        "/",
    )

    expression = expression.replace(
        "−",
        "-",
    )

    expression = expression.replace(
        "–",
        "-",
    )

    expression = expression.replace(
        "—",
        "-",
    )

    # ^ means power in calculator mode.
    expression = expression.replace(
        "^",
        "**",
    )

    # Remove commas from big numbers.
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
    expression = clean_calculator_expression(
        text
    )

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
    if not looks_like_calculation(
        question
    ):
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
# MATH HELPERS
# ============================================================

def normalize_math_text(text):
    text = text.replace(
        "svg",
        "√",
    )

    text = text.replace(
        "SVG",
        "√",
    )

    superscript_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )

    return text.translate(
        superscript_map
    )


def fmt(value):
    if abs(
        value - round(value)
    ) < 1e-10:
        return str(
            int(round(value))
        )

    return (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
    )


# ============================================================
# MATH: x + 1/x = a
# ============================================================

def solve_power_recurrence(question):
    """
    Supports:
        x^5
        x**5
        x⁵

    Example:
        If x + 1/x = 5,
        find x⁵ + 1/x⁵.
    """

    q = normalize_math_text(
        question.lower()
    )

    # Find x + 1/x = number
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

    # After superscript conversion x⁵ becomes x5.
    #
    # Recognizes:
    # x^5
    # x**5
    # x5
    #
    # and the same exponent after 1/x.
    target = re.search(
        r"x\s*(?:(?:\^|\*\*)\s*)?(\d+)"
        r"\s*\+\s*"
        r"1\s*/\s*x\s*"
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

    # Recurrence:
    # S0 = 2
    # S1 = a
    # S(n+1) = a*S(n) - S(n-1)

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
        "We use:",
        "",
        "Sₙ₊₁ = (x + 1/x)Sₙ − Sₙ₋₁",
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
# MATH: KNOWN SURDS
# ============================================================

def solve_known_surds(question):
    q = normalize_math_text(
        question
    )

    compact = re.sub(
        r"\s+",
        "",
        q,
    )

    has_a = (
        "√(5+2√6)" in compact
        or "√(5+2√(6))" in compact
    )

    has_b = (
        "√(5-2√6)" in compact
        or "√(5-2√(6))" in compact
    )

    if not has_a or not has_b:
        return None

    a = math.sqrt(
        5 + 2 * math.sqrt(6)
    )

    b = math.sqrt(
        5 - 2 * math.sqrt(6)
    )

    if (
        "√(5+2√6)+√(5-2√6)"
        in compact
        and
        "√(5+2√6)-√(5-2√6)"
        in compact
    ):

        value = (
            a + b
        ) / (
            a - b
        )

        return f"""
### Solution

√(5 + 2√6) = √3 + √2

√(5 − 2√6) = √3 − √2

Therefore:

N = [(√3 + √2) + (√3 − √2)]
    / [(√3 + √2) − (√3 − √2)]

N = 2√3 / 2√2

N = √(3/2)

### Final Answer

**√(3/2) ≈ {fmt(value)} ✅**
""".strip()

    return None


# ============================================================
# PHYSICS: SERIES + PARALLEL CIRCUIT
# ============================================================

def solve_circuit(question):
    q = question.lower()

    if (
        "series" not in q
        or "parallel" not in q
    ):
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

    if not voltage_match:
        return None

    if len(resistor_values) < 3:
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

    r1 = float(
        match.group(1)
    )

    r2 = float(
        match.group(2)
    )

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

    p1 = (
        vp ** 2
        / r1
    )

    p2 = (
        vp ** 2
        / r2
    )

    total_power = (
        voltage
        * total_i
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
        f"Rₜ = {' + '.join(fmt(x) for x in others)}"
        f" + {fmt(rp)}",
        "",
        f"**Rₜ = {fmt(rt)} Ω**",
        "",
        "**3. Total current**",
        "",
        "I = V/Rₜ",
        "",
        f"I = {fmt(voltage)}/{fmt(rt)}",
        "",
        f"**I = {fmt(total_i)} A**",
        "",
    ]

    step = 4

    for resistor, drop in zip(
        others,
        series_voltages,
    ):
        lines.extend(
            [
                f"**{step}. Voltage across {fmt(resistor)} Ω**",
                "",
                "V = IR",
                "",
                f"V = {fmt(total_i)} × {fmt(resistor)}",
                "",
                f"**V = {fmt(drop)} V**",
                "",
            ]
        )

        step += 1

    lines.extend(
        [
            f"**{step}. Parallel-section voltage**",
            "",
            f"**Vₚ = {fmt(vp)} V**",
            "",
        ]
    )

    step += 1

    lines.extend(
        [
            f"**{step}. Current through {fmt(r1)} Ω**",
            "",
            "I₁ = Vₚ/R₁",
            "",
            f"**I₁ = {fmt(i1)} A**",
            "",
        ]
    )

    step += 1

    lines.extend(
        [
            f"**{step}. Current through {fmt(r2)} Ω**",
            "",
            "I₂ = Vₚ/R₂",
            "",
            f"**I₂ = {fmt(i2)} A**",
            "",
        ]
    )

    step += 1

    lines.extend(
        [
            f"**{step}. Power dissipated**",
            "",
        ]
    )

    for resistor, power in zip(
        others,
        series_powers,
    ):
        lines.extend(
            [
                f"Power in {fmt(resistor)} Ω = I²R",
                "",
                f"**P = {fmt(power)} W**",
                "",
            ]
        )

    lines.extend(
        [
            f"Power in {fmt(r1)} Ω = V²/R",
            "",
            f"**P = {fmt(p1)} W**",
            "",
            f"Power in {fmt(r2)} Ω = V²/R",
            "",
            f"**P = {fmt(p2)} W**",
            "",
        ]
    )

    step += 1

    lines.extend(
        [
            f"**{step}. Total power**",
            "",
            "P = VI",
            "",
            f"P = {fmt(voltage)} × {fmt(total_i)}",
            "",
            f"**P = {fmt(total_power)} W**",
            "",
            "### Verification",
            "",
            f"Component powers = {fmt(component_power)} W",
            "",
            f"Battery power = {fmt(total_power)} W",
            "",
            "**Power check ✅**",
            "",
            "### Final Answer",
            "",
            f"- **Parallel resistance = {fmt(rp)} Ω**",
            f"- **Total resistance = {fmt(rt)} Ω**",
            f"- **Total current = {fmt(total_i)} A**",
            f"- **Parallel voltage = {fmt(vp)} V**",
            f"- **Total power = {fmt(total_power)} W ✅**",
        ]
    )

    return "\n".join(lines)


# ============================================================
# CHEMISTRY: FE + HCL
# ============================================================

def solve_iron_hcl(question):
    q = question.lower()

    if (
        "fe" not in q
        or "hcl" not in q
    ):
        return None

    if not (
        "limiting" in q
        or "hydrogen" in q
        or "h2" in q
        or "fecl2" in q
    ):
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*?"
        r"(?:iron|fe)",
        question,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mL.*?"
        r"(?:HCl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M.*?"
        r"(?:HCl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    if (
        not mass_match
        or not volume_match
        or not molarity_match
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

    fe_moles = (
        mass_fe / 56
    )

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

n = {fmt(molarity_hcl)} × {fmt(volume_hcl)}

**n(HCl) = {fmt(hcl_moles)} mol**

**3. Limiting reagent**

HCl required = 2 × {fmt(fe_moles)}

= **{fmt(hcl_required)} mol**

Available HCl = **{fmt(hcl_moles)} mol**

**Limiting reagent = {limiting}**

**4. Moles of H₂**

**n(H₂) = {fmt(h2_moles)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M(FeCl₂) = 56 + 2(35.5)

= 127 g mol⁻¹

**Mass(FeCl₂) = {fmt(fecl2_mass)} g**

### Final Answer

- **Fe = {fmt(fe_moles)} mol**
- **HCl = {fmt(hcl_moles)} mol**
- **Limiting reagent = {limiting}**
- **H₂ = {fmt(h2_moles)} mol**
- **H₂ at STP = {fmt(h2_volume)} L**
- **FeCl₂ = {fmt(fecl2_mass)} g ✅**
""".strip()


# ============================================================
# CHEMISTRY: NaCl
# ============================================================

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

    molar_mass = 58.5
    moles = (
        mass / molar_mass
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
# HUGGING FACE TEXT AI
# ============================================================

def stream_text_ai(question, subject):
    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ Online AI is unavailable because HF_TOKEN is missing.\n\n"
            "Local calculator and exact solvers are still available."
        )
        return

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
                    "content": question,
                },
            ],
            stream=True,
            temperature=0.15,
            max_tokens=2000,
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
                yield text

    except Exception as error:
        print(
            "Hugging Face text error:",
            repr(error),
        )

        yield (
            "❌ Online AI could not answer right now."
        )


# ============================================================
# VISION AI
# ============================================================

def analyze_image(image_data, prompt):
    if not HF_TOKEN or hf_client is None:
        return (
            "❌ Image AI is unavailable because HF_TOKEN is missing."
        )

    if not image_data.startswith(
        "data:image/"
    ):
        return (
            "❌ Invalid image data."
        )

    # Prevent very large payloads.
    if len(image_data) > 8_000_000:
        return (
            "❌ Image is too large. "
            "Please use a smaller image."
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
                                "Read the image and solve the question shown in it."
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
            max_tokens=2000,
        )

        return (
            response.choices[0]
            .message
            .content
        )

    except Exception as error:
        print(
            "Hugging Face vision error:",
            repr(error),
        )

        return (
            "❌ I couldn't analyze the image right now.\n\n"
            "Please check the Hugging Face connection "
            "and try again."
        )


# ============================================================
# HOME / LOGIN PAGE
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

    # Email OR phone.
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

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if image_data:

        prompt = (
            original_question
            if original_question
            else
            "Read this image and solve the question shown in it."
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

    # --------------------------------------------------------
    # EMPTY
    # --------------------------------------------------------

    if not original_question:
        return Response(
            "Please type a message or select an image.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # BASIC
    # --------------------------------------------------------

    simple = basic_response(
        original_question
    )

    if simple:

        save_history(
            original_question,
            simple,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # LOCAL CALCULATOR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOCAL SOLVERS
    #
    # IMPORTANT:
    # These use only the new question.
    # --------------------------------------------------------

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

    result = solve_known_surds(
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

    # --------------------------------------------------------
    # ONLINE AI
    # --------------------------------------------------------

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
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# IMPROVE / CHECK / EXPLAIN / SHORT
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

    if (
        not question
        or not old_answer
    ):
        return Response(
            "Missing question or answer.",
            mimetype="text/plain",
        )

    instructions = {
        "improve": (
            "Improve the answer and make it "
            "clearer, more accurate and complete."
        ),
        "check": (
            "Check the answer carefully for "
            "mistakes and correct them."
        ),
        "explain": (
            "Explain the answer in more detail "
            "while keeping it student-friendly."
        ),
        "short": (
            "Make the answer shorter while "
            "keeping the important steps and result."
        ),
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
- Do not use broken raw LaTeX.
- Do not invent information.
- Finish completely.
"""

    return Response(
        stream_with_context(
            stream_text_ai(
                prompt,
                subject,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MY AI")
    print("=" * 70)
    print(
        "Hugging Face configured:",
        bool(HF_TOKEN),
    )
    print("Creator information     : ON")
    print("Login system            : ON")
    print("Create account          : ON")
    print("Password hashing        : ON")
    print("Calculator              : ON")
    print("Math solver             : ON")
    print("Physics solver          : ON")
    print("Chemistry solver        : ON")
    print("Biology AI              : ON")
    print("Image questions         : ON")
    print("Follow-up context       : ON")
    print("Improve tools           : ON")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
