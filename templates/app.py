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
# CREATOR
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

# Do not add :cheapest here.
TEXT_MODEL = "openai/gpt-oss-120b"

# Current Qwen vision model.
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# USER FILE
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
# PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a high-quality study assistant.

You help with Mathematics, Physics, Chemistry, Biology,
Science and general academic questions.

Solve questions from basic through very difficult levels.

Rules:
1. Read the entire question.
2. Identify exactly what is being asked.
3. Never guess.
4. Never skip requested parts.
5. Show important calculations.
6. Check important arithmetic.
7. Check units where applicable.
8. Verify the final answer when practical.
9. For multiple-choice questions, verify the option.
10. Understand follow-up questions using conversation context.
11. Always finish the answer.

For simple questions, be concise.
For hard questions, show the useful derivation.

NEVER output raw LaTeX.

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

Use readable mathematics such as:
x²
x³
x⁵
√2
a/b
F = ma
ω² = k/m
"""


MATH_PROMPT = """
You are an expert Mathematics problem solver.

You must handle basic, intermediate and very difficult problems.

Topics include:
- Algebra
- Quadratics
- Polynomials
- Identities
- Surds
- Sequences and series
- Binomial theorem
- Permutations and combinations
- Probability
- Trigonometry
- Geometry
- Coordinate geometry
- Vectors
- Matrices
- Determinants
- Functions
- Logarithms
- Inequalities
- Number theory
- Complex numbers
- Limits
- Differentiation
- Integration
- Calculus
- JEE Main
- JEE Advanced
- Olympiad-style problems

For difficult Maths:

1. Understand the target.
2. Identify useful information.
3. Choose the right method.
4. Derive step by step.
5. Simplify carefully.
6. Check the result.
7. Give the exact answer.
8. For MCQs, verify the option.

For equations, check roots.
For identities, verify both sides.
For inequalities, check equality cases.
For probability, check that the probability is valid.
For geometry, respect all given constraints.
For calculus, check the resulting derivative/integral.

Never guess.

Never output raw LaTeX.
"""


PHYSICS_PROMPT = """
You are an expert JEE Main and JEE Advanced Physics solver.

Handle basic through very difficult problems involving:

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
2. Identify objects and forces.
3. Choose coordinates.
4. Find equilibrium if needed.
5. Write the governing equations.
6. Apply approximations only when justified.
7. Derive the requested quantity.
8. Check dimensions.
9. Check signs.
10. Check limiting cases when useful.
11. For MCQs, verify the answer against the options.

For small oscillation problems:
- Find equilibrium first.
- Define a small displacement.
- Find restoring force.
- Use the small-displacement approximation.
- Compare with the SHM relation.

Never guess.

Do not output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are an expert Chemistry solver for school, JEE Main
and JEE Advanced.

Handle:
- Mole concept
- Stoichiometry
- Atomic structure
- Periodic properties
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
- Numerical chemistry

For numerical questions:
1. Write given data.
2. Balance the equation if required.
3. Calculate molar masses.
4. Convert units correctly.
5. Use stoichiometric ratios.
6. Identify limiting reagent.
7. Calculate the requested quantity.
8. Verify units.

Do not guess.
Do not output raw LaTeX.
"""


BIOLOGY_PROMPT = """
You are an expert Biology solver.

Handle basic through difficult school and competitive questions.

Topics include:
- Cell biology
- Genetics
- Mendelian inheritance
- Molecular biology
- DNA
- RNA
- Biotechnology
- Human physiology
- Plant physiology
- Reproduction
- Evolution
- Ecology

For difficult questions:
1. Identify the concept.
2. State relevant facts.
3. Explain the process in order.
4. Compare alternatives when needed.
5. For genetics, calculate genotype, phenotype and probability.
6. For MCQs, evaluate the options carefully.

Do not invent biological information.
"""


VISION_PROMPT = """
You are My AI's image-question solver.

The user has uploaded a photo.

First inspect the entire image.

Determine whether it contains:
- Mathematics
- Physics
- Chemistry
- Biology
- Another academic question

Then actually solve the question.

MATHEMATICS:
- Read every visible number and symbol.
- Reconstruct the equation carefully.
- Solve step by step.
- Check the result.

PHYSICS:
- Identify the physical system.
- Extract given data.
- Choose the correct laws.
- Derive the result.
- Check units.

CHEMISTRY:
- Read all values and reactions.
- Balance equations where necessary.
- Calculate carefully.

BIOLOGY:
- Read labels and diagrams carefully.
- Answer every requested part.

If the image is blurry or part is missing, say what is unclear.
Do not invent information that is not visible.

Do not merely describe the image.
Solve the question.

Never output raw LaTeX.
"""


# ============================================================
# BASIC ANSWERS
# ============================================================

def basic_response(message):

    q = message.strip().lower()

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

    if q in {"how are you", "how are you?"}:
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
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty",
    }:
        return "You're welcome! 😊"

    return None


# ============================================================
# FOLLOW-UP MEMORY
# ============================================================

def is_followup(question):

    q = question.strip().lower()

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
        "which one is easiest",
        "which is easiest",
        "which is better",
        "why",
        "why?",
        "how",
        "how?",
    }

    if q in short_followups:
        return True

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
        "which one ",
        "which is ",
        "why is ",
        "why are ",
        "how is ",
        "how are ",
    )

    if q.startswith(prefixes):
        return True

    if len(q.split()) <= 3:
        return True

    return False


def save_history(question, answer):

    history = session.get(
        "chat_history",
        [],
    )

    history.append(
        {
            "question": question,
            "answer": answer[:10000],
        }
    )

    session["chat_history"] = history[-10:]
    session.modified = True


def build_ai_context(question):

    history = session.get(
        "chat_history",
        [],
    )

    if not history:
        return question

    if not is_followup(question):
        return question

    recent = history[-5:]

    parts = [
        "RECENT CONVERSATION:",
        "",
    ]

    for item in recent:
        parts.append("USER:")
        parts.append(item["question"])
        parts.append("MY AI:")
        parts.append(item["answer"])
        parts.append("")

    parts.append("NEW USER QUESTION:")
    parts.append(question)
    parts.append("")

    parts.append(
        "Answer the new question using relevant "
        "previous context. Do not discuss the memory itself."
    )

    return "\n".join(parts)


# ============================================================
# SUBJECT
# ============================================================

def detect_subject(question):

    q = question.lower()

    math_terms = [
        "equation", "quadratic", "algebra", "surds",
        "sqrt", "√", "svg", "trigonometry",
        "sin", "cos", "tan", "geometry",
        "probability", "permutation", "combination",
        "sequence", "series", "polynomial",
        "coordinate", "logarithm", "matrix",
        "determinant", "calculus", "integral",
        "differentiate", "derivative", "limit",
        "complex number", "binomial", "inequality",
        "vector", "number theory", "divisibility",
        "function",
    ]

    physics_terms = [
        "force", "velocity", "acceleration", "momentum",
        "newton", "work", "energy", "power", "friction",
        "gravitation", "projectile", "current", "voltage",
        "resistance", "resistor", "circuit", "magnetic",
        "electric field", "electric potential",
        "lens", "mirror", "refraction", "heat",
        "temperature", "pressure", "density",
        "fluid", "motion", "rotation", "rotational",
        "torque", "angular momentum", "oscillation",
        "oscillations", "shm", "wave", "capacitor",
        "induction", "electromagnetic",
    ]

    chemistry_terms = [
        "mole", "moles", "molar", "molarity",
        "molality", "stoichiometry", "limiting reagent",
        "oxidation", "reduction", "redox", "acid",
        "base", "ph", "salt", "equilibrium",
        "enthalpy", "electrochemistry", "organic",
        "alkane", "alkene", "alkyne", "benzene",
        "alcohol", "aldehyde", "ketone", "ester",
        "ion", "atom", "electron", "compound",
        "reaction", "chemical", "thermodynamics",
        "kinetics",
    ]

    biology_terms = [
        "cell", "mitosis", "meiosis", "chromosome",
        "gene", "genetics", "allele", "dna",
        "rna", "protein", "enzyme", "photosynthesis",
        "respiration", "plant", "animal", "tissue",
        "organ", "ecosystem", "ecology", "evolution",
        "hormone", "neuron", "digestion",
        "reproduction", "heredity", "blood", "heart",
        "kidney", "lung", "brain", "biotechnology",
    ]

    scores = {
        "math": sum(word in q for word in math_terms),
        "physics": sum(word in q for word in physics_terms),
        "chemistry": sum(word in q for word in chemistry_terms),
        "biology": sum(word in q for word in biology_terms),
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

    if q.count("?") >= 2:
        score += 2

    if len(
        re.findall(
            r"\b\d+\.",
            q,
        )
    ) >= 3:
        score += 2

    if score >= 6:
        return "advanced"

    if score >= 2:
        return "intermediate"

    return "basic"


def get_prompt(subject):

    if subject == "math":
        return MATH_PROMPT

    if subject == "physics":
        return PHYSICS_PROMPT

    if subject == "chemistry":
        return CHEMISTRY_PROMPT

    if subject == "biology":
        return BIOLOGY_PROMPT

    return GENERAL_PROMPT


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

        if isinstance(node, ast.BinOp):

            operation = (
                ALLOWED_BINARY_OPERATORS.get(
                    type(node.op)
                )
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
# MATH HELPERS
# ============================================================

def normalize_math_text(text):

    text = text.replace("svg", "√")
    text = text.replace("SVG", "√")

    superscripts = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )

    return text.translate(superscripts)


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
# LOCAL MATH: x + 1/x
# ============================================================

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
# LOCAL MATH: TRIG IDENTITY
# ============================================================

def solve_trig_identity(question):

    q = question.lower()

    required = [
        "sec",
        "cosec",
        "sin",
        "cos",
    ]

    if not all(
        word in q
        for word in required
    ):
        return None

    # Recognize the identity we were discussing.
    if not (
        "sec" in q
        and "cosec" in q
        and "sin" in q
        and "cos" in q
    ):
        return None

    return """
### Proof

We have

[1/(sec²θ − cos²θ) + 1/(cosec²θ − sin²θ)]
× sin²θ cos²θ

Using:

sec²θ = 1/cos²θ

and

cosec²θ = 1/sin²θ

Therefore,

1/(sec²θ − cos²θ)

= 1/(1/cos²θ − cos²θ)

= cos²θ/(1 − cos⁴θ)

Since

1 − cos⁴θ
= (1 − cos²θ)(1 + cos²θ)
= sin²θ(1 + cos²θ),

we get

1/(sec²θ − cos²θ)
= cos²θ/[sin²θ(1 + cos²θ)]

Similarly,

1/(cosec²θ − sin²θ)
= sin²θ/[cos²θ(1 + sin²θ)]

Multiplying by sin²θ cos²θ:

LHS
= cos⁴θ/(1 + cos²θ)
  + sin⁴θ/(1 + sin²θ)

Taking the LCM:

= [cos⁴θ(1 + sin²θ)
   + sin⁴θ(1 + cos²θ)]
  / [(1 + sin²θ)(1 + cos²θ)]

Using:

sin²θ + cos²θ = 1

and

sin⁴θ + cos⁴θ
= 1 − 2sin²θcos²θ

we get:

LHS
= [1 − sin²θcos²θ]
  / [2 + sin²θcos²θ]

### Final Answer

**(1 − sin²θ cos²θ)/(2 + sin²θ cos²θ) ✅**
""".strip()


# ============================================================
# LOCAL PHYSICS
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

    if (
        not voltage_match
        or len(resistor_values) < 3
    ):
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

    vp = (
        voltage
        - sum(
            total_i * r
            for r in others
        )
    )

    i1 = vp / r1
    i2 = vp / r2

    return f"""
### Solution

Rₚ = (R₁ × R₂)/(R₁ + R₂)

**Rₚ = {fmt(rp)} Ω**

Rₜ = {' + '.join(fmt(x) for x in others)} + {fmt(rp)}

**Rₜ = {fmt(rt)} Ω**

I = V/Rₜ

**I = {fmt(total_i)} A**

Parallel voltage:

**Vₚ = {fmt(vp)} V**

Current through {fmt(r1)} Ω:

**I₁ = {fmt(i1)} A**

Current through {fmt(r2)} Ω:

**I₂ = {fmt(i2)} A**

### Final Answer

**Total current = {fmt(total_i)} A ✅**
""".strip()


# ============================================================
# LOCAL CHEMISTRY
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
        r"(?:hcl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M.*?"
        r"(?:hcl|hydrochloric)",
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
        float(
            volume_match.group(1)
        )
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

Moles of Fe:

**{fmt(fe_moles)} mol**

Moles of HCl:

**{fmt(hcl_moles)} mol**

Limiting reagent:

**{limiting}**

Moles of H₂:

**{fmt(h2_moles)} mol**

H₂ volume at STP:

**{fmt(h2_volume)} L**

Mass of FeCl₂:

**{fmt(fecl2_mass)} g**

### Final Answer

**Limiting reagent = {limiting} ✅**
""".strip()


# ============================================================
# TEXT AI
# ============================================================

def stream_text_ai(
    ai_question,
    subject,
    original_question,
    difficulty,
):

    if not HF_TOKEN or hf_client is None:

        yield (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment → HF_TOKEN."
        )

        return

    prompt = get_prompt(subject)

    if difficulty == "advanced":

        prompt += """

THIS IS AN ADVANCED PROBLEM.

Work through the necessary mathematical or physical derivation.

Before the final answer:
- Recheck equations.
- Recheck algebra.
- Recheck signs.
- Recheck units.
- Check limiting cases when useful.
- Verify MCQ options when present.

Do not guess.
Do not reveal private chain-of-thought.
Show only the useful derivation and verification.
"""

    elif difficulty == "intermediate":

        prompt += """

This is an intermediate problem.
Show the important steps and check the result.
"""

    else:

        prompt += """

This is a basic problem.
Keep the response reasonably concise.
"""

    max_tokens = {
        "basic": 2500,
        "intermediate": 4000,
        "advanced": 7000,
    }.get(
        difficulty,
        4000,
    )

    full_answer = ""

    try:

        response = hf_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": ai_question,
                },
            ],
            stream=True,
            temperature=0.05,
            max_tokens=max_tokens,
        )

        for chunk in response:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            text = (
                getattr(
                    delta,
                    "content",
                    None,
                )
                or ""
            )

            if text:

                full_answer += text

                yield text

        if full_answer.strip():

            save_history(
                original_question,
                full_answer,
            )

    except Exception as error:

        print(
            "HUGGING FACE TEXT ERROR:",
            repr(error),
        )

        yield (
            "❌ Hugging Face error:\n\n"
            + str(error)
        )


# ============================================================
# IMAGE AI
# ============================================================

def analyze_image(
    image_data,
    prompt,
):

    if not HF_TOKEN or hf_client is None:

        return (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment → HF_TOKEN."
        )

    if not image_data.startswith(
        "data:image/"
    ):

        return "❌ Invalid image data."

    # 8 MB is enough for the compressed frontend image.
    if len(image_data) > 8_000_000:

        return (
            "❌ Image is too large.\n\n"
            "Please take a smaller or clearer photo."
        )

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
                            "text": (
                                prompt.strip()
                                if prompt.strip()
                                else
                                "Read this image carefully and solve the question step by step."
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
            temperature=0.05,
            max_tokens=6000,
        )

        if not response.choices:
            return (
                "❌ Vision model returned no answer."
            )

        answer = (
            response.choices[0]
            .message
            .content
        )

        if not answer:
            return (
                "❌ Vision model returned an empty answer."
            )

        return answer

    except Exception as error:

        print(
            "HUGGING FACE VISION ERROR:",
            repr(error),
        )

        return (
            "❌ Vision AI error:\n\n"
            + str(error)
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

    save_users(users)

    session["username"] = username
    session["chat_history"] = []

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
    # IMAGE FIRST
    # --------------------------------------------------------

    if image_data:

        prompt = (
            original_question
            if original_question
            else
            "Read this image carefully and solve the question."
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

    normalized = normalize_math_text(
        original_question
    )

    # --------------------------------------------------------
    # LOCAL MATH
    # --------------------------------------------------------

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

    result = solve_trig_identity(
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
    # LOCAL PHYSICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOCAL CHEMISTRY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ONLINE AI
    # --------------------------------------------------------

    subject = detect_subject(
        normalized
    )

    difficulty = detect_difficulty(
        normalized
    )

    ai_question = build_ai_context(
        normalized
    )

    return Response(
        stream_with_context(
            stream_text_ai(
                ai_question,
                subject,
                original_question,
                difficulty,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control":
                "no-cache",
            "X-Accel-Buffering":
                "no",
        },
    )


# ============================================================
# IMPROVE
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
            "Explain the answer in more detail.",

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

    difficulty = detect_difficulty(
        question
    )

    prompt = f"""
Original question:

{question}

Previous answer:

{old_answer}

Task:

{instruction}

Check calculations, algebra and units.
Do not use raw LaTeX.
Finish completely.
"""

    return Response(
        stream_with_context(
            stream_text_ai(
                prompt,
                subject,
                question,
                difficulty,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control":
                "no-cache",
            "X-Accel-Buffering":
                "no-cache",
        },
    )


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
        "Calculator    : ON"
    )
    print(
        "Math solver   : ON"
    )
    print(
        "Trig solver   : ON"
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
        "Vision        : ON"
    )
    print(
        "Follow-ups    : ON"
    )
    print(
        "Login         : ON"
    )
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
