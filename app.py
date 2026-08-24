from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context,
    session,
    redirect,
    url_for
)

import os
import re
import math
import ast
import operator
import json

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from openai import OpenAI


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "my-ai-development-secret"
)

USERS_FILE = "users.json"


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "openai/gpt-oss-120b:cheapest"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN
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
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users):
    with open(
        USERS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            users,
            file,
            indent=2
        )


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are My AI, a helpful study assistant.

You help with:
Mathematics, Physics, Chemistry, Biology, Science and general questions.

Rules:
- Read the entire question.
- Answer every requested part.
- Never stop halfway through a sentence.
- Never leave an equation unfinished.
- Show important calculations.
- Use simple student-friendly language.
- Do not invent missing information.
- For follow-up questions, use the previous context.
- Always finish with a clear Final Answer.
"""

MATH_SYSTEM_PROMPT = """
You are My AI's Mathematics specialist.

Formatting rules:
- Do not use raw LaTeX commands such as \\frac, \\boxed, \\begin or \\end.
- Do not put equations inside [ ... ].
- Use readable forms such as x², x³, x⁵ and √.
- Never replace √ with x.
- Show important algebraic steps.
- Check calculations.
- Answer every requested part.
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

Use correct units and complete every requested part.
"""

CHEMISTRY_SYSTEM_PROMPT = """
You are My AI's Chemistry specialist.

For numerical questions:
- Balance reactions when necessary.
- Calculate molar mass.
- Convert units correctly.
- Calculate moles and molarity.
- Use stoichiometric ratios.
- Identify limiting reagent when required.
- Include units.
- Complete every requested part.
"""

BIOLOGY_SYSTEM_PROMPT = """
You are My AI's Biology specialist.

Explain biology clearly and accurately.

For comparisons:
compare each requested point.

For processes:
explain them in correct order.

For genetics:
show genotype, phenotype and probability clearly.

Do not invent facts.
"""


# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

    if q in {
        "hi", "hello", "hey",
        "hii", "hiii", "helo"
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

    if q in {
        "how are you",
        "how are you?"
    }:
        return "I'm doing great! 🤖 Ask me a difficult question."

    if q in {
        "who are you",
        "who are you?"
    }:
        return (
            "I'm My AI 🤖, a study assistant for "
            "Mathematics, Physics, Chemistry and Biology."
        )

    if q in {
        "what can you do",
        "what can you do?"
    }:
        return (
            "I can help with:\n\n"
            "🧮 Mathematics\n"
            "⚡ Physics\n"
            "🧪 Chemistry\n"
            "🧬 Biology\n"
            "🧮 Large calculations\n"
            "📚 General questions"
        )

    if q in {
        "thanks",
        "thank you",
        "thankyou"
    }:
        return "You're welcome! 😊"

    if q in {
        "ok",
        "okay",
        "cool",
        "nice",
        "great"
    }:
        return "👍 Great! Ask your next question."

    return None


# ============================================================
# SUBJECT DETECTION
# ============================================================

def detect_subject(question):
    q = question.lower()

    math_words = [
        "equation", "quadratic", "algebra", "surds",
        "sqrt", "√", "svg", "trigonometry",
        "sin", "cos", "tan", "geometry",
        "probability", "sequence", "series",
        "polynomial", "coordinate", "logarithm",
        "matrix", "calculus", "integral"
    ]

    physics_words = [
        "force", "velocity", "acceleration",
        "momentum", "newton", "work", "energy",
        "power", "friction", "gravitation",
        "projectile", "current", "voltage",
        "resistance", "resistor", "circuit",
        "magnetic", "electric field", "lens",
        "mirror", "refraction", "heat",
        "temperature", "pressure", "density"
    ]

    chemistry_words = [
        "mole", "moles", "molar", "molarity",
        "molality", "stoichiometry",
        "limiting reagent", "oxidation",
        "reduction", "redox", "acid", "base",
        "ph", "salt", "equilibrium",
        "enthalpy", "electrochemistry",
        "organic", "alkane", "alkene",
        "alkyne", "benzene", "alcohol",
        "aldehyde", "ketone", "ion",
        "atom", "electron", "compound",
        "reaction"
    ]

    biology_words = [
        "cell", "mitosis", "meiosis",
        "chromosome", "gene", "genetics",
        "allele", "dna", "rna", "protein",
        "enzyme", "photosynthesis",
        "respiration", "plant", "animal",
        "tissue", "organ", "ecosystem",
        "ecology", "evolution", "hormone",
        "neuron", "digestion", "reproduction",
        "heredity", "blood", "heart",
        "kidney", "lung", "brain"
    ]

    scores = {
        "math": sum(x in q for x in math_words),
        "physics": sum(x in q for x in physics_words),
        "chemistry": sum(x in q for x in chemistry_words),
        "biology": sum(x in q for x in biology_words)
    }

    subject = max(
        scores,
        key=scores.get
    )

    return (
        subject
        if scores[subject] > 0
        else "general"
    )


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
    ast.Pow: operator.pow
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}


def safe_calculate(expression):
    expression = expression.strip()

    if not expression or len(expression) > 1000:
        return None

    try:
        tree = ast.parse(
            expression,
            mode="eval"
        )
    except (SyntaxError, ValueError):
        return None

    def calculate(node):

        if isinstance(node, ast.Constant):
            if isinstance(
                node.value,
                (int, float)
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

            # Avoid absurdly large powers.
            if (
                isinstance(node.op, ast.Pow)
                and abs(right) > 10000
            ):
                raise ValueError()

            return operation(
                left,
                right
            )

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
        ZeroDivisionError
    ):
        return None


def clean_calculator_expression(text):
    expression = text.strip()

    expression = expression.replace(
        "×", "*"
    )

    expression = expression.replace(
        "÷", "/"
    )

    expression = expression.replace(
        "−", "-"
    )

    expression = expression.replace(
        "–", "-"
    )

    expression = expression.replace(
        "^", "**"
    )

    # 999,999,999 -> 999999999
    expression = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        expression
    )

    expression = re.sub(
        r"^(what is|calculate|find|solve|evaluate)\s+",
        "",
        expression,
        flags=re.IGNORECASE
    )

    return expression.strip()


def looks_like_calculation(text):
    expression = clean_calculator_expression(text)

    if not re.fullmatch(
        r"[0-9\s\.\+\-\*\/%\(\)]+",
        expression
    ):
        return False

    return bool(
        re.search(
            r"[\+\-\*\/%]",
            expression
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
# MATH SOLVER
# ============================================================

def normalize_math_text(text):
    text = text.replace(
        "svg",
        "√"
    )

    superscript_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789"
    )

    text = text.translate(
        superscript_map
    )

    return text


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
        q
    )

    if not given:
        return None

    a = float(
        given.group(1)
    )

    target = re.search(
        r"x\s*(?:\^|\*\*)\s*(\d+)"
        r"\s*\+\s*1\s*/\s*x"
        r"\s*(?:\^|\*\*)\s*\1",
        q
    )

    if not target:
        return None

    n = int(
        target.group(1)
    )

    if n < 1 or n > 50:
        return None

    s = {
        0: 2,
        1: a
    }

    for k in range(1, n):
        s[k + 1] = (
            a * s[k]
            - s[k - 1]
        )

    lines = [
        "### Solution",
        "",
        f"Given: x + 1/x = {fmt(a)}",
        "",
        "Let:",
        "",
        "Sₙ = xⁿ + 1/xⁿ",
        "",
        "S₀ = 2",
        "",
        f"S₁ = {fmt(a)}",
        ""
    ]

    for k in range(2, n + 1):
        lines.extend([
            f"**Step {k - 1}:**",
            "",
            f"S{k} = {fmt(a)}S{k - 1} − S{k - 2}",
            "",
            (
                f"S{k} = {fmt(a)}({fmt(s[k - 1])}) "
                f"− {fmt(s[k - 2])}"
            ),
            "",
            f"**S{k} = {fmt(s[k])}**",
            ""
        ])

    lines.extend([
        "### Final Answer",
        "",
        f"**x{n} + 1/x{n} = {fmt(s[n])} ✅**"
    ])

    return "\n".join(lines)


# ============================================================
# PHYSICS CIRCUIT SOLVER
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
        q
    )

    resistor_values = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*"
            r"(?:Ω|ohm|ohms)",
            question,
            re.IGNORECASE
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
        re.IGNORECASE
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
        voltage / rt
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

    total_power = (
        voltage * total_i
    )

    power_parts = [
        total_i ** 2 * r
        for r in others
    ]

    power_parts.append(
        vp ** 2 / r1
    )

    power_parts.append(
        vp ** 2 / r2
    )

    resistor_power = sum(
        power_parts
    )

    return f"""
### Solution

**1. Parallel resistance**

Rₚ = (R₁ × R₂)/(R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)})/
({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(rp)} Ω**

**2. Total resistance**

Rₜ = {' + '.join(fmt(r) for r in others)} + {fmt(rp)}

**Rₜ = {fmt(rt)} Ω**

**3. Total current**

I = V/Rₜ

I = {fmt(voltage)}/{fmt(rt)}

**I = {fmt(total_i)} A**

**4. Parallel voltage**

**Vₚ = {fmt(vp)} V**

**5. Current through {fmt(r1)} Ω**

**I₁ = {fmt(i1)} A**

**6. Current through {fmt(r2)} Ω**

**I₂ = {fmt(i2)} A**

**7. Total power**

P = VI

**P = {fmt(total_power)} W**

### Verification

Sum of resistor powers =
**{fmt(resistor_power)} W**

Battery power =
**{fmt(total_power)} W**

**Power check ✅**

### Final Answer

- Parallel resistance = **{fmt(rp)} Ω**
- Total resistance = **{fmt(rt)} Ω**
- Total current = **{fmt(total_i)} A**
- Parallel voltage = **{fmt(vp)} V**
- Total power = **{fmt(total_power)} W ✅**
""".strip()


# ============================================================
# CHEMISTRY FE + HCL
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
        re.IGNORECASE
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mL.*?HCl",
        question,
        re.IGNORECASE
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M.*?HCl",
        question,
        re.IGNORECASE
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

    fe_moles = mass_fe / 56
    hcl_moles = (
        molarity_hcl
        * volume_hcl
    )

    hcl_required = 2 * fe_moles

    if hcl_moles < hcl_required:
        limiting = "HCl"
        h2_moles = hcl_moles / 2
        fe_used = h2_moles

    else:
        limiting = "Fe"
        h2_moles = fe_moles
        fe_used = fe_moles

    h2_volume = h2_moles * 22.4
    fecl2_mass = fe_used * 127

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

**{fmt(h2_moles)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M = 56 + 2(35.5)

M = 127 g mol⁻¹

**Mass = {fmt(fecl2_mass)} g**

### Final Answer

- Fe = **{fmt(fe_moles)} mol**
- HCl = **{fmt(hcl_moles)} mol**
- Limiting reagent = **{limiting}**
- H₂ = **{fmt(h2_moles)} mol**
- H₂ volume = **{fmt(h2_volume)} L**
- FeCl₂ = **{fmt(fecl2_mass)} g ✅**
""".strip()


# ============================================================
# HUGGING FACE
# ============================================================

def stream_ai(question, subject):
    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ Online AI is unavailable because HF_TOKEN is missing.\n\n"
            "Local calculator and local science solvers can still work."
        )
        return

    try:
        stream = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": get_prompt(subject)
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True,
            temperature=0.15,
            max_tokens=2000
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
            "Hugging Face error:",
            repr(error)
        )

        yield (
            "❌ Online AI could not answer right now."
        )


# ============================================================
# LOGIN PAGE
# ============================================================

@app.route("/")
def home():
    if "username" not in session:
        return render_template(
            "index.html",
            logged_in=False
        )

    return render_template(
        "index.html",
        logged_in=True,
        username=session["username"]
    )


# ============================================================
# CREATE ACCOUNT
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

    if not username or not email or not password:
        return {
            "success": False,
            "message": "Username, email and password are required."
        }, 400

    if len(password) < 6:
        return {
            "success": False,
            "message": "Password must be at least 6 characters."
        }, 400

    users = load_users()

    # username exists
    if username.lower() in {
        u.lower()
        for u in users.keys()
    }:
        return {
            "success": False,
            "message": "Username already exists."
        }, 400

    # email exists
    for user in users.values():
        if user.get("email", "").lower() == email:
            return {
                "success": False,
                "message": "Email already registered."
            }, 400

    users[username] = {
        "email": email,
        "phone": phone,
        "password": generate_password_hash(
            password
        )
    }

    save_users(users)

    session["username"] = username

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

    users = load_users()

    found_username = None
    found_user = None

    for username, user in users.items():

        if (
            username.lower()
            == login_value.lower()
            or
            user.get("email", "").lower()
            == login_value.lower()
            or
            user.get("phone", "")
            == login_value
        ):
            found_username = username
            found_user = user
            break

    if not found_user:
        return {
            "success": False,
            "message": "Account not found."
        }, 401

    if not check_password_hash(
        found_user.get("password", ""),
        password
    ):
        return {
            "success": False,
            "message": "Incorrect password."
        }, 401

    session["username"] = found_username

    return {
        "success": True,
        "message": "Login successful."
    }


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
    methods=["POST"]
)
def chat():

    if "username" not in session:
        return Response(
            "Please login first.",
            mimetype="text/plain",
            status=401
        )

    data = request.get_json(
        silent=True
    ) or {}

    original_question = data.get(
        "message",
        ""
    ).strip()

    if not original_question:
        return Response(
            "Please type a message.",
            mimetype="text/plain"
        )

    # Basic response
    simple = basic_response(
        original_question
    )

    if simple:
        return Response(
            simple,
            mimetype="text/plain"
        )

    # Local calculator FIRST
    result = solve_calculation(
        original_question
    )

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # Math
    normalized = normalize_math_text(
        original_question
    )

    result = solve_power_recurrence(
        normalized
    )

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # Physics
    result = solve_circuit(
        normalized
    )

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # Chemistry
    result = solve_iron_hcl(
        normalized
    )

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # Online AI fallback
    subject = detect_subject(
        normalized
    )

    return Response(
        stream_with_context(
            stream_ai(
                normalized,
                subject
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MY AI")
    print("=" * 60)
    print(
        "HF configured:",
        bool(HF_TOKEN)
    )
    print("Login system       : ON")
    print("Create account     : ON")
    print("Password hashing   : ON")
    print("Calculator         : ON")
    print("Math solver        : ON")
    print("Physics solver     : ON")
    print("Chemistry solver   : ON")
    print("Biology AI         : ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
