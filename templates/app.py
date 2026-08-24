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

# Leave the provider policy automatic.
TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

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
        ) as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_users(users):
    with open(
        USERS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(users, f, indent=2)


# ============================================================
# PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a study assistant.

Subjects:
Mathematics, Physics, Chemistry, Biology and general science.

Answer from basic through difficult levels.

Rules:
- Read the entire question.
- Identify exactly what is being asked.
- Do not guess.
- Answer every requested part.
- Show important calculations.
- Verify important results.
- Understand short follow-up questions from recent conversation.
- For simple questions, stay concise.
- For difficult questions, provide a complete useful derivation.

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
- Geometry
- Coordinate geometry
- Polynomials
- Quadratics
- Surds
- Sequences and series
- Probability
- Permutations and combinations
- Binomial theorem
- Functions
- Logarithms
- Inequalities
- Vectors
- Matrices
- Determinants
- Complex numbers
- Number theory
- Limits
- Differentiation
- Integration
- Calculus
- JEE Main
- JEE Advanced
- Olympiad-style mathematics

For difficult problems:

1. Understand the exact target.
2. Identify useful information.
3. Choose an appropriate method.
4. Derive step by step.
5. Simplify carefully.
6. Check the result.
7. For MCQs, verify the selected option.

For identities, prove the identity rather than only stating it.

Never guess.

Do not output raw LaTeX.
"""


PHYSICS_PROMPT = """
You are an expert Physics solver for school, JEE Main and JEE Advanced.

Handle:
- Kinematics
- Newton's laws
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
- Optics
- Thermodynamics
- Modern physics

For difficult problems:

1. Understand the physical system.
2. Identify all important forces and constraints.
3. Choose coordinates.
4. Find equilibrium when needed.
5. Write governing equations.
6. Apply approximations only when justified.
7. Derive the requested quantity.
8. Check dimensions and signs.
9. Check limiting cases where useful.
10. Verify MCQ options.

For small oscillation problems:
find equilibrium first, then derive the restoring-force relation.

Never guess.

Do not output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are an expert Chemistry solver.

Handle school, JEE Main and JEE Advanced level questions.

Topics:
- Mole concept
- Stoichiometry
- Atomic structure
- Chemical bonding
- Thermodynamics
- Equilibrium
- Ionic equilibrium
- Electrochemistry
- Kinetics
- Redox
- Inorganic chemistry
- Organic chemistry
- Isomerism
- Reaction mechanisms

For numerical problems:
Given → formula/reaction → substitution → calculation → verification → final answer.

Do not guess.

Do not output raw LaTeX.
"""


BIOLOGY_PROMPT = """
You are an expert Biology solver.

Handle:
- Cell biology
- Genetics
- Molecular biology
- DNA/RNA
- Biotechnology
- Human physiology
- Plant physiology
- Reproduction
- Evolution
- Ecology

For difficult questions:
identify the concept, explain the process in order, compare alternatives
when necessary, and calculate genetics probabilities carefully.

Do not invent facts.
"""


VISION_PROMPT = """
You are My AI's academic image-question solver.

The user has uploaded a photograph or screenshot.

First inspect the entire image.

Determine whether it is:
Mathematics, Physics, Chemistry, Biology or another academic question.

Then actually solve the question.

MATHEMATICS:
- Read every visible number and symbol.
- Reconstruct the question.
- Solve step by step.
- Check algebra and arithmetic.

PHYSICS:
- Identify the physical system.
- Read all given values.
- Choose the relevant laws.
- Derive the result.
- Check units.

CHEMISTRY:
- Read equations and values.
- Balance reactions when needed.
- Calculate carefully.

BIOLOGY:
- Read labels and diagrams.
- Answer every requested part.

IMPORTANT:
- Do not invent text that is not visible.
- If something is blurry, explicitly say what is unclear.
- Do not merely describe the image.
- Solve the question.
- Do not output raw LaTeX.
"""


# ============================================================
# BASIC RESPONSES
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


def save_history(question, answer):
    history = session.get(
        "chat_history",
        [],
    )

    history.append({
        "question": question,
        "answer": answer[:10000],
    })

    session["chat_history"] = history[-10:]
    session.modified = True


def build_ai_context(question):
    history = session.get(
        "chat_history",
        [],
    )

    if not history or not is_followup(question):
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
        "Answer the new question using relevant previous context."
    )

    return "\n".join(parts)


# ============================================================
# SUBJECT
# ============================================================

def detect_subject(question):
    q = question.lower()

    math_words = [
        "equation", "algebra", "quadratic", "polynomial",
        "trigonometry", "trigonometric", "sin", "cos",
        "tan", "sec", "cosec", "geometry", "surds",
        "probability", "sequence", "series", "matrix",
        "determinant", "calculus", "integral", "derivative",
        "logarithm", "vector", "complex number",
        "binomial", "inequality", "number theory",
    ]

    physics_words = [
        "force", "velocity", "acceleration", "momentum",
        "newton", "work", "energy", "power", "friction",
        "gravitation", "projectile", "current", "voltage",
        "resistance", "resistor", "circuit", "electric",
        "magnetic", "lens", "mirror", "refraction",
        "heat", "temperature", "pressure", "density",
        "rotation", "torque", "oscillation", "shm",
        "wave", "capacitor", "induction",
    ]

    chemistry_words = [
        "mole", "molarity", "stoichiometry",
        "limiting reagent", "oxidation", "reduction",
        "acid", "base", "ph", "equilibrium",
        "enthalpy", "electrochemistry", "organic",
        "alkane", "alkene", "alkyne", "benzene",
        "alcohol", "aldehyde", "ketone", "reaction",
        "atom", "electron", "compound", "kinetics",
    ]

    biology_words = [
        "cell", "mitosis", "meiosis", "chromosome",
        "gene", "genetics", "allele", "dna", "rna",
        "protein", "enzyme", "photosynthesis",
        "respiration", "plant", "animal", "tissue",
        "organ", "ecosystem", "evolution", "hormone",
        "neuron", "reproduction", "heredity",
    ]

    scores = {
        "math": sum(x in q for x in math_words),
        "physics": sum(x in q for x in physics_words),
        "chemistry": sum(x in q for x in chemistry_words),
        "biology": sum(x in q for x in biology_words),
    }

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
        "multi-step",
        "multiple correct",
        "integer answer",
        "constraint",
        "optimization",
        "differential equation",
        "complex number",
        "number theory",
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


def get_prompt(subject):
    prompts = {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }

    return prompts.get(
        subject,
        GENERAL_PROMPT,
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

    def calc(node):
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
                calc(node.operand)
            )

        if isinstance(node, ast.BinOp):
            operation = ALLOWED_BINARY_OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError()

            left = calc(node.left)
            right = calc(node.right)

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
        return calc(tree.body)
    except (
        ArithmeticError,
        ValueError,
        OverflowError,
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
# LOCAL x + 1/x SOLVER
# ============================================================

def solve_power_recurrence(question):
    q = question.lower()

    match = re.search(
        r"x\s*\+\s*1\s*/\s*x\s*=\s*"
        r"(-?\d+(?:\.\d+)?)",
        q,
    )

    if not match:
        return None

    a = float(match.group(1))

    target = re.search(
        r"x\s*(?:\^|\*\*)?(\d+)"
        r"\s*\+\s*1\s*/\s*x\s*(?:\^|\*\*)?\1",
        q,
    )

    if not target:
        return None

    n = int(target.group(1))

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
            f"S{k} = {a:g}({values[k-1]:g}) − {values[k-2]:g}",
            f"**S{k} = {values[k]:g}**",
            "",
        ])

    lines.extend([
        "### Final Answer",
        "",
        f"**x^{n} + 1/x^{n} = {values[n]:g} ✅**",
    ])

    return "\n".join(lines)


# ============================================================
# LOCAL TRIG IDENTITY
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

    # Recognize natural language like:
    # 1 upon sec square theta...
    trig_markers = [
        "upon",
        "sec square",
        "cosec square",
        "sin square",
        "cos square",
        "bracket",
    ]

    if not any(
        marker in q
        for marker in trig_markers
    ):
        return None

    return """
### Proof

We need to simplify:

[1/(sec²θ − cos²θ) + 1/(cosec²θ − sin²θ)]
× sin²θ × cos²θ

Using:

sec²θ = 1/cos²θ

and

cosec²θ = 1/sin²θ

Therefore:

1/(sec²θ − cos²θ)
= 1/(1/cos²θ − cos²θ)

= cos²θ/(1 − cos⁴θ)

Now:

1 − cos⁴θ
= (1 − cos²θ)(1 + cos²θ)

= sin²θ(1 + cos²θ)

Therefore:

1/(sec²θ − cos²θ)
= cos²θ/[sin²θ(1 + cos²θ)]

Similarly:

1/(cosec²θ − sin²θ)
= sin²θ/[cos²θ(1 + sin²θ)]

Multiply by sin²θ cos²θ:

LHS
= cos⁴θ/(1 + cos²θ)
+ sin⁴θ/(1 + sin²θ)

Taking the LCM:

LHS
= [cos⁴θ(1 + sin²θ)
+ sin⁴θ(1 + cos²θ)]
/[(1 + sin²θ)(1 + cos²θ)]

Using:

sin²θ + cos²θ = 1

and

sin⁴θ + cos⁴θ
= 1 − 2sin²θcos²θ

we obtain:

LHS
= [1 − sin²θcos²θ]
/[2 + sin²θcos²θ]

### Final Answer

**(1 − sin²θ cos²θ)/(2 + sin²θ cos²θ) ✅**
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
This is an ADVANCED question.

Use a careful derivation.
Check algebra, arithmetic, units and signs.
For MCQs, verify the option.
Do not guess.
"""

    elif difficulty == "intermediate":
        prompt += """
This is an intermediate question.
Show the important steps and verify the result.
"""

    else:
        prompt += """
This is a basic question.
Keep the answer clear and reasonably concise.
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
        stream = hf_client.chat.completions.create(
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

        for chunk in stream:
            if not chunk.choices:
                continue

            text = getattr(
                chunk.choices[0].delta,
                "content",
                None,
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
            "TEXT AI ERROR:",
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
    question,
):
    if not HF_TOKEN or hf_client is None:
        return (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment → HF_TOKEN."
        )

    if not image_data.startswith("data:image/"):
        return "❌ Invalid image data."

    if len(image_data) > 10_000_000:
        return (
            "❌ Image is too large.\n\n"
            "Please capture a smaller or clearer image."
        )

    user_text = (
        question.strip()
        if question.strip()
        else
        "Read this image carefully and solve the academic question step by step."
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
                            "text": user_text,
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
            "VISION AI ERROR:",
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

    question = data.get(
        "message",
        "",
    ).strip()

    image_data = data.get(
        "image",
        "",
    )

    # Image has priority.
    if image_data:

        answer = analyze_image(
            image_data,
            question,
        )

        save_history(
            question or "Image question",
            answer,
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

    # Basic answers
    simple = basic_response(question)

    if simple:
        save_history(
            question,
            simple,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # Local calculator
    result = solve_calculation(question)

    if result:
        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # Local x + 1/x
    result = solve_power_recurrence(question)

    if result:
        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # Local trig identity
    result = solve_trig_identity(question)

    if result:
        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # Online AI
    subject = detect_subject(question)
    difficulty = detect_difficulty(question)
    context_question = build_ai_context(
        question
    )

    return Response(
        stream_with_context(
            stream_text_ai(
                context_question,
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
            "Check the answer for mistakes and correct them.",

        "explain":
            "Explain the answer in more detail.",

        "short":
            "Make the answer shorter while preserving the important steps.",
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

Recalculate important values.
Check algebra.
Check units.
Do not output raw LaTeX.
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
        bool(HF_TOKEN),
    )
    print("Text AI      : ON")
    print("Vision AI    : ON")
    print("Calculator   : ON")
    print("Math solver  : ON")
    print("Trig solver  : ON")
    print("Memory       : ON")
    print("Camera       : ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
