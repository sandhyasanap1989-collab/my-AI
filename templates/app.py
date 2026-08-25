from flask import Flask, render_template, request, Response, session, redirect, url_for
import os, re, json, ast, math, operator, threading
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

USERS_FILE = "users.json"
QUESTION_SETS_FILE = "question_sets.json"
LOCK = threading.Lock()

# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

hf_client = (
    OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    if HF_TOKEN
    else None
)

# ============================================================
# CREATOR
# ============================================================

CREATOR_RESPONSE = """My creator is Soham Chandrahas Sanap.

He is 15 years old and is studying in Class 10 in 2026
at Nimbark English School in Beed district, Maharashtra, India.

His main interests are Mathematics and web development.

He built My AI as an AI study assistant to help students
with Mathematics, Physics, Chemistry and Biology."""

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
    "who is your father",
    "who is your maker?",
    "tell me about your creator",
    "tell me about your creator?",
    "who is soham chandrahas sanap",
    "who is soham chandrahas sanap?",
}

# ============================================================
# MATH FORMATTING
# ============================================================

PLAIN_MATH = r"""
Never use LaTeX.

Use plain readable mathematics such as:
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
x²
x³
10⁻³

Never output:
\frac
\sqrt
\sin
\cos
\tan
\left
\right
\[
\]
\(
\)
$$
"""

# ============================================================
# PROMPTS
# ============================================================

GENERAL = """
You are My AI, a careful student study assistant.

Answer the user's actual question, not unrelated profile information.

Rules:
1. Read the complete question.
2. Identify exactly what is being asked.
3. Never guess.
4. Answer every requested part.
5. Check important arithmetic.
6. Check units when relevant.
7. Verify important results.
8. Use relevant context only.
9. Never invent user information.
10. Finish the answer completely.
"""

MATH = GENERAL + """
You are also an expert Mathematics solver for school,
JEE Main, JEE Advanced and olympiad-style problems.

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
- Probability
- Permutations and combinations
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
- Calculus

For difficult problems:
1. Identify the target.
2. Extract given information.
3. Choose the correct method.
4. Derive step by step.
5. Check algebra.
6. Check arithmetic.
7. Check domains and conditions.
8. Verify the final result.
9. Verify MCQ options.
10. For proofs, actually prove the statement.
"""

PHYSICS = GENERAL + """
You are also an expert Physics solver for school,
JEE Main and JEE Advanced.

Handle:
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
2. Identify bodies, forces and constraints.
3. Choose coordinates.
4. Find equilibrium when required.
5. Write governing equations.
6. Derive carefully.
7. Check signs.
8. Check units.
9. Check dimensions.
10. Check limiting cases when useful.
11. Verify the final result.

For SHM/small oscillations:
- Find equilibrium first.
- Define displacement.
- Find restoring force.
- Apply the small-displacement approximation.
- Obtain the SHM equation.
"""

CHEM = GENERAL + """
You are also an expert Chemistry solver.

Check:
- chemical equations
- stoichiometry
- units
- limiting reagent
- reaction logic
- numerical calculations
"""

BIO = GENERAL + """
You are also an expert Biology solver.

Use established concepts.
Explain processes in the correct order.
Do not invent facts.
"""

VISION = GENERAL + """
Read the complete image before answering.

Identify whether the image contains:
- Mathematics
- Physics
- Chemistry
- Biology

Do not invent unreadable text.

Actually solve the question shown in the image.

For Mathematics:
Check algebra and arithmetic.

For Physics:
Check equations, signs, units and dimensions.

For Chemistry:
Check reactions and numerical values.

For Biology:
Read diagrams and labels carefully.
"""

def prompt_for(s):
    return {
        "math": MATH,
        "physics": PHYSICS,
        "chemistry": CHEM,
        "biology": BIO,
        "general": GENERAL,
    }.get(s, GENERAL) + "\n\n" + PLAIN_MATH

# ============================================================
# CLEAN OUTPUT
# ============================================================

def clean(text):
    if not text:
        return ""

    t = str(text)

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

    for a, b in replacements.items():
        t = t.replace(a, b)

    t = re.sub(
        r"\\(?:d)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"(\1/\2)",
        t,
    )

    t = re.sub(
        r"\\sqrt\s*\{([^{}]*)\}",
        r"√(\1)",
        t,
    )

    t = re.sub(
        r"\\text\s*\{([^{}]*)\}",
        r"\1",
        t,
    )

    t = re.sub(
        r"\\(?:mathrm|mathbf|mathit|boxed)\s*\{([^{}]*)\}",
        r"\1",
        t,
    )

    t = re.sub(
        r"\\[A-Za-z]+",
        lambda m: m.group(0)[1:],
        t,
    )

    for a, b in {
        "^2": "²",
        "^3": "³",
        "^4": "⁴",
        "^5": "⁵",
        "_1": "₁",
        "_2": "₂",
        "_3": "₃",
        "_4": "₄",
        "_5": "₅",
    }.items():
        t = t.replace(a, b)

    return re.sub(
        r"\n{3,}",
        "\n\n",
        re.sub(
            r"[ \t]{2,}",
            " ",
            t,
        ),
    ).strip()

# ============================================================
# FILE STORAGE
# ============================================================

def load_json(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

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
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    os.replace(
        temp,
        path,
    )


def users():
    return load_json(
        USERS_FILE,
        {},
    )


def save_users(data):
    with LOCK:
        save_json(
            USERS_FILE,
            data,
        )

# ============================================================
# QUESTION SET STORAGE
# ============================================================

def save_qset(
    request_text,
    answer,
    subject,
    difficulty,
):
    username = session.get(
        "username"
    )

    if not username:
        return

    data = load_json(
        QUESTION_SETS_FILE,
        {},
    )

    data[username] = {
        "request": request_text,
        "answer": answer,
        "subject": subject,
        "difficulty": difficulty,
    }

    with LOCK:
        save_json(
            QUESTION_SETS_FILE,
            data,
        )


def qset():
    username = session.get(
        "username"
    )

    if not username:
        return None

    return load_json(
        QUESTION_SETS_FILE,
        {},
    ).get(username)


def clear_qset():
    username = session.get(
        "username"
    )

    if not username:
        return

    data = load_json(
        QUESTION_SETS_FILE,
        {},
    )

    if username in data:
        del data[username]

        with LOCK:
            save_json(
                QUESTION_SETS_FILE,
                data,
            )

# ============================================================
# NORMAL CHAT HISTORY
# ============================================================

def history():
    return session.get(
        "chat_history",
        [],
    )


def save_history(
    q,
    a,
    category="normal",
):
    h = history()

    h.append({
        "question": q,
        "answer": a[:10000],
        "category": category,
    })

    session["chat_history"] = h[-10:]
    session.modified = True

# ============================================================
# CREATOR
# ============================================================

def creator_question(q):
    return q.strip().lower() in CREATOR_QUESTIONS


def basic(q):
    x = q.strip().lower()

    if creator_question(q):
        return CREATOR_RESPONSE

    if x in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
    }:
        return (
            "Hello! 👋 I'm My AI.\n\n"
            "Ask Mathematics, Physics, Chemistry, "
            "Biology or general questions."
        )

    if x in {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty",
    }:
        return "You're welcome! 😊"

    if x in {
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
# SUBJECT DETECTION
# ============================================================

def subject(q):
    x = q.lower()

    scores = {
        "math": 0,
        "physics": 0,
        "chemistry": 0,
        "biology": 0,
    }

    words = {
        "math": [
            "trigonometry",
            "trigonometric",
            "algebra",
            "quadratic",
            "equation",
            "identity",
            "integral",
            "derivative",
            "calculus",
            "geometry",
            "probability",
            "sequence",
            "series",
            "matrix",
            "determinant",
            "vector",
            "complex",
            "logarithm",
            "inequality",
            "polynomial",
            "surds",
            "number theory",
            "prove",
            "proof",
            "sin",
            "cos",
            "tan",
            "sec",
            "cosec",
        ],

        "physics": [
            "physics",
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
            "atom",
            "electron",
            "compound",
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
            "evolution",
            "ecology",
            "hormone",
            "neuron",
            "reproduction",
        ],
    }

    for s, ws in words.items():
        scores[s] = sum(
            w in x
            for w in ws
        )

    detected = max(
        scores,
        key=scores.get,
    )

    return (
        detected
        if scores[detected] > 0
        else "general"
    )

# ============================================================
# DIFFICULTY
# ============================================================

def difficulty(q):
    x = q.lower()

    score = sum(
        x.count(w) * 2
        for w in [
            "jee advanced",
            "hard",
            "hardest",
            "difficult",
            "challenging",
            "derive",
            "prove",
            "show that",
            "olympiad",
            "multi-step",
            "optimization",
            "constraint",
            "small oscillation",
            "multiple correct",
            "integer answer",
        ]
    )

    if len(x) > 350:
        score += 2

    if score >= 6:
        return "advanced"

    if score >= 2:
        return "intermediate"

    return "basic"

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
        + PLAIN_MATH
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

        return clean(
            response.output_text or ""
        )

    except Exception as error:

        print(
            "RESPONSES ERROR:",
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

        return clean(
            response.choices[0]
            .message
            .content
            or ""
        )

# ============================================================
# CAMERA / IMAGE ANALYSIS
# ============================================================

def vision_ai(
    image_data,
    question,
):
    """
    IMPORTANT:
    Uses Chat Completions for vision input.
    The camera can send a base64 data:image/... URL.
    """

    if not hf_client:
        return (
            "❌ HF_TOKEN is missing."
        )

    if not image_data.startswith(
        "data:image/"
    ):
        return (
            "❌ Invalid image data."
        )

    if len(image_data) > 12_000_000:
        return (
            "❌ Image is too large.\n\n"
            "Please use a smaller or clearer image."
        )

    user_question = (
        question.strip()
        if question.strip()
        else
        "Read the complete image and solve the academic "
        "question step by step."
    )

    try:

        response = hf_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        VISION
                        + "\n\n"
                        + PLAIN_MATH
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_question,
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

        return clean(answer)

    except Exception as error:

        print(
            "VISION ERROR:",
            repr(error),
        )

        return (
            "❌ I could not analyze that image right now.\n\n"
            "Vision error: "
            + str(error)
        )

# ============================================================
# QUESTION GENERATION
# ============================================================

def extract_nums(text):
    return sorted({
        int(n)
        for n in re.findall(
            r"(?:^|\n)\s*(\d+)\.\s",
            text,
        )
    })


def last_num(text):
    nums = extract_nums(text)

    return (
        max(nums)
        if nums
        else 0
    )


def question_count(q):
    match = re.search(
        r"\b(\d+)\s+(?:questions?|mcqs?)\b",
        q.lower(),
    )

    if not match:
        return None

    return min(
        max(
            int(match.group(1)),
            1,
        ),
        50,
    )


def continuation_count(q):
    x = q.strip().lower()

    patterns = [
        r"^(?:other|another|next|more|remaining)\s+(\d+)\s+questions?$",
        r"^give (?:me )?(?:the )?other\s+(\d+)\s+questions?$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            x,
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


def qset_command(q):
    x = q.strip().lower()

    if x in {
        "ans",
        "answer",
        "answers",
        "ans of all",
        "answers of all",
        "answer key",
        "show answers",
        "give answers",
        "give ans",
        "give ans of all",
    }:
        return "answers"

    if x in {
        "with options",
        "add options",
        "give options",
    }:
        return "options"

    if x in {
        "explain all",
        "explain answers",
        "solve all",
        "solutions",
        "explain all questions",
    }:
        return "explain"

    if re.match(
        r"^(?:make|set|change|increase|decrease) "
        r"(?:them|the questions) "
        r"(?:harder|easier)$",
        x,
    ):
        return "difficulty"

    if re.match(
        r"^(?:explain|solve|answer) "
        r"(?:q|question)\s*\d+",
        x,
    ):
        return "single"

    return None


def is_generation(q):
    x = q.lower()

    if question_count(q) is not None:
        return True

    return any(
        phrase in x
        for phrase in [
            "give me questions",
            "generate questions",
            "make questions",
            "create questions",
            "mcq",
            "mcqs",
            "multiple choice",
            "question paper",
        ]
    )


def batch_prompt(
    subject_name,
    diff,
    start,
    end,
    context="",
):
    count = (
        end
        - start
        + 1
    )

    return prompt_for(
        subject_name
    ) + f"""

QUESTION GENERATION MODE

Generate exactly {count} multiple-choice questions,
numbered {start} through {end}.

STRICT FORMAT:

{start}. Question text
A. Option text
B. Option text
C. Option text
D. Option text

Each next question must follow exactly the same format.

Rules:
- NEVER use a Markdown table.
- NEVER use the | character for formatting.
- NEVER use LaTeX.
- Every question must be complete.
- Every question must have A, B, C and D.
- Keep numbering continuous.
- Do not restart numbering.
- Do not give the answer key.
- Do not add an introduction.
- Do not add a conclusion.
- Keep the same topic.
- Difficulty: {diff}
- Check the question numbers before finishing.

Previous context:
{context}
"""


def validate_batch(
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
        extract_nums(text)
    )

    missing = expected - actual

    if missing:
        return (
            False,
            "Missing numbers: "
            + str(
                sorted(missing)
            ),
        )

    blocks = re.split(
        r"(?:^|\n)\s*\d+\.\s+",
        text,
    )[1:]

    if len(blocks) < (
        end - start + 1
    ):
        return (
            False,
            "Missing question block",
        )

    for block in blocks[
        : end - start + 1
    ]:

        options = set(
            re.findall(
                r"(?:^|\n)\s*([ABCD])\.\s+",
                block,
                re.MULTILINE,
            )
        )

        if options != {
            "A",
            "B",
            "C",
            "D",
        }:
            return (
                False,
                "Missing A/B/C/D",
            )

    return True, ""


def generate_set(
    subject_name,
    diff,
    total,
    start=1,
    context="",
):
    parts = []

    current = start

    final_number = (
        start
        + total
        - 1
    )

    while current <= final_number:

        batch_end = min(
            current + 4,
            final_number,
        )

        prompt = batch_prompt(
            subject_name,
            diff,
            current,
            batch_end,
            context,
        )

        batch = text_ai(
            prompt,
            prompt,
            reasoning="medium",
            max_tokens=7500,
        )

        valid, reason = validate_batch(
            batch,
            current,
            batch_end,
        )

        attempts = 0

        while (
            not valid
            and attempts < 2
        ):

            attempts += 1

            repair = f"""
Regenerate ONLY questions
{current} through {batch_end}.

Problem detected:
{reason}

Use ONLY this format:

N. Question
A. Option
B. Option
C. Option
D. Option

No Markdown table.
No | characters.
No LaTeX.
No answer key.
"""

            batch = text_ai(
                prompt_for(
                    subject_name
                ) + "\n" + repair,
                repair,
                reasoning="medium",
                max_tokens=6500,
            )

            valid, reason = validate_batch(
                batch,
                current,
                batch_end,
            )

        if not valid:
            raise RuntimeError(
                f"Could not validate questions "
                f"{current}-{batch_end}: "
                f"{reason}"
            )

        parts.append(batch)

        current = (
            batch_end
            + 1
        )

    return clean(
        "\n\n".join(parts)
    )

# ============================================================
# VERIFY HARD MATH / PHYSICS
# ============================================================

def verify_solution(
    question,
    answer,
    subject_name,
):
    if subject_name not in {
        "math",
        "physics",
    }:
        return answer

    verifier = prompt_for(
        subject_name
    ) + """

VERIFICATION STAGE

Check the proposed answer carefully.

Check:
- equations
- algebra
- arithmetic
- signs
- dimensions
- units
- conditions
- final answer
- MCQ option if present

If there is an error, correct it.

Return only the corrected educational solution.
Do not reveal private chain-of-thought.
"""

    try:

        return text_ai(
            verifier,
            "QUESTION:\n"
            + question
            + "\n\nPROPOSED ANSWER:\n"
            + answer,
            reasoning="high",
            max_tokens=10000,
        )

    except Exception:
        return answer

# ============================================================
# NORMAL FOLLOW-UP CONTEXT
# ============================================================

def normal_context(q):
    # Long questions normally stand on their own.
    if len(q.split()) > 6:
        return q

    usable = [
        item
        for item in history()
        if item.get("category") == "normal"
    ]

    if not usable:
        return q

    recent = usable[-4:]

    parts = [
        "RELEVANT RECENT CONVERSATION:",
    ]

    for item in recent:
        parts.extend([
            "USER:",
            item["question"],
            "MY AI:",
            item["answer"],
        ])

    parts.extend([
        "NEW USER QUESTION:",
        q,
        "Use previous context only when clearly relevant.",
    ])

    return "\n".join(parts)

# ============================================================
# SAFE CALCULATOR
# ============================================================

ALLOWED_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UN = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calc(expr):
    try:
        tree = ast.parse(
            expr,
            mode="eval",
        )

    except Exception:
        return None

    def ev(node):

        if isinstance(
            node,
            ast.Constant,
        ) and isinstance(
            node.value,
            (int, float),
        ):
            return node.value

        if (
            isinstance(
                node,
                ast.UnaryOp,
            )
            and type(node.op)
            in ALLOWED_UN
        ):
            return ALLOWED_UN[
                type(node.op)
            ](
                ev(node.operand)
            )

        if (
            isinstance(
                node,
                ast.BinOp,
            )
            and type(node.op)
            in ALLOWED_BIN
        ):

            right = ev(
                node.right
            )

            if (
                isinstance(
                    node.op,
                    ast.Pow,
                )
                and abs(right) > 10000
            ):
                raise ValueError

            return ALLOWED_BIN[
                type(node.op)
            ](
                ev(node.left),
                right,
            )

        raise ValueError

    try:
        return ev(
            tree.body
        )

    except Exception:
        return None


def calculation(q):
    x = (
        q.strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )

    x = re.sub(
        r"^(what is|calculate|find|solve|evaluate)\s+",
        "",
        x,
        flags=re.I,
    )

    if not re.fullmatch(
        r"[0-9\s.()+\-*/%*]+",
        x,
    ):
        return None

    if not re.search(
        r"[+\-*/%]",
        x,
    ):
        return None

    result = safe_calc(x)

    if result is None:
        return None

    if (
        isinstance(result, float)
        and result.is_integer()
    ):
        result = int(result)

    return (
        f"**{x} = {result} ✅**"
    )

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

    data_users = users()

    for name, user in data_users.items():

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
                    "Phone already registered.",
            }, 400

    data_users[username] = {
        "email": email,
        "phone": phone,
        "password":
            generate_password_hash(
                password
            ),
    }

    save_users(
        data_users
    )

    session["username"] = username
    session["chat_history"] = []

    clear_qset()

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

    value = data.get(
        "login",
        "",
    ).strip()

    password = data.get(
        "password",
        "",
    )

    data_users = users()

    for username, user in data_users.items():

        matches = (
            username.lower()
            == value.lower()
            or user.get(
                "email",
                "",
            ).lower()
            == value.lower()
            or user.get(
                "phone",
                "",
            )
            == value
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

                clear_qset()

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

    clear_qset()

    return {
        "success": True
    }

# ============================================================
# MAIN CHAT
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

    q = data.get(
        "message",
        "",
    ).strip()

    image = data.get(
        "image",
        "",
    )

    # --------------------------------------------------------
    # CAMERA / IMAGE
    # --------------------------------------------------------

    if image:

        answer = vision_ai(
            image,
            q,
        )

        save_history(
            q or "Image question",
            answer,
            "normal",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    if not q:

        return Response(
            "Please type a question or capture an image.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # BASIC / CREATOR
    # --------------------------------------------------------

    simple = basic(q)

    if simple:

        category = (
            "creator"
            if creator_question(q)
            else "normal"
        )

        simple = clean(
            simple
        )

        save_history(
            q,
            simple,
            category,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # QUESTION SET MEMORY
    # --------------------------------------------------------

    active_set = qset()

    command = qset_command(q)

    if command:

        if not active_set:

            return Response(
                "There is no active question set "
                "for this conversation. First generate "
                "a question set.",
                mimetype="text/plain",
            )

        set_subject = active_set[
            "subject"
        ]

        if command == "answers":

            answer = text_ai(
                prompt_for(
                    set_subject
                )
                + """
Answer every question in the existing question set.

Use ONLY the existing question set.
Keep the original numbering.
Give the correct option and a brief explanation.
Do not discuss the creator.
Do not generate unrelated questions.
Do not skip questions.
""",
                active_set[
                    "answer"
                ],
                reasoning="high",
                max_tokens=16000,
            )

        elif command == "options":

            answer = text_ai(
                prompt_for(
                    set_subject
                )
                + """
Add A, B, C and D to every existing question.

Keep:
- all questions
- original numbering
- same subject
- same difficulty

Do not answer the questions.
Do not use a table.
Do not use LaTeX.
""",
                active_set[
                    "answer"
                ],
                reasoning="medium",
                max_tokens=16000,
            )

            save_qset(
                active_set[
                    "request"
                ],
                answer,
                active_set[
                    "subject"
                ],
                active_set[
                    "difficulty"
                ],
            )

        elif command == "explain":

            answer = text_ai(
                prompt_for(
                    set_subject
                )
                + """
Solve every question in the existing set.

Keep original numbering.
Do not skip any question.
Show the important solution steps.
""",
                active_set[
                    "answer"
                ],
                reasoning="high",
                max_tokens=18000,
            )

        elif command == "difficulty":

            answer = text_ai(
                prompt_for(
                    set_subject
                )
                + """
Keep the same topic and question count.

Change the difficulty according to the user's request.

Return the complete updated question set.
""",
                active_set[
                    "answer"
                ]
                + "\n\nUSER REQUEST:\n"
                + q,
                reasoning="medium",
                max_tokens=16000,
            )

            save_qset(
                active_set[
                    "request"
                ],
                answer,
                active_set[
                    "subject"
                ],
                active_set[
                    "difficulty"
                ],
            )

        elif command == "single":

            match = re.search(
                r"(?:explain|solve|answer)"
                r"\s+(?:q|question)\s*(\d+)",
                q.lower(),
            )

            number = (
                match.group(1)
                if match
                else ""
            )

            answer = text_ai(
                prompt_for(
                    set_subject
                )
                + f"""
From the existing question set, answer ONLY Question {number}.

Do not answer other questions.
Do not discuss the creator.
""",
                active_set[
                    "answer"
                ],
                reasoning="high",
                max_tokens=7000,
            )

        else:

            answer = text_ai(
                prompt_for(
                    set_subject
                ),
                active_set[
                    "answer"
                ],
                reasoning="medium",
                max_tokens=7000,
            )

        answer = clean(
            answer
        )

        save_history(
            q,
            answer,
            "question_generation",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # QUESTION-SET CONTINUATION
    # --------------------------------------------------------

    continuation = continuation_count(
        q
    )

    if continuation is not None:

        if not active_set:

            return Response(
                "There is no active question set to continue.",
                mimetype="text/plain",
            )

        start = (
            last_num(
                active_set[
                    "answer"
                ]
            )
            + 1
        )

        answer = generate_set(
            active_set[
                "subject"
            ],
            active_set[
                "difficulty"
            ],
            continuation,
            start,
            active_set[
                "answer"
            ],
        )

        combined = clean(
            active_set[
                "answer"
            ]
            + "\n\n"
            + answer
        )

        save_qset(
            active_set[
                "request"
            ],
            combined,
            active_set[
                "subject"
            ],
            active_set[
                "difficulty"
            ],
        )

        save_history(
            q,
            answer,
            "question_generation",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # CONTINUE / MORE
    # --------------------------------------------------------

    if q.lower() in {
        "continue",
        "more",
        "another",
        "another one",
    } and active_set:

        start = (
            last_num(
                active_set[
                    "answer"
                ]
            )
            + 1
        )

        answer = generate_set(
            active_set[
                "subject"
            ],
            active_set[
                "difficulty"
            ],
            5,
            start,
            active_set[
                "answer"
            ],
        )

        combined = clean(
            active_set[
                "answer"
            ]
            + "\n\n"
            + answer
        )

        save_qset(
            active_set[
                "request"
            ],
            combined,
            active_set[
                "subject"
            ],
            active_set[
                "difficulty"
            ],
        )

        save_history(
            q,
            answer,
            "question_generation",
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    calc = calculation(
        q
    )

    if calc:

        save_history(
            q,
            calc,
            "normal",
        )

        return Response(
            calc,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # NEW QUESTION SET
    # --------------------------------------------------------

    if is_generation(q):

        count = question_count(
            q
        )

        if count:

            detected_subject = subject(
                q
            )

            detected_difficulty = difficulty(
                q
            )

            answer = generate_set(
                detected_subject,
                detected_difficulty,
                count,
                1,
                "",
            )

            save_qset(
                q,
                answer,
                detected_subject,
                detected_difficulty,
            )

            save_history(
                q,
                answer,
                "question_generation",
            )

            return Response(
                answer,
                mimetype="text/plain",
            )

    # --------------------------------------------------------
    # NORMAL / HARD SOLVING
    # --------------------------------------------------------

    detected_subject = subject(
        q
    )

    detected_difficulty = difficulty(
        q
    )

    context = normal_context(
        q
    )

    system_prompt = prompt_for(
        detected_subject
    )

    if detected_difficulty == "advanced":

        system_prompt += """

ADVANCED MODE

Solve very carefully.

Before finalizing:
- Recheck equations.
- Recheck algebra.
- Recheck arithmetic.
- Recheck units.
- Recheck signs.
- Recheck dimensions.
- Check the final result.
- Check MCQ options if present.

Never guess.
"""

        answer = text_ai(
            system_prompt,
            context,
            reasoning="high",
            max_tokens=9500,
        )

        if detected_subject in {
            "math",
            "physics",
        }:

            answer = verify_solution(
                q,
                answer,
                detected_subject,
            )

    elif detected_difficulty == "intermediate":

        answer = text_ai(
            system_prompt,
            context,
            reasoning="medium",
            max_tokens=6000,
        )

    else:

        answer = text_ai(
            system_prompt,
            context,
            reasoning="low",
            max_tokens=3500,
        )

    answer = clean(
        answer
    )

    save_history(
        q,
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

    q = data.get(
        "question",
        "",
    )

    answer = data.get(
        "answer",
        "",
    )

    action = data.get(
        "action",
        "improve",
    )

    detected_subject = subject(
        q
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

    result = text_ai(
        prompt_for(
            detected_subject
        )
        + "\n"
        + task,
        "QUESTION:\n"
        + q
        + "\n\nANSWER:\n"
        + answer,
        reasoning=(
            "high"
            if detected_subject
            in {
                "math",
                "physics",
            }
            else "medium"
        ),
        max_tokens=9000,
    )

    return Response(
        clean(result),
        mimetype="text/plain",
    )

# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "hf_configured": bool(
            HF_TOKEN
        ),
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
        "HF configured:",
        bool(HF_TOKEN),
    )
    print(
        "Hard Math: ON"
    )
    print(
        "Hard Physics: ON"
    )
    print(
        "Question Sets: ON"
    )
    print(
        "Follow-up Memory: ON"
    )
    print(
        "Vision / Camera: ON"
    )
    print(
        "Math Formatting: ON"
    )
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
