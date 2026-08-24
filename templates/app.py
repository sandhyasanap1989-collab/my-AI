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
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Automatic provider selection.
TEXT_MODEL = "openai/gpt-oss-120b"

# Vision model for uploaded/captured photos.
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# CREATOR INFORMATION
# ============================================================

CREATOR_RESPONSE = """
My creator is Soham Chandrahas Sanap.

He is 15 years old and is studying in Class 10 in 2026
at Nimbark English School in Beed district, Maharashtra, India.

His main interests are Mathematics and web development.

He built My AI as an AI study assistant to help students
with Mathematics, Physics, Chemistry and Biology.
""".strip()


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

        return {}

    except (
        OSError,
        json.JSONDecodeError,
    ):
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
You are My AI, a student study assistant.

You help with:
- Mathematics
- Physics
- Chemistry
- Biology
- General science
- General academic questions

Handle questions from basic to very difficult levels.

Rules:
1. Read the complete question.
2. Identify exactly what is being asked.
3. Do not guess.
4. Answer every requested part.
5. Show important calculations.
6. Verify important results.
7. Check units when applicable.
8. Understand short follow-up questions from conversation context.
9. For simple questions, be concise.
10. For hard questions, give a complete useful derivation.
11. Always finish the response.

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
You are My AI's expert Mathematics solver.

Handle basic through extremely difficult Mathematics.

Topics include:
- Algebra
- Trigonometry
- Trigonometric identities
- Geometry
- Coordinate geometry
- Polynomials
- Quadratics
- Surds
- Sequences and series
- Binomial theorem
- Permutations and combinations
- Probability
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
- Olympiad-style problems

For difficult problems:

1. Understand the exact target.
2. Identify useful information.
3. Choose the correct method.
4. Derive step by step.
5. Simplify carefully.
6. Check the result.
7. Give the exact answer.
8. For MCQs, verify the selected option.
9. For identities, prove both sides or simplify correctly.
10. Never guess.

For equations, check roots.

For probability, check that the result is valid.

For inequalities, check equality conditions.

For geometry, respect all constraints.

For calculus, check the resulting derivative/integral.

When generating questions:
- Generate EXACTLY the number requested.
- Number continuously from 1 to the requested count.
- Give complete questions.
- Give A, B, C, D options when requested.
- Do not stop early.
- Do not give answers unless asked.
- Before finishing, verify that the requested count has actually been reached.

Never output raw LaTeX.
"""


PHYSICS_PROMPT = """
You are My AI's expert Physics solver for school, JEE Main
and JEE Advanced level.

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
2. Identify all important bodies, forces and constraints.
3. Choose coordinates.
4. Find equilibrium when needed.
5. Write governing equations.
6. Apply approximations only when justified.
7. Derive the requested quantity.
8. Check dimensions.
9. Check signs.
10. Check limiting cases when useful.
11. Verify MCQ options.

For small oscillations:
- Find equilibrium first.
- Define a small displacement.
- Find the restoring force.
- Use the small-displacement approximation.
- Compare with the SHM equation.

For hard JEE questions:
- Do not guess.
- Show the useful derivation.
- Verify the final result.

When generating questions:
- Generate EXACTLY the requested number.
- Give A, B, C, D options when requested.
- Make questions complete and challenging.
- Do not stop before the requested count.

Never output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are My AI's expert Chemistry solver.

Handle basic through advanced school, JEE Main and JEE Advanced questions.

Topics include:
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

For numerical problems:

Given
Formula/reaction
Substitution
Calculation
Verification
Final Answer

Check:
- units
- molar masses
- stoichiometric ratios
- limiting reagent
- significant arithmetic

When generating questions:
- Generate EXACTLY the requested number.
- Give A, B, C, D options when requested.
- Do not stop early.

Never output raw LaTeX.
"""


BIOLOGY_PROMPT = """
You are My AI's expert Biology solver.

Handle basic through advanced questions.

Topics:
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
6. For MCQs, evaluate all relevant options.

Do not invent biological facts.

When generating questions:
- Generate EXACTLY the requested number.
- Give A, B, C, D options when requested.
- Do not stop early.
"""


VISION_PROMPT = """
You are My AI's academic image-question solver.

The user has uploaded a photograph or screenshot.

First inspect the entire image.

Determine whether it contains:
- Mathematics
- Physics
- Chemistry
- Biology
- General academic content

Then solve the question.

For Mathematics:
- Read every visible number and symbol.
- Reconstruct the question carefully.
- Solve step by step.
- Check algebra and arithmetic.

For Physics:
- Identify the physical system.
- Extract given values.
- Choose the correct laws.
- Derive the result.
- Check units.

For Chemistry:
- Read equations and numerical data.
- Balance reactions when needed.
- Calculate carefully.

For Biology:
- Read labels and diagrams.
- Answer all requested parts.

IMPORTANT:
- Do not invent information not visible in the image.
- If something is blurry or cut off, say what is unclear.
- Actually solve the question.
- Do not merely describe the image.
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
        "previous context. Do not mention the memory."
    )

    return "\n".join(parts)


# ============================================================
# DETECT SUBJECT
# ============================================================

def detect_subject(question):

    q = question.lower()

    math_words = [
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
        "surds",
        "probability",
        "permutation",
        "combination",
        "sequence",
        "series",
        "matrix",
        "determinant",
        "calculus",
        "integral",
        "derivative",
        "logarithm",
        "vector",
        "complex number",
        "binomial",
        "inequality",
        "number theory",
        "divisibility",
        "function",
        "proof",
        "identity",
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
        "angular momentum",
        "oscillation",
        "oscillations",
        "shm",
        "wave",
        "capacitor",
        "induction",
    ]

    chemistry_words = [
        "mole",
        "moles",
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
        "kinetics",
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


# ============================================================
# DETECT DIFFICULTY
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

    if score >= 6:
        return "advanced"

    if score >= 2:
        return "intermediate"

    return "basic"


# ============================================================
# DETECT QUESTION GENERATION
# ============================================================

def is_question_generation(question):

    q = question.lower()

    generation_phrases = [
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
        for phrase in generation_phrases
    )


def requested_question_count(question):

    q = question.lower()

    patterns = [
        r"\b(\d+)\s+questions?\b",
        r"\btop\s+(\d+)\b",
        r"\b(\d+)\s+mcqs?\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
        )

        if match:

            try:
                number = int(
                    match.group(1)
                )

                # Safety for accidentally huge requests.
                return max(
                    1,
                    min(
                        number,
                        50,
                    ),
                )

            except ValueError:
                pass

    return None


# ============================================================
# GET PROMPT
# ============================================================

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
                calculate(node.operand)
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

        if result.is_integer():

            answer = f"{int(result):,}"

        else:

            answer = f"{result:.12g}"

    return (
        "### Calculation\n\n"
        f"**{q} = {answer} ✅**"
    )


# ============================================================
# LOCAL TRIGONOMETRY EXAMPLE SOLVER
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

    if not (
        "upon" in q
        or "bracket" in q
        or "sec square" in q
    ):
        return None

    return """
### Proof

We need to simplify:

[1/(sec²θ − cos²θ) + 1/(cosec²θ − sin²θ)]
× sin²θ × cos²θ

Using:

sec²θ = 1/cos²θ

and:

cosec²θ = 1/sin²θ

Therefore:

1/(sec²θ − cos²θ)
= cos²θ/(1 − cos⁴θ)

Since:

1 − cos⁴θ
= sin²θ(1 + cos²θ)

we get:

1/(sec²θ − cos²θ)
= cos²θ/[sin²θ(1 + cos²θ)]

Similarly:

1/(cosec²θ − sin²θ)
= sin²θ/[cos²θ(1 + sin²θ)]

Multiplying by sin²θ cos²θ:

LHS
= cos⁴θ/(1 + cos²θ)
+ sin⁴θ/(1 + sin²θ)

Using:

sin²θ + cos²θ = 1

and:

sin⁴θ + cos⁴θ
= 1 − 2sin²θcos²θ

we obtain:

LHS
= [1 − sin²θcos²θ]
  / [2 + sin²θcos²θ]

### Final Answer

**(1 − sin²θ cos²θ)/(2 + sin²θ cos²θ) ✅**
""".strip()


# ============================================================
# LOCAL x + 1/x SOLVER
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
                    f"({values[k-1]:g})"
                    f" − {values[k-2]:g}"
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

    prompt = get_prompt(
        subject
    )

    # Detect question generation.
    generation_request = (
        is_question_generation(
            original_question
        )
    )

    count = requested_question_count(
        original_question
    )

    # ========================================================
    # LONG QUESTION GENERATION
    # ========================================================

    if generation_request:

        prompt += """

THIS IS A QUESTION-GENERATION REQUEST.

Generate exactly the number of questions requested.

IMPORTANT:
- Do NOT stop early.
- Number continuously.
- Make every question complete.
- Include A, B, C and D when options are requested.
- Keep every option complete.
- Do not accidentally switch to answering the questions.
- Do not give the answer key unless the user asks.
- Before finishing, count the questions again.
"""

        if count:

            prompt += (
                f"\nThe requested count is exactly {count}."
            )

        max_tokens = max(
            10000,
            600 * (count or 10),
        )

        # Avoid unnecessarily huge requests.
        max_tokens = min(
            max_tokens,
            20000,
        )

        temperature = 0.4

    # ========================================================
    # ADVANCED
    # ========================================================

    elif difficulty == "advanced":

        prompt += """

THIS IS AN ADVANCED PROBLEM.

Solve carefully.

Before finalizing:
- Recheck equations.
- Recheck algebra.
- Recheck arithmetic.
- Recheck units.
- Recheck signs.
- Check limiting cases when useful.
- Verify MCQ options.
- Never guess.

Do not reveal private chain-of-thought.
Show the useful derivation and verification only.
"""

        max_tokens = 7000
        temperature = 0.05

    # ========================================================
    # INTERMEDIATE
    # ========================================================

    elif difficulty == "intermediate":

        prompt += """

This is an intermediate problem.

Show the important steps and verify the final result.
"""

        max_tokens = 4000
        temperature = 0.08

    # ========================================================
    # BASIC
    # ========================================================

    else:

        prompt += """

This is a basic question.

Keep the answer clear and reasonably concise.
"""

        max_tokens = 2500
        temperature = 0.1

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
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for chunk in stream:

            if not chunk.choices:
                continue

            delta = (
                chunk.choices[0]
                .delta
            )

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

        # Save COMPLETE answer so follow-ups work.
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
# IMAGE / CAMERA AI
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

    if not image_data.startswith(
        "data:image/"
    ):

        return (
            "❌ Invalid image data."
        )

    # Limit request size.
    if len(image_data) > 10_000_000:

        return (
            "❌ Image is too large.\n\n"
            "Please take a smaller or clearer photo."
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
            response
            .choices[0]
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
    # CAMERA / IMAGE
    # ========================================================

    if image_data:

        answer = analyze_image(
            image_data,
            question,
        )

        save_history(
            question
            or
            "Image question",
            answer,
        )

        return Response(
            answer,
            mimetype="text/plain",
        )

    # ========================================================
    # EMPTY
    # ========================================================

    if not question:

        return Response(
            "Please type a question or capture an image.",
            mimetype="text/plain",
        )

    # ========================================================
    # BASIC RESPONSE
    # ========================================================

    simple = basic_response(
        question
    )

    if simple:

        save_history(
            question,
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
        question
    )

    if result:

        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL MATH SOLVER
    # ========================================================

    result = solve_power_recurrence(
        question
    )

    if result:

        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL TRIG SOLVER
    # ========================================================

    result = solve_trig_identity(
        question
    )

    if result:

        save_history(
            question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # ========================================================
    # ONLINE AI
    # ========================================================

    subject = detect_subject(
        question
    )

    difficulty = detect_difficulty(
        question
    )

    context_question = (
        build_ai_context(
            question
        )
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
        "Basic answers : ON"
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
        "Advanced AI   : ON"
    )

    print(
        "Question gen  : ON"
    )

    print(
        "Vision AI     : ON"
    )

    print(
        "Camera backend: ON"
    )

    print(
        "Memory        : ON"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
