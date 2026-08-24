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
import json
import ast
import math
import operator

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


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Automatic fastest provider selection.
TEXT_MODEL = "openai/gpt-oss-120b"

# Better vision model for image questions.
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

His interests include Mathematics and web development.

He built My AI as a study assistant for Mathematics,
Physics, Chemistry and Biology.
""".strip()


# ============================================================
# PROMPTS
# ============================================================

GENERAL_PROMPT = """
You are My AI, a high-quality student study assistant.

You answer:
- Mathematics
- Physics
- Chemistry
- Biology
- General science
- Academic questions

You must handle basic, intermediate, advanced and very difficult questions.

GENERAL RULES:
1. Read the complete question.
2. Identify exactly what is being asked.
3. Never guess.
4. Answer every requested part.
5. Show useful calculations.
6. Check important arithmetic.
7. Check units when relevant.
8. Verify the final result whenever practical.
9. For MCQs, verify the option.
10. Understand follow-up questions using conversation context.
11. Always finish the response.

Never output raw LaTeX such as:
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

Use readable forms such as:
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

Handle everything from school mathematics to very difficult
JEE Main, JEE Advanced and Olympiad-style mathematics.

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
- Permutations
- Combinations
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

FOR DIFFICULT PROBLEMS:
1. Understand the target.
2. Identify the relevant information.
3. Choose the correct method.
4. Derive step by step.
5. Simplify carefully.
6. Check algebra.
7. Check the final result.
8. For MCQs, verify the option.
9. For identities, actually prove the identity.
10. Never guess.

For equations, check roots.

For inequalities, check equality cases.

For probability, check that the result is valid.

For geometry, respect all constraints.

For calculus, check the result where practical.

Do not output raw LaTeX.
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

FOR DIFFICULT PROBLEMS:
1. Understand the physical setup.
2. Identify all relevant bodies.
3. Identify forces and constraints.
4. Choose coordinates.
5. Find equilibrium if required.
6. Write governing equations.
7. Apply approximations only when justified.
8. Derive the requested quantity.
9. Check signs.
10. Check dimensions.
11. Check limiting cases when useful.
12. Verify MCQ options.

FOR SMALL OSCILLATION PROBLEMS:
- Find equilibrium first.
- Define the small displacement.
- Find the restoring force.
- Use the correct small-displacement approximation.
- Compare with the SHM relation.

Never guess.
Do not output raw LaTeX.
"""


CHEMISTRY_PROMPT = """
You are My AI's expert Chemistry solver.

Handle school, JEE Main and JEE Advanced questions.

Topics:
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
→ Formula/reaction
→ Substitution
→ Calculation
→ Verification
→ Final Answer

Check units, molar masses, stoichiometric ratios and limiting reagents.

Do not guess.
Do not output raw LaTeX.
"""


BIOLOGY_PROMPT = """
You are My AI's expert Biology solver.

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
3. Explain the mechanism in order.
4. Compare alternatives when needed.
5. For genetics, calculate probabilities carefully.
6. For MCQs, evaluate the options.

Do not invent facts.
"""


VISION_PROMPT = """
You are My AI's image-question solver.

The user has uploaded a photograph or screenshot.

Read the entire image carefully.

Identify whether it is:
- Mathematics
- Physics
- Chemistry
- Biology
- General academic content

Then solve it.

MATHEMATICS:
- Read every number and symbol.
- Reconstruct the question.
- Solve carefully.
- Check algebra and arithmetic.

PHYSICS:
- Identify the physical system.
- Read all given values.
- Choose the correct physical laws.
- Derive the result.
- Check dimensions and units.

CHEMISTRY:
- Read equations and values.
- Balance reactions when needed.
- Calculate carefully.

BIOLOGY:
- Read labels and diagrams.
- Answer all requested parts.

IMPORTANT:
- Do not invent text that is not visible.
- If something is unclear, say what is unclear.
- Do not merely describe the image.
- Actually solve the question.
- Do not output raw LaTeX.
"""


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
    ) as f:
        json.dump(
            users,
            f,
            indent=2,
        )


# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(question):
    q = question.strip().lower()

    creator_words = {
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
        "who is your father",
        "tell me about your creator",
        "tell me about your creator?",
        "who is soham chandrahas sanap",
        "who is soham chandrahas sanap?",
    }

    if q in creator_words:
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
# FOLLOW-UP DETECTION
# ============================================================

def is_followup(question):
    q = question.strip().lower()

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
        "continue please",
    }

    if q in short:
        return True

    patterns = [
        r"^other\s+\d+\s+questions?$",
        r"^another\s+\d+\s+questions?$",
        r"^next\s+\d+\s+questions?$",
        r"^more\s+\d+\s+questions?$",
        r"^remaining\s+\d+\s+questions?$",
        r"^give\s+(?:me\s+)?(?:the\s+)?other\s+\d+\s+questions?$",
        r"^give\s+(?:me\s+)?another\s+\d+\s+questions?$",
    ]

    for pattern in patterns:
        if re.match(pattern, q):
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
        m = re.match(pattern, q)

        if m:
            return min(
                max(
                    int(m.group(1)),
                    1,
                ),
                50,
            )

    return None


def save_history(question, answer):
    history = session.get(
        "chat_history",
        [],
    )

    history.append(
        {
            "question": question,
            "answer": answer[:12000],
        }
    )

    session["chat_history"] = history[-10:]
    session.modified = True


def last_question_number(answer):
    numbers = re.findall(
        r"(?:^|\n)\s*(?:\*\*)?(\d+)\.",
        answer,
    )

    if not numbers:
        return 0

    return max(
        int(n)
        for n in numbers
    )


def build_context(question):
    history = session.get(
        "chat_history",
        [],
    )

    if not history:
        return question

    count = continuation_count(
        question
    )

    if count is not None:
        last = history[-1]

        previous_number = last_question_number(
            last["answer"]
        )

        start = previous_number + 1
        end = previous_number + count

        return f"""
PREVIOUS QUESTION-GENERATION REQUEST:
{last["question"]}

PREVIOUS GENERATED RESPONSE:
{last["answer"]}

NEW USER REQUEST:
{question}

THIS IS A CONTINUATION.

Generate exactly {count} additional questions.

Start numbering at {start}.
End numbering at {end}.

Keep the same subject and style as the previous set.
Do not restart from Question 1.
Do not change to an unrelated subject.
Give options if the previous request asked for options.
Do not provide an answer key unless requested.
""".strip()

    if not is_followup(question):
        return question

    recent = history[-5:]

    parts = [
        "RECENT CONVERSATION:",
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
        "\nUse relevant context to answer the new question."
    )

    return "\n".join(parts)


# ============================================================
# SUBJECT
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
    ]

    chemistry_words = [
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
        "reproduction",
        "heredity",
        "blood",
        "heart",
        "kidney",
        "lung",
        "brain",
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

    if scores[subject] == 0:
        return "general"

    return subject


# ============================================================
# DIFFICULTY
# ============================================================

def detect_difficulty(question):
    q = question.lower()

    score = 0

    advanced_terms = [
        "jee advanced",
        "jee adv",
        "olympiad",
        "hard",
        "hardest",
        "difficult",
        "challenging",
        "prove",
        "derive",
        "derivation",
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

    for term in advanced_terms:
        if term in q:
            score += 2

    if len(q) >= 300:
        score += 2

    if len(q) >= 700:
        score += 2

    if q.count("?") >= 2:
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
        m = re.search(
            pattern,
            q,
        )

        if m:
            return min(
                max(
                    int(m.group(1)),
                    1,
                ),
                50,
            )

    return None


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

    def calc(node):
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
                calc(node.operand)
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

            left = calc(node.left)
            right = calc(node.right)

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
        return calc(
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
# LOCAL x + 1/x SOLVER
# ============================================================

def solve_power_recurrence(question):
    q = question.lower()

    m = re.search(
        r"x\s*\+\s*1\s*/\s*x\s*=\s*"
        r"(-?\d+(?:\.\d+)?)",
        q,
    )

    if not m:
        return None

    a = float(
        m.group(1)
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

    if n < 1 or n > 50:
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
# TEXT MODEL STREAM
# ============================================================

def stream_responses_api(
    instructions,
    user_input,
    reasoning_effort,
    max_output_tokens,
):
    """
    Streams output_text.delta events from the Responses API.
    Hugging Face documents this event-based streaming format.
    """

    stream = hf_client.responses.create(
        model=TEXT_MODEL,
        instructions=instructions,
        input=user_input,
        reasoning={
            "effort": reasoning_effort,
        },
        max_output_tokens=max_output_tokens,
        stream=True,
    )

    full_answer = ""

    for event in stream:

        event_type = getattr(
            event,
            "type",
            "",
        )

        if event_type == (
            "response.output_text.delta"
        ):
            delta = getattr(
                event,
                "delta",
                "",
            )

            if delta:
                full_answer += delta
                yield delta

        elif event_type == "response.failed":

            error = getattr(
                event,
                "error",
                None,
            )

            if error:
                yield (
                    "\n\n❌ AI error: "
                    + str(error)
                )

    return


# ============================================================
# NORMAL / HARD TEXT AI
# ============================================================

def stream_text_ai(
    user_question,
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

    base_prompt = {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }.get(
        subject,
        GENERAL_PROMPT,
    )

    generation = is_question_generation(
        original_question
    )

    count = requested_question_count(
        original_question
    )

    continuation = continuation_count(
        original_question
    )

    if generation:

        prompt = base_prompt + """

QUESTION GENERATION MODE

Generate EXACTLY the requested number of questions.

Rules:
- Do not stop early.
- Number continuously.
- Give complete questions.
- Use A, B, C and D when options are requested.
- Keep the same subject.
- Keep the requested difficulty.
- Do not provide the answer key unless asked.
- Before ending, count the questions and make sure the requested
  number has been completed.
"""

        if continuation is not None:

            history = session.get(
                "chat_history",
                [],
            )

            previous_end = 0

            if history:
                previous_end = last_question_number(
                    history[-1]["answer"]
                )

            start = previous_end + 1
            end = previous_end + continuation

            prompt += f"""

CONTINUATION REQUEST:

Generate exactly {continuation} more questions.

Start at Question {start}.
End at Question {end}.

Do not restart at Question 1.
"""

        elif count is not None:

            prompt += f"""

The requested number is exactly {count} questions.
"""

        max_tokens = min(
            18000,
            max(
                8000,
                600 * (
                    count
                    or continuation
                    or 10
                ),
            ),
        )

        reasoning = "medium"

    elif difficulty == "advanced":

        prompt = base_prompt + """

ADVANCED REASONING MODE

This is a difficult problem.

Work carefully through the mathematical or physical logic.

Before finalizing:
- Recheck equations.
- Recheck algebra.
- Recheck arithmetic.
- Recheck units.
- Recheck signs.
- Check limiting cases where useful.
- Verify MCQ options.
- Never guess.

Do not reveal private chain-of-thought.
Give the useful derivation, checks and final answer.
"""

        max_tokens = 9000
        reasoning = "high"

    elif difficulty == "intermediate":

        prompt = base_prompt + """

INTERMEDIATE MODE

Show the important steps.
Check the result before finishing.
"""

        max_tokens = 5000
        reasoning = "medium"

    else:

        prompt = base_prompt + """

BASIC MODE

Give a clear, concise answer.
"""

        max_tokens = 3000
        reasoning = "low"

    try:

        # Stream primary answer.
        stream = stream_responses_api(
            instructions=prompt,
            user_input=user_question,
            reasoning_effort=reasoning,
            max_output_tokens=max_tokens,
        )

        # The generator yields the actual answer.
        for piece in stream:
            yield piece

    except Exception as error:

        print(
            "RESPONSES API ERROR:",
            repr(error),
        )

        # Fallback to Chat Completions so a Responses API
        # provider failure does not immediately break the app.
        try:

            completion = (
                hf_client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": user_question,
                        },
                    ],
                    temperature=0.05,
                    max_tokens=max_tokens,
                    stream=True,
                )
            )

            for chunk in completion:

                if not chunk.choices:
                    continue

                text = getattr(
                    chunk.choices[0].delta,
                    "content",
                    None,
                )

                if text:
                    yield text

        except Exception as fallback_error:

            print(
                "CHAT COMPLETIONS FALLBACK ERROR:",
                repr(fallback_error),
            )

            yield (
                "❌ Hugging Face error:\n\n"
                + str(fallback_error)
            )


# ============================================================
# COMPLETE ANSWER FOR VERIFICATION
# ============================================================

def get_complete_answer(
    user_question,
    subject,
    difficulty,
):
    """
    Non-streaming call used only for the verification pass.
    """

    prompt = {
        "math": MATH_PROMPT,
        "physics": PHYSICS_PROMPT,
        "chemistry": CHEMISTRY_PROMPT,
        "biology": BIOLOGY_PROMPT,
        "general": GENERAL_PROMPT,
    }.get(
        subject,
        GENERAL_PROMPT,
    )

    prompt += """

Return the complete final solution.

Show the useful derivation.
Check the result carefully.
Do not output raw LaTeX.
"""

    response = hf_client.responses.create(
        model=TEXT_MODEL,
        instructions=prompt,
        input=user_question,
        reasoning={
            "effort": (
                "high"
                if difficulty == "advanced"
                else "medium"
            ),
        },
        max_output_tokens=9000,
    )

    return response.output_text


# ============================================================
# VERIFY ADVANCED ANSWER
# ============================================================

def verify_advanced_answer(
    question,
    answer,
    subject,
):
    """
    A second pass checks advanced Math/Physics answers.

    It does not expose private reasoning.
    """

    if subject not in {
        "math",
        "physics",
    }:
        return answer

    if len(answer) < 20:
        return answer

    verifier_prompt = """
You are a strict verifier for an educational AI.

Check the supplied problem and solution.

Your task:
1. Determine whether the final result is correct.
2. Recalculate important mathematics.
3. Check Physics equations, dimensions and signs if applicable.
4. If the answer is wrong, provide a corrected solution.
5. If it is correct, return the solution with only necessary corrections.
6. Do not discuss private chain-of-thought.
7. Do not output raw LaTeX.

Return only the corrected final educational answer.
"""

    try:

        response = hf_client.responses.create(
            model=TEXT_MODEL,
            instructions=verifier_prompt,
            input=f"""
QUESTION:

{question}

PROPOSED ANSWER:

{answer}
""",
            reasoning={
                "effort": "high",
            },
            max_output_tokens=9000,
        )

        verified = response.output_text

        if verified and verified.strip():
            return verified

    except Exception as error:

        print(
            "VERIFICATION ERROR:",
            repr(error),
        )

    return answer


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

    if not image_data.startswith(
        "data:image/"
    ):
        return "❌ Invalid image data."

    if len(image_data) > 12_000_000:
        return (
            "❌ Image is too large.\n\n"
            "Please capture a smaller or clearer image."
        )

    user_text = (
        question.strip()
        if question.strip()
        else
        "Read the image carefully and solve the question step by step."
    )

    try:

        # Responses API multimodal input.
        response = hf_client.responses.create(
            model=VISION_MODEL,
            instructions=VISION_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text,
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

        answer = response.output_text

        if not answer:
            return (
                "❌ Vision model returned no answer."
            )

        return answer

    except Exception as error:

        print(
            "VISION API ERROR:",
            repr(error),
        )

        # Vision fallback through Chat Completions.
        try:

            response = (
                hf_client.chat.completions.create(
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
                    max_tokens=7000,
                )
            )

            if response.choices:

                answer = (
                    response.choices[0]
                    .message
                    .content
                )

                if answer:
                    return answer

        except Exception as fallback_error:

            print(
                "VISION FALLBACK ERROR:",
                repr(fallback_error),
            )

            return (
                "❌ Vision AI error:\n\n"
                + str(fallback_error)
            )

        return (
            "❌ Vision AI could not analyze this image."
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
# CHAT ROUTE
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
    # BASIC
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

    local_result = solve_calculation(
        question
    )

    if local_result:

        save_history(
            question,
            local_result,
        )

        return Response(
            local_result,
            mimetype="text/plain",
        )

    # ========================================================
    # LOCAL RECURRING MATH
    # ========================================================

    local_result = solve_power_recurrence(
        question
    )

    if local_result:

        save_history(
            question,
            local_result,
        )

        return Response(
            local_result,
            mimetype="text/plain",
        )

    # ========================================================
    # ONLINE REASONING
    # ========================================================

    subject = detect_subject(
        question
    )

    difficulty = detect_difficulty(
        question
    )

    context_question = build_context(
        question
    )

    # For advanced Math/Physics, use a complete
    # verification pass after the main answer.
    #
    # For normal/basic answers, stream directly.
    if (
        difficulty == "advanced"
        and subject in {
            "math",
            "physics",
        }
        and not is_question_generation(question)
    ):

        try:

            draft = get_complete_answer(
                context_question,
                subject,
                difficulty,
            )

            verified = verify_advanced_answer(
                question,
                draft,
                subject,
            )

            save_history(
                question,
                verified,
            )

            return Response(
                verified,
                mimetype="text/plain",
            )

        except Exception as error:

            print(
                "ADVANCED VERIFIED PATH ERROR:",
                repr(error),
            )

            # Fall through to the streaming path.
            pass

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

    instruction_map = {
        "improve":
            "Improve the answer and make it clearer and more complete.",

        "check":
            "Check the answer for mistakes and correct them.",

        "explain":
            "Explain the answer in more detail.",

        "short":
            "Make the answer shorter while keeping important steps.",
    }

    instruction = instruction_map.get(
        action,
        instruction_map["improve"],
    )

    subject = detect_subject(
        question
    )

    prompt = f"""
You are improving an existing educational answer.

QUESTION:
{question}

CURRENT ANSWER:
{old_answer}

TASK:
{instruction}

Check:
- mathematical correctness
- physics equations
- arithmetic
- units
- final answer

Do not output raw LaTeX.
Return the improved final answer only.
"""

    if (
        subject in {
            "math",
            "physics",
        }
    ):

        try:

            response = hf_client.responses.create(
                model=TEXT_MODEL,
                instructions=prompt,
                input=prompt,
                reasoning={
                    "effort": "high",
                },
                max_output_tokens=8000,
            )

            result = response.output_text

            return Response(
                result,
                mimetype="text/plain",
            )

        except Exception as error:

            print(
                "IMPROVE ERROR:",
                repr(error),
            )

    return Response(
        stream_with_context(
            stream_text_ai(
                prompt,
                subject,
                question,
                detect_difficulty(question),
            )
        ),
        mimetype="text/plain",
    )


# ============================================================
# HEALTH CHECK
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
        "Text reasoning: ON"
    )
    print(
        "Hard Math: ON"
    )
    print(
        "Hard Physics: ON"
    )
    print(
        "Chemistry: ON"
    )
    print(
        "Biology: ON"
    )
    print(
        "Vision: ON"
    )
    print(
        "Camera backend: ON"
    )
    print(
        "Question generation: ON"
    )
    print(
        "Continuation memory: ON"
    )
    print(
        "Advanced verification: ON"
    )
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
