from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context,
    session,
)
import os
import re
import math

from openai import OpenAI


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

# IMPORTANT:
# Set FLASK_SECRET_KEY in Render Environment Variables.
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "my-ai-development-secret",
)


# ============================================================
# HUGGING FACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = "openai/gpt-oss-120b:cheapest"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# GENERAL AI PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are My AI, a helpful study assistant.

You help with:

- Mathematics
- Physics
- Chemistry
- Biology
- General Science
- General knowledge

IMPORTANT COMPLETION RULES
==========================

1. Read the complete question before answering.

2. If there are numbered parts, answer EVERY part.

3. Never stop in the middle of a sentence.

4. Never leave an equation incomplete.

5. Never omit a requested calculation.

6. Before finishing, check:
   - Did I answer every part?
   - Did I include units?
   - Did I check important calculations?
   - Did I provide a clear final answer?

7. For numerical questions use:

### Solution

**1. Step**

Formula

Calculation

**2. Step**

Formula

Calculation

### Final Answer

State all final results clearly.

8. If a question is a follow-up such as:
   "and world"
   "and China"
   "what about Japan"
   understand it using the previous conversation.

9. For current facts, do not pretend an old value is an exact live value.
   Use approximate/estimated wording when appropriate.

10. Do not invent missing information.

11. Use simple student-friendly language.

12. Always finish the answer completely.

13. Always end with a complete:

### Final Answer

**Answer ✅**
"""


# ============================================================
# MATH PROMPT
# ============================================================

MATH_SYSTEM_PROMPT = r"""
You are My AI's mathematics specialist.

Solve mathematics questions carefully and completely.

MATH FORMATTING RULES
=====================

- NEVER put equations inside [ ... ].
- NEVER output raw LaTeX commands such as:
  \frac
  \boxed
  \begin
  \end
  \tag
- Do not use broken expressions like x^{,n}.
- Use simple readable school-style mathematics.
- Use x², x³, x⁵ where possible.
- Use √ for square roots.
- For fractions, use (a/b) when simple.
- Put equations on separate lines.
- Never replace √ with x.
- In this project, copied text "svg" means √.

SOLVING RULES
=============

- Show the important steps.
- Do not skip transformations.
- For algebra, show the identity being used.
- For quadratic equations, show the method.
- For trigonometry, show the identity/formula.
- For geometry, mention the theorem/formula.
- For sequences and series, show the relevant formula.
- For coordinate geometry, show the coordinate formula.
- For probability, clearly define favorable and total outcomes.
- For surds, simplify exactly where possible.
- Check the final answer when useful.
- Answer every numbered part.

Always finish with:

### Final Answer

**answer ✅**
"""


# ============================================================
# PHYSICS PROMPT
# ============================================================

PHYSICS_SYSTEM_PROMPT = r"""
You are My AI's physics specialist.

Solve physics questions completely.

For numerical questions use:

### Given

List known quantities and units.

### Formula

State the relevant law/formula.

### Solution

Substitute values and calculate.

### Verification

Check units or the physical relation.

### Final Answer

Give the final answer with units.

Important:
- Use SI units where appropriate.
- Show signs/directions where relevant.
- Do not invent missing data.
- For circuits, verify current/voltage/power.
- For mechanics, verify laws and units.
- For work-energy, check joules.
- For heat, check mass units.
- For optics, use an appropriate sign convention.
- For electricity, distinguish voltage, current, resistance and power.
- Complete all requested parts.
"""


# ============================================================
# CHEMISTRY PROMPT
# ============================================================

CHEMISTRY_SYSTEM_PROMPT = r"""
You are My AI's chemistry specialist.

Solve chemistry questions completely.

For numerical problems:
- Write the reaction equation when relevant.
- Balance the equation if necessary.
- Calculate molar mass carefully.
- Convert mass to moles.
- Convert mL to L.
- Use M = n/V correctly.
- Use stoichiometric ratios.
- Identify limiting reagent.
- Calculate excess reagent when asked.
- Calculate gas volume using the stated conditions.
- Use Avogadro's number when required.
- Distinguish atoms, ions, molecules and moles.
- Include units.

For conceptual questions:
- Give correct definitions.
- Explain reactions clearly.
- Do not invent information.

Always answer every numbered part.

Always finish with:

### Final Answer
"""


# ============================================================
# BIOLOGY PROMPT
# ============================================================

BIOLOGY_SYSTEM_PROMPT = r"""
You are My AI's biology specialist.

Answer biology questions accurately and completely.

For definitions:
Give the definition first, followed by a simple explanation.

For processes:
Explain the steps in the correct order.

For comparisons:
Give clear point-by-point differences.

For genetics:
- Identify genotype and phenotype.
- Show crosses clearly.
- Calculate ratios and probabilities.

For cell biology:
- Identify structures and their functions.

For human physiology:
- Explain organs, tissues and systems logically.

For ecology:
- Explain energy flow, trophic levels, food chains and food webs.

Use correct biological terminology.

Do not invent facts.

Answer every requested part.

Always finish with:

### Final Answer
"""


# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

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
            "You can ask me Mathematics, Physics, Chemistry, "
            "Biology, Science or general questions."
        )

    if q in {
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return f"{q.title()}! 👋 How can I help you?"

    if q in {
        "how are you",
        "how are you?",
        "how r u",
        "how r u?",
    }:
        return (
            "I'm doing great! 🤖\n\n"
            "Give me a difficult PCMB question."
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
            "📚 General Science\n"
            "✏️ Step-by-step numerical problems"
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
        "okk",
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
# FOLLOW-UP MEMORY
# ============================================================

def is_followup(question):
    q = question.strip().lower()

    if q in {
        "and world",
        "world",
        "and china",
        "china",
        "and india",
        "and usa",
        "and the world",
        "what about the world",
        "what about china",
        "what about india",
    }:
        return True

    prefixes = (
        "and ",
        "what about ",
        "how about ",
        "also ",
        "compare ",
        "what about ",
        "then ",
    )

    return q.startswith(prefixes)


def build_contextual_question(question):
    history = session.get("chat_history", [])

    if not history:
        return question

    if not is_followup(question):
        return question

    previous_question = history[-1].get(
        "question",
        "",
    )

    previous_answer = history[-1].get(
        "answer",
        "",
    )

    return f"""
Previous question:
{previous_question}

Previous answer:
{previous_answer}

New follow-up:
{question}

Understand the follow-up in the context of the previous conversation.
Answer only the new follow-up, but use the earlier context when necessary.
"""


def save_turn(question, answer):
    history = session.get(
        "chat_history",
        [],
    )

    history.append(
        {
            "question": question,
            "answer": answer[:6000],
        }
    )

    # Keep only the most recent few turns
    session["chat_history"] = history[-6:]

    # Needed for Flask session changes
    session.modified = True


# ============================================================
# HELPERS
# ============================================================

def fmt(value, decimals=5):
    if value is None:
        return "N/A"

    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def normalize_math_text(text):
    text = text.replace("svg", "√")
    text = text.replace("SVG", "√")

    for char in [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060",
    ]:
        text = text.replace(char, "")

    return text


def get_problem_text(question):
    patterns = [
        r"\bcalculate the following\s*:",
        r"\bfind the following\s*:",
        r"\bdetermine the following\s*:",
        r"\bcalculate\s*:",
        r"\bfind\s*:",
        r"\bdetermine\s*:",
    ]

    positions = []

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        if match:
            positions.append(match.start())

    if positions:
        return question[: min(positions)]

    return question


def extract_mcq_options(text):
    options = re.findall(
        r"(?:^|\n)\s*[A-D]\s*[\)\.:\-]\s*(.+)",
        text,
        re.IGNORECASE,
    )

    return [x.strip() for x in options]


def detect_subject(question):
    q = question.lower()

    math_words = [
        "algebra",
        "equation",
        "quadratic",
        "polynomial",
        "surds",
        "sqrt",
        "√",
        "svg",
        "trigonometry",
        "sin",
        "cos",
        "tan",
        "coordinate",
        "geometry",
        "probability",
        "permutation",
        "combination",
        "sequence",
        "series",
        "logarithm",
        "matrix",
        "determinant",
        "vector",
        "calculus",
        "differentiate",
        "integration",
        "integral",
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
        "capacitor",
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
        "math": sum(word in q for word in math_words),
        "physics": sum(word in q for word in physics_words),
        "chemistry": sum(word in q for word in chemistry_words),
        "biology": sum(word in q for word in biology_words),
    }

    best_subject = max(
        scores,
        key=scores.get,
    )

    if scores[best_subject] == 0:
        return "general"

    return best_subject


def subject_prompt(subject):
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
# EXACT MATH SOLVERS
# ============================================================

def solve_known_surds(question):
    q = normalize_math_text(question)
    compact = re.sub(r"\s+", "", q)

    has_a = (
        "√(5+2√6)" in compact
        or "√(5+2√(6))" in compact
    )

    has_b = (
        "√(5-2√6)" in compact
        or "√(5-2√(6))" in compact
    )

    if not (has_a and has_b):
        return None

    a = math.sqrt(
        5 + 2 * math.sqrt(6)
    )

    b = math.sqrt(
        5 - 2 * math.sqrt(6)
    )

    if (
        "√(5+2√6)+√(5-2√6)" in compact
        and
        "√(5+2√6)-√(5-2√6)" in compact
    ):
        value = (a + b) / (a - b)

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


def solve_power_recurrence(question):
    """
    Handles:
    If x + 1/x = a, find x^n + 1/x^n
    for small n.
    """

    q = question.lower()

    match = re.search(
        r"x\s*\+\s*1/x\s*=\s*(-?\d+(?:\.\d+)?)",
        q,
    )

    n_match = re.search(
        r"x\s*(?:\^|\*\*)\s*(\d+)"
        r"\s*\+\s*1\s*/\s*x\s*(?:\^|\*\*)\s*\1",
        q,
    )

    if not match or not n_match:
        return None

    a = float(match.group(1))
    n = int(n_match.group(1))

    if n < 1 or n > 20:
        return None

    values = {
        0: 2,
        1: a,
    }

    if n >= 2:
        values[2] = a * a - 2

    for k in range(2, n):
        values[k + 1] = (
            a * values[k]
            - values[k - 1]
        )

    value = values[n]

    steps = []

    for k in range(2, n + 1):
        if k == 2:
            steps.append(
                f"S₂ = ({fmt(a)})² − 2 = {fmt(values[2])}"
            )
        else:
            steps.append(
                f"S{k} = {fmt(a)}S{k-1} − S{k-2} = {fmt(values[k])}"
            )

    return f"""
### Solution

Given:

x + 1/x = {fmt(a)}

Let:

Sₙ = xⁿ + 1/xⁿ

We use:

Sₙ₊₁ = (x + 1/x)Sₙ − Sₙ₋₁

Also:

S₀ = 2

S₁ = {fmt(a)}

{' '.join(steps)}

Therefore:

S{n} = {fmt(value)}

### Final Answer

**xⁿ + 1/xⁿ = {fmt(value)} ✅**
""".strip()


# ============================================================
# PHYSICS SOLVER
# ============================================================

def solve_circuit(question):
    problem = get_problem_text(question)
    q = problem.lower()

    if "parallel" not in q or "series" not in q:
        return None

    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:V|volt|volts)\b",
        problem,
        re.IGNORECASE,
    )

    if not voltage_match:
        return None

    voltage = float(
        voltage_match.group(1)
    )

    resistors = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)\b",
            problem,
            re.IGNORECASE,
        )
    ]

    if len(resistors) < 3:
        return None

    match = re.search(
        r"parallel\s+combination\s+of\s+"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)"
        r"\s*(?:and|&)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)",
        problem,
        re.IGNORECASE,
    )

    if not match:
        return None

    r1 = float(match.group(1))
    r2 = float(match.group(2))

    remaining = resistors.copy()

    try:
        remaining.remove(r1)
        remaining.remove(r2)
    except ValueError:
        return None

    if not remaining:
        return None

    series = remaining

    rp = (r1 * r2) / (
        r1 + r2
    )

    rt = sum(series) + rp

    current = voltage / rt

    series_voltage = [
        current * r
        for r in series
    ]

    vp = voltage - sum(
        series_voltage
    )

    i1 = vp / r1
    i2 = vp / r2

    series_power = [
        current ** 2 * r
        for r in series
    ]

    p1 = vp ** 2 / r1
    p2 = vp ** 2 / r2

    total_power = voltage * current
    component_power = (
        sum(series_power)
        + p1
        + p2
    )

    lines = [
        "### Solution",
        "",
        "**1. Parallel resistance**",
        "",
        "Rₚ = (R₁ × R₂) / (R₁ + R₂)",
        "",
        f"Rₚ = ({fmt(r1)} × {fmt(r2)}) / "
        f"({fmt(r1)} + {fmt(r2)})",
        "",
        f"**Rₚ = {fmt(rp)} Ω**",
        "",
        "**2. Total resistance**",
        "",
        f"Rₜ = {' + '.join(fmt(r) for r in series)} + {fmt(rp)}",
        "",
        f"**Rₜ = {fmt(rt)} Ω**",
        "",
        "**3. Total current**",
        "",
        "I = V / Rₜ",
        "",
        f"I = {fmt(voltage)} / {fmt(rt)}",
        "",
        f"**I = {fmt(current)} A**",
        "",
    ]

    step = 4

    for resistor, drop in zip(
        series,
        series_voltage,
    ):
        lines.extend([
            f"**{step}. Voltage across {fmt(resistor)} Ω**",
            "",
            "V = IR",
            "",
            f"V = {fmt(current)} × {fmt(resistor)}",
            "",
            f"**V = {fmt(drop)} V**",
            "",
        ])

        step += 1

    lines.extend([
        f"**{step}. Voltage across parallel section**",
        "",
        f"**Vₚ = {fmt(vp)} V**",
        "",
    ])

    step += 1

    lines.extend([
        f"**{step}. Current through {fmt(r1)} Ω**",
        "",
        "I₁ = Vₚ / R₁",
        "",
        f"**I₁ = {fmt(i1)} A**",
        "",
    ])

    step += 1

    lines.extend([
        f"**{step}. Current through {fmt(r2)} Ω**",
        "",
        "I₂ = Vₚ / R₂",
        "",
        f"**I₂ = {fmt(i2)} A**",
        "",
    ])

    step += 1

    lines.extend([
        f"**{step}. Power dissipated**",
        "",
    ])

    for resistor, power in zip(
        series,
        series_power,
    ):
        lines.extend([
            f"Power in {fmt(resistor)} Ω = I²R",
            "",
            f"**P = {fmt(power)} W**",
            "",
        ])

    lines.extend([
        f"Power in {fmt(r1)} Ω = V²/R",
        "",
        f"**P = {fmt(p1)} W**",
        "",
        f"Power in {fmt(r2)} Ω = V²/R",
        "",
        f"**P = {fmt(p2)} W**",
        "",
    ])

    step += 1

    lines.extend([
        f"**{step}. Total power**",
        "",
        "P = VI",
        "",
        f"P = {fmt(voltage)} × {fmt(current)}",
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
        f"- **Total current = {fmt(current)} A**",
        f"- **Parallel voltage = {fmt(vp)} V**",
        f"- **Total power = {fmt(total_power)} W ✅**",
    ])

    return "\n".join(lines)


def solve_basic_electricity(question):
    q = question.lower()
    problem = get_problem_text(question)

    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:V|volt|volts)\b",
        problem,
        re.IGNORECASE,
    )

    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:W|watt|watts)\b",
        problem,
        re.IGNORECASE,
    )

    resistors = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)\b",
            problem,
            re.IGNORECASE,
        )
    ]

    voltage = (
        float(voltage_match.group(1))
        if voltage_match
        else None
    )

    power = (
        float(power_match.group(1))
        if power_match
        else None
    )

    if (
        voltage is not None
        and power is not None
        and "resistance" in q
    ):
        resistance = (
            voltage ** 2
            / power
        )

        return f"""
### Solution

P = V²/R

R = V²/P

R = {fmt(voltage)}² / {fmt(power)}

**R = {fmt(resistance)} Ω**

### Final Answer

**{fmt(resistance)} Ω ✅**
""".strip()

    if (
        voltage is not None
        and power is not None
        and "current" in q
    ):
        current = power / voltage

        return f"""
### Solution

P = VI

I = P/V

I = {fmt(power)} / {fmt(voltage)}

**I = {fmt(current)} A**

### Final Answer

**{fmt(current)} A ✅**
""".strip()

    if (
        "series" in q
        and "resistance" in q
        and len(resistors) >= 2
    ):
        total = sum(resistors)

        return f"""
### Solution

For series resistors:

Rₜ = R₁ + R₂ + ...

Rₜ = {' + '.join(fmt(x) for x in resistors)}

**Rₜ = {fmt(total)} Ω**

### Final Answer

**{fmt(total)} Ω ✅**
""".strip()

    if (
        "parallel" in q
        and "resistance" in q
        and len(resistors) == 2
    ):
        r1, r2 = resistors

        rp = (
            r1 * r2
            / (r1 + r2)
        )

        return f"""
### Solution

Rₚ = (R₁ × R₂)/(R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)})
     / ({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(rp)} Ω**

### Final Answer

**{fmt(rp)} Ω ✅**
""".strip()

    return None


def solve_heat_question(question):
    problem = get_problem_text(question)

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(g|kg)",
        problem,
        re.IGNORECASE,
    )

    c_match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:J|j)"
        r"\s*/\s*"
        r"(kg|g)"
        r"\s*(?:°|deg)?\s*C",
        problem,
        re.IGNORECASE,
    )

    temp_match = re.search(
        r"from\s+(-?\d+(?:\.\d+)?)\s*°?\s*C"
        r"\s*(?:to|up to)\s*"
        r"(-?\d+(?:\.\d+)?)\s*°?\s*C",
        problem,
        re.IGNORECASE,
    )

    if not mass_match:
        return None

    if not c_match:
        return None

    if not temp_match:
        return None

    mass = float(
        mass_match.group(1)
    )

    mass_unit = (
        mass_match.group(2)
        .lower()
    )

    c = float(
        c_match.group(1)
    )

    c_unit = (
        c_match.group(2)
        .lower()
    )

    t1 = float(
        temp_match.group(1)
    )

    t2 = float(
        temp_match.group(2)
    )

    if mass_unit == "g":
        mass_kg = mass / 1000
    else:
        mass_kg = mass

    if c_unit == "g":
        c = c * 1000

    delta_t = abs(
        t2 - t1
    )

    heat = (
        mass_kg
        * c
        * delta_t
    )

    return f"""
### Solution

Q = mcΔT

m = {fmt(mass_kg)} kg

c = {fmt(c)} J kg⁻¹ °C⁻¹

ΔT = {fmt(delta_t)} °C

Q = {fmt(mass_kg)} × {fmt(c)} × {fmt(delta_t)}

**Q = {fmt(heat)} J**

### Final Answer

**Heat required = {fmt(heat)} J ✅**
""".strip()


# ============================================================
# CHEMISTRY SOLVERS
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
        or "fecl2" in q
    ):
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g"
        r".*?"
        r"(?:iron|fe)",
        question,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mL"
        r".*?"
        r"(?:hcl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M"
        r".*?"
        r"(?:hcl|hydrochloric)",
        question,
        re.IGNORECASE,
    )

    if not mass_match:
        return None

    if not volume_match:
        return None

    if not molarity_match:
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

    moles_fe = mass_fe / 56
    moles_hcl = (
        molarity_hcl
        * volume_hcl
    )

    hcl_needed = (
        2 * moles_fe
    )

    if moles_hcl < hcl_needed:
        limiting = "HCl"
        h2_moles = moles_hcl / 2
        fe_used = h2_moles
    else:
        limiting = "Fe"
        h2_moles = moles_fe
        fe_used = moles_fe

    h2_volume = (
        h2_moles
        * 22.4
    )

    fecl2_mass = (
        fe_used
        * 126.9
    )

    return f"""
### Solution

Reaction:

Fe + 2HCl → FeCl₂ + H₂

**1. Moles of Fe**

n = m/M

n = {fmt(mass_fe)} / 56

**n(Fe) = {fmt(moles_fe)} mol**

**2. Moles of HCl**

n = M × V

V = {fmt(volume_hcl)} L

n = {fmt(molarity_hcl)} × {fmt(volume_hcl)}

**n(HCl) = {fmt(moles_hcl)} mol**

**3. Limiting reagent**

1 mol Fe requires 2 mol HCl.

HCl required:

2 × {fmt(moles_fe)}

= {fmt(hcl_needed)} mol

Available HCl:

{fmt(moles_hcl)} mol

Therefore:

**Limiting reagent = {limiting}**

**4. Moles of H₂**

**n(H₂) = {fmt(h2_moles)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

V = {fmt(h2_moles)} × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M(FeCl₂) = 56 + 2(35.45)

= 126.9 g mol⁻¹

Mass = n × M

Mass = {fmt(fe_used)} × 126.9

**Mass(FeCl₂) = {fmt(fecl2_mass)} g**

### Final Answer

- **Fe = {fmt(moles_fe)} mol**
- **HCl = {fmt(moles_hcl)} mol**
- **Limiting reagent = {limiting}**
- **H₂ = {fmt(h2_moles)} mol**
- **H₂ at STP = {fmt(h2_volume)} L**
- **FeCl₂ = {fmt(fecl2_mass)} g ✅**
""".strip()


def solve_mole_question(question):
    q = question.lower()

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g",
        question,
        re.IGNORECASE,
    )

    if not mass_match:
        return None

    mass = float(
        mass_match.group(1)
    )

    # NaCl special case
    if "nacl" in q:
        molar_mass = 58.5

        moles = mass / molar_mass

        return f"""
### Solution

Molar mass of NaCl:

M = 23 + 35.5

M = 58.5 g mol⁻¹

Moles:

n = m/M

n = {fmt(mass)} / 58.5

**n = {fmt(moles)} mol**

### Final Answer

**{fmt(moles)} mol NaCl ✅**
""".strip()

    return None


def solve_molarity(question):
    q = question.lower()

    if "molarity" not in q:
        return None

    moles_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mol",
        question,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(mL|L)",
        question,
        re.IGNORECASE,
    )

    if not moles_match:
        return None

    if not volume_match:
        return None

    moles = float(
        moles_match.group(1)
    )

    volume = float(
        volume_match.group(1)
    )

    unit = (
        volume_match.group(2)
        .lower()
    )

    if unit == "ml":
        volume /= 1000

    if volume <= 0:
        return None

    molarity = (
        moles / volume
    )

    return f"""
### Solution

M = n/V

n = {fmt(moles)} mol

V = {fmt(volume)} L

M = {fmt(moles)} / {fmt(volume)}

**M = {fmt(molarity)} mol L⁻¹**

### Final Answer

**{fmt(molarity)} M ✅**
""".strip()


# ============================================================
# AI ROUTING
# ============================================================

def stream_huggingface(
    question,
    subject="general",
):
    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment."
        )
        return

    try:
        system_prompt = subject_prompt(
            subject
        )

        stream = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
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
            "Hugging Face error:",
            repr(error),
        )

        yield (
            "❌ I couldn't connect to the online AI right now.\n\n"
            "Please check HF_TOKEN and the "
            "Inference Providers permission."
        )


# ============================================================
# IMPROVE ANSWER
# ============================================================

def improve_answer(
    question,
    old_answer,
    action,
):
    instructions = {
        "improve": (
            "Improve the answer and make it clearer, "
            "more accurate and more complete."
        ),
        "check": (
            "Check the previous answer for "
            "mathematical, physics, chemistry and biology "
            "mistakes. Correct them."
        ),
        "explain": (
            "Explain the answer in more detail while "
            "remaining student-friendly."
        ),
        "short": (
            "Make the answer shorter but keep every "
            "important step and the final answer."
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

Important:
- Answer every requested part.
- Keep equations readable.
- Do not use raw LaTeX commands.
- Check all numerical calculations.
- Do not invent information.
- Never stop halfway through.
- End with a complete Final Answer.
"""

    yield from stream_huggingface(
        prompt,
        subject,
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"],
)
def chat():
    data = request.get_json(
        silent=True
    )

    if not data:
        return Response(
            "Please send a message.",
            mimetype="text/plain",
        )

    original_question = data.get(
        "message",
        "",
    ).strip()

    if not original_question:
        return Response(
            "Please type a message.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # BASIC RESPONSES
    # --------------------------------------------------------

    simple = basic_response(
        original_question
    )

    if simple:
        save_turn(
            original_question,
            simple,
        )

        return Response(
            simple,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # FOLLOW-UP CONTEXT
    # --------------------------------------------------------

    question = build_contextual_question(
        original_question
    )

    # --------------------------------------------------------
    # NORMALIZE MATH COPY ARTIFACT
    # --------------------------------------------------------

    question = normalize_math_text(
        question
    )

    # --------------------------------------------------------
    # EXACT MATH
    # --------------------------------------------------------

    result = solve_known_surds(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_power_recurrence(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # EXACT PHYSICS
    # --------------------------------------------------------

    result = solve_circuit(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_basic_electricity(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_heat_question(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # EXACT CHEMISTRY
    # --------------------------------------------------------

    result = solve_iron_hcl(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_molarity(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_mole_question(
        question
    )

    if result:
        save_turn(
            original_question,
            result,
        )

        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    subject = detect_subject(
        question
    )

    return Response(
        stream_with_context(
            stream_huggingface(
                question,
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
# IMPROVE ROUTE
# ============================================================

@app.route(
    "/improve",
    methods=["POST"],
)
def improve():
    data = request.get_json(
        silent=True
    )

    if not data:
        return Response(
            "Invalid request.",
            mimetype="text/plain",
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

    return Response(
        stream_with_context(
            improve_answer(
                question,
                old_answer,
                action,
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MY AI")
    print("=" * 70)
    print(
        "Hugging Face configured:",
        bool(HF_TOKEN),
    )
    print("Mathematics specialist : ON")
    print("Physics specialist     : ON")
    print("Chemistry specialist   : ON")
    print("Biology specialist     : ON")
    print("Basic responses        : ON")
    print("Follow-up memory       : ON")
    print("Long answers           : ON")
    print("Improve Answer         : ON")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
