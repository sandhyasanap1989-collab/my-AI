from flask import Flask, render_template, request, Response, stream_with_context
import ollama
import re
import math

app = Flask(__name__)

# ============================================================
# MODEL SETTINGS
# ============================================================

BIG_MODEL = "llama3.2"
SMALL_MODEL = "llama3.2:1b"
KEEP_ALIVE = "30m"


# ============================================================
# GENERAL AI PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are My AI, a helpful, accurate and concise study assistant.

ANSWER STYLE
------------

Simple question:
Give a short direct answer.

Numerical question:
Use:

### Solution

**1. Step**
Formula and short calculation.

**2. Step**
Formula and short calculation.

### Final Answer

**Answer ✅**

For multiple-choice questions:
- Solve the question.
- Compare with the options.
- Clearly state the correct option.

For multi-part questions:
- Answer every requested part.
- Use numbered steps.
- Do not repeat answers unnecessarily.
- Do not write one huge paragraph.

IMPORTANT:
- Do not invent missing information.
- Do not change a verified numerical result from Python.
- Use simple student-friendly language.
"""


# ============================================================
# MATH-SPECIALIST PROMPT
# ============================================================

MATH_SYSTEM_PROMPT = r"""
You are My AI's mathematics specialist.

The user's copied mathematical questions sometimes use the text
"svg" where a square-root symbol √ was originally displayed.

IMPORTANT:
- Treat "svg" as √ ONLY in this project.
- Never treat "svg" as x.
- Preserve the original numbers.
- Carefully identify fractions, parentheses, exponents and square roots.
- Do not invent numbers.
- If the copied layout is partially flattened, use the surrounding
  expression and MCQ options to reconstruct the most likely intended
  expression.
- If you cannot reconstruct it confidently, state exactly what part
  is ambiguous.

For Real Numbers / Surds:
- Simplify exact forms where possible.
- Use identities such as:
  √(a + 2√b) = √m + √n
  when m+n=a and mn=b.
- Check the result numerically.
- For MCQs, compare the result with the options.

Answer format:

### Solution

[clear short steps]

### Final Answer

**[answer] ✅**
"""


# ============================================================
# HELPERS
# ============================================================

def fmt(value, decimals=4):
    if value is None:
        return "N/A"

    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def close(a, b, tolerance=1e-7):
    return math.isclose(
        a,
        b,
        rel_tol=tolerance,
        abs_tol=tolerance
    )


def extract_voltage(text):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:V|volt|volts)\b",
        text,
        re.IGNORECASE
    )

    return float(match.group(1)) if match else None


def extract_resistors(text):
    return [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)\b",
            text,
            re.IGNORECASE
        )
    ]


def get_problem_text(question):
    """
    Ignore the list of requested outputs after:
    Calculate:
    Find:
    Determine:
    """

    patterns = [
        r"\bcalculate the following\s*:",
        r"\bfind the following\s*:",
        r"\bdetermine the following\s*:",
        r"\bcalculate\s*:",
        r"\bfind\s*:",
        r"\bdetermine\s*:"
    ]

    positions = []

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:
            positions.append(match.start())

    if positions:
        return question[:min(positions)]

    return question


# ============================================================
# SVG / COPY-FORMATTING NORMALIZATION
# ============================================================

def normalize_math_text(text):
    """
    Convert the project's known copied-text artifact:

        svg  ->  √

    Also remove zero-width/invisible characters.
    """

    text = text.replace("svg", "√")
    text = text.replace("SVG", "√")

    for char in [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060"
    ]:
        text = text.replace(char, "")

    return text


# ============================================================
# CASUAL MESSAGE
# ============================================================

def is_casual(message):
    return message.strip().lower() in {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "hi!",
        "hello!",
        "hey!",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay"
    }


# ============================================================
# MCQ EXTRACTION
# ============================================================

def extract_mcq_options(text):
    options = re.findall(
        r"(?:^|\n)\s*[A-D]\s*[\)\.:\-]\s*(.+)",
        text,
        re.IGNORECASE
    )

    return [x.strip() for x in options]


# ============================================================
# KNOWN SURD SOLVER
# ============================================================

def solve_known_surds(question):
    """
    Handles common exact nested-surds patterns.

    Most useful example:

    (√(5+2√6) + √(5-2√6))
    /
    (√(5+2√6) - √(5-2√6))

    and the reversed +/- version.
    """

    q = normalize_math_text(question)
    compact = re.sub(r"\s+", "", q)

    # --------------------------------------------------------
    # Pattern recognition
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Explicit plus-over-minus
    # --------------------------------------------------------

    if (
        "√(5+2√6)+√(5-2√6)" in compact
        and
        "√(5+2√6)-√(5-2√6)" in compact
    ):

        numerator = a + b
        denominator = a - b

        if abs(denominator) < 1e-12:
            return None

        value = numerator / denominator

        return f"""
### Solution

Let

A = √(5 + 2√6)

B = √(5 − 2√6)

We use:

√(5 + 2√6) = √3 + √2

and

√(5 − 2√6) = √3 − √2

Therefore,

N = (A + B) / (A − B)

N = [(√3 + √2) + (√3 − √2)]
    / [(√3 + √2) − (√3 − √2)]

N = 2√3 / 2√2

N = √(3/2)

### Final Answer

**√(3/2) ≈ {fmt(value)} ✅**
""".strip()

    # --------------------------------------------------------
    # Explicit minus-over-plus
    # --------------------------------------------------------

    if (
        "√(5+2√6)-√(5-2√6)" in compact
        and
        "√(5+2√6)+√(5-2√6)" in compact
    ):

        numerator = a - b
        denominator = a + b

        if abs(denominator) < 1e-12:
            return None

        value = numerator / denominator

        return f"""
### Solution

Let

A = √(5 + 2√6)

B = √(5 − 2√6)

We use:

√(5 + 2√6) = √3 + √2

and

√(5 − 2√6) = √3 − √2

Therefore,

N = (A − B) / (A + B)

N = [(√3 + √2) − (√3 − √2)]
    / [(√3 + √2) + (√3 − √2)]

N = 2√2 / 2√3

N = √(2/3)

### Final Answer

**√(2/3) ≈ {fmt(value)} ✅**
""".strip()

    return None


# ============================================================
# MATH FALLBACK
# ============================================================

def stream_math_ai(question):
    """
    Dedicated mathematics route.

    This handles:
    - roots
    - surds
    - algebra
    - real numbers
    - MCQs
    - copied svg -> √ text
    """

    normalized = normalize_math_text(question)

    options = extract_mcq_options(normalized)

    options_text = ""

    if options:
        options_text = (
            "\n\nThe question includes these options:\n"
            + "\n".join(
                f"{chr(65+i)}) {value}"
                for i, value in enumerate(options[:4])
            )
        )

    prompt = f"""
Solve the following mathematics question carefully.

IMPORTANT:
In this project, copied text 'svg' means √.

Original question:
{normalized}

{options_text}

Instructions:
1. Reconstruct the mathematical expression as faithfully as possible.
2. Do not replace √ with x.
3. Show a short solution.
4. For surds, simplify exactly when possible.
5. If it is an MCQ, select one of the given options if correct.
6. Check the final result numerically where useful.
7. End with:
### Final Answer
**answer ✅**
"""

    try:

        stream = ollama.chat(
            model=BIG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": MATH_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={
                "temperature": 0.05,
                "num_predict": 500
            }
        )

        for chunk in stream:
            text = chunk["message"]["content"]

            if text:
                yield text

    except Exception as error:

        print("Math AI error:", error)

        yield (
            "❌ I couldn't solve the mathematical expression."
        )


def looks_like_math(question):
    q = question.lower()

    math_terms = [
        "sqrt",
        "√",
        "svg",
        "solve",
        "simplify",
        "find the value",
        "real number",
        "surds",
        "quadratic",
        "equation",
        "polynomial",
        "root",
        "fraction"
    ]

    return any(
        term in q
        for term in math_terms
    )


# ============================================================
# MIXED CIRCUIT SOLVER
# ============================================================

def solve_mixed_circuit(question):

    problem = get_problem_text(question)
    q = problem.lower()

    if "parallel combination" not in q:
        return None

    if "series" not in q:
        return None

    voltage = extract_voltage(problem)

    if voltage is None:
        return None

    parallel_match = re.search(
        r"parallel\s+combination\s+of\s+"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)"
        r"\s*(?:and|&)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)",
        problem,
        re.IGNORECASE
    )

    if not parallel_match:
        return None

    r1 = float(parallel_match.group(1))
    r2 = float(parallel_match.group(2))

    all_resistors = extract_resistors(problem)

    if len(all_resistors) < 3:
        return None

    remaining = all_resistors.copy()

    try:
        remaining.remove(r1)
        remaining.remove(r2)
    except ValueError:
        return None

    series_resistors = remaining

    if not series_resistors:
        return None

    rp = (
        r1 * r2
    ) / (
        r1 + r2
    )

    total_resistance = (
        sum(series_resistors) + rp
    )

    total_current = (
        voltage / total_resistance
    )

    voltage_drops = [
        total_current * r
        for r in series_resistors
    ]

    parallel_voltage = (
        voltage - sum(voltage_drops)
    )

    i1 = parallel_voltage / r1
    i2 = parallel_voltage / r2

    series_powers = [
        total_current ** 2 * r
        for r in series_resistors
    ]

    p1 = (
        parallel_voltage ** 2
    ) / r1

    p2 = (
        parallel_voltage ** 2
    ) / r2

    total_power = voltage * total_current

    component_power = (
        sum(series_powers) + p1 + p2
    )

    if not close(
        sum(voltage_drops) + parallel_voltage,
        voltage
    ):
        return None

    if not close(
        i1 + i2,
        total_current
    ):
        return None

    if not close(
        component_power,
        total_power
    ):
        return None

    lines = ["### Solution"]

    lines.append(
        f"""**1. Parallel resistance**

Rₚ = (R₁ × R₂) / (R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)}) / ({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(rp)} Ω**"""
    )

    series_expression = " + ".join(
        fmt(x)
        for x in series_resistors
    )

    lines.append(
        f"""**2. Total resistance**

Rₜ = {series_expression} + {fmt(rp)}

**Rₜ = {fmt(total_resistance)} Ω**"""
    )

    lines.append(
        f"""**3. Total current**

I = V / Rₜ

I = {fmt(voltage)} / {fmt(total_resistance)}

**I = {fmt(total_current)} A**"""
    )

    step = 4

    for resistor, drop in zip(
        series_resistors,
        voltage_drops
    ):
        lines.append(
            f"""**{step}. Voltage across {fmt(resistor)} Ω**

V = IR

V = {fmt(total_current)} × {fmt(resistor)}

**V = {fmt(drop)} V**"""
        )

        step += 1

    lines.append(
        f"""**{step}. Voltage across the parallel section**

**Vₚ = {fmt(parallel_voltage)} V**

Both parallel branches have the same voltage."""
    )

    step += 1

    lines.append(
        f"""**{step}. Current through {fmt(r1)} Ω**

I₁ = Vₚ / R₁

**I₁ = {fmt(i1)} A**"""
    )

    step += 1

    lines.append(
        f"""**{step}. Current through {fmt(r2)} Ω**

I₂ = Vₚ / R₂

**I₂ = {fmt(i2)} A**"""
    )

    step += 1

    for resistor, power in zip(
        series_resistors,
        series_powers
    ):
        lines.append(
            f"""**{step}. Power in {fmt(resistor)} Ω**

P = I²R

**P = {fmt(power)} W**"""
        )

        step += 1

    lines.append(
        f"""**{step}. Power in {fmt(r1)} Ω**

P = Vₚ² / R

**P = {fmt(p1)} W**"""
    )

    step += 1

    lines.append(
        f"""**{step}. Power in {fmt(r2)} Ω**

P = Vₚ² / R

**P = {fmt(p2)} W**"""
    )

    step += 1

    lines.append(
        f"""**{step}. Total power**

P = VI

**P = {fmt(total_power)} W**"""
    )

    lines.append(
        f"""### Verification

**Voltage:** {fmt(sum(voltage_drops) + parallel_voltage)} V = {fmt(voltage)} V ✅

**Current:** {fmt(i1 + i2)} A = {fmt(total_current)} A ✅

**Power:** {fmt(component_power)} W = {fmt(total_power)} W ✅"""
    )

    final = [
        "### Final Answer",
        "",
        f"- **Parallel resistance = {fmt(rp)} Ω**",
        f"- **Total resistance = {fmt(total_resistance)} Ω**",
        f"- **Total current = {fmt(total_current)} A**"
    ]

    for resistor, drop in zip(
        series_resistors,
        voltage_drops
    ):
        final.append(
            f"- **Voltage across {fmt(resistor)} Ω = {fmt(drop)} V**"
        )

    final.extend([
        f"- **Parallel voltage = {fmt(parallel_voltage)} V**",
        f"- **Current through {fmt(r1)} Ω = {fmt(i1)} A**",
        f"- **Current through {fmt(r2)} Ω = {fmt(i2)} A**"
    ])

    for resistor, power in zip(
        series_resistors,
        series_powers
    ):
        final.append(
            f"- **Power in {fmt(resistor)} Ω = {fmt(power)} W**"
        )

    final.extend([
        f"- **Power in {fmt(r1)} Ω = {fmt(p1)} W**",
        f"- **Power in {fmt(r2)} Ω = {fmt(p2)} W**",
        f"- **Total power = {fmt(total_power)} W ✅**"
    ])

    lines.append("\n".join(final))

    return "\n\n".join(lines)


# ============================================================
# HEAT SOLVER
# ============================================================

def solve_heat_question(question):

    problem = get_problem_text(question)

    masses = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*g(?:ram|rams)?\b",
            problem,
            re.IGNORECASE
        )
    ]

    specific_heats = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*J\s*/\s*kg\s*(?:°|deg)?\s*C",
            problem,
            re.IGNORECASE
        )
    ]

    temperature_pairs = re.findall(
        r"(\d+(?:\.\d+)?)\s*°?\s*C"
        r"\s*(?:to|-)\s*"
        r"(\d+(?:\.\d+)?)\s*°?\s*C",
        problem,
        re.IGNORECASE
    )

    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:W|watt|watts)\b",
        problem,
        re.IGNORECASE
    )

    loss_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:heat\s*)?loss",
        problem,
        re.IGNORECASE
    )

    if not masses or not specific_heats:
        return None

    if not temperature_pairs:
        return None

    count = min(
        len(masses),
        len(specific_heats),
        len(temperature_pairs)
    )

    total_heat = 0
    blocks = []

    for i in range(count):

        mass = masses[i] / 1000
        specific_heat = specific_heats[i]

        initial = float(
            temperature_pairs[i][0]
        )

        final = float(
            temperature_pairs[i][1]
        )

        delta = abs(
            final - initial
        )

        heat = (
            mass
            * specific_heat
            * delta
        )

        total_heat += heat

        blocks.append(
            {
                "mass": mass,
                "c": specific_heat,
                "delta": delta,
                "heat": heat
            }
        )

    heater_power = (
        float(power_match.group(1))
        if power_match
        else None
    )

    loss_percent = (
        float(loss_match.group(1))
        if loss_match
        else 0
    )

    loss_fraction = loss_percent / 100

    if loss_fraction < 1:
        supplied_heat = (
            total_heat / (1 - loss_fraction)
        )
    else:
        supplied_heat = total_heat

    time_without_loss = (
        total_heat / heater_power
        if heater_power
        else None
    )

    time_with_loss = (
        supplied_heat / heater_power
        if heater_power
        else None
    )

    lines = ["### Solution"]

    for i, block in enumerate(
        blocks,
        start=1
    ):
        lines.append(
            f"""**{i}. Heat required for block {i}**

Q = mcΔT

Q = {fmt(block["mass"])} × {fmt(block["c"])} × {fmt(block["delta"])}

**Q = {fmt(block["heat"])} J**"""
        )

    step = len(blocks) + 1

    lines.append(
        f"""**{step}. Total useful heat**

**Q = {fmt(total_heat)} J**"""
    )

    step += 1

    if heater_power:
        lines.append(
            f"""**{step}. Time without heat loss**

t = Q / P

t = {fmt(total_heat)} / {fmt(heater_power)}

**t = {fmt(time_without_loss)} s**"""
        )
        step += 1

    if loss_percent > 0:

        lines.append(
            f"""**{step}. Heat required with {fmt(loss_percent)}% heat loss**

Useful fraction = {fmt(1 - loss_fraction)}

Qₛ = Q / useful fraction

**Qₛ = {fmt(supplied_heat)} J**"""
        )

        step += 1

        if heater_power:
            lines.append(
                f"""**{step}. Time with heat loss**

t = Qₛ / P

t = {fmt(supplied_heat)} / {fmt(heater_power)}

**t = {fmt(time_with_loss)} s**"""
            )

    final = [
        "### Final Answer",
        ""
    ]

    for i, block in enumerate(
        blocks,
        start=1
    ):
        final.append(
            f"- **Block {i} heat = {fmt(block['heat'])} J**"
        )

    final.append(
        f"- **Total useful heat = {fmt(total_heat)} J**"
    )

    if heater_power:
        final.append(
            f"- **Time without loss = {fmt(time_without_loss)} s**"
        )

    if loss_percent > 0:
        final.append(
            f"- **Heat with loss = {fmt(supplied_heat)} J**"
        )

        if heater_power:
            final.append(
                f"- **Time with loss = {fmt(time_with_loss)} s ✅**"
            )

    lines.append(
        "\n".join(final)
    )

    return "\n\n".join(lines)


# ============================================================
# BULB SOLVER
# ============================================================

def solve_bulb_question(question):

    q = question.lower()
    problem = get_problem_text(question)

    if "bulb" not in q:
        return None

    powers = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:W|watt|watts)\b",
            problem,
            re.IGNORECASE
        )
    ]

    voltage = extract_voltage(problem)

    if voltage is None or len(powers) < 2:
        return None

    if (
        "minimum resistance" in q
        or "least resistance" in q
        or "lowest resistance" in q
    ):
        highest_power = max(powers)

        return f"""### Solution

R = V² / P

At the same voltage, resistance is inversely proportional to
rated power.

Highest rated power = **{fmt(highest_power)} W**

### Final Answer

**{fmt(highest_power)} W ✅**
"""

    if (
        "maximum resistance" in q
        or "greatest resistance" in q
        or "highest resistance" in q
    ):
        lowest_power = min(powers)

        return f"""### Solution

R = V² / P

At the same voltage, lower rated power means greater resistance.

Lowest rated power = **{fmt(lowest_power)} W**

### Final Answer

**{fmt(lowest_power)} W ✅**
"""

    return None


# ============================================================
# BASIC ELECTRICITY
# ============================================================

def solve_basic_electricity(question):

    q = question.lower()
    problem = get_problem_text(question)

    voltage = extract_voltage(problem)
    resistors = extract_resistors(problem)

    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:W|watt|watts)\b",
        problem,
        re.IGNORECASE
    )

    power = (
        float(power_match.group(1))
        if power_match
        else None
    )

    if (
        voltage is not None
        and power is not None
        and "bulb" in q
        and "resistance" in q
    ):

        resistance = voltage ** 2 / power

        return f"""### Solution

R = V² / P

R = {fmt(voltage)}² / {fmt(power)}

**R = {fmt(resistance)} Ω**

### Final Answer

**{fmt(resistance)} Ω ✅**
"""

    if (
        "series" in q
        and "parallel combination" not in q
        and "resistance" in q
        and len(resistors) >= 2
    ):

        total = sum(resistors)

        return f"""### Solution

Rₜ = R₁ + R₂ + ...

Rₜ = {' + '.join(fmt(x) for x in resistors)}

**Rₜ = {fmt(total)} Ω**

### Final Answer

**{fmt(total)} Ω ✅**
"""

    if (
        "parallel" in q
        and "parallel combination" not in q
        and "resistance" in q
        and len(resistors) == 2
    ):

        r1, r2 = resistors

        equivalent = (
            r1 * r2
        ) / (
            r1 + r2
        )

        return f"""### Solution

Rₚ = (R₁ × R₂) / (R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)}) /
     ({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(equivalent)} Ω**

### Final Answer

**{fmt(equivalent)} Ω ✅**
"""

    if (
        voltage is not None
        and power is not None
        and "current" in q
    ):

        current = power / voltage

        return f"""### Solution

P = VI

I = P / V

I = {fmt(power)} / {fmt(voltage)}

**I = {fmt(current)} A**

### Final Answer

**{fmt(current)} A ✅**
"""

    return None


# ============================================================
# PERCENTAGE POWER
# ============================================================

def solve_percentage_power(question):

    q = question.lower()

    if (
        "current" in q
        and "power" in q
        and "resistor" in q
        and "increased by 100%" in q
    ):

        return """### Solution

P = I²R

Current increases by 100%:

I' = 2I

P' = (2I)²R = 4P

Increase:

4P − P = 3P = 300%

### Final Answer

**300% ✅**
"""

    return None


# ============================================================
# GENERAL OLLAMA STREAM
# ============================================================

def stream_ollama(question):

    model = (
        SMALL_MODEL
        if is_casual(question)
        else BIG_MODEL
    )

    try:

        stream = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={
                "temperature": 0.2,
                "num_predict": 400
            }
        )

        for chunk in stream:

            text = chunk["message"]["content"]

            if text:
                yield text

    except Exception as error:

        print("Ollama error:", error)

        yield (
            "❌ I couldn't connect to my AI brain. "
            "Please make sure Ollama is running."
        )


# ============================================================
# IMPROVE ANSWER
# ============================================================

def improve_answer(question, old_answer, action):

    instructions = {
        "improve":
            "Improve the answer and make it clearer and more accurate.",

        "check":
            "Check the previous answer for mathematical, physics and factual mistakes. "
            "Correct any mistakes.",

        "explain":
            "Give a slightly more detailed explanation while remaining concise.",

        "short":
            "Make the answer shorter while keeping the essential steps and final answer."
    }

    instruction = instructions.get(
        action,
        instructions["improve"]
    )

    normalized = normalize_math_text(question)

    prompt = f"""
Original question:

{normalized}

Previous answer:

{old_answer}

Task:

{instruction}

Rules:
- Treat svg as √ if it appears in this project.
- Never replace svg with x.
- Verify calculations carefully.
- Do not invent missing numbers.
- Use Markdown.
- End with a clear Final Answer.
"""

    try:

        stream = ollama.chat(
            model=BIG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": MATH_SYSTEM_PROMPT
                    if looks_like_math(normalized)
                    else SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
            keep_alive=KEEP_ALIVE,
            options={
                "temperature": 0.1,
                "num_predict": 500
            }
        )

        for chunk in stream:

            text = chunk["message"]["content"]

            if text:
                yield text

    except Exception as error:

        print("Improve error:", error)

        yield "❌ I couldn't improve the answer."


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# CHAT
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True)

    if not data:
        return Response(
            "Please send a message.",
            mimetype="text/plain"
        )

    question = data.get(
        "message",
        ""
    ).strip()

    if not question:
        return Response(
            "Please type a message.",
            mimetype="text/plain"
        )

    # --------------------------------------------------------
    # Normalize known copied math artifact.
    # --------------------------------------------------------

    question = normalize_math_text(
        question
    )

    # --------------------------------------------------------
    # Real Numbers / Surds
    # --------------------------------------------------------

    result = solve_known_surds(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # --------------------------------------------------------
    # Fast physics
    # --------------------------------------------------------

    result = solve_mixed_circuit(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_heat_question(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_percentage_power(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_bulb_question(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_basic_electricity(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # --------------------------------------------------------
    # Math fallback before general AI.
    # --------------------------------------------------------

    if looks_like_math(question):
        return Response(
            stream_with_context(
                stream_math_ai(question)
            ),
            mimetype="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    # --------------------------------------------------------
    # General AI fallback.
    # --------------------------------------------------------

    return Response(
        stream_with_context(
            stream_ollama(question)
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# IMPROVE ROUTE
# ============================================================

@app.route("/improve", methods=["POST"])
def improve():

    data = request.get_json(silent=True)

    if not data:
        return Response(
            "Invalid request.",
            mimetype="text/plain"
        )

    question = data.get(
        "question",
        ""
    ).strip()

    old_answer = data.get(
        "answer",
        ""
    ).strip()

    action = data.get(
        "action",
        "improve"
    ).strip().lower()

    if not question or not old_answer:
        return Response(
            "Missing question or answer.",
            mimetype="text/plain"
        )

    return Response(
        stream_with_context(
            improve_answer(
                question,
                old_answer,
                action
            )
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MY AI")
    print("=" * 70)
    print("SVG -> √ normalization : ON")
    print("Real Numbers / Surds   : ON")
    print("Physics engine         : ON")
    print("Circuit verification   : ON")
    print("Heat solver            : ON")
    print("Math AI fallback       : ON")
    print("Improve Answer         : ON")
    print("Streaming              : ON")
    print("Keep alive             :", KEEP_ALIVE)
    print("Big model              :", BIG_MODEL)
    print("Small model            :", SMALL_MODEL)
    print("Website                : http://127.0.0.1:5000")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )