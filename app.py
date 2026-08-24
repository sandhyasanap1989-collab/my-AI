from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context
)

import os
import re
import math

from openai import OpenAI


# ============================================================
# APP
# ============================================================

app = Flask(__name__)


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
# PROMPTS
# ============================================================

SYSTEM_PROMPT = r"""
You are My AI, a helpful, accurate and concise study assistant.

Answer style:

For simple questions:
Give a short direct answer.

For numerical questions:

### Solution

**1. Step**
Formula and calculation.

**2. Step**
Formula and calculation.

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

Important:
- Do not invent missing information.
- Use simple student-friendly language.
- Check calculations carefully.
"""


MATH_SYSTEM_PROMPT = r"""
You are My AI's mathematics specialist.

Important:
- In this project, copied text "svg" means √.
- Never treat "svg" as x.
- Preserve the original numbers.
- Carefully identify fractions, parentheses, exponents and square roots.
- Do not invent numbers.
- Show clear short steps.
- Simplify surds exactly when possible.
- For MCQs, compare with the options.

End with:

### Final Answer

**answer ✅**
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


def normalize_math_text(text):
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


def get_problem_text(question):
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
        "fraction",
        "algebra"
    ]

    return any(term in q for term in math_terms)


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

        denominator = a - b

        if abs(denominator) < 1e-12:
            return None

        value = (a + b) / denominator

        return f"""
### Solution

We use:

√(5 + 2√6) = √3 + √2

and

√(5 − 2√6) = √3 − √2

Therefore,

N = [(√3 + √2) + (√3 − √2)]
    / [(√3 + √2) − (√3 − √2)]

N = 2√3 / 2√2

N = √(3/2)

### Final Answer

**√(3/2) ≈ {fmt(value)} ✅**
""".strip()

    return None


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

    rp = (r1 * r2) / (r1 + r2)

    total_resistance = sum(series_resistors) + rp

    total_current = voltage / total_resistance

    voltage_drops = [
        total_current * r
        for r in series_resistors
    ]

    parallel_voltage = (
        voltage - sum(voltage_drops)
    )

    i1 = parallel_voltage / r1
    i2 = parallel_voltage / r2

    if not close(
        sum(voltage_drops) + parallel_voltage,
        voltage
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
        fmt(x) for x in series_resistors
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

    for index, (resistor, drop) in enumerate(
        zip(series_resistors, voltage_drops),
        start=4
    ):
        lines.append(
            f"""**{index}. Voltage across {fmt(resistor)} Ω**

V = IR

V = {fmt(total_current)} × {fmt(resistor)}

**V = {fmt(drop)} V**"""
        )

    next_step = 4 + len(series_resistors)

    lines.append(
        f"""**{next_step}. Parallel-section voltage**

**Vₚ = {fmt(parallel_voltage)} V**

Both parallel branches have the same voltage."""
    )

    lines.append(
        f"""**{next_step + 1}. Current through {fmt(r1)} Ω**

I₁ = Vₚ / R₁

**I₁ = {fmt(i1)} A**"""
    )

    lines.append(
        f"""**{next_step + 2}. Current through {fmt(r2)} Ω**

I₂ = Vₚ / R₂

**I₂ = {fmt(i2)} A**"""
    )

    lines.append(
        f"""### Verification

**Voltage:** {fmt(sum(voltage_drops) + parallel_voltage)} V = {fmt(voltage)} V ✅

**Current:** {fmt(i1 + i2)} A = {fmt(total_current)} A ✅"""
    )

    lines.append(
        f"""### Final Answer

- **Parallel resistance = {fmt(rp)} Ω**
- **Total resistance = {fmt(total_resistance)} Ω**
- **Total current = {fmt(total_current)} A**
- **Parallel voltage = {fmt(parallel_voltage)} V**"""
    )

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

        delta = abs(final - initial)

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

    supplied_heat = (
        total_heat / (1 - loss_fraction)
        if loss_fraction < 1
        else total_heat
    )

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
            f"""**{i}. Heat required**

Q = mcΔT

Q = {fmt(block["mass"])} × {fmt(block["c"])} × {fmt(block["delta"])}

**Q = {fmt(block["heat"])} J**"""
        )

    lines.append(
        f"""**Total useful heat**

**Q = {fmt(total_heat)} J**"""
    )

    if heater_power:
        lines.append(
            f"""**Time without heat loss**

t = Q / P

**t = {fmt(time_without_loss)} s**"""
        )

    if loss_percent > 0:
        lines.append(
            f"""**Heat required with {fmt(loss_percent)}% loss**

**Q = {fmt(supplied_heat)} J**"""
        )

        if heater_power:
            lines.append(
                f"""**Time with heat loss**

**t = {fmt(time_with_loss)} s**"""
            )

    final = [
        "### Final Answer",
        f"**Total useful heat = {fmt(total_heat)} J**"
    ]

    if heater_power:
        final.append(
            f"**Time without loss = {fmt(time_without_loss)} s**"
        )

    if loss_percent > 0:
        final.append(
            f"**Heat with loss = {fmt(supplied_heat)} J**"
        )

        if heater_power:
            final.append(
                f"**Time with loss = {fmt(time_with_loss)} s ✅**"
            )

    lines.append("\n\n".join(final))

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

    if voltage is None:
        return None

    if not powers:
        return None

    if (
        "minimum resistance" in q
        or "least resistance" in q
        or "lowest resistance" in q
    ):

        highest_power = max(powers)

        return f"""### Solution

R = V² / P

At the same voltage, resistance is inversely proportional to power.

Highest rated power = **{fmt(highest_power)} W**

### Final Answer

**{fmt(highest_power)} W ✅**
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

Rₚ = ({fmt(r1)} × {fmt(r2)})
     / ({fmt(r1)} + {fmt(r2)})

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

Current increases by 100%.

Therefore:

I' = 2I

P' = (2I)²R

P' = 4P

Increase:

4P − P = 3P

Percentage increase = 300%

### Final Answer

**300% ✅**
"""

    return None


# ============================================================
# HUGGING FACE STREAM
# ============================================================

def stream_huggingface(question, system_prompt=None):

    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ HF_TOKEN is missing.\n\n"
            "Please add HF_TOKEN in Render → Environment."
        )
        return

    prompt = system_prompt or SYSTEM_PROMPT

    model = HF_MODEL

    try:

        stream = hf_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True,
            temperature=0.2,
            max_tokens=500
        )

        for chunk in stream:

            if not chunk.choices:
                continue

            text = chunk.choices[0].delta.content

            if text:
                yield text

    except Exception as error:

        print(
            "Hugging Face error:",
            repr(error)
        )

        yield (
            "❌ I couldn't connect to the online AI.\n\n"
            "Please check the Hugging Face token and "
            "Inference permission in Render."
        )


def stream_math_ai(question):

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

Original question:

{normalized}

{options_text}

Instructions:

1. Reconstruct the mathematical expression faithfully.
2. Treat svg as √.
3. Show a short solution.
4. Simplify exactly where possible.
5. For MCQs, select the correct option.
6. Check the final result where useful.

End with:

### Final Answer

**answer ✅**
"""

    yield from stream_huggingface(
        prompt,
        MATH_SYSTEM_PROMPT
    )


# ============================================================
# IMPROVE ANSWER
# ============================================================

def improve_answer(question, old_answer, action):

    instructions = {
        "improve":
            "Improve the answer and make it clearer and more accurate.",

        "check":
            "Check the previous answer for mathematical, physics and factual mistakes. Correct any mistakes.",

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

- Treat svg as √.
- Never replace svg with x.
- Verify calculations carefully.
- Do not invent missing numbers.
- Use Markdown.
- End with a clear Final Answer.
"""

    system = (
        MATH_SYSTEM_PROMPT
        if looks_like_math(normalized)
        else SYSTEM_PROMPT
    )

    yield from stream_huggingface(
        prompt,
        system
    )


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

    question = normalize_math_text(question)

    # Fast exact solvers
    result = solve_known_surds(question)
    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

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

    # Math AI
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

    # General AI
    return Response(
        stream_with_context(
            stream_huggingface(question)
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# IMPROVE
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
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MY AI")
    print("=" * 60)
    print("Hugging Face AI       :", bool(HF_TOKEN))
    print("Math solver           : ON")
    print("Physics solver        : ON")
    print("Circuit solver        : ON")
    print("Heat solver           : ON")
    print("Improve Answer        : ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )
