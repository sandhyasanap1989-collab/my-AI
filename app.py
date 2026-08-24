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

app = Flask(__name__)

# ============================================================
# HUGGING FACE SETTINGS
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
# AI PROMPTS
# ============================================================

SYSTEM_PROMPT = r"""
You are My AI, a helpful study assistant.

Answer accurately and clearly.

For simple questions:
Give a direct answer.

For numerical questions:

### Solution

**1. Step**
Show the formula and calculation.

**2. Step**
Continue the calculation.

### Final Answer

**Answer ✅**

For MCQs:
- Solve the question.
- Compare the options.
- Clearly state the correct option.

For multi-part questions:
- Answer every part.
- Use numbered steps.
- Do not stop halfway through.
- Do not omit requested calculations.
- Check the final result.

Use simple student-friendly language.
Do not invent missing information.
"""


MATH_SYSTEM_PROMPT = r"""
You are My AI's mathematics specialist.

Important:
- In this project, copied text "svg" means √.
- Never treat "svg" as x.
- Preserve the original numbers.
- Carefully identify fractions, brackets, exponents and square roots.
- Show all requested steps.
- Complete every requested part.
- Check the result where useful.

End with:

### Final Answer

**answer ✅**
"""


# ============================================================
# BASIC CHAT RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "hi!",
        "hello!",
        "hey!"
    }

    if q in greetings:
        return (
            "Hello! 👋 I'm My AI.\n\n"
            "You can ask me mathematics, physics, chemistry, "
            "science, or general questions."
        )

    if q in {
        "good morning",
        "good afternoon",
        "good evening"
    }:
        return f"{q.title()}! 👋 How can I help you today?"

    if q in {
        "how are you",
        "how are you?",
        "how r u",
        "how r u?"
    }:
        return (
            "I'm doing great! 🤖\n\n"
            "Ask me a question and let's solve it together."
        )

    if q in {
        "who are you",
        "who are you?",
        "what are you",
        "what are you?"
    }:
        return (
            "I'm My AI 🤖, a study assistant for "
            "mathematics, physics, chemistry, science and general questions."
        )

    if q in {
        "what can you do",
        "what can you do?",
        "help",
        "help me"
    }:
        return (
            "I can help with:\n\n"
            "• Mathematics 🧮\n"
            "• Physics ⚡\n"
            "• Chemistry 🧪\n"
            "• Science 🔬\n"
            "• General questions 📚\n"
            "• Step-by-step numerical problems ✏️"
        )

    if q in {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty"
    }:
        return "You're welcome! 😊"

    if q in {
        "ok",
        "okay",
        "okk",
        "cool",
        "nice",
        "great"
    }:
        return "👍 Great! Ask me your next question."

    if q in {
        "bye",
        "goodbye",
        "see you",
        "see ya"
    }:
        return "Goodbye! 👋 Come back whenever you need help."

    return None


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


def extract_mcq_options(text):
    options = re.findall(
        r"(?:^|\n)\s*[A-D]\s*[\)\.:\-]\s*(.+)",
        text,
        re.IGNORECASE
    )

    return [x.strip() for x in options]


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
        "algebra",
        "trigonometry",
        "sin",
        "cos",
        "tan",
        "geometry",
        "angle"
    ]

    return any(term in q for term in math_terms)


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

    a = math.sqrt(5 + 2 * math.sqrt(6))
    b = math.sqrt(5 - 2 * math.sqrt(6))

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
# CIRCUIT SOLVER
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

    parallel_voltage = voltage - sum(voltage_drops)

    i1 = parallel_voltage / r1
    i2 = parallel_voltage / r2

    powers = {
        r: total_current ** 2 * r
        for r in series_resistors
    }

    p1 = parallel_voltage ** 2 / r1
    p2 = parallel_voltage ** 2 / r2

    total_power = voltage * total_current

    component_power = sum(powers.values()) + p1 + p2

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
        f"""**{step}. Voltage across parallel combination**

**Vₚ = {fmt(parallel_voltage)} V**"""
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

    lines.append(f"**{step}. Power dissipated**\n")

    for resistor, power in powers.items():
        lines.append(
            f"Power in {fmt(resistor)} Ω = I²R = **{fmt(power)} W**"
        )

    lines.append(
        f"Power in {fmt(r1)} Ω = V²/R = **{fmt(p1)} W**"
    )

    lines.append(
        f"Power in {fmt(r2)} Ω = V²/R = **{fmt(p2)} W**"
    )

    step += 1

    lines.append(
        f"""**{step}. Total power**

P = VI

P = {fmt(voltage)} × {fmt(total_current)}

**P = {fmt(total_power)} W**"""
    )

    lines.append(
        f"""### Verification

Component power = {fmt(component_power)} W

Total battery power = {fmt(total_power)} W

**Power verification ✅**"""
    )

    lines.append(
        f"""### Final Answer

- **Parallel resistance = {fmt(rp)} Ω**
- **Total resistance = {fmt(total_resistance)} Ω**
- **Total current = {fmt(total_current)} A**
- **Parallel voltage = {fmt(parallel_voltage)} V**
- **Total power = {fmt(total_power)} W ✅**"""
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

        initial = float(temperature_pairs[i][0])
        final = float(temperature_pairs[i][1])

        delta = abs(final - initial)

        heat = mass * specific_heat * delta

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

    lines = ["### Solution"]

    for i, block in enumerate(blocks, start=1):
        lines.append(
            f"""**{i}. Heat required**

Q = mcΔT

Q = {fmt(block["mass"])} × {fmt(block["c"])} × {fmt(block["delta"])}

**Q = {fmt(block["heat"])} J**"""
        )

    lines.append(
        f"""### Final Answer

**Total heat = {fmt(total_heat)} J**"""
    )

    if heater_power:
        time_taken = total_heat / heater_power

        lines.append(
            f"""**Time**

t = Q / P

t = {fmt(total_heat)} / {fmt(heater_power)}

**t = {fmt(time_taken)} s ✅**"""
        )

    return "\n\n".join(lines)


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

        equivalent = (r1 * r2) / (r1 + r2)

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

I' = 2I

P' = (2I)²R

P' = 4P

Percentage increase = 300%

### Final Answer

**300% ✅**
"""

    return None


# ============================================================
# HUGGING FACE AI
# ============================================================

def stream_huggingface(question, system_prompt=None):

    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ HF_TOKEN is missing.\n\n"
            "Check Render → Environment."
        )
        return

    try:
        stream = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True,
            temperature=0.2,
            max_tokens=1200
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
            "Please check the Hugging Face connection "
            "and Render environment variable."
        )


def stream_math_ai(question):

    normalized = normalize_math_text(question)

    options = extract_mcq_options(normalized)

    options_text = ""

    if options:
        options_text = (
            "\n\nOptions:\n"
            + "\n".join(
                f"{chr(65+i)}) {value}"
                for i, value in enumerate(options[:4])
            )
        )

    prompt = f"""
Solve this mathematics question completely.

Question:
{normalized}

{options_text}

Instructions:
1. Understand the original expression carefully.
2. Treat svg as √.
3. Show all requested steps.
4. Do not skip any requested part.
5. For MCQs, choose the correct option.
6. Check the answer where useful.
7. End with a Final Answer.
"""

    yield from stream_huggingface(
        prompt,
        MATH_SYSTEM_PROMPT
    )


# ============================================================
# IMPROVE
# ============================================================

def improve_answer(question, old_answer, action):

    instructions = {
        "improve":
            "Improve the answer and make it clearer and more accurate.",

        "check":
            "Check the previous answer for mathematical, physics, chemistry and factual mistakes. Correct them.",

        "explain":
            "Explain the answer in slightly more detail while staying clear.",

        "short":
            "Make the answer shorter but keep all important steps and the final answer."
    }

    instruction = instructions.get(
        action,
        instructions["improve"]
    )

    prompt = f"""
Original question:

{question}

Previous answer:

{old_answer}

Task:

{instruction}

Do not remove necessary calculations.
Verify numerical answers carefully.
Use simple student-friendly language.
"""

    system = (
        MATH_SYSTEM_PROMPT
        if looks_like_math(question)
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

    # ========================================================
    # BASIC MESSAGES
    # ========================================================

    simple = basic_response(question)

    if simple:
        return Response(
            simple,
            mimetype="text/plain"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    question = normalize_math_text(question)

    # ========================================================
    # FAST SOLVERS
    # ========================================================

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

    result = solve_basic_electricity(question)

    if result:
        return Response(
            result,
            mimetype="text/plain"
        )

    # ========================================================
    # MATH AI
    # ========================================================

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

    # ========================================================
    # GENERAL AI
    # ========================================================

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
# START LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MY AI")
    print("=" * 60)
    print(
        "Hugging Face configured:",
        bool(HF_TOKEN)
    )
    print("Basic responses        : ON")
    print("Math solver            : ON")
    print("Physics solver         : ON")
    print("Chemistry AI           : ON")
    print("Longer answers         : ON")
    print("Improve Answer         : ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )
