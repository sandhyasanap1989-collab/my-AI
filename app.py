from flask import Flask, render_template, request, Response, session
import os
import re
import math
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# SECURITY / SESSION
# ============================================================

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "my-ai-development-secret"
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
        api_key=HF_TOKEN
    )

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are My AI, a helpful study assistant.

You help with:
- Mathematics
- Physics
- Chemistry
- Biology
- Science
- General questions

IMPORTANT:
- Answer every requested part.
- Never stop halfway through a sentence.
- Never leave an equation unfinished.
- Show important calculations.
- Use simple student-friendly language.
- Do not invent missing information.
- For follow-up questions, use previous conversation context.
- Always finish with a clear Final Answer.
"""

MATH_SYSTEM_PROMPT = """
You are My AI's mathematics specialist.

IMPORTANT FORMATTING:
- Do not output raw LaTeX commands such as \\frac, \\boxed, \\begin or \\end.
- Do not put equations inside [ ... ].
- Use readable notation such as x², x³, x⁵ and √.
- Never replace √ with x.
- In this project "svg" means √.
- Show all important algebraic steps.
- Answer every part.
- Check the result.

Always finish with:

### Final Answer
**answer ✅**
"""

PHYSICS_SYSTEM_PROMPT = """
You are My AI's physics specialist.

For numerical questions:
1. Given
2. Formula
3. Substitution
4. Calculation
5. Verification
6. Final Answer

Use correct SI units and complete every requested part.
"""

CHEMISTRY_SYSTEM_PROMPT = """
You are My AI's chemistry specialist.

For numerical questions:
- Write/balance the reaction when required.
- Calculate molar masses carefully.
- Convert mL to L.
- Calculate moles and molarity correctly.
- Use stoichiometric ratios.
- Identify limiting reagent.
- Include units.
- Complete every numbered part.
"""

BIOLOGY_SYSTEM_PROMPT = """
You are My AI's biology specialist.

Explain biology accurately and clearly.
For processes, give the steps in order.
For comparisons, compare each requested point.
For genetics, show ratios and probabilities clearly.
Do not invent facts.
"""

# ============================================================
# BASIC RESPONSES
# ============================================================

def basic_response(message):
    q = message.strip().lower()

    if q in {
        "hi", "hello", "hey", "hii", "hiii",
        "helo", "hi!", "hello!", "hey!"
    }:
        return (
            "Hello! 👋 I'm My AI.\n\n"
            "Ask me Mathematics, Physics, Chemistry, Biology "
            "or general questions."
        )

    if q in {"good morning", "good afternoon", "good evening"}:
        return f"{q.title()}! 👋 How can I help you?"

    if q in {"how are you", "how are you?", "how r u", "how r u?"}:
        return "I'm doing great! 🤖 Ask me a difficult question."

    if q in {"who are you", "who are you?", "what are you", "what are you?"}:
        return (
            "I'm My AI 🤖, a study assistant for "
            "Mathematics, Physics, Chemistry and Biology."
        )

    if q in {"what can you do", "what can you do?", "help", "help me"}:
        return (
            "I can help with:\n\n"
            "🧮 Mathematics\n"
            "⚡ Physics\n"
            "🧪 Chemistry\n"
            "🧬 Biology\n"
            "📚 General Science"
        )

    if q in {"thanks", "thank you", "thankyou", "thx", "ty"}:
        return "You're welcome! 😊"

    if q in {"ok", "okay", "okk", "cool", "nice", "great"}:
        return "👍 Great! Ask your next question."

    if q in {"bye", "goodbye", "see you", "see ya"}:
        return "Goodbye! 👋"

    return None

# ============================================================
# FOLLOW-UP CONTEXT
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
        "what about india"
    }:
        return True

    return q.startswith((
        "and ",
        "what about ",
        "how about ",
        "also ",
        "compare ",
        "then "
    ))

def contextual_question(question):
    history = session.get("history", [])

    if not history or not is_followup(question):
        return question

    previous = history[-1]

    return (
        f"Previous question:\n{previous['question']}\n\n"
        f"Previous answer:\n{previous['answer']}\n\n"
        f"Follow-up:\n{question}\n\n"
        f"Answer the follow-up using the previous context."
    )

def save_history(question, answer):
    history = session.get("history", [])

    history.append({
        "question": question,
        "answer": answer[:5000]
    })

    session["history"] = history[-6:]
    session.modified = True

# ============================================================
# HELPERS
# ============================================================

def fmt(value, decimals=6):
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

def normalize_math_text(text):
    text = text.replace("svg", "√")
    text = text.replace("SVG", "√")

    for c in [
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060"
    ]:
        text = text.replace(c, "")

    return text

def detect_subject(question):
    q = question.lower()

    math_words = [
        "equation", "quadratic", "algebra", "surds",
        "sqrt", "√", "svg", "trigonometry", "sin",
        "cos", "tan", "geometry", "probability",
        "permutation", "combination", "sequence",
        "series", "polynomial", "coordinate",
        "logarithm", "matrix", "calculus", "integral"
    ]

    physics_words = [
        "force", "velocity", "acceleration", "momentum",
        "newton", "work", "energy", "power",
        "friction", "gravitation", "projectile",
        "current", "voltage", "resistance", "resistor",
        "circuit", "magnetic", "electric field",
        "lens", "mirror", "refraction", "heat",
        "temperature", "pressure", "density"
    ]

    chemistry_words = [
        "mole", "moles", "molar", "molarity", "molality",
        "stoichiometry", "limiting reagent", "oxidation",
        "reduction", "redox", "acid", "base", "ph",
        "salt", "equilibrium", "enthalpy",
        "electrochemistry", "organic", "alkane",
        "alkene", "alkyne", "benzene", "alcohol",
        "aldehyde", "ketone", "ion", "atom",
        "electron", "compound", "reaction"
    ]

    biology_words = [
        "cell", "mitosis", "meiosis", "chromosome",
        "gene", "genetics", "allele", "dna", "rna",
        "protein", "enzyme", "photosynthesis",
        "respiration", "plant", "animal", "tissue",
        "organ", "ecosystem", "ecology", "evolution",
        "hormone", "neuron", "digestion",
        "reproduction", "heredity", "blood", "heart",
        "kidney", "lung", "brain"
    ]

    scores = {
        "math": sum(x in q for x in math_words),
        "physics": sum(x in q for x in physics_words),
        "chemistry": sum(x in q for x in chemistry_words),
        "biology": sum(x in q for x in biology_words)
    }

    subject = max(scores, key=scores.get)

    if scores[subject] == 0:
        return "general"

    return subject

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
# MATH SOLVER
# ============================================================

def solve_power_recurrence(question):
    """
    Solves questions like:

    If x + 1/x = 5,
    find x^5 + 1/x^5.
    """

    q = normalize_math_text(question.lower())

    # x + 1/x = number
    given = re.search(
        r"x\s*\+\s*1\s*/\s*x\s*=\s*(-?\d+(?:\.\d+)?)",
        q
    )

    if not given:
        return None

    # Find x^n + 1/x^n
    target = re.search(
        r"x\s*(?:\^|\*\*)\s*(\d+)"
        r"\s*\+\s*1\s*/\s*x\s*(?:\^|\*\*)\s*\1",
        q
    )

    if not target:
        return None

    a = float(given.group(1))
    n = int(target.group(1))

    if n < 1 or n > 50:
        return None

    s = {
        0: 2,
        1: a
    }

    for k in range(1, n):
        s[k + 1] = a * s[k] - s[k - 1]

    lines = [
        "### Solution",
        "",
        f"Given:",
        "",
        f"x + 1/x = {fmt(a)}",
        "",
        "Let:",
        "",
        "Sₙ = xⁿ + 1/xⁿ",
        "",
        "We use:",
        "",
        "Sₙ₊₁ = (x + 1/x)Sₙ − Sₙ₋₁",
        "",
        f"S₀ = 2",
        "",
        f"S₁ = {fmt(a)}",
        ""
    ]

    for k in range(2, n + 1):
        lines.extend([
            f"**Step {k - 1}: Find S{k}**",
            "",
            f"S{k} = {fmt(a)}S{k - 1} − S{k - 2}",
            "",
            f"S{k} = {fmt(a)}({fmt(s[k - 1])}) − {fmt(s[k - 2])}",
            "",
            f"**S{k} = {fmt(s[k])}**",
            ""
        ])

    lines.extend([
        "### Final Answer",
        "",
        f"**xⁿ + 1/xⁿ = {fmt(s[n])} ✅**"
    ])

    return "\n".join(lines)

# ============================================================
# SURD SOLVER
# ============================================================

def solve_known_surds(question):
    q = normalize_math_text(question)
    compact = re.sub(r"\s+", "", q)

    if not (
        "√(5+2√6)" in compact
        and
        "√(5-2√6)" in compact
    ):
        return None

    a = math.sqrt(5 + 2 * math.sqrt(6))
    b = math.sqrt(5 - 2 * math.sqrt(6))

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

# ============================================================
# PHYSICS: CIRCUIT
# ============================================================

def solve_circuit(question):
    q = question.lower()

    if "series" not in q or "parallel" not in q:
        return None

    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:v|volt|volts)\b",
        q
    )

    resistor_values = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)",
            question,
            re.IGNORECASE
        )
    ]

    if not voltage_match or len(resistor_values) < 3:
        return None

    voltage = float(
        voltage_match.group(1)
    )

    match = re.search(
        r"parallel\s+combination\s+of\s+"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)"
        r"\s*(?:and|&)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)",
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

    rp = (r1 * r2) / (r1 + r2)
    rt = sum(others) + rp
    total_i = voltage / rt

    series_voltage = [
        total_i * r
        for r in others
    ]

    vp = voltage - sum(series_voltage)

    i1 = vp / r1
    i2 = vp / r2

    series_power = [
        total_i ** 2 * r
        for r in others
    ]

    p1 = vp ** 2 / r1
    p2 = vp ** 2 / r2

    total_power = voltage * total_i

    return f"""
### Solution

**1. Parallel resistance**

Rₚ = (R₁ × R₂)/(R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)})/({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(rp)} Ω**

**2. Total resistance**

Rₜ = {sum(others)} + {fmt(rp)}

**Rₜ = {fmt(rt)} Ω**

**3. Total current**

I = V/Rₜ

I = {fmt(voltage)}/{fmt(rt)}

**I = {fmt(total_i)} A**

**4. Voltage across series resistor(s)**

{chr(10).join(
    f"{fmt(r)} Ω resistor: V = IR = {fmt(v)} V"
    for r, v in zip(others, series_voltage)
)}

**5. Voltage across parallel section**

**Vₚ = {fmt(vp)} V**

**6. Current through {fmt(r1)} Ω**

I₁ = Vₚ/R₁

**I₁ = {fmt(i1)} A**

**7. Current through {fmt(r2)} Ω**

I₂ = Vₚ/R₂

**I₂ = {fmt(i2)} A**

**8. Power**

{chr(10).join(
    f"{fmt(r)} Ω resistor: **{fmt(p)} W**"
    for r, p in zip(others, series_power)
)}

{fmt(r1)} Ω resistor: **{fmt(p1)} W**

{fmt(r2)} Ω resistor: **{fmt(p2)} W**

**9. Total power**

P = VI

**P = {fmt(total_power)} W**

### Verification

Current in parallel:

{fmt(i1 + i2)} A = {fmt(total_i)} A ✅

Power is conserved:

**Sum of resistor powers = {fmt(
    sum(series_power) + p1 + p2
)} W**

**Battery power = {fmt(total_power)} W ✅**

### Final Answer

- **Parallel resistance = {fmt(rp)} Ω**
- **Total resistance = {fmt(rt)} Ω**
- **Total current = {fmt(total_i)} A**
- **Parallel voltage = {fmt(vp)} V**
- **Total power = {fmt(total_power)} W ✅**
""".strip()

# ============================================================
# PHYSICS: SIMPLE ELECTRICITY
# ============================================================

def solve_basic_electricity(question):
    q = question.lower()

    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:v|volt|volts)\b",
        q
    )

    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:w|watt|watts)\b",
        q
    )

    resistors = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)",
            question,
            re.IGNORECASE
        )
    ]

    voltage = (
        float(voltage_match.group(1))
        if voltage_match else None
    )

    power = (
        float(power_match.group(1))
        if power_match else None
    )

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

I = {fmt(power)}/{fmt(voltage)}

**I = {fmt(current)} A**

### Final Answer

**{fmt(current)} A ✅**
""".strip()

    if (
        voltage is not None
        and power is not None
        and "resistance" in q
    ):
        resistance = voltage ** 2 / power

        return f"""
### Solution

P = V²/R

R = V²/P

R = {fmt(voltage)}²/{fmt(power)}

**R = {fmt(resistance)} Ω**

### Final Answer

**{fmt(resistance)} Ω ✅**
""".strip()

    if (
        "series" in q
        and "resistance" in q
        and len(resistors) >= 2
    ):
        total = sum(resistors)

        return f"""
### Solution

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
        equivalent = (
            r1 * r2 / (r1 + r2)
        )

        return f"""
### Solution

Rₚ = (R₁ × R₂)/(R₁ + R₂)

Rₚ = ({fmt(r1)} × {fmt(r2)})/({fmt(r1)} + {fmt(r2)})

**Rₚ = {fmt(equivalent)} Ω**

### Final Answer

**{fmt(equivalent)} Ω ✅**
""".strip()

    return None

# ============================================================
# CHEMISTRY: FE + HCL
# ============================================================

def solve_iron_hcl(question):
    q = question.lower()

    if "fe" not in q or "hcl" not in q:
        return None

    if (
        "limiting" not in q
        and "h2" not in q
        and "hydrogen" not in q
        and "fecl2" not in q
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

    if not mass_match or not volume_match or not molarity_match:
        return None

    mass_fe = float(mass_match.group(1))
    volume_hcl = float(volume_match.group(1)) / 1000
    molarity_hcl = float(molarity_match.group(1))

    fe_moles = mass_fe / 56
    hcl_moles = molarity_hcl * volume_hcl

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
    fecl2_mass = fe_used * 126.9

    return f"""
### Solution

Reaction:

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

For 1 mol Fe, 2 mol HCl are required.

HCl required = 2 × {fmt(fe_moles)}

= **{fmt(hcl_required)} mol**

Available HCl = **{fmt(hcl_moles)} mol**

Therefore:

**Limiting reagent = {limiting}**

**4. Moles of H₂**

**n(H₂) = {fmt(h2_moles)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

V = {fmt(h2_moles)} × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M(FeCl₂) = 56 + 2(35.5)

= 127 g mol⁻¹

Mass = {fmt(fe_used)} × 127

**Mass(FeCl₂) = {fmt(fecl2_mass, 2)} g**

### Final Answer

- **Fe = {fmt(fe_moles)} mol**
- **HCl = {fmt(hcl_moles)} mol**
- **Limiting reagent = {limiting}**
- **H₂ = {fmt(h2_moles)} mol**
- **H₂ at STP = {fmt(h2_volume)} L**
- **FeCl₂ = {fmt(fecl2_mass, 2)} g ✅**
""".strip()

# ============================================================
# CHEMISTRY: NaCl MOLES
# ============================================================

def solve_nacl(question):
    q = question.lower()

    if "nacl" not in q:
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*nacl",
        q
    )

    if not mass_match:
        return None

    mass = float(
        mass_match.group(1)
    )

    molar_mass = 58.5
    moles = mass / molar_mass

    return f"""
### Solution

Molar mass of NaCl:

M = 23 + 35.5

M = 58.5 g mol⁻¹

Moles:

n = m/M

n = {fmt(mass)}/58.5

**n = {fmt(moles)} mol**

### Final Answer

**{fmt(moles)} mol NaCl ✅**
""".strip()

# ============================================================
# BIOLOGY / OTHER HARD QUESTIONS
# ============================================================
# Unusual biology questions are deliberately sent to the
# Biology specialist instead of using fragile hard-coded rules.

# ============================================================
# HUGGING FACE
# ============================================================

def stream_ai(question, subject):
    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ HF_TOKEN is missing.\n\n"
            "Only questions that need the online AI are affected.\n"
            "Check Render → Environment → HF_TOKEN."
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

            text = chunk.choices[0].delta.content

            if text:
                yield text

    except Exception as error:
        print("Hugging Face error:", repr(error))

        yield (
            "❌ I couldn't connect to the online AI right now.\n\n"
            "Check HF_TOKEN and the Hugging Face Inference permission."
        )

# ============================================================
# IMPROVE ANSWER
# ============================================================

def improve_answer(question, old_answer, action):
    actions = {
        "improve":
            "Improve the answer and make it clearer and more complete.",

        "check":
            "Check the previous answer for mistakes and correct them.",

        "explain":
            "Explain the answer in more detail.",

        "short":
            "Make the answer shorter while keeping important steps."
    }

    instruction = actions.get(
        action,
        actions["improve"]
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

Rules:
- Check every calculation.
- Answer every requested part.
- Keep mathematical formatting readable.
- Do not use broken raw LaTeX.
- Do not invent information.
- Finish the answer completely.
"""

    yield from stream_ai(
        prompt,
        subject
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

    original_question = data.get(
        "message",
        ""
    ).strip()

    if not original_question:
        return Response(
            "Please type a message.",
            mimetype="text/plain"
        )

    # BASIC
    simple = basic_response(
        original_question
    )

    if simple:
        save_history(
            original_question,
            simple
        )

        return Response(
            simple,
            mimetype="text/plain"
        )

    # FOLLOW-UP CONTEXT
    question = contextual_question(
        original_question
    )

    question = normalize_math_text(
        question
    )

    # ========================================================
    # EXACT LOCAL SOLVERS
    # IMPORTANT: These happen BEFORE HF_TOKEN is checked.
    # ========================================================

    result = solve_power_recurrence(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_known_surds(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_circuit(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_basic_electricity(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_iron_hcl(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    result = solve_nacl(
        question
    )

    if result:
        save_history(
            original_question,
            result
        )

        return Response(
            result,
            mimetype="text/plain"
        )

    # ========================================================
    # CLOUD AI FALLBACK
    # ========================================================

    subject = detect_subject(
        question
    )

    return Response(
        stream_ai(
            question,
            subject
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
    data = request.get_json(
        silent=True
    )

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
        improve_answer(
            question,
            old_answer,
            action
        ),
        mimetype="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MY AI")
    print("=" * 60)
    print("HF configured:", bool(HF_TOKEN))
    print("Math solver: ON")
    print("Physics solver: ON")
    print("Chemistry solver: ON")
    print("Biology AI: ON")
    print("Follow-up memory: ON")
    print("Basic responses: ON")
    print("Long answers: ON")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )
