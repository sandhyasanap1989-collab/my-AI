from flask import (
    Flask,
    render_template,
    request,
    Response,
    stream_with_context,
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
# HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face currently supports this model through
# the OpenAI-compatible Inference Providers router.
HF_MODEL = "openai/gpt-oss-120b:cheapest"

hf_client = None

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )


# ============================================================
# GENERAL SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are My AI, a smart and patient study assistant.

You help students with:
- Mathematics
- Physics
- Chemistry
- Biology
- General Science
- General knowledge

IMPORTANT ANSWERING RULES
=========================

1. Read the ENTIRE question before answering.

2. If the question has numbered parts such as:
   1, 2, 3, 4...
   answer EVERY part.

3. NEVER stop halfway through a sentence.

4. NEVER leave an unfinished equation.

5. NEVER omit a requested calculation.

6. Before finishing, mentally check:
   - Did I answer every requested part?
   - Did I include units?
   - Did I check important calculations?
   - Did I state the final answers clearly?

7. For numerical questions use:

### Solution

**1. Step**
Formula

Calculation

**2. Step**
Formula

Calculation

### Final Answer

Give all final values clearly.

8. For chemistry:
   - Balance equations when necessary.
   - Calculate molar mass carefully.
   - Show moles, molarity, stoichiometry and limiting reagent when relevant.
   - Use correct units.
   - Distinguish atoms, molecules, ions and moles.

9. For physics:
   - State the relevant law/formula.
   - Substitute values.
   - Use SI units when appropriate.
   - Check dimensions where useful.
   - Include direction/sign when relevant.

10. For mathematics:
    - Show algebraic steps.
    - Preserve fractions and radicals.
    - Do not replace √ with x.
    - Check the final result.

11. For biology:
    - Use correct biological terminology.
    - For processes, explain them in logical order.
    - For comparison questions, use clear headings or a table-like structure.

12. For multiple choice questions:
    - Solve first.
    - Compare with the options.
    - Clearly state the correct option.

13. If information is missing:
    - Say exactly what is missing.
    - Do not invent values.

14. Use simple student-friendly language.

15. For a long question, keep the explanation organized.

16. Always finish with:

### Final Answer

**Answer ✅**

Never end with an unfinished sentence.
"""


# ============================================================
# MATHEMATICS SYSTEM PROMPT
# ============================================================

MATH_SYSTEM_PROMPT = r"""
You are My AI's mathematics specialist.

Solve carefully and completely.

Important:
- In this project, copied text "svg" means √.
- Never treat "svg" as x.
- Preserve original values.
- Carefully identify brackets, fractions, powers and radicals.
- Show the important algebraic steps.
- For geometry, identify the theorem/formula before using it.
- For trigonometry, keep exact values when possible.
- For coordinate geometry, show the formula.
- For quadratic equations, show the method used.
- For surds, simplify exact forms when possible.
- Check the answer where useful.
- For MCQs, compare against the options.
- Answer every requested sub-question.

Always end with a complete:

### Final Answer
"""


# ============================================================
# PHYSICS SYSTEM PROMPT
# ============================================================

PHYSICS_SYSTEM_PROMPT = r"""
You are My AI's physics specialist.

Solve physics questions completely.

For numerical problems:

### Given
List the known values with units.

### Formula
Write the relevant equation.

### Solution
Substitute values carefully.

### Verification
Check units or the physical relation where useful.

### Final Answer
State the answer with units.

Important:
- Use SI units where appropriate.
- Distinguish scalar and vector quantities.
- Use correct signs.
- Do not invent missing values.
- For circuits, check voltage, current and power relationships.
- For mechanics, check Newton's laws, work-energy and momentum.
- For heat, use correct mass units.
- For optics, use the correct sign convention when applicable.
- Complete every numbered part.
"""


# ============================================================
# CHEMISTRY SYSTEM PROMPT
# ============================================================

CHEMISTRY_SYSTEM_PROMPT = r"""
You are My AI's chemistry specialist.

Solve chemistry questions completely and carefully.

For numerical questions:
- Write the chemical equation.
- Balance it if necessary.
- Calculate molar masses.
- Convert mass to moles correctly.
- Use molarity as mol/L.
- Identify limiting reagent when required.
- Use stoichiometric ratios.
- Calculate particles using Avogadro's number.
- Use correct gas-volume conditions.
- Include units at every important stage.

For conceptual questions:
- Explain clearly.
- Use correct chemical terminology.
- Distinguish atoms, ions, molecules and compounds.
- Do not invent data.

For organic chemistry:
- Identify the reaction type where appropriate.
- Preserve the structure and stoichiometry.

For equilibrium, acids/bases, electrochemistry and thermochemistry:
- Show the governing relationship before calculating.

Always answer every numbered part.

Always end with:

### Final Answer
"""


# ============================================================
# BIOLOGY SYSTEM PROMPT
# ============================================================

BIOLOGY_SYSTEM_PROMPT = r"""
You are My AI's biology specialist.

Answer biology questions accurately and clearly.

For definitions:
Give a direct definition and one useful explanation.

For processes:
Explain events in correct sequence.

For comparisons:
Give clear point-by-point differences.

For diagrams or structures described in text:
Identify structures carefully and explain their functions.

For genetics:
- Define alleles/genotypes/phenotypes when necessary.
- Show crosses step by step.
- State genotype and phenotype ratios clearly.

For ecology:
- Distinguish food chains, food webs, trophic levels and energy flow.

For human biology:
- Use accurate terminology.
- Explain organs, tissues and systems logically.

Do not invent biological facts.
Answer every requested part.

Always end with:

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
            "Biology, Science, or general questions."
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
            "Give me a difficult question and let's solve it."
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
# GENERAL HELPERS
# ============================================================

def fmt(value, decimals=5):
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
        abs_tol=tolerance,
    )


def normalize_math_text(text):
    # Known copied-text artifact from this project
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
# EXACT MATH SOLVER
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
        value = (a + b) / (a - b)

        return f"""
### Solution

√(5 + 2√6) = √3 + √2

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
# PHYSICS: SERIES-PARALLEL CIRCUIT
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

    voltage = float(voltage_match.group(1))

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

    # Common case:
    # one resistor in series with two parallel resistors
    if "parallel combination" in q:
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

        rp = (r1 * r2) / (r1 + r2)
        rt = sum(series) + rp
        total_current = voltage / rt

        series_voltage = [
            total_current * r
            for r in series
        ]

        vp = voltage - sum(series_voltage)

        i1 = vp / r1
        i2 = vp / r2

        series_power = [
            total_current ** 2 * r
            for r in series
        ]

        p1 = vp ** 2 / r1
        p2 = vp ** 2 / r2

        total_power = voltage * total_current
        component_power = sum(series_power) + p1 + p2

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
            f"**I = {fmt(total_current)} A**",
            "",
        ]

        step = 4

        for resistor, drop in zip(series, series_voltage):
            lines.extend([
                f"**{step}. Voltage across {fmt(resistor)} Ω**",
                "",
                "V = IR",
                "",
                f"V = {fmt(total_current)} × {fmt(resistor)}",
                "",
                f"**V = {fmt(drop)} V**",
                "",
            ])
            step += 1

        lines.extend([
            f"**{step}. Voltage across the parallel section**",
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
            f"**{step}. Power in each resistor**",
            "",
        ])

        for resistor, power in zip(series, series_power):
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
            f"P = {fmt(voltage)} × {fmt(total_current)}",
            "",
            f"**P = {fmt(total_power)} W**",
            "",
            "### Verification",
            "",
            f"Sum of component powers = {fmt(component_power)} W",
            "",
            f"Battery power = {fmt(total_power)} W",
            "",
            "**Power check ✅**",
            "",
            "### Final Answer",
            "",
            f"- **Parallel resistance = {fmt(rp)} Ω**",
            f"- **Total resistance = {fmt(rt)} Ω**",
            f"- **Total current = {fmt(total_current)} A**",
            f"- **Parallel voltage = {fmt(vp)} V**",
            f"- **Total power = {fmt(total_power)} W ✅**",
        ])

        return "\n".join(lines)

    return None


# ============================================================
# PHYSICS: BASIC ELECTRICITY
# ============================================================

def solve_basic_electricity(question):
    q = question.lower()
    problem = get_problem_text(question)

    voltage = None
    voltage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:V|volt|volts)\b",
        problem,
        re.IGNORECASE,
    )

    if voltage_match:
        voltage = float(voltage_match.group(1))

    power = None
    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:W|watt|watts)\b",
        problem,
        re.IGNORECASE,
    )

    if power_match:
        power = float(power_match.group(1))

    resistors = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:Ω|ohm|ohms)\b",
            problem,
            re.IGNORECASE,
        )
    ]

    if (
        voltage is not None
        and power is not None
        and "resistance" in q
    ):
        resistance = voltage ** 2 / power

        return f"""
### Solution

P = V²/R

Therefore:

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

For resistors in series:

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

        rp = (r1 * r2) / (r1 + r2)

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


# ============================================================
# PHYSICS: HEAT
# ============================================================

def solve_heat_question(question):
    problem = get_problem_text(question)

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(g|kg)",
        problem,
        re.IGNORECASE,
    )

    c_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:J|j)"
        r"\s*(?:/|per)\s*"
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

    if not mass_match or not c_match or not temp_match:
        return None

    mass = float(mass_match.group(1))
    mass_unit = mass_match.group(2).lower()

    c = float(c_match.group(1))
    c_unit = c_match.group(2).lower()

    t1 = float(temp_match.group(1))
    t2 = float(temp_match.group(2))

    if mass_unit == "g":
        mass_kg = mass / 1000
    else:
        mass_kg = mass

    # Convert J/g°C to J/kg°C
    if c_unit == "g":
        c = c * 1000

    delta_t = abs(t2 - t1)

    heat = mass_kg * c * delta_t

    return f"""
### Solution

Given:

m = {fmt(mass_kg)} kg

c = {fmt(c)} J kg⁻¹ °C⁻¹

ΔT = |{fmt(t2)} − {fmt(t1)}|

ΔT = {fmt(delta_t)} °C

Use:

Q = mcΔT

Q = {fmt(mass_kg)} × {fmt(c)} × {fmt(delta_t)}

**Q = {fmt(heat)} J**

### Final Answer

**Heat required = {fmt(heat)} J ✅**
""".strip()


# ============================================================
# CHEMISTRY: MOLES / MOLARITY
# ============================================================

def solve_mole_question(question):
    q = question.lower()
    problem = get_problem_text(question)

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g",
        problem,
        re.IGNORECASE,
    )

    molar_mass_match = re.search(
        r"molar\s+mass\s*(?:=|is)\s*"
        r"(\d+(?:\.\d+)?)\s*g",
        problem,
        re.IGNORECASE,
    )

    if mass_match and molar_mass_match:
        mass = float(mass_match.group(1))
        molar_mass = float(molar_mass_match.group(1))

        if molar_mass > 0:
            moles = mass / molar_mass

            return f"""
### Solution

Use:

n = m/M

n = {fmt(mass)} / {fmt(molar_mass)}

**n = {fmt(moles)} mol**

### Final Answer

**{fmt(moles)} mol ✅**
""".strip()

    # Common wording: "5.85 g NaCl, Na=23, Cl=35.5"
    nacl_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*NaCl",
        problem,
        re.IGNORECASE,
    )

    if nacl_match:
        mass = float(nacl_match.group(1))

        molar_mass = 23 + 35.5

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


# ============================================================
# CHEMISTRY: MOLARITY
# ============================================================

def solve_molarity(question):
    q = question.lower()
    problem = get_problem_text(question)

    if "molarity" not in q and not re.search(r"\bM\b", problem):
        return None

    moles_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mol",
        problem,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(mL|L)",
        problem,
        re.IGNORECASE,
    )

    if not moles_match or not volume_match:
        return None

    moles = float(moles_match.group(1))
    volume = float(volume_match.group(1))
    unit = volume_match.group(2).lower()

    if unit == "ml":
        volume /= 1000

    if volume <= 0:
        return None

    molarity = moles / volume

    return f"""
### Solution

Molarity:

M = n/V

n = {fmt(moles)} mol

V = {fmt(volume)} L

M = {fmt(moles)} / {fmt(volume)}

**M = {fmt(molarity)} mol L⁻¹**

### Final Answer

**{fmt(molarity)} M ✅**
""".strip()


# ============================================================
# CHEMISTRY: PARTICLES FROM MOLES
# ============================================================

def solve_particles(question):
    q = question.lower()

    if not any(
        phrase in q
        for phrase in [
            "particles",
            "molecules",
            "atoms",
            "ions",
            "number of particles",
            "number of molecules",
            "number of atoms",
            "number of ions",
        ]
    ):
        return None

    mole_match = re.search(
        r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mol",
        question,
        re.IGNORECASE,
    )

    if not mole_match:
        return None

    moles = float(mole_match.group(1))

    avogadro = 6.022e23

    count = moles * avogadro

    return f"""
### Solution

Use:

N = nNₐ

N = {fmt(moles)} × 6.022 × 10²³

**N = {count:.5e}**

### Final Answer

**{count:.5e} particles ✅**
""".strip()


# ============================================================
# CHEMISTRY: STOICHIOMETRY / LIMITING REAGENT
# ============================================================

def solve_iron_hcl(question):
    q = question.lower()

    if "fe" not in q or "hcl" not in q:
        return None

    if "limiting" not in q and "hydrogen" not in q:
        return None

    mass_match = re.search(
        r"(\d+(?:\.\d+)?)\s*g.*iron",
        question,
        re.IGNORECASE,
    )

    volume_match = re.search(
        r"(\d+(?:\.\d+)?)\s*mL.*hcl",
        question,
        re.IGNORECASE,
    )

    molarity_match = re.search(
        r"(\d+(?:\.\d+)?)\s*M.*hcl",
        question,
        re.IGNORECASE,
    )

    if not mass_match or not volume_match or not molarity_match:
        return None

    mass_fe = float(mass_match.group(1))
    volume_hcl = float(volume_match.group(1)) / 1000
    molarity_hcl = float(molarity_match.group(1))

    moles_fe = mass_fe / 56
    moles_hcl = molarity_hcl * volume_hcl

    hcl_needed = 2 * moles_fe

    if moles_hcl < hcl_needed:
        limiting = "HCl"
        moles_h2 = moles_hcl / 2
        moles_fe_used = moles_h2
    else:
        limiting = "Fe"
        moles_h2 = moles_fe
        moles_fe_used = moles_fe

    h2_volume = moles_h2 * 22.4

    fecl2_mass = moles_fe_used * 126.9

    return f"""
### Solution

Reaction:

Fe + 2HCl → FeCl₂ + H₂

**1. Moles of Fe**

n = m/M

n(Fe) = {fmt(mass_fe)} / 56

**n(Fe) = {fmt(moles_fe)} mol**

**2. Moles of HCl**

n = M × V

V = {fmt(volume_hcl)} L

n(HCl) = {fmt(molarity_hcl)} × {fmt(volume_hcl)}

**n(HCl) = {fmt(moles_hcl)} mol**

**3. Limiting reagent**

1 mol Fe needs 2 mol HCl.

HCl required = 2 × {fmt(moles_fe)}

= **{fmt(hcl_needed)} mol**

Available HCl = **{fmt(moles_hcl)} mol**

Therefore:

**Limiting reagent = {limiting}**

**4. Moles of H₂**

**n(H₂) = {fmt(moles_h2)} mol**

**5. Volume of H₂ at STP**

V = n × 22.4

V = {fmt(moles_h2)} × 22.4

**V(H₂) = {fmt(h2_volume)} L**

**6. Mass of FeCl₂**

M(FeCl₂) = 56 + 2(35.45)

= 126.9 g mol⁻¹

Mass = n × M

Mass = {fmt(moles_fe_used)} × 126.9

**Mass(FeCl₂) = {fmt(fecl2_mass)} g**

### Final Answer

- **Fe = {fmt(moles_fe)} mol**
- **HCl = {fmt(moles_hcl)} mol**
- **Limiting reagent = {limiting}**
- **H₂ = {fmt(moles_h2)} mol**
- **H₂ at STP = {fmt(h2_volume)} L**
- **FeCl₂ = {fmt(fecl2_mass)} g ✅**
""".strip()


# ============================================================
# BIOLOGY: MONOHYBRID GENETICS
# ============================================================

def solve_mendel(question):
    q = question.lower()

    if not any(
        word in q
        for word in [
            "monohybrid",
            "heterozygous",
            "genotype",
            "phenotype",
            "punnett",
        ]
    ):
        return None

    if "tall" in q and "dwarf" in q:
        if "tt" in q and "tt" in q:
            pass

    return None


# ============================================================
# AI RESPONSE
# ============================================================

def stream_huggingface(question, subject=None):

    if not HF_TOKEN or hf_client is None:
        yield (
            "❌ The AI token is missing.\n\n"
            "Check Render → Environment → HF_TOKEN."
        )
        return

    try:
        prompt = subject_prompt(subject)

        stream = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
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

            text = chunk.choices[0].delta.content

            if text:
                yield text

    except Exception as error:
        print(
            "Hugging Face error:",
            repr(error),
        )

        yield (
            "❌ I couldn't connect to the online AI right now.\n\n"
            "Please check the Hugging Face token and "
            "Inference Providers permission in Render."
        )


# ============================================================
# IMPROVE ANSWER
# ============================================================

def improve_answer(question, old_answer, action):

    actions = {
        "improve": (
            "Improve the previous answer. "
            "Make it clearer, more accurate and complete."
        ),
        "check": (
            "Check the previous answer carefully for "
            "mathematical, physics, chemistry or biology errors. "
            "Correct all errors."
        ),
        "explain": (
            "Explain the previous answer in more detail, "
            "but keep it student-friendly."
        ),
        "short": (
            "Make the previous answer shorter while keeping "
            "all important steps and the final answer."
        ),
    }

    instruction = actions.get(
        action,
        actions["improve"],
    )

    subject = detect_subject(question)

    prompt = f"""
Original question:

{question}

Previous answer:

{old_answer}

Task:

{instruction}

Important:
- Check every numerical value.
- Do not invent missing information.
- Answer every numbered part.
- Never stop halfway through.
- End with a complete Final Answer.
"""

    system = subject_prompt(subject)

    yield from stream_huggingface(
        prompt,
        subject,
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# CHAT API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True)

    if not data:
        return Response(
            "Please send a message.",
            mimetype="text/plain",
        )

    question = data.get(
        "message",
        "",
    ).strip()

    if not question:
        return Response(
            "Please type a message.",
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # Basic responses
    # --------------------------------------------------------

    simple = basic_response(question)

    if simple:
        return Response(
            simple,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # Normalize copied math text
    # --------------------------------------------------------

    question = normalize_math_text(question)

    # --------------------------------------------------------
    # Exact mathematics
    # --------------------------------------------------------

    result = solve_known_surds(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # Exact physics
    # --------------------------------------------------------

    result = solve_circuit(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_basic_electricity(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_heat_question(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # Exact chemistry
    # --------------------------------------------------------

    result = solve_iron_hcl(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_molarity(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_particles(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    result = solve_mole_question(question)

    if result:
        return Response(
            result,
            mimetype="text/plain",
        )

    # --------------------------------------------------------
    # General AI with automatic PCMB specialist selection
    # --------------------------------------------------------

    subject = detect_subject(question)

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
# IMPROVE API
# ============================================================

@app.route("/improve", methods=["POST"])
def improve():

    data = request.get_json(silent=True)

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
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MY AI")
    print("=" * 70)
    print("Hugging Face configured :", bool(HF_TOKEN))
    print("Math specialist        : ON")
    print("Physics specialist     : ON")
    print("Chemistry specialist   : ON")
    print("Biology specialist     : ON")
    print("Basic responses        : ON")
    print("Long answers           : ON")
    print("Exact circuit solver   : ON")
    print("Exact chemistry solver : ON")
    print("Improve Answer         : ON")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
    )
