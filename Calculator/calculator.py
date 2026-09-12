import math
from flask import Flask, render_template_string, request

app = Flask(__name__)

OP_NAMES = {
    "1": "Addition",
    "2": "Subtraction",
    "3": "Multiplication",
    "4": "Division",
    "5": "Exponent",
    "6": "Factorial",
    "7": "Square Root",
    "8": "Cube Root",
}

OP_SYMBOLS = {
    "1": "+",
    "2": "\u2212",
    "3": "\u00d7",
    "4": "\u00f7",
    "5": "x^y",
    "6": "n!",
    "7": "&radic;",
    "8": "&#8731;",
}


def format_result(value):
    """Format a result, switching to scientific notation if it has more than 10 digits."""
    if isinstance(value, int):
        digit_count = len(str(abs(value)))
        if digit_count > 10:
            return f"{value:.9e}"
        return str(value)

    # float
    formatted = f"{value:g}"
    digit_count = sum(ch.isdigit() for ch in formatted)
    if digit_count > 10 or abs(value) >= 1e10:
        return f"{value:.9e}"
    return formatted

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Calculator</title>
  <style>
    /* ---- Edit these to restyle the whole page ---- */
    :root {
      --page-bg: #000000;
      --calc-body: #2b2b2e;
      --screen-bg: #1c1d1f;
      --screen-text: #fff700;
      --screen-label: #b9bdc6;
      --key-bg: #e35835;
      --key-text: #f2f2f2;
      --key-hover: #b98b3c;
      --key-selected: #8e8415;
      --key-selected-text: #fffafa;
      --equals-bg: #f97e0b;
      --equals-hover: #ffb266;
      --clear-bg: #3a3a3d;
      --clear-hover: #4d4d51;
      --clear-text: #f2f2f2;
      --error-color: #ff6b6b;
      --result-color: #7daefc;
      --font-family: "Segoe UI", Arial, sans-serif;
      --mono-font: Arial;
      --radius: 16px;
    }

    body {
      font-family: var(--font-family);
      background-color: var(--page-bg);
      margin: 0;
      padding: 40px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .abba {
      font-weight: bold;
      color: #e3dddd;
    }

    h1 {
      background: -webkit-linear-gradient(#ffc800, #ff7300);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 3px #ffc800;
    }

    .calculator {
      width: 340px;
      max-width: 100%;
      background: var(--calc-body);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    /* ---- Screen ---- */
    .display {
      background: var(--screen-bg);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 16px;
    }

    .display-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 6px 0;
    }

    .display-row label {
      color: var(--screen-label);
      font-size: 12px;
    }

    .display-row input {
      background: #0f0f10;
      border: 1px solid #3a3a3d;
      border-radius: 6px;
      color: var(--screen-text);
      font-family: var(--mono-font);
      font-size: 22px;
      text-align: right;
      width: 100%;
      padding: 8px 12px;
      box-sizing: border-box;
    }

    .display-row input:focus {
      outline: none;
      border-color: var(--equals-bg);
    }

    /* Hide the second number field for unary operations (Factorial, Square Root, Cube Root) */
    .calculator:has(input[name="operation"][value="6"]:checked) .second-number-row,
    .calculator:has(input[name="operation"][value="7"]:checked) .second-number-row,
    .calculator:has(input[name="operation"][value="8"]:checked) .second-number-row {
      display: none;
    }

    /* ---- Keypad ---- */
    .keypad {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-bottom: 14px;
    }

    .key {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: var(--key-bg);
      color: var(--key-text);
      border-radius: 10px;
      padding: 10px 4px;
      cursor: pointer;
      user-select: none;
      transition: background-color 0.15s ease;
    }

    .key input {
      display: none;
    }

    .key span {
      font-size: 16px;
      font-weight: bold;
      line-height: 1;
    }

    .key small {
      font-size: 12px;
      margin-top: 4px;
      color: inherit;
      opacity: 0.8;
      text-align: center;
    }

    .key:hover {
      background: var(--key-hover);
    }

    .key:has(input:checked) {
      background: var(--key-selected);
      color: var(--key-selected-text);
    }

    /* ---- Action buttons row (Calculate / Clear) ---- */
    .actions {
      display: flex;
      gap: 8px;
    }

    .equals,
    .clear-btn {
      flex: 1;
      padding: 14px;
      border: none;
      border-radius: 10px;
      font-size: 16px;
      font-weight: bold;
      cursor: pointer;
      text-align: center;
      box-sizing: border-box;
    }

    /* ---- Equals / submit button ---- */
    .equals {
      background: var(--equals-bg);
      color: #1c1d1f;
    }

    .equals:hover {
      background: var(--equals-hover);
    }

    /* ---- Clear button ---- */
    .clear-btn {
      background: var(--clear-bg);
      color: var(--clear-text);
      text-decoration: none;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .clear-btn:hover {
      background: var(--clear-hover);
    }

    /* ---- Result / error readouts ---- */
    .result,
    .error {
      margin-top: 14px;
      margin-bottom: 0;
      padding: 14px 16px;
      border-radius: 8px;
      font-family: var(--mono-font);
      text-align: center;
      font-weight: bold;
    }

    .result {
      background: rgba(125, 140, 252, 0.12);
      color: var(--result-color);
      font-size: 28px;
      letter-spacing: 0.5px;
      text-shadow: 0 0 8px rgba(125, 140, 252, 0.5);
      word-break: break-word;
    }

    .error {
      background: rgba(255, 107, 107, 0.1);
      color: var(--error-color);
      font-size: 14px;
    }
  </style>
</head>
<body>

  <h1>Calculator</h1>

  <form method="POST" action="/" class="calculator">

    <div class="display">
      <div class="display-row">
        <label for="first_number" class="abba">First number</label>
        <input type="number" step="any" name="first_number" id="first_number"
          value="{{ first_number if first_number is not none else '' }}">
      </div>
      <div class="display-row second-number-row">
        <label for="second_number" class="abba">Second number</label>
        <input type="number" step="any" name="second_number" id="second_number"
          value="{{ second_number if second_number is not none else '' }}">
      </div>
    </div>

    <div class="keypad">
      {% for key, name in op_names.items() %}
        <label class="key">
          <input type="radio" name="operation" value="{{ key }}"
            {% if selected_operation == key %}checked{% endif %}>
          <span>{{ op_symbols[key] | safe }}</span>
          <small>{{ name }}</small>
        </label>
      {% endfor %}
    </div>

    <div class="actions">
      <button type="submit" class="equals">Calculate</button>
      <a href="/" class="clear-btn">Clear</a>
    </div>

    {% if error %}
      <p class="error">{{ error }}</p>
    {% elif result is not none %}
      <p class="result">{{ result }}</p>
    {% endif %}

  </form>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def calculator():
    result = None
    error = None
    operation = "1"  # Default to Addition so a key is always pre-selected
    first_number = None
    second_number = None

    if request.method == "POST":
        operation = request.form.get("operation")
        first_number_raw = request.form.get("first_number")
        second_number_raw = request.form.get("second_number")

        # ---- Same structure as the original script, just fed by form fields ----

        if operation in ["1", "2", "3", "4", "5"]:
            try:
                first_number = float(first_number_raw)
                second_number = float(second_number_raw)
            except (TypeError, ValueError):
                error = "Please enter valid numbers for both fields."

        elif operation in ["6", "7", "8"]:
            try:
                first_number = float(first_number_raw)
            except (TypeError, ValueError):
                error = "Please enter a valid number."

        if error is None:
            if operation == "1":
                result = float(first_number) + float(second_number)

            elif operation == "2":
                result = float(first_number) - float(second_number)

            elif operation == "3":
                result = float(first_number) * float(second_number)

            elif operation == "4":
                if second_number == 0:
                    error = "Cannot divide by zero."
                else:
                    result = float(first_number) / float(second_number)

            elif operation == "5":
                result = float(first_number) ** float(second_number)

            elif operation == "6":
                # Factorial is only valid for non-negative integers
                if first_number.is_integer() and first_number >= 0:
                    result = math.factorial(int(first_number))
                else:
                    error = "Factorial is only valid for non-negative integers."

            elif operation == "7":
                if first_number < 0:
                    error = "Cannot take the square root of a negative number."
                else:
                    result = float(math.sqrt(first_number))

            elif operation == "8":
                result = float(math.cbrt(first_number))

            else:
                error = "INVALID INPUT! TRY AGAIN!"

    return render_template_string(
        PAGE,
        op_names=OP_NAMES,
        op_symbols=OP_SYMBOLS,
        selected_operation=operation,
        first_number=first_number,
        second_number=second_number,
        result=format_result(result) if result is not None else None,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
