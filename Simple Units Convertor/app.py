from flask import Flask, render_template_string, request

app = Flask(__name__)

def convert(value, conversion):
    if conversion == "kilometersToMiles":
        return value * 0.621371
    elif conversion == "milesToKilometers":
        return value / 0.621371
    elif conversion == "feetToMeters":
        return value * 0.3048
    elif conversion == "metersToFeet":
        return value / 0.3048
    return value

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Convertor</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #0f1117;
            color: #cbd5e1;
            font-size: 18px;
        }

        nav {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 0 2rem;
            height: 60px;
            background-color: #0b0d13;
            color: white;
            border-bottom: 1px solid #1e2330;
            box-shadow: 0 1px 8px rgba(0, 0, 0, 0.4);
        }

        nav h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-right: auto;
            color: #e2e8f0;
        }

        nav a {
            color: #94a3b8;
            text-decoration: none;
            font-size: 17px;
            font-weight: 500;
            padding: 0.4rem 0.75rem;
            border-radius: 6px;
        }

        nav a:hover {
            color: #e2e8f0;
        }

        nav a.active {
            color: #6366f1;
        }

        .convertor {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: #13161f;
            border-radius: 8px;
            border: 1px solid #1e2330;
        }

        input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #1e2330;
            border-radius: 4px;
            background-color: #1a1d28;
            color: #e2e8f0;
            font-size: 17px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        input:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 5px rgba(99, 102, 241, 0.35);
        }

        select {
            width: 100%;
            padding: 16px;
            padding-left: 5px;
            border: 1px solid #1e2330;
            border-radius: 4px;
            background-color: #1a1d28;
            color: #e2e8f0;
            font-size: 17px;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        button {
            width: 100%;
            padding: 0.75rem;
            background-color: #6366f1;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 17px;
        }

        button:hover {
            background-color: #4f52d6;
        }

        #result {
            margin-top: 1rem;
            font-size: 1.2rem;
            color: #e2e8f0;
            text-align: center;
        }

        span {
            font-weight: 600;
            color: #6366f1;
        }

        @media (max-width: 850px) {
            nav {
                padding: 0 1rem;
                gap: 1rem;
                height: auto;
                flex-wrap: wrap;
                padding-top: 0.75rem;
                padding-bottom: 0.75rem;
            }

            nav h1 {
                font-size: 1.4rem;
                width: 100%;
            }

            nav a {
                font-size: 14px;
                padding: 0.3rem 0.5rem;
            }

            .convertor {
                margin: 1rem;
                padding: 1.25rem;
            }

            input, select {
                font-size: 17px;
            }

            button {
                font-size: 17px;
            }

            #result {
                font-size: 1rem;
            }
        }
    </style>
</head>
<body>
    <nav>
        <h1>Units Convertor</h1>
    </nav>
    <div class="convertor">
        <form method="post">
            <label for="inputValue">Enter value:</label>
            <input type="text" id="inputValue" name="inputValue" value="{{ input_value or '' }}" required>
            <label for="conversionType">Select conversion type:</label>
            <select id="conversionType" name="conversionType" required>
                <option value="feetToMeters"      {% if conversion_type == "feetToMeters"      %}selected{% endif %}>Feet to Meters</option>
                <option value="metersToFeet"      {% if conversion_type == "metersToFeet"      %}selected{% endif %}>Meters to Feet</option>
                <option value="kilometersToMiles" {% if conversion_type == "kilometersToMiles" %}selected{% endif %}>Kilometers to Miles</option>
                <option value="milesToKilometers" {% if conversion_type == "milesToKilometers" %}selected{% endif %}>Miles to Kilometers</option>
            </select>
            <button type="submit">Convert</button>
        </form>
        <div id="result">
            {% if result is not none %}
                <span>Result:</span> {{ result }}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    conversion_type = None
    input_value = None

    if request.method == "POST":
        try:
            input_value = request.form["inputValue"]
            conversion_type = request.form["conversionType"]
            if float(input_value) < 0:
                result = "Error: Please enter a positive value."
            else:
                result = round(convert(float(input_value), conversion_type), 2)
        except ValueError:
            result = "Error: Please enter a valid number."

    return render_template_string(TEMPLATE, result=result, conversion_type=conversion_type, input_value=input_value)

if __name__ == "__main__":
    app.run(debug=True)
