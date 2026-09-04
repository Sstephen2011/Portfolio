import random
from flask import Flask, request, render_template_string

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Rock, Paper, Scissors</title>
<style>
  body { font-family: sans-serif; text-align: center; margin-top: 60px; color: white; background-color: #1a1a1a; }
  .button { font-size: 20px; padding: 10px 20px; margin: 5px; width: 180px; border-radius: 5px; cursor: pointer; }
  .win { color: lightgreen; }
  .lose { color: red; }
  .tie { color: lightgray; }
  .btn1 { background-color: #f44336; color: white; }
  .btn2 { background-color: #4CAF50; color: white; }
  .btn3 { background-color: #008CBA; color: white; }
  h1 { font-size: 40px; color: gold; }
</style>
</head>
<body>

 
  <h1>Rock, Paper or Scissors?</h1>
   <p>Game Not Rigged :)</p><br>

  <form method="POST">
    <button class="btn1 button" name="choice" value="1">Rock 🪨</button>
    <button class="btn2 button" name="choice" value="2">Paper 📄</button>
    <button class="btn3 button" name="choice" value="3">Scissors ✂️</button><br>
  </form>

  {% if human_line %}
    <p>{{ human_line }}</p>
    <p>{{ computer_line }}</p>
    <br>
    <h2 class="{{ result }}">{{ outcome }}</h2>
  {% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    human_line = computer_line = outcome = result = None

    if request.method == "POST":
        human = request.form.get("choice")

        # Failsafe
        if human not in ("1", "2", "3"):
            return render_template_string(PAGE)

        # Human Choice Namer
        if human == "1":
            human_line = "You chose rock!"
        elif human == "2":
            human_line = "You chose paper!"
        elif human == "3":
            human_line = "You chose scissors!"

        # Computer's Choice
        computer = random.randint(1, 3)
        if computer == 1:
            computer_line = "Computer chooses rock"
        elif computer == 2:
            computer_line = "Computer chooses paper"
        elif computer == 3:
            computer_line = "Computer chooses scissors"

        # Logic
        if (computer == 1 and human == "1") or (computer == 2 and human == "2") or (computer == 3 and human == "3"):
            outcome = "Tie!"
            result = "tie"
        elif (computer == 1 and human == "2") or (computer == 2 and human == "3") or (computer == 3 and human == "1"):
            outcome = "You Won!"
            result = "win"
        elif (human == "1" and computer == 2) or (human == "2" and computer == 3) or (human == "3" and computer == 1):
            outcome = "You Lost!"
            result = "lose"

    return render_template_string(
        PAGE,
        human_line=human_line,
        computer_line=computer_line,
        outcome=outcome,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
