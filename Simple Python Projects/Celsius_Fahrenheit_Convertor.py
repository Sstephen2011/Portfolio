print("Welcome to the Fahrenheit to Celsius converter")
op = float(input("1. Celsius to Fahrenheit\n2. Fahrenheit to Celsius"))
a = float(input("Enter Number"))

if op == 1:
    result = (9/5 * a ) + 32
    print(result)
elif op == 2:
    result = (a - 32) * 5/9
    print(result)
else:
    print("INVALID INPUT!")
