import math 

print("Welcome to the calculator")

operation = input("Enter the operation you want to perform: \n 1.Addition \n 2.Subtraction \n 3.Multiplication \n 4.Division \n 5.Exponent \n 6.Factorial \n 7.Square Root \n 8.Cube Root")

if operation in ["1", "2", "3", "4", "5"]:
    first_number = float(input("Enter the first number:"))
    second_number = float(input("Enter the second number:"))

elif operation in ["6", "7", "8"]:
    first_number = float(input("Enter the number:"))


if operation == "1":
    result = float(first_number) + float(second_number)
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "2":
    result = float(first_number) - float(second_number)
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "3":
    result = float(first_number) * float(second_number)
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "4":
    result = float(first_number) / float(second_number)
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "5":
    result = float(first_number) ** float(second_number)
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "6":
    # Factorial is only valid for non-negative integers
    if first_number.is_integer() and first_number >= 0:
        result = math.factorial(int(first_number))  
        print(result)
    else:
        print("Factorial is only valid for non-negative integers.")
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "7":
    result = float(math.sqrt(first_number))
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

elif operation == "8":
    result = float(math.cbrt(first_number))
    print("The result is", result)
    if result > 0: 
        print("The Number is positive ")
    elif result == 0: 
        print(" The Number is zero ")
    elif result < 0: 
        print(" The Number is negative")
    if result.is_integer() and int(result) % 2 == 0:
        print("The Number is exactly divisible by two")
    else:
        print("The Number is not exactly divisible by two")

else: 
    print("INVALID INPUT! TRY AGAIN!")

