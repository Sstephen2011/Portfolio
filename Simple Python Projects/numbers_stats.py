count = 0
sum = 0
largest = None
smallest = None
evencount = 0
oddcount = 0

while True:
    num = input("Enter a number ")
    if num == "done":
        break
    try:
        num = int(num)
        count = count + 1 
        sum = sum + num
        if num % 2 == 0:
            evencount = evencount + 1
        if num % 2 != 0:
            oddcount = oddcount + 1
    except:
        print("Invalid Input")
        continue
    if smallest == None or num < smallest:
        smallest = num
    if largest == None or num > largest: 
        largest = num
    average = sum / count

print("Largest is", largest)
print("Smallest is", smallest)
print("Total is", sum)
print("Count is", count)
print("Average is", average)
print("Even count is", evencount)
print("Odd count is", oddcount)
