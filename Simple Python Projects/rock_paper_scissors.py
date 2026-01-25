import random
import sys

#While Loop Logic
while True:

#Basics
    print("Game Not Rigged :)")
    human = input("Rock, Paper or Scissors? Type '1' for Rock, '2' for Paper, '3' for Scissors ")

#Failsafe
    if human != "1" and human != "2" and human != "3":
        print("Error! Please enter '1', '2', or '3'.")
        continue
    

#Human Choice Namer
    if human == "1":
        print("You chose rock!")
    elif human == "2":
        print("You chose paper!")
    elif human == "3":
        print("You chose scissors!")
    
#Computer's Choice
    computer = random.randint(1,3)

    if computer == 1:
        print("Computer chooses rock")
    elif computer == 2: 
        print("Computer chooses paper")
    elif computer == 3:
        print("Computer chooses scissors")
    
#Logic
    if (computer == 1 and human == "1") or (computer == 2 and human == "2") or (computer == 3 and human == "3"):
        print("Tie!")
    elif (computer == 1 and human == "2") or (computer == 2 and human == "3") or (computer == 3 and human == "1"):
        print("You Won!")
    elif (human == "1" and computer == 2) or (human == "2" and computer == 3) or (human == "3" and computer == 1):
        print("You Lost!")

#End, and Loop Finish
    print("\n ~~~~Thanks for playing!~~~~")
    choice = input("Do you want to play again? (Y/N)")
    if choice.upper() != "Y":
        break
    else: 
        continue
