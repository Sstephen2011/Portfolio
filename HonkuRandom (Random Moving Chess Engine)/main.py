# Hello! This is a random move playing chess bot developed by StephenPS

import chess #Importing chess library
import random# Importing random library

def uci_loop(): #MAIN UCI LOOP
    board = chess.Board() #defining board object
    while True:
        line = input().strip() # Reads innput from GUI 

        if line == "uci":  # UCI Information
            print("id name HonkuRandom") 
            print("id author StephenPS")
            print("uciok")

        elif line == "isready": #UCI COMMAND
            print("readyok")

        elif line.startswith("position"): #Uci command to set position
            tokens = line.split()
            move_index = tokens.index("moves") if "moves" in tokens else -1 #Index of all moves

            if "startpos" in tokens:
                board.set_fen(chess.STARTING_FEN) #If startpos is in input, it sets the board to starting position
            elif "fen" in tokens:
                fen = " ".join(tokens[2:8])  # FEN has 6 parts, 2-8. 
                board.set_fen(fen) #Setting the board according to the FEN 

            if move_index != -1: # If there are moves to play (-1 indicates that there are no moves to play)
                moves = tokens[move_index + 1:]
                for move in moves:
                    board.push_uci(move) #Pushes move

        elif line.startswith("go"):
            movetime = 1000  # Default move time: 1 second
            tokens = line.split()

            if "movetime" in tokens: #Checks movetime in input
                idx = tokens.index("movetime")
                if idx + 1 < len(tokens):
                    movetime = int(tokens[idx + 1]) 

            elif "wtime" in tokens and "btime" in tokens:
                try:
                    wtime = int(tokens[tokens.index("wtime") + 1]) #Wtime
                    btime = int(tokens[tokens.index("btime") + 1]) #Btime
                    movetime = min(wtime, btime) // 30 #Movetime Calculation
                except:
                    pass

            legal_moves = list(board.legal_moves) #List of legal moves
            if legal_moves:
                move = random.choice(legal_moves) #This is the brain of the bot :p
                print(f"bestmove {move.uci()}") #Prints the best move in UCI format
            else:
                print("bestmove 0000")  # Null move 

        elif line == "ucinewgame":
            board.reset() #Resets the board after new game command

        elif line == "quit":
            break #Quit command

if __name__ == "__main__":
    uci_loop()

#If you want to play against this bot, you can copy the code and make it an .exe file using pyinstaller, and play with any GUI
#If you want to play it online, check @HonkuRandom on lichess.org
#This bot is developed by StephenPS
