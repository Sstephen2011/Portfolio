"""
HonkuWeak Chess Engine - UCI Compatible
========================================
Strong single-file Python chess engine targeting 2400+ Lichess blitz.

Features:
  - Negamax + Alpha-Beta + PVS (Principal Variation Search)
  - Iterative Deepening
  - Aspiration Windows
  - Null Move Pruning (NMP)
  - Late Move Reductions (LMR)
  - Futility Pruning
  - Razoring
  - Delta Pruning (in QSearch)
  - Transposition Table (Zobrist, 128MB default)
  - Move Ordering: TT move, MVV-LVA, Killers (2 slots), History, Countermove
  - Evaluation: Material, PST, Mobility, King Safety, Pawn Structure,
                Passed Pawns, Rook on open/semi-open file, Bishop pair
  - Time Management: adaptive, handles sudden death / increment / movestogo
  - Pondering (infinite search, stopped by UCI stop)

To compile to .exe:
  pip install pyinstaller python-chess
  pyinstaller --onefile blastfish.py
"""

import sys
import time
import threading
from typing import Optional

try:
    import chess
    import chess.polyglot
except ImportError:
    print("pip install python-chess", file=sys.stderr)
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

ENGINE_NAME   = "HonkuWeak"
ENGINE_AUTHOR = "StephenPS"
VERSION       = "2.0"

INF         = 10_000_000
MATE_SCORE  = 9_000_000
DRAW_SCORE  = 0
MAX_DEPTH   = 64
MAX_PLY     = 128

# Piece values in centipawns (middlegame / endgame)
MG_VALUES = {
    chess.PAWN:   82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK:   477,
    chess.QUEEN:  1025,
    chess.KING:   0,
}
EG_VALUES = {
    chess.PAWN:   94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK:   512,
    chess.QUEEN:  936,
    chess.KING:   0,
}

# Game phase weights per piece (for tapered eval)
PHASE_WEIGHTS = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  4,
    chess.KING:   0,
}
MAX_PHASE = 24  # 4 knights + 4 bishops + 4 rooks + 2 queens = 4+4+8+8 = 24

# ───────────────────────────────────────────────────────────────────────────────
# PIECE-SQUARE TABLES  (white perspective, a1=0, h8=63)
# Stored rank 1..8 bottom to top
# ───────────────────────────────────────────────────────────────────────────────

# Helper: PST stored as rank8..rank1 (top to bottom visually), we'll flip
def _make_pst(table):
    """Flip ranks so index 0 = a1."""
    out = [0] * 64
    for sq in range(64):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        row  = 7 - rank  # visual row (0=rank8, 7=rank1)
        out[sq] = table[row * 8 + file]
    return out

_MG_PAWN_RAW = [
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_EG_PAWN_RAW = [
     0,  0,  0,  0,  0,  0,  0,  0,
   178,173,158,134,147,132,165,187,
    94,100, 85, 67, 56, 53, 82, 84,
    32, 24, 13,  5, -2,  4, 17, 17,
    13,  9, -3, -7, -7, -8,  3, -1,
     4,  7, -6,  1,  0, -5, -1, -8,
    13,  8,  8, 10, 13,  0,  2, -7,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_MG_KNIGHT_RAW = [
   -167,-89,-34,-49, 61,-97,-15,-107,
    -73,-41, 72, 36, 23, 62,  7, -17,
    -47, 60, 37, 65, 84,129, 73,  44,
     -9, 17, 19, 53, 37, 69, 18,  22,
    -13,  4, 16, 13, 28, 19, 21,  -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29,-53,-12, -3, -1, 18,-14, -19,
   -105,-21,-58,-33,-17,-28,-19, -23,
]
_EG_KNIGHT_RAW = [
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20, 10,  9, -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64,
]
_MG_BISHOP_RAW = [
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21,
]
_EG_BISHOP_RAW = [
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17,
]
_MG_ROOK_RAW = [
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26,
]
_EG_ROOK_RAW = [
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  3,  8,  3,
      7,  7,  7,  5,  4, -3, -5, -3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8, -11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20,
]
_MG_QUEEN_RAW = [
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14,  2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50,
]
_EG_QUEEN_RAW = [
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41,
]
_MG_KING_RAW = [
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14,
]
_EG_KING_RAW = [
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4,-5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43,
]

MG_PST = {
    chess.PAWN:   _make_pst(_MG_PAWN_RAW),
    chess.KNIGHT: _make_pst(_MG_KNIGHT_RAW),
    chess.BISHOP: _make_pst(_MG_BISHOP_RAW),
    chess.ROOK:   _make_pst(_MG_ROOK_RAW),
    chess.QUEEN:  _make_pst(_MG_QUEEN_RAW),
    chess.KING:   _make_pst(_MG_KING_RAW),
}
EG_PST = {
    chess.PAWN:   _make_pst(_EG_PAWN_RAW),
    chess.KNIGHT: _make_pst(_EG_KNIGHT_RAW),
    chess.BISHOP: _make_pst(_EG_BISHOP_RAW),
    chess.ROOK:   _make_pst(_EG_ROOK_RAW),
    chess.QUEEN:  _make_pst(_EG_QUEEN_RAW),
    chess.KING:   _make_pst(_EG_KING_RAW),
}

# ───────────────────────────────────────────────────────────────────────────────
# TRANSPOSITION TABLE
# ───────────────────────────────────────────────────────────────────────────────

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "best_move", "age")
    def __init__(self, key, depth, score, flag, best_move, age):
        self.key       = key
        self.depth     = depth
        self.score     = score
        self.flag      = flag
        self.best_move = best_move
        self.age       = age

class TT:
    def __init__(self, mb: int = 128):
        self.size  = (mb * 1024 * 1024) // 48
        self.table = [None] * self.size
        self.age   = 0

    def index(self, key: int) -> int:
        return key % self.size

    def get(self, key: int) -> Optional[TTEntry]:
        e = self.table[self.index(key)]
        if e is not None and e.key == key:
            return e
        return None

    def put(self, key: int, depth: int, score: int, flag: int, best_move, age: int):
        idx = self.index(key)
        e   = self.table[idx]
        # Replace if: empty, same key, older entry, or shallower depth
        if (e is None or e.key == key or e.age < age or e.depth <= depth):
            self.table[idx] = TTEntry(key, depth, score, flag, best_move, age)

    def clear(self):
        self.table = [None] * self.size
        self.age   = 0

    def new_search(self):
        self.age += 1


# ───────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ───────────────────────────────────────────────────────────────────────────────

# Precompute file masks
FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
RANK_MASKS = [chess.BB_RANKS[r] for r in range(8)]

def _adjacent_files(f: int) -> int:
    mask = 0
    if f > 0: mask |= FILE_MASKS[f-1]
    if f < 7: mask |= FILE_MASKS[f+1]
    return mask

ADJ_FILE_MASKS = [_adjacent_files(f) for f in range(8)]

# Passed pawn masks: squares in front of pawn + adjacent files in front
def _passed_mask(sq: int, color: chess.Color) -> int:
    f    = chess.square_file(sq)
    r    = chess.square_rank(sq)
    mask = 0
    if color == chess.WHITE:
        for rr in range(r+1, 8):
            for ff in [f-1, f, f+1]:
                if 0 <= ff <= 7:
                    mask |= chess.BB_SQUARES[chess.square(ff, rr)]
    else:
        for rr in range(r-1, -1, -1):
            for ff in [f-1, f, f+1]:
                if 0 <= ff <= 7:
                    mask |= chess.BB_SQUARES[chess.square(ff, rr)]
    return mask

PASSED_MASK = {
    chess.WHITE: [_passed_mask(sq, chess.WHITE) for sq in chess.SQUARES],
    chess.BLACK: [_passed_mask(sq, chess.BLACK) for sq in chess.SQUARES],
}

PASSED_BONUS_MG = [0, 10, 17, 15, 62,168, 276, 0]  # by rank (0=rank1,7=rank8 for white)
PASSED_BONUS_EG = [0, 28, 33, 41,112,198, 312, 0]

ISOLATED_PENALTY_MG = 15
ISOLATED_PENALTY_EG = 20
DOUBLED_PENALTY_MG  = 10
DOUBLED_PENALTY_EG  = 20

BISHOP_PAIR_MG = 22
BISHOP_PAIR_EG = 30

ROOK_OPEN_MG   = 25
ROOK_SEMI_MG   = 12
ROOK_OPEN_EG   = 20
ROOK_SEMI_EG   = 10

MOBILITY_MG = {
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK:   2,
    chess.QUEEN:  1,
}
MOBILITY_EG = {
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK:   2,
    chess.QUEEN:  2,
}

# King safety attack weights
ATTACK_WEIGHT = {
    chess.KNIGHT: 20,
    chess.BISHOP: 20,
    chess.ROOK:   40,
    chess.QUEEN:  80,
}


def evaluate(board: chess.Board) -> int:
    """
    Full tapered evaluation. Returns score from perspective of side to move.
    All SquareSet objects cast to int before bitwise ops.
    """
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return DRAW_SCORE

    mg = [0, 0]
    eg = [0, 0]
    phase = 0

    white_pawns = int(board.pawns) & int(board.occupied_co[chess.WHITE])
    black_pawns = int(board.pawns) & int(board.occupied_co[chess.BLACK])

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    king_zone = [
        int(chess.BB_KING_ATTACKS[bk]) | int(chess.BB_SQUARES[bk]),
        int(chess.BB_KING_ATTACKS[wk]) | int(chess.BB_SQUARES[wk]),
    ]
    king_attack_count  = [0, 0]
    king_attack_weight = [0, 0]

    for color in (chess.WHITE, chess.BLACK):
        c   = int(color)
        occ = int(board.occupied_co[color])

        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            bb = int(board.pieces(pt, color))
            tmp = bb
            while tmp:
                sq   = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                phase += PHASE_WEIGHTS.get(pt, 0)

                pst_sq = sq if color == chess.WHITE else (sq ^ 56)
                mg[c] += MG_VALUES[pt] + MG_PST[pt][pst_sq]
                eg[c] += EG_VALUES[pt] + EG_PST[pt][pst_sq]

                if pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    atk = int(board.attacks(sq)) & ~occ
                    mob = bin(atk).count("1")
                    mg[c] += MOBILITY_MG[pt] * mob
                    eg[c] += MOBILITY_EG[pt] * mob

                    if int(board.attacks(sq)) & king_zone[c]:
                        king_attack_count[c]  += 1
                        king_attack_weight[c] += ATTACK_WEIGHT[pt]

        my_pawns  = white_pawns if color == chess.WHITE else black_pawns
        opp_pawns = black_pawns if color == chess.WHITE else white_pawns

        tmp = my_pawns
        while tmp:
            sq   = (tmp & -tmp).bit_length() - 1
            tmp &= tmp - 1
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            rank_idx = r if color == chess.WHITE else (7 - r)

            if not (PASSED_MASK[color][sq] & opp_pawns):
                mg[c] += PASSED_BONUS_MG[rank_idx]
                eg[c] += PASSED_BONUS_EG[rank_idx]

            if not (int(ADJ_FILE_MASKS[f]) & my_pawns):
                mg[c] -= ISOLATED_PENALTY_MG
                eg[c] -= ISOLATED_PENALTY_EG

            front_ranks = 0
            if color == chess.WHITE:
                for rr in range(r + 1, 8):
                    front_ranks |= int(chess.BB_RANKS[rr])
            else:
                for rr in range(0, r):
                    front_ranks |= int(chess.BB_RANKS[rr])
            if int(FILE_MASKS[f]) & front_ranks & my_pawns:
                mg[c] -= DOUBLED_PENALTY_MG // 2
                eg[c] -= DOUBLED_PENALTY_EG // 2

        if bin(int(board.bishops) & occ).count("1") >= 2:
            mg[c] += BISHOP_PAIR_MG
            eg[c] += BISHOP_PAIR_EG

        rooks = int(board.rooks) & occ
        tmp   = rooks
        while tmp:
            sq   = (tmp & -tmp).bit_length() - 1
            tmp &= tmp - 1
            f  = chess.square_file(sq)
            fm = int(FILE_MASKS[f])
            if not (fm & (white_pawns | black_pawns)):
                mg[c] += ROOK_OPEN_MG
                eg[c] += ROOK_OPEN_EG
            elif not (fm & my_pawns):
                mg[c] += ROOK_SEMI_MG
                eg[c] += ROOK_SEMI_EG

    for c in (0, 1):
        if king_attack_count[c] >= 2:
            w = king_attack_weight[c]
            mg[c] -= (w * w) // 400

    phase    = min(phase, MAX_PHASE)
    mg_score = mg[1] - mg[0]
    eg_score = eg[1] - eg[0]
    score    = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    if board.turn == chess.WHITE:
        return score
    else:
        return -score


# ───────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING
# ───────────────────────────────────────────────────────────────────────────────

CAPTURE_BONUS   = 1_000_000
KILLER_BONUS    = 900_000
COUNTER_BONUS   = 800_000
PROMO_BONUS     = 950_000

MVV_LVA = {}
for victim in range(1, 7):
    for attacker in range(1, 7):
        MVV_LVA[(attacker, victim)] = victim * 10 - attacker

class MoveOrderer:
    def __init__(self):
        # killers[ply] = [move1, move2]
        self.killers     = [[None, None] for _ in range(MAX_PLY)]
        # history[from][to] = score
        self.history     = [[0] * 64 for _ in range(64)]
        # countermove[from][to] = move
        self.countermove = [[None] * 64 for _ in range(64)]

    def store_killer(self, move: chess.Move, ply: int):
        if ply < MAX_PLY and move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def store_history(self, move: chess.Move, depth: int):
        self.history[move.from_square][move.to_square] += depth * depth

    def store_counter(self, prev_move: Optional[chess.Move], move: chess.Move):
        if prev_move is not None:
            self.countermove[prev_move.from_square][prev_move.to_square] = move

    def get_counter(self, prev_move: Optional[chess.Move]) -> Optional[chess.Move]:
        if prev_move is None:
            return None
        return self.countermove[prev_move.from_square][prev_move.to_square]

    def score(self, board: chess.Board, move: chess.Move, ply: int,
              tt_move: Optional[chess.Move], prev_move: Optional[chess.Move]) -> int:

        if move == tt_move:
            return 2_000_000

        # Promotions
        if move.promotion and move.promotion == chess.QUEEN:
            return PROMO_BONUS

        captured = board.piece_at(move.to_square)
        if captured:
            attacker = board.piece_at(move.from_square)
            att_type = attacker.piece_type if attacker else chess.PAWN
            score    = CAPTURE_BONUS + MVV_LVA.get((att_type, captured.piece_type), 0)
            # SEE-lite: penalize losing captures slightly
            if att_type > captured.piece_type:
                score -= 50
            return score

        if ply < MAX_PLY:
            if move == self.killers[ply][0]:
                return KILLER_BONUS
            if move == self.killers[ply][1]:
                return KILLER_BONUS - 10

        if move == self.get_counter(prev_move):
            return COUNTER_BONUS

        return self.history[move.from_square][move.to_square]

    def ordered(self, board: chess.Board, moves, ply: int,
                tt_move: Optional[chess.Move], prev_move: Optional[chess.Move]):
        return sorted(
            moves,
            key=lambda m: self.score(board, m, ply, tt_move, prev_move),
            reverse=True,
        )

    def reset(self):
        self.killers     = [[None, None] for _ in range(MAX_PLY)]
        # Age history (don't clear fully for continuity)
        for f in range(64):
            for t in range(64):
                self.history[f][t] //= 8
        self.countermove = [[None] * 64 for _ in range(64)]


# ───────────────────────────────────────────────────────────────────────────────
# SEARCH
# ───────────────────────────────────────────────────────────────────────────────

# LMR table: reductions[depth][move_index]
LMR_TABLE = [[0] * 64 for _ in range(MAX_DEPTH)]
for _d in range(1, MAX_DEPTH):
    for _m in range(1, 64):
        import math as _math
        LMR_TABLE[_d][_m] = max(0, int(0.75 + _math.log(_d) * _math.log(_m) / 2.25))

class Searcher:
    def __init__(self):
        self.tt      = TT(128)
        self.orderer = MoveOrderer()

        self.stop_event  = threading.Event()
        self.nodes       = 0
        self.start_time  = 0.0
        self.time_limit  = 0.0
        self.best_move: Optional[chess.Move] = None
        self.best_score  = 0
        self.seldepth    = 0

        # Stack for prev_move context (indexed by ply)
        self._prev_move = [None] * MAX_PLY

    def should_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        if self.nodes & 4095 == 0:
            if time.time() - self.start_time >= self.time_limit:
                return True
        return False

    # ── Quiescence ───────────────────────────────────────────────────────────

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes    += 1
        self.seldepth  = max(self.seldepth, ply)

        stand_pat = evaluate(board)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        DELTA = 975  # delta pruning margin (queen value ≈ 975)

        for move in self.orderer.ordered(
            board,
            [m for m in board.legal_moves if board.is_capture(m) or m.promotion],
            ply, None, None
        ):
            # Delta pruning
            captured = board.piece_at(move.to_square)
            if captured:
                gain = MG_VALUES.get(captured.piece_type, 0)
                if stand_pat + gain + DELTA < alpha:
                    continue

            board.push(move)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ── Main negamax ─────────────────────────────────────────────────────────

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int, is_pv: bool, cut_node: bool) -> int:

        if self.should_stop():
            return 0

        self.nodes += 1

        in_check = board.is_check()

        # Check extension
        if in_check:
            depth += 1

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        # ── Draw detection ────────────────────────────────────────────────────
        if ply > 0:
            if board.is_repetition(2) or board.is_fifty_moves():
                return DRAW_SCORE
            if board.is_insufficient_material():
                return DRAW_SCORE

        # ── Mate distance pruning ─────────────────────────────────────────────
        alpha = max(alpha, -MATE_SCORE + ply)
        beta  = min(beta,   MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        # ── Transposition table ───────────────────────────────────────────────
        key      = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(key)
        tt_move  = None
        orig_alpha = alpha

        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth and ply > 0:
                s = tt_entry.score
                if tt_entry.flag == TT_EXACT:
                    return s
                elif tt_entry.flag == TT_LOWER:
                    alpha = max(alpha, s)
                elif tt_entry.flag == TT_UPPER:
                    beta = min(beta, s)
                if alpha >= beta:
                    return s

        # ── Static eval for pruning ───────────────────────────────────────────
        static_eval = evaluate(board)

        # ── Razoring ─────────────────────────────────────────────────────────
        if (not is_pv and not in_check and depth <= 3
                and static_eval + 300 * depth < alpha):
            score = self.quiesce(board, alpha, beta, ply)
            if score < alpha:
                return score

        # ── Futility pruning ──────────────────────────────────────────────────
        FUTILITY_MARGIN = [0, 100, 200, 300, 400]
        if (not is_pv and not in_check and depth <= 4
                and static_eval + FUTILITY_MARGIN[depth] <= alpha
                and static_eval > -MATE_SCORE + MAX_PLY):
            pass  # Will prune individual moves below

        # ── Null move pruning ─────────────────────────────────────────────────
        if (not is_pv and not in_check and depth >= 3
                and static_eval >= beta
                and bin(board.occupied_co[board.turn]).count("1") > 3):
            R = 3 + depth // 4 + min(3, (static_eval - beta) // 200)
            board.push(chess.Move.null())
            score = -self.negamax(board, depth - 1 - R, -beta, -beta + 1,
                                  ply + 1, False, not cut_node)
            board.pop()
            if score >= beta:
                if score >= MATE_SCORE - MAX_PLY:
                    score = beta
                return score

        # ── IIR: Internal Iterative Reduction ────────────────────────────────
        if is_pv and tt_move is None and depth >= 4:
            depth -= 1

        # ── Move loop ─────────────────────────────────────────────────────────
        prev_move = self._prev_move[ply - 1] if ply > 0 else None
        moves     = list(board.legal_moves)

        if not moves:
            if in_check:
                return -MATE_SCORE + ply
            return DRAW_SCORE

        ordered   = self.orderer.ordered(board, moves, ply, tt_move, prev_move)

        best_score = -INF
        best_move  = None
        moves_done = 0
        quiets_tried = []

        for move in ordered:
            is_capture = board.is_capture(move)
            is_promo   = move.promotion is not None
            gives_check = board.gives_check(move)
            is_quiet   = not is_capture and not is_promo

            # ── Futility pruning (quiet moves) ───────────────────────────────
            if (not is_pv and not in_check and is_quiet
                    and depth <= 4 and moves_done > 0
                    and static_eval + FUTILITY_MARGIN[depth] <= alpha
                    and best_score > -MATE_SCORE + MAX_PLY):
                continue

            # ── Late move pruning ─────────────────────────────────────────────
            if (not is_pv and not in_check and is_quiet
                    and depth <= 5 and moves_done >= 3 + depth * depth
                    and not gives_check):
                continue

            self._prev_move[ply] = move
            board.push(move)
            moves_done += 1

            if is_quiet:
                quiets_tried.append(move)

            # ── PVS + LMR ─────────────────────────────────────────────────────
            if moves_done == 1:
                score = -self.negamax(board, depth - 1, -beta, -alpha,
                                      ply + 1, is_pv, False)
            else:
                # LMR
                reduction = 0
                if (depth >= 3 and moves_done >= 3 and is_quiet
                        and not in_check and not gives_check):
                    reduction = LMR_TABLE[min(depth, 63)][min(moves_done, 63)]
                    if is_pv:
                        reduction = max(0, reduction - 1)
                    if cut_node:
                        reduction += 1

                # Reduced search
                score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha,
                                      ply + 1, False, True)

                # If LMR succeeded, re-search at full depth
                if score > alpha and reduction > 0:
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha,
                                          ply + 1, False, not cut_node)

                # PVS re-search if inside window
                if score > alpha and score < beta:
                    score = -self.negamax(board, depth - 1, -beta, -alpha,
                                          ply + 1, True, False)

            board.pop()

            if self.should_stop():
                break

            if score > best_score:
                best_score = score
                best_move  = move
                if ply == 0:
                    self.best_move  = move
                    self.best_score = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff
                if not is_capture:
                    self.orderer.store_killer(move, ply)
                    self.orderer.store_history(move, depth)
                    self.orderer.store_counter(prev_move, move)
                    # Penalize quiets that didn't cause cutoff
                    for q in quiets_tried[:-1]:
                        self.orderer.history[q.from_square][q.to_square] -= depth * depth
                break

        if self.should_stop():
            return best_score

        # ── Store TT ──────────────────────────────────────────────────────────
        if best_score <= orig_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        self.tt.put(key, depth, best_score, flag, best_move, self.tt.age)

        return best_score

    # ── Iterative deepening ───────────────────────────────────────────────────

    def search(self, board: chess.Board, time_limit: float,
               max_depth: int = MAX_DEPTH) -> Optional[chess.Move]:
        self.stop_event.clear()
        self.nodes      = 0
        self.seldepth   = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.best_move  = None
        self.best_score = 0
        self.orderer.reset()
        self.tt.new_search()

        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        # Aspiration window
        asp_delta  = 25
        prev_score = 0

        for depth in range(1, max_depth + 1):
            self.seldepth = 0

            if depth <= 4:
                score = self.negamax(board, depth, -INF, INF, 0, True, False)
            else:
                # Aspiration windows
                alpha = prev_score - asp_delta
                beta  = prev_score + asp_delta
                while True:
                    score = self.negamax(board, depth, alpha, beta, 0, True, False)
                    if self.should_stop():
                        break
                    if score <= alpha:
                        alpha -= asp_delta
                        asp_delta *= 2
                    elif score >= beta:
                        beta += asp_delta
                        asp_delta *= 2
                    else:
                        break
                asp_delta = 25

            if self.should_stop():
                break

            prev_score = score
            elapsed    = time.time() - self.start_time
            nps        = int(self.nodes / elapsed) if elapsed > 0 else 0

            # Format score
            if abs(score) >= MATE_SCORE - MAX_PLY:
                moves_to_mate = (MATE_SCORE - abs(score) + 1) // 2
                score_str = f"mate {moves_to_mate if score > 0 else -moves_to_mate}"
            else:
                score_str = f"cp {score}"

            bm = self.best_move.uci() if self.best_move else ""
            print(
                f"info depth {depth} seldepth {self.seldepth} "
                f"score {score_str} nodes {self.nodes} nps {nps} "
                f"time {int(elapsed*1000)} pv {bm}",
                flush=True,
            )

            # Time check: stop if we've used most of our time
            if elapsed >= time_limit * 0.6:
                break

        return self.best_move or moves[0]

    def stop(self):
        self.stop_event.set()


# ───────────────────────────────────────────────────────────────────────────────
# TIME MANAGEMENT
# ───────────────────────────────────────────────────────────────────────────────

def calc_time(wtime: int, btime: int, winc: int, binc: int,
              movestogo: int, color: chess.Color, move_number: int,
              complexity: float = 1.0) -> float:
    """
    Returns seconds to think for this move.
    complexity: 1.0 = normal, >1.0 = spend more time
    """
    my_time_ms = wtime if color == chess.WHITE else btime
    my_inc_ms  = winc  if color == chess.WHITE else binc

    if my_time_ms <= 0:
        return 0.05

    t  = my_time_ms / 1000.0
    inc = my_inc_ms / 1000.0

    if movestogo > 0:
        # Moves-to-go time control
        target = t / (movestogo + 2) + inc * 0.8
    else:
        # Increment / sudden death
        # Estimate moves remaining based on move number
        # Typical game ~40 moves, so remaining ≈ max(10, 40 - move_number)
        moves_left = max(10, 45 - move_number // 2)
        target = t / moves_left + inc * 0.85

    # Hard cap at 20% of remaining time
    hard_cap = t * 0.20
    target   = min(target, hard_cap)

    # Complexity bonus (up to 2x)
    target *= min(2.0, complexity)

    # Absolute floor
    target = max(target, 0.05)

    return target


# ───────────────────────────────────────────────────────────────────────────────
# UCI LOOP
# ───────────────────────────────────────────────────────────────────────────────

def uci_loop():
    board         = chess.Board()
    searcher      = Searcher()
    search_thread: Optional[threading.Thread] = None
    ponder_move: Optional[chess.Move] = None

    def stop_search():
        nonlocal search_thread
        searcher.stop()
        if search_thread and search_thread.is_alive():
            search_thread.join(timeout=3)

    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            continue

        parts = line.split()
        cmd   = parts[0]

        # ── uci ───────────────────────────────────────────────────────────────
        if cmd == "uci":
            print(f"id name {ENGINE_NAME} {VERSION}")
            print(f"id author {ENGINE_AUTHOR}")
            print("option name Hash type spin default 128 min 1 max 4096")
            print("option name Threads type spin default 1 min 1 max 1")
            print("option name Move Overhead type spin default 30 min 0 max 500")
            print("uciok", flush=True)

        # ── isready ───────────────────────────────────────────────────────────
        elif cmd == "isready":
            print("readyok", flush=True)

        # ── ucinewgame ────────────────────────────────────────────────────────
        elif cmd == "ucinewgame":
            stop_search()
            board = chess.Board()
            searcher.tt.clear()
            searcher.orderer.reset()

        # ── setoption ─────────────────────────────────────────────────────────
        elif cmd == "setoption":
            try:
                name_idx  = parts.index("name")
                value_idx = parts.index("value")
                opt_name  = " ".join(parts[name_idx+1:value_idx]).strip().lower()
                opt_val   = parts[value_idx+1]
                if opt_name == "hash":
                    searcher.tt = TT(int(opt_val))
            except (ValueError, IndexError):
                pass

        # ── position ──────────────────────────────────────────────────────────
        elif cmd == "position":
            board = chess.Board()
            idx   = 1
            if idx < len(parts) and parts[idx] == "startpos":
                idx += 1
            elif idx < len(parts) and parts[idx] == "fen":
                idx += 1
                fen_parts = []
                while idx < len(parts) and parts[idx] != "moves":
                    fen_parts.append(parts[idx])
                    idx += 1
                try:
                    board.set_fen(" ".join(fen_parts))
                except ValueError:
                    pass

            if idx < len(parts) and parts[idx] == "moves":
                idx += 1
                while idx < len(parts):
                    try:
                        m = chess.Move.from_uci(parts[idx])
                        if m in board.legal_moves:
                            board.push(m)
                    except ValueError:
                        pass
                    idx += 1

        # ── go ────────────────────────────────────────────────────────────────
        elif cmd == "go":
            wtime     = 0
            btime     = 0
            winc      = 0
            binc      = 0
            movestogo = 0
            movetime  = 0
            depth_lim = MAX_DEPTH
            infinite  = False
            ponder    = False
            overhead  = 30  # ms

            i = 1
            while i < len(parts):
                p = parts[i]
                if   p == "wtime"     and i+1 < len(parts): wtime     = int(parts[i+1]); i += 2
                elif p == "btime"     and i+1 < len(parts): btime     = int(parts[i+1]); i += 2
                elif p == "winc"      and i+1 < len(parts): winc      = int(parts[i+1]); i += 2
                elif p == "binc"      and i+1 < len(parts): binc      = int(parts[i+1]); i += 2
                elif p == "movestogo" and i+1 < len(parts): movestogo = int(parts[i+1]); i += 2
                elif p == "movetime"  and i+1 < len(parts): movetime  = int(parts[i+1]); i += 2
                elif p == "depth"     and i+1 < len(parts): depth_lim = int(parts[i+1]); i += 2
                elif p == "infinite":  infinite = True; i += 1
                elif p == "ponder":    ponder   = True; i += 1
                else: i += 1

            # Apply overhead
            wtime = max(0, wtime - overhead)
            btime = max(0, btime - overhead)

            if movetime > 0:
                time_limit = (movetime - overhead) / 1000.0
            elif infinite or ponder:
                time_limit = 3600.0
            else:
                time_limit = calc_time(
                    wtime, btime, winc, binc,
                    movestogo, board.turn, board.fullmove_number,
                )

            stop_search()

            board_copy = board.copy()

            def run_search():
                move = searcher.search(board_copy, time_limit, depth_lim)
                if not ponder:
                    uci_str = move.uci() if move else "0000"
                    print(f"bestmove {uci_str}", flush=True)

            search_thread = threading.Thread(target=run_search, daemon=True)
            search_thread.start()

        # ── stop ──────────────────────────────────────────────────────────────
        elif cmd == "stop":
            stop_search()
            bm = searcher.best_move
            print(f"bestmove {bm.uci() if bm else '0000'}", flush=True)

        # ── quit ──────────────────────────────────────────────────────────────
        elif cmd == "quit":
            stop_search()
            sys.exit(0)

        # ── debug helpers ─────────────────────────────────────────────────────
        elif cmd == "d":
            print(board)
            print(f"FEN: {board.fen()}", flush=True)

        elif cmd == "eval":
            e = evaluate(board)
            print(f"info string static eval {e} cp (side to move)", flush=True)

        elif cmd == "perft":
            depth = int(parts[1]) if len(parts) > 1 else 1
            def _perft(b, d):
                if d == 0: return 1
                n = 0
                for m in b.legal_moves:
                    b.push(m)
                    n += _perft(b, d-1)
                    b.pop()
                return n
            t0 = time.time()
            nodes = _perft(board, depth)
            elapsed = time.time() - t0
            print(f"Nodes: {nodes}  Time: {elapsed:.3f}s  NPS: {int(nodes/elapsed) if elapsed > 0 else 0}", flush=True)


# ───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uci_loop()
