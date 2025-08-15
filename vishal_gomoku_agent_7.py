import json
from typing import Tuple, List
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VishalGomokuLLMAgent7(Agent):
    """Simplified, efficient Gomoku agent: minimal helpers, strong tactical rules."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        return (
            "You are a tactical Gomoku expert on 8x8. Priorities: "
            "1) Win now. 2) Block opponent win. 3) Block opponent two-ply threats (their move this turn creates a forced win next turn). "
            "4) Create your own two-ply threat. 5) Prefer center and forks.\n\n"
            "Always scan rows, columns, and diagonals (\\ and //). "
            "Avoid any move that gives opponent an immediate win next move. "
            "Return JSON only: {\"reasoning\": \"...\", \"row\": <int>, \"col\": <int>}"
        )

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        me = game_state.current_player.value
        opp = 'O' if me == 'X' else 'X'
        legal: List[Tuple[int, int]] = game_state.get_legal_moves()
        board_str = game_state.format_board(formatter="standard")

        # Parse board (inline)
        board: List[List[str]] = []
        for line in board_str.strip().split('\n'):
            row = [ch for ch in line if ch in ['X', 'O', '.']]
            if len(row) == 8:
                board.append(row)
        while len(board) < 8:
            board.append(['.'] * 8)
        board = board[:8]

        dirs = [(0,1), (1,0), (1,1), (1,-1)]

        def five_if_place(r: int, c: int, player: str) -> bool:
            if board[r][c] != '.':
                return False
            board[r][c] = player
            won = False
            for dr, dc in dirs:
                cnt = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
                    cnt += 1
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
                    cnt += 1
                    rr -= dr
                    cc -= dc
                if cnt >= 5:
                    won = True
                    break
            board[r][c] = '.'
            return won

        def opponent_wins_if_we_skip_at(pos: Tuple[int, int]) -> bool:
            r, c = pos
            if board[r][c] != '.':
                return False
            # If opponent can win by playing here now, this is a must-block square
            return five_if_place(r, c, opp)

        def gives_opp_immediate_win_after(move: Tuple[int,int]) -> bool:
            r, c = move
            if board[r][c] != '.':
                return True
            board[r][c] = me
            # After my move, opponent must not have any winning reply
            for rr, cc in legal:
                if board[rr][cc] == '.' and five_if_place(rr, cc, opp):
                    board[r][c] = '.'
                    return True
            board[r][c] = '.'
            return False

        def best_key(pos: Tuple[int,int]) -> Tuple[int, float]:
            r, c = pos
            # score by longest line created for me
            if board[r][c] != '.':
                return (-1, 99.0)
            board[r][c] = me
            best = 0
            for dr, dc in dirs:
                cnt = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == me:
                    cnt += 1
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == me:
                    cnt += 1
                    rr -= dr
                    cc -= dc
                best = max(best, cnt)
            board[r][c] = '.'
            center = abs(r - 3.5) + abs(c - 3.5)
            return (-best, center)

        # 1) Win now
        wins = [m for m in legal if five_if_place(m[0], m[1], me)]
        if wins:
            return min(wins, key=best_key)

        # 2) Block opponent immediate win (must-block squares)
        must_blocks = [m for m in legal if opponent_wins_if_we_skip_at(m)]
        if must_blocks:
            safe = [m for m in must_blocks if not gives_opp_immediate_win_after(m)]
            return min(safe or must_blocks, key=best_key)

        # 3) Block opponent two-ply threats: squares where, if opponent plays there now,
        # they will have a winning follow-up next turn
        two_ply_blocks: List[Tuple[int,int]] = []
        for m in legal:
            r, c = m
            board[r][c] = opp
            threat = False
            for rr in range(8):
                for cc in range(8):
                    if board[rr][cc] == '.' and five_if_place(rr, cc, opp):
                        threat = True
                        break
                if threat:
                    break
            board[r][c] = '.'
            if threat:
                two_ply_blocks.append(m)
        if two_ply_blocks:
            safe = [m for m in two_ply_blocks if not gives_opp_immediate_win_after(m)]
            if safe:
                return min(safe, key=best_key)
            return min(two_ply_blocks, key=best_key)

        # 4) Create our own two-ply threat: play a move such that we will have a winning reply next turn
        creators: List[Tuple[int,int]] = []
        for m in legal:
            r, c = m
            board[r][c] = me
            kill_next = False
            for rr in range(8):
                for cc in range(8):
                    if board[rr][cc] == '.' and five_if_place(rr, cc, me):
                        kill_next = True
                        break
                if kill_next:
                    break
            board[r][c] = '.'
            if kill_next:
                creators.append(m)
        if creators:
            safe = [m for m in creators if not gives_opp_immediate_win_after(m)]
            return min(safe or creators, key=best_key)

        # 5) Otherwise pick safest, best-scoring central move
        pool = [m for m in legal if not gives_opp_immediate_win_after(m)] or legal
        return min(pool, key=best_key)
