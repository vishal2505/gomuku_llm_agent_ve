import json
import random
import re
from typing import List, Tuple
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VishalGomokuLLMAgent7(Agent):
    """LLM-powered Gomoku agent with stronger diagonal threat capture and blunder-avoidance."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VishalGomokuLLMAgent7: {agent_id}")

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        return (
            "You are an elite Gomoku (8x8, five-in-a-row) tactician.\n"
            "Play a strong central-backbone style: build a central cross/backbone, extend straight lines to 5,\n"
            "and continuously create dual threats. Always pick from the provided legal_moves only. Coordinates are 0-indexed.\n\n"
            "Hard priority order (never violate):\n"
            "1) WIN NOW: If any legal move makes 5-in-a-row for current player, choose it.\n"
            "2) BLOCK NOW: If opponent could make 5-in-a-row next move, block that exact square.\n"
            "3) POWER THREATS: Prefer moves that create open-fours (.XXXX. after your move) or create two simultaneous winning threats (double-three/double-four).\n"
            "4) BACKBONE THROUGH CENTER: Extend the longest straight line (row/column/diagonal) of your stones that passes near the center.\n"
            "5) CENTER BIAS + FORKS: Prefer central squares and moves that extend two directions at once (forks).\n"
            "6) NO BLUNDERS: Do not choose a move that immediately allows opponent a 5-in-a-row on their very next move.\n\n"
            "Evaluation guidance: scan all rows, columns, and both diagonals using sliding windows of length 5.\n"
            "Explicitly identify: (a) your immediate wins, (b) opponent's immediate wins, (c) open-fours and strong double threats.\n\n"
            "Opening rules: If you are X on an empty board, play (4, 4). If you are O and (4,4) is taken by X, play (3, 3).\n\n"
            "Respond ONLY with JSON in this exact schema and nothing else: {\"row\": <int>, \"col\": <int>}\n"
        )

    # ===== Agent5-style lightweight tactical analysis helpers =====
    def _parse_board_from_string(self, board_str: str) -> List[List[str]]:
        rows: List[List[str]] = []
        for line in board_str.strip().split('\n'):
            tokens = [ch for ch in line if ch in ['X', 'O', '.']]
            if len(tokens) == 8:
                rows.append(tokens)
        while len(rows) < 8:
            rows.append(['.'] * 8)
        return rows[:8]

    def _check_line_for_threat(self, board: List[List[str]], start_r: int, start_c: int,
                               dr: int, dc: int, player: str, target_count: int) -> List[Tuple[int, int]]:
        threat_positions: List[Tuple[int, int]] = []
        line: List[Tuple[int, int]] = []
        r, c = start_r, start_c
        while 0 <= r - dr < 8 and 0 <= c - dc < 8:
            r -= dr
            c -= dc
        while 0 <= r < 8 and 0 <= c < 8:
            line.append((r, c))
            r += dr
            c += dc
        opp = 'O' if player == 'X' else 'X'
        for i in range(0, max(0, len(line) - 4)):
            window = line[i:i+5]
            pieces = [board[rr][cc] for rr, cc in window]
            player_count = pieces.count(player)
            empty_count = pieces.count('.')
            opp_count = pieces.count(opp)
            if target_count == 4 and player_count == 4 and empty_count == 1 and opp_count == 0:
                empty_idx = pieces.index('.')
                threat_positions.append(window[empty_idx])
                continue
            if target_count == 3 and player_count == 3 and empty_count == 2 and opp_count == 0:
                if pieces[0] == '.' and pieces[4] == '.' and pieces[1] == player and pieces[2] == player and pieces[3] == player:
                    threat_positions.append(window[0]); threat_positions.append(window[4])
                else:
                    for idx, val in enumerate(pieces):
                        if val == '.':
                            threat_positions.append(window[idx])
                continue
            if target_count == 3 and player_count == 3 and empty_count == 1 and opp_count == 1:
                empty_idx = pieces.index('.')
                threat_positions.append(window[empty_idx])
        if target_count == 3 and len(line) >= 4:
            for i in range(0, len(line) - 3):
                window4 = line[i:i+4]
                pieces4 = [board[rr][cc] for rr, cc in window4]
                if pieces4.count(player) == 3 and pieces4.count('.') == 1 and pieces4.count(opp) == 0:
                    if pieces4[0] == '.' and pieces4[1] == player and pieces4[2] == player and pieces4[3] == player:
                        threat_positions.append(window4[0])
                    elif pieces4[3] == '.' and pieces4[0] == player and pieces4[1] == player and pieces4[2] == player:
                        threat_positions.append(window4[3])
        return threat_positions

    def _find_all_threats(self, board: List[List[str]], player: str, target_count: int) -> List[Tuple[int, int]]:
        threats = set()
        for r in range(8):
            for c in range(8):
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                    threats.update(self._check_line_for_threat(board, r, c, dr, dc, player, target_count))
        return list(threats)

    def _pick_best_center(self, candidates: List[Tuple[int, int]], legal_moves: List[Tuple[int, int]]) -> Tuple[int, int] | None:
        legal = [m for m in candidates if m in legal_moves]
        if not legal:
            return None
        return min(legal, key=lambda pos: abs(pos[0] - 3.5) + abs(pos[1] - 3.5))

    def _get_strategic_move(self, board: List[List[str]], me: str, opp: str, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int] | None:
        win_moves = self._find_all_threats(board, me, 4)
        best = self._pick_best_center(win_moves, legal_moves)
        if best:
            return best
        block_moves = self._find_all_threats(board, opp, 4)
        best = self._pick_best_center(block_moves, legal_moves)
        if best:
            return best
        opp_threes = self._find_all_threats(board, opp, 3)
        best = self._pick_best_center(opp_threes, legal_moves)
        if best:
            return best
        my_threes = self._find_all_threats(board, me, 3)
        best = self._pick_best_center(my_threes, legal_moves)
        if best:
            return best
        return None

    async def get_move(self, game_state: 'GameState') -> Tuple[int, int]:  # type: ignore
        # Build user prompt with board and legal moves to keep LLM on rails
        try:
            board_str = game_state.format_board(formatter="standard")
        except Exception:
            board_str = ""

        try:
            legal_moves: List[Tuple[int, int]] = list(game_state.get_legal_moves())  # type: ignore
        except Exception:
            legal_moves = []

        # Quick fallback if no LLM or no legal moves
        if not legal_moves:
            return self._fallback_centerish(game_state)

        current_player = getattr(getattr(game_state, "current_player", object()), "value", "?")
        opp = 'O' if current_player == 'X' else 'X'

        # Algorithmic safeguards first
        board = self._parse_board_from_string(board_str)
        strat = self._get_strategic_move(board, current_player, opp, legal_moves)
        if strat:
            return strat

        user_prompt = (
            f"Current player: {current_player}\n"
            f"Board (standard):\n{board_str}\n\n"
            f"legal_moves (choose exactly one of these): {legal_moves}\n\n"
            "Output only valid JSON: {\"row\": <int>, \"col\": <int>} from legal_moves."
        )

        # If LLM unavailable, go to deterministic fallback
        if getattr(self, 'llm', None) is None:
            return self._center_backbone_fallback(game_state, legal_moves)

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await self.llm.complete(messages=messages, temperature=0.0, max_tokens=150)
            move = self._parse_move(response)
            if move in legal_moves:
                return move
        except Exception:
            pass

        return self._center_backbone_fallback(game_state, legal_moves)

    def _parse_move(self, text: str) -> Tuple[int, int]:
        try:
            m = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if not m:
                raise ValueError("no JSON found")
            data = json.loads(m.group(0))
            return int(data["row"]), int(data["col"])
        except Exception:
            return (-1, -1)

    # Heuristic fallback that plays a center-backbone style without deep simulation
    def _center_backbone_fallback(self, game_state: 'GameState', legal_moves: List[Tuple[int, int]]) -> Tuple[int, int]:  # type: ignore
        # 1) Opening book
        try:
            board = getattr(game_state, "board", None)
            is_empty = False
            if board is not None and hasattr(board, "size") and hasattr(board, "grid"):
                size = getattr(board, "size", 8)
                grid = getattr(board, "grid", [])
                is_empty = all(cell == "." for row in grid for cell in row)
            if is_empty:
                if (4, 4) in legal_moves:
                    return (4, 4)
        except Exception:
            pass

        # 2) Prefer extending a central backbone and center proximity
        def center_dist_sq(r: int, c: int) -> int:
            return (r - 3.5) ** 2 + (c - 3.5) ** 2  # type: ignore

        # Simple central plus-shape preference
        plus_targets = {(4, 4), (3, 4), (5, 4), (4, 3), (4, 5), (3, 3), (3, 5), (5, 3), (5, 5)}
        plus_moves = [m for m in legal_moves if m in plus_targets]
        if plus_moves:
            plus_moves.sort(key=lambda rc: center_dist_sq(*rc))
            return plus_moves[0]

        # 3) Otherwise choose the most central legal move
        legal_moves.sort(key=lambda rc: center_dist_sq(*rc))
        return legal_moves[0]

    def _fallback_centerish(self, game_state: 'GameState') -> Tuple[int, int]:  # type: ignore
        try:
            legal_moves: List[Tuple[int, int]] = list(game_state.get_legal_moves())  # type: ignore
            if not legal_moves:
                return (0, 0)
            legal_moves.sort(key=lambda rc: (abs(rc[0]-4)+abs(rc[1]-4), rc))
            return legal_moves[0]
        except Exception:
            return (4, 4)
