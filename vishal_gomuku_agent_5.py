import json
import re
import random
from typing import Tuple, List
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VishalGomokuLLMAgent5(Agent):
    """LLM-powered Gomoku agent for 8x8 five-in-a-row tournament."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VishalGomokuLLMAgent: {agent_id}")

    def _setup(self):
        """Setup LLM client and system prompt."""
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        """Enhanced strategic Gomoku system prompt with explicit examples."""
        return """
You are a TACTICAL Gomoku expert. FIRST priority: analyze for wins and blocks.

## EXACT ANALYSIS STEPS:

1. **SCAN FOR YOUR WINS**: Check if placing your piece anywhere creates 5-in-a-row
2. **SCAN FOR OPPONENT THREATS**: Check if opponent has 4-in-a-row that MUST be blocked
3. **SCAN FOR DANGEROUS 3s**: Block opponent's open-ended three-in-a-row patterns
4. **BUILD YOUR ATTACK**: Extend your strongest sequences

## CRITICAL THREAT PATTERNS TO DETECT:
- XXXX. or .XXXX = 4-in-a-row (EMERGENCY BLOCK!)
- .XXX. = open three (VERY DANGEROUS!)
- XXX. or .XXX = closed three (still dangerous)

## BOARD ANALYSIS METHOD:
For EVERY row, column, and BOTH diagonals (\\ and //):
- Count consecutive pieces
- Check if adding one piece creates 5-in-a-row
- Check if opponent could win next move
- Pay special attention to diagonal .XXX. and XXXX. patterns

## OUTPUT FORMAT (STRICT):
Return JSON only, no extra text:
{
  "reasoning": "STEP 1: [what you checked]. STEP 2: [what you found]. DECISION: [why this move]",
  "row": <number>,
  "col": <number>
}
""".strip()

    def _parse_board_from_string(self, board_str: str) -> List[List[str]]:
        """Parse board string into 2D array - robust across formats."""
        rows = []
        for line in board_str.strip().split('\n'):
            tokens = [ch for ch in line if ch in ['X', 'O', '.']]
            if len(tokens) == 8:
                rows.append(tokens)
        # Ensure 8 rows by padding if needed (shouldn't happen but for safety)
        while len(rows) < 8:
            rows.append(['.'] * 8)
        return rows[:8]

    def _check_line_for_threat(self, board: List[List[str]], start_r: int, start_c: int, 
                               dr: int, dc: int, player: str, target_count: int) -> List[Tuple[int, int]]:
        """Check a line direction for threats that need exact count."""
        threat_positions = []
        
        # Start from the given cell and extend both directions to collect a maximal line
        line = []
        r, c = start_r, start_c
        
        # Move backwards to the edge in this direction
        while 0 <= r - dr < 8 and 0 <= c - dc < 8:
            r -= dr
            c -= dc
        
        # Collect the entire line forward until edge
        while 0 <= r < 8 and 0 <= c < 8:
            line.append((r, c))
            r += dr
            c += dc
        
        opp = 'O' if player == 'X' else 'X'
        
        # Slide a window of size 5 across this line
        for i in range(0, max(0, len(line) - 4)):
            window = line[i:i+5]
            pieces = [board[rr][cc] for rr, cc in window]
            player_count = pieces.count(player)
            empty_count = pieces.count('.')
            opp_count = pieces.count(opp)

            # Immediate win/block patterns: XXXX. / .XXXX (no opponent piece in window)
            if target_count == 4 and player_count == 4 and empty_count == 1 and opp_count == 0:
                empty_idx = pieces.index('.')
                threat_positions.append(window[empty_idx])
                continue

            # Open-three patterns: ONLY match contiguous .XXX. (two empties at both ends)
            if target_count == 3 and player_count == 3 and empty_count == 2 and opp_count == 0:
                if pieces[0] == '.' and pieces[1] == player and pieces[2] == player and pieces[3] == player and pieces[4] == '.':
                    threat_positions.append(window[0])
                    threat_positions.append(window[4])
                else:
                    # Also consider broken-threes like X.XX or XX.X (return both empties)
                    for idx, val in enumerate(pieces):
                        if val == '.':
                            threat_positions.append(window[idx])
                continue

            # Optional: closed-three (one empty, one opponent) - lower priority
            if target_count == 3 and player_count == 3 and empty_count == 1 and opp_count == 1:
                empty_idx = pieces.index('.')
                threat_positions.append(window[empty_idx])
        
        # Additionally, for lines near the border (length 4 windows), catch XXX. or .XXX when target is 3
        if target_count == 3 and len(line) >= 4:
            for i in range(0, len(line) - 3):
                window4 = line[i:i+4]
                pieces4 = [board[rr][cc] for rr, cc in window4]
                player_count4 = pieces4.count(player)
                empty_count4 = pieces4.count('.')
                opp_count4 = pieces4.count(opp)
                if player_count4 == 3 and empty_count4 == 1 and opp_count4 == 0:
                    # Ensure contiguity: match .XXX or XXX.
                    if pieces4[0] == '.' and pieces4[1] == player and pieces4[2] == player and pieces4[3] == player:
                        threat_positions.append(window4[0])
                    elif pieces4[0] == player and pieces4[1] == player and pieces4[2] == player and pieces4[3] == '.':
                        threat_positions.append(window4[3])
        
        return threat_positions

    def _find_all_threats(self, board: List[List[str]], player: str, target_count: int) -> List[Tuple[int, int]]:
        """Find all threat positions for a player with target_count pieces."""
        threats = set()
        
        # Check all positions and directions
        for r in range(8):
            for c in range(8):
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:  # 4 main directions
                    threat_positions = self._check_line_for_threat(board, r, c, dr, dc, player, target_count)
                    threats.update(threat_positions)
        
        return list(threats)

    def _score_move(self, board: List[List[str]], r: int, c: int, me: str) -> int:
        """Score a move by the longest contiguous line it creates for 'me' and fork potential."""
        if board[r][c] != '.':
            return -1
        board[r][c] = me
        best = 0
        fork_bonus = 0
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
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
            if cnt >= 3:
                fork_bonus += 1
        board[r][c] = '.'
        # Weight best line more, add small fork bonus
        return best * 10 + fork_bonus

    def _pick_best(self, candidates: List[Tuple[int, int]], legal_moves: List[Tuple[int, int]], board: List[List[str]], me: str) -> Tuple[int, int] | None:
        """Pick the best candidate: maximize our line length, then prefer center."""
        legal = [m for m in candidates if m in legal_moves]
        if not legal:
            return None
        def key(pos: Tuple[int, int]):
            score = self._score_move(board, pos[0], pos[1], me)
            center = abs(pos[0] - 3.5) + abs(pos[1] - 3.5)
            return (-score, center)
        return min(legal, key=key)

    def _get_strategic_move(self, board: List[List[str]], me: str, opp: str, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int] | None:
        """Get strategic move using explicit threat detection. Always return a single (row,col)."""
        # 1. Check for immediate wins (4 pieces + 1 empty = 5)
        win_moves = self._find_all_threats(board, me, 4)
        best = self._pick_best(win_moves, legal_moves, board, me)
        if best:
            return best

        # 2. Block opponent's immediate wins
        block_moves = self._find_all_threats(board, opp, 4)
        best = self._pick_best(block_moves, legal_moves, board, me)
        if best:
            return best

        # 3. Block opponent's strong threats (open three .XXX.) BEFORE creating your own
        opp_strong_threats = self._find_all_threats(board, opp, 3)
        best = self._pick_best(opp_strong_threats, legal_moves, board, me)
        if best:
            return best

        # 4. Create your own strong threats (open three .XXX.)
        my_strong_threats = self._find_all_threats(board, me, 3)
        best = self._pick_best(my_strong_threats, legal_moves, board, me)
        if best:
            return best

        # 5. Build from existing pieces (2 pieces)
        my_extensions = self._find_all_threats(board, me, 2)
        best = self._pick_best(my_extensions, legal_moves, board, me)
        if best:
            return best

        return None

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Enhanced move selection with better threat detection."""
        try:
            me = game_state.current_player.value
            opp = "O" if me == "X" else "X"
            legal_moves = game_state.get_legal_moves()
            
            # Parse board
            board_str = game_state.format_board(formatter="standard")
            board = self._parse_board_from_string(board_str)

            # SAFEGUARD: Use algorithmic threat detection first
            strategic_move = self._get_strategic_move(board, me, opp, legal_moves)
            if strategic_move and strategic_move in legal_moves:
                print(f"STRATEGIC MOVE: {strategic_move}")
                return strategic_move

            # Enhanced LLM prompt with board analysis
            move_count = sum(row.count('X') + row.count('O') for row in board)
            
            user_prompt = f"""
=== GOMOKU BATTLE ANALYSIS ===
You: {me} | Opponent: {opp} | Move #{move_count + 1}

CURRENT BOARD:
{board_str}

CRITICAL ANALYSIS REQUIRED:
1. Check EVERY row for patterns like "XXXX." or ".XXXX" (4-in-a-row to block)
2. Check EVERY column for vertical 4-in-a-row patterns  
3. Check EVERY diagonal (\\ and //) for diagonal 4-in-a-row and .XXX. patterns
4. Look for your own winning opportunities

Legal moves: {legal_moves}

Provide JSON only.
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM with deterministic settings
            response = await self.llm.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=150,
            )

            # Enhanced JSON extraction
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_data = json.loads(response[start:end])
                
                r, c = int(json_data["row"]), int(json_data["col"])
                
                if (r, c) in legal_moves:
                    reasoning = json_data.get("reasoning", "No reasoning")
                    print(f"LLM move: ({r},{c}) - {reasoning}")
                    return r, c

        except Exception as e:
            print(f"Agent error: {e}")

        # Smart fallback: prefer center
        return self._get_smart_fallback(legal_moves)

    def _get_smart_fallback(self, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Fallback that prefers central positions."""
        return min(legal_moves, key=lambda pos: abs(pos[0] - 3.5) + abs(pos[1] - 3.5))