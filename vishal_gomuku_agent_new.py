import json
import re
import random
from typing import Tuple, List
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VishalGomokuLLMAgent2(Agent):
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
For EVERY row, column, and diagonal:
- Count consecutive pieces
- Check if adding one piece creates 5-in-a-row
- Check if opponent could win next move

## OUTPUT FORMAT:
{
  "reasoning": "STEP 1: [what you checked]. STEP 2: [what you found]. DECISION: [why this move]",
  "row": <number>,
  "col": <number>
}

NEVER miss a 4-in-a-row threat! Losing to obvious threats = FAILURE.
""".strip()

    def _parse_board_from_string(self, board_str: str) -> List[List[str]]:
        """Parse board string into 2D array - FIXED VERSION"""
        lines = board_str.strip().split('\n')
        board = []
        
        for line in lines:
            # Handle both spaced and non-spaced formats
            if ' ' in line:
                row = line.split()
            else:
                row = list(line)
            
            # Only take valid board characters
            filtered_row = [cell for cell in row if cell in ['X', 'O', '.']]
            if len(filtered_row) == 8:  # Valid row
                board.append(filtered_row)
        
        return board

    def _check_line_for_threat(self, board: List[List[str]], start_r: int, start_c: int, 
                               dr: int, dc: int, player: str, target_count: int) -> List[Tuple[int, int]]:
        """Check a line direction for threats that need exact count."""
        threat_positions = []
        
        # Scan along the line direction
        for start_offset in range(-4, 5):  # Check all possible 5-piece windows
            positions = []
            for i in range(5):
                r = start_r + start_offset * dr + i * dr
                c = start_c + start_offset * dc + i * dc
                if 0 <= r < 8 and 0 <= c < 8:
                    positions.append((r, c))
            
            if len(positions) == 5:
                pieces = [board[r][c] for r, c in positions]
                player_count = pieces.count(player)
                empty_count = pieces.count('.')
                
                # Check if this could be a threat (target_count pieces + 1 empty = win)
                if player_count == target_count and empty_count == 1:
                    empty_pos = next(pos for pos, piece in zip(positions, pieces) if piece == '.')
                    threat_positions.append(empty_pos)
        
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

    def _get_strategic_move(self, board: List[List[str]], me: str, opp: str) -> Tuple[int, int] | None:
        """Get strategic move using explicit threat detection."""
        
        # 1. Check for immediate wins (4 pieces + 1 empty = 5)
        win_moves = self._find_all_threats(board, me, 4)
        if win_moves:
            return win_moves[0]
        
        # 2. Block opponent's immediate wins
        block_moves = self._find_all_threats(board, opp, 4)
        if block_moves:
            return block_moves
        
        # 3. Create your own strong threats (3 pieces)
        my_strong_threats = self._find_all_threats(board, me, 3)
        if my_strong_threats:
            return my_strong_threats
        
        # 4. Block opponent's strong threats
        opp_strong_threats = self._find_all_threats(board, opp, 3)
        if opp_strong_threats:
            return opp_strong_threats
        
        # 5. Build from existing pieces (2 pieces)
        my_extensions = self._find_all_threats(board, me, 2)
        if my_extensions:
            return my_extensions
        
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
            strategic_move = self._get_strategic_move(board, me, opp)
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
3. Check EVERY diagonal for diagonal 4-in-a-row patterns
4. Look for your own winning opportunities

PREVIOUS LOSSES TO LEARN FROM:
- Lost Game 1: Missed blocking (0,4) when opponent had XXXX in row 0
- Lost Game 2: Missed blocking (4,2) when opponent had XXXX in row 4  
- Lost Game 3: Missed blocking threats that led to diagonal/vertical wins

Legal moves: {legal_moves}

ANALYZE SYSTEMATICALLY - don't just play random center moves!
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Call LLM with better settings
            response = await self.llm.complete(
                messages=messages,
                temperature=0.1,  # Low temp for tactical decisions
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
        center_distance = lambda pos: abs(pos[0] - 3.5) + abs(pos - 3.5)
        return min(legal_moves, key=center_distance)