import json
import re
import random
from typing import Tuple

# The competition framework should provide these imports
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient


class VishalGomokuLLMAgent(Agent):
    """LLM-powered Gomoku agent for 8x8 five-in-a-row tournament."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VishalGomokuLLMAgent: {agent_id}")

    def _setup(self):
        """Setup LLM client and system prompt."""
        self.system_prompt = self._create_system_prompt()

        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        """Enhanced strategic Gomoku system prompt."""
        return """
You are an EXPERT Gomoku (8×8, five-in-a-row) strategist competing in a tournament.
You play either X or O on an 8×8 board. Empty cells are '.'.
Coordinates are 0-indexed: row ∈ [0..7], col ∈ [0..7].

## CRITICAL SUCCESS PRINCIPLES
1. NEVER make isolated moves - always build connected patterns
2. ALWAYS analyze the ENTIRE board for threats before moving
3. FOCUS on the CENTER area (rows 2-5, cols 2-5) in early game
4. PRIORITIZE extending your longest sequences over starting new ones

## MANDATORY PRIORITY SYSTEM (Execute in EXACT order)

### LEVEL 1: IMMEDIATE THREATS (Check these first!)
1) **INSTANT WIN**: Systematically check if ANY legal move creates exactly 5-in-a-row:
   - For each legal move, check all 8 directions (horizontal, vertical, 4 diagonals)
   - Count: your pieces + the new piece + your pieces in opposite direction
   - If total = 5, PLAY IT IMMEDIATELY (this is the winning move!)
   
2) **CRITICAL BLOCK**: If opponent has 4-in-a-row with open end(s), BLOCK immediately:
   - Scan all opponent pieces for 4-in-a-row patterns
   - Block the empty end(s) that would complete their 5-in-a-row

### LEVEL 2: STRONG TACTICAL MOVES  
3) **CREATE WINNING THREAT**: Make 4-in-a-row to force win next turn:
   - Look for your 3-in-a-row that can become 4-in-a-row
   - Prefer open-four (both ends free) over closed-four
   
4) **FORK CREATION**: Create double-threat (two separate ways to win)
5) **BLOCK OPPONENT THREATS**: Stop opponent's dangerous 3-in-a-row patterns

### LEVEL 3: POSITIONAL ADVANTAGE
6) **EXTEND LONGEST**: Add to your longest connected sequence (3+ pieces)
7) **BUILD CENTER CONTROL**: Occupy central squares (3,3), (3,4), (4,3), (4,4)  
8) **CREATE OPEN-THREE**: Make .XXX. pattern with room to extend

### LEVEL 4: FALLBACK STRATEGY
9) **ADJACENT PLACEMENT**: Place next to existing pieces (maintain connectivity)
10) **CENTER PREFERENCE**: Choose moves closer to center (4,4) if no other priority

## PATTERN RECOGNITION EXAMPLES
- Winning: XXXXX (any direction)
- Open-four: .XXXX. or XXXX. or .XXXX
- Open-three: .XXX. 
- Closed-three: OXXX. or .XXXO (less valuable)
- Fork: Two open-three patterns intersecting

## BOARD ANALYSIS METHOD - WINNING DETECTION
**CRITICAL**: For EVERY legal move, check if it creates 5-in-a-row:

1. **Horizontal Check**: Place your piece, count left + center + right
   Example: .XXX[new].  → Check: left pieces + new + right pieces = 5?
   
2. **Vertical Check**: Place your piece, count up + center + down  
   Example: 
   X
   X  
   X
   [new]
   . → Check: up pieces + new + down pieces = 5?
   
3. **Diagonal Check**: Both diagonal directions (/ and \)
   Count pieces in both diagonal directions from new position

4. **Verification**: If ANY direction gives exactly 5 connected pieces, THAT'S THE WINNING MOVE!

## PATTERN RECOGNITION EXAMPLES - WINNING SITUATIONS
- Horizontal win: XXXX. → place at . for XXXXX
- Vertical win: X/X/X/X/. → place at . for 5 vertical
- Diagonal win: X..../X..../X..../X..../..... → place at bottom-right for diagonal 5
- Gap-fill win: XX.XX → place at middle . for XXXXX

## STRATEGIC FOCUS AREAS
Early game (moves 1-10): Control center (rows 2-5, cols 2-5)
Mid game (moves 11-30): Build connected groups, block opponent patterns
Late game (moves 31+): Force wins, prevent opponent forks

## OUTPUT FORMAT
Respond with JSON only:
{
  "reasoning": "Priority level used and specific tactic (e.g., 'Level 1: Block opponent 4-in-a-row at (2,3)')",
  "row": <int>,
  "col": <int>
}

## CRITICAL REMINDERS
- SCAN for 4-in-a-row threats FIRST (yours and opponent's)
- NEVER ignore opponent's growing patterns (3+ in a row)
- ALWAYS prefer connected moves over isolated placements
- The (row, col) MUST be from `legal_moves` list
- Think like a tournament champion - every move must have strategic purpose
""".strip()

    def _check_immediate_win(self, game_state: GameState, player: str) -> Tuple[int, int] | None:
        """Check if there's an immediate winning move for the given player."""
        legal_moves = game_state.get_legal_moves()
        board = [['.'] * 8 for _ in range(8)]
        
        # Reconstruct board from game state
        board_str = game_state.format_board(formatter="standard")
        lines = board_str.strip().split('\n')
        for i, line in enumerate(lines):
            for j, cell in enumerate(line):
                if cell in ['X', 'O']:
                    board[i][j] = cell
        
        # Check each legal move
        for row, col in legal_moves:
            # Temporarily place the piece
            board[row][col] = player
            
            # Check all 8 directions for 5-in-a-row
            directions = [
                (0, 1), (1, 0), (1, 1), (1, -1),  # horizontal, vertical, diagonals
                (0, -1), (-1, 0), (-1, -1), (-1, 1)  # opposite directions
            ]
            
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # 4 main directions
                count = 1  # count the placed piece
                
                # Count in positive direction
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                    count += 1
                    r, c = r + dr, c + dc
                
                # Count in negative direction  
                r, c = row - dr, col - dc
                while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                    count += 1
                    r, c = r - dr, c - dc
                
                if count >= 5:
                    board[row][col] = '.'  # restore board
                    return (row, col)
            
            # Restore board
            board[row][col] = '.'
        
        return None

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Return the next move coordinates as (row, col)."""
        try:
            me = game_state.current_player.value
            opp = "O" if me == "X" else "X"
            
            # SAFEGUARD: Check for immediate winning moves first
            winning_move = self._check_immediate_win(game_state, me)
            if winning_move:
                print(f"SAFEGUARD: Immediate winning move found: {winning_move}")
                return winning_move
            
            # SAFEGUARD: Check if we need to block opponent's winning move
            opponent_winning_move = self._check_immediate_win(game_state, opp)
            if opponent_winning_move:
                print(f"SAFEGUARD: Blocking opponent's winning move: {opponent_winning_move}")
                return opponent_winning_move
            
            # Continue with LLM-based strategy
            board_str = game_state.format_board(formatter="standard")
            legal_moves = game_state.get_legal_moves()
            
            # Count pieces for game phase analysis
            board_lines = board_str.strip().split('\n')
            move_count = sum(line.count('X') + line.count('O') for line in board_lines)
            game_phase = "EARLY" if move_count <= 10 else "MID" if move_count <= 30 else "LATE"

            # Enhanced user prompt with board analysis
            user_prompt = (
                f"=== GOMOKU MOVE ANALYSIS ===\n"
                f"You are player: {me}\n"
                f"Opponent is: {opp}\n"
                f"Game phase: {game_phase} (move #{move_count + 1})\n"
                f"Legal moves available: {len(legal_moves)} positions\n\n"
                
                f"CURRENT BOARD STATE:\n"
                f"{board_str}\n\n"
                
                f"CRITICAL ANALYSIS CHECKLIST (MUST DO IN ORDER):\n"
                f"1. *** WINNING CHECK ***: For EACH legal move, imagine placing {me} there\n"
                f"   - Count horizontal: pieces left + new + pieces right = 5?\n"
                f"   - Count vertical: pieces up + new + pieces down = 5?\n" 
                f"   - Count diagonal /: pieces + new + pieces = 5?\n"
                f"   - Count diagonal \\: pieces + new + pieces = 5?\n"
                f"   - If ANY equals 5, THAT IS THE WINNING MOVE!\n\n"
                f"2. *** BLOCK CHECK ***: Scan for {opp} 4-in-a-row patterns that need immediate blocking\n"
                f"3. Look for {me} 3-in-a-row that can become threatening 4-in-a-row\n"  
                f"4. Look for {opp} 3-in-a-row patterns that need blocking\n"
                f"5. Extend your longest connected sequences\n"
                f"6. Maintain connectivity - never play isolated moves\n\n"
                
                f"LEGAL MOVES (row,col): {legal_moves}\n\n"
                
                f"INSTRUCTIONS:\n"
                f"- FIRST: Check EVERY legal move for immediate 5-in-a-row wins\n"
                f"- Example: If you have XX.XX horizontally, placing in middle . = XXXXX = WIN!\n"
                f"- Example: If you have XXXX. pattern, placing at . = XXXXX = WIN!\n"
                f"- Use the MANDATORY PRIORITY SYSTEM from your training\n"
                f"- State which priority level (1-4) you're using in reasoning\n"
                f"- NEVER make isolated moves unless forced\n"
                f"- Return JSON format ONLY\n"
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            kwargs = dict(
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent strategic play
                max_tokens=150,   # Slightly more tokens for reasoning
                response_format={"type": "json_object"},
            )

            # Enhanced retry loop with better error handling
            for attempt in range(3):  # More attempts for better reliability
                try:
                    response = await self.llm.complete(**kwargs)
                    data = json.loads(response)
                    
                    # Validate required keys
                    if "row" not in data or "col" not in data:
                        raise ValueError("Missing row or col in response")
                    
                    r, c = int(data["row"]), int(data["col"])

                    if (r, c) in legal_moves:
                        # Log the reasoning for debugging
                        reasoning = data.get("reasoning", "No reasoning provided")
                        print(f"Agent move: ({r},{c}) - {reasoning}")
                        return r, c

                    # If illegal move, provide specific feedback
                    messages.append({
                        "role": "user",
                        "content": (f"ILLEGAL MOVE: ({r},{c}) is not in legal moves.\n"
                                  f"You MUST choose from: {legal_moves}\n"
                                  f"Analyze the board again and pick a legal position.")
                    })

                except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                    error_msg = (f"ERROR in attempt {attempt + 1}: {str(parse_err)}\n"
                               f"You must respond with valid JSON containing 'row' and 'col' integers.\n"
                               f"Legal moves: {legal_moves}")
                    messages.append({"role": "user", "content": error_msg})

        except Exception as e:
            print(f"Agent error: {e}")

        # Enhanced fallback: prefer center moves over random
        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Enhanced fallback: prefer center positions over random."""
        legal_moves = game_state.get_legal_moves()
        
        # Prefer center positions (Manhattan distance from center)
        center_row, center_col = 3.5, 3.5
        legal_moves_sorted = sorted(legal_moves, key=lambda pos: 
            abs(pos[0] - center_row) + abs(pos[1] - center_col))
        
        # Take the most central legal move
        print(f"Fallback move: {legal_moves_sorted[0]} (most central)")
        return legal_moves_sorted[0]
