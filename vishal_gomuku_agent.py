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
You are an EXPERT Gomoku strategist. Your ONLY goal is to WIN by getting 5-in-a-row OR prevent opponent from getting 5-in-a-row.

## CRITICAL RULES (CHECK IN EXACT ORDER):

1. **INSTANT WIN CHECK**: If ANY legal move gives you 5-in-a-row, PLAY IT IMMEDIATELY!
   - Check horizontal: count your pieces left + new piece + your pieces right = 5?
   - Check vertical: count your pieces up + new piece + your pieces down = 5?  
   - Check diagonals: count pieces in both diagonal directions = 5?

2. **EMERGENCY BLOCK**: If opponent has 4-in-a-row anywhere, BLOCK IT IMMEDIATELY!
   - Look for patterns like: XXXX., .XXXX, or XX.XX (opponent pieces)
   - Block the empty spot that would give them 5-in-a-row

3. **PREVENT THREATS**: Block opponent's dangerous 3-in-a-row patterns:
   - Look for .XXX. (open three - very dangerous!)
   - Look for XXX. or .XXX (closed three - still dangerous)
   - Block one end to prevent them from reaching 4-in-a-row

4. **BUILD YOUR ATTACK**: Extend your longest sequences:
   - If you have 3+ pieces in a row, try to extend them
   - Create your own threats while blocking opponent

## EXAMPLES OF CRITICAL SITUATIONS:
- Opponent has (0,0)-(0,1)-(0,2)-(0,3): MUST block (0,4) immediately!
- Opponent has (4,2)-(4,3)-(4,4)-(4,5): MUST block (4,1) or (4,6) immediately!
- You have pieces at (3,2)-(3,3)-(3,4)-(3,5): Playing (3,6) = WIN!

## WHAT TO LOOK FOR ON BOARD:
- Horizontal lines: Count pieces in same row
- Vertical lines: Count pieces in same column  
- Diagonal lines: Count pieces on same diagonal (/ or \\)

## OUTPUT FORMAT:
{
  "reasoning": "Explain WHY this move (win/block/threat/extend)",
  "row": <number>,
  "col": <number>
}

REMEMBER: Blocking opponent threats is MORE IMPORTANT than creating your own!
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

            # Simplified but powerful user prompt
            user_prompt = (
                f"=== GOMOKU BATTLE ===\n"
                f"You are: {me}\n"
                f"Opponent: {opp}\n"
                f"Move #{move_count + 1}\n\n"
                
                f"BOARD:\n{board_str}\n\n"
                
                f"URGENT CHECKS (do these in order):\n"
                f"1. Can YOU win in 1 move? Check if placing {me} anywhere creates 5-in-a-row!\n"
                f"2. Can OPPONENT win in 1 move? Check if they have 4-in-a-row to block!\n"
                f"3. Does opponent have dangerous 3-in-a-row patterns? Block them!\n"
                f"4. Can you extend your longest sequence?\n\n"
                
                f"EXAMPLES FROM YOUR LOSING GAMES:\n"
                f"- SuperDuper won with (0,0)→(0,1)→(0,2)→(0,3)→(0,4) horizontally\n"
                f"- GomokuRobot won with (4,4)→(4,3)→(4,5)→(4,6)→(4,2) horizontally\n"
                f"- Don't let this happen again!\n\n"
                
                f"Legal moves: {legal_moves}\n\n"
                
                f"RESPOND WITH JSON:\n"
                f"- First check for wins and blocks\n"
                f"- Explain your reasoning clearly\n"
                f"- Choose coordinates from legal moves only\n"
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            kwargs = dict(
                messages=messages,
                temperature=0.0,  # Most deterministic for tactical decisions
                max_tokens=100,   # Shorter responses, more focused
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
