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
1) **INSTANT WIN**: If ANY move creates exactly 5-in-a-row, play it immediately
2) **CRITICAL BLOCK**: If opponent has 4-in-a-row with open end(s), BLOCK immediately

### LEVEL 2: STRONG TACTICAL MOVES
3) **CREATE WINNING THREAT**: Make open-four (_.XXXX._ or .XXXX._) to threaten win next turn
4) **FORK CREATION**: Create double-threat (two ways to win simultaneously)
5) **BLOCK OPPONENT THREATS**: Stop opponent's open-three (.OOO.) or growing patterns

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

## BOARD ANALYSIS METHOD
1. Scan ALL 8 directions from each existing piece
2. Count consecutive pieces + available extensions
3. Identify immediate threats (4-in-a-row)
4. Look for pattern completion opportunities

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

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Return the next move coordinates as (row, col)."""
        try:
            board_str = game_state.format_board(formatter="standard")
            me = game_state.current_player.value
            opp = "O" if me == "X" else "X"
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
                
                f"ANALYSIS CHECKLIST:\n"
                f"1. Scan for ANY {me} 4-in-a-row that needs one more piece to win\n"
                f"2. Scan for ANY {opp} 4-in-a-row that MUST be blocked immediately\n"
                f"3. Look for {me} 3-in-a-row patterns that can become 4-in-a-row\n"
                f"4. Look for {opp} 3-in-a-row patterns that need blocking\n"
                f"5. Find moves that extend your longest connected sequences\n"
                f"6. Prefer center area in early game, connectivity always\n\n"
                
                f"LEGAL MOVES (row,col): {legal_moves}\n\n"
                
                f"INSTRUCTIONS:\n"
                f"- Use the MANDATORY PRIORITY SYSTEM from your training\n"
                f"- State which priority level (1-4) you're using\n"
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
