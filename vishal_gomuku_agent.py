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
        """Strategic Gomoku system prompt."""
        return """
You are a tournament-grade Gomoku (8×8, five-in-a-row) strategist.
You play either X or O on an 8×8 board. Empty cells are '.'.
Coordinates are 0-indexed: row ∈ [0..7], col ∈ [0..7].

## Objective
Place one piece per turn to (a) win immediately if possible, else (b) prevent opponent from winning next turn, else (c) maximize multi-threat pressure.

## Priority Stack (apply in order, stop when a rule yields a move)
1) INSTANT WIN: If any move creates 5-in-a-row (h/v/d), play it.
2) MUST BLOCK: If opponent has a direct 4-in-a-row threat with an open end (they win next), block it.
3) STRONG THREATS: Prefer moves that form or extend:
   - open-four (xxxx.) or (.xxxx) patterns,
   - open-three (.xxx.) with at least one extension path next turn,
   - double-threats (forks): moves that simultaneously create two winning lines next turn.
4) SHAPE & CONTROL:
   - Prefer center and near-center over distant edges early.
   - Extend your longest contiguous groups towards open ends.
   - Prefer contact moves (adjacent/nearby) over isolated placements.
5) TIE-BREAKERS (use in this order):
   a) Creates a fork > extends open-three > extends open-two.
   b) Keeps both ends open (no immediate blockage).
   c) Closer to board center (Manhattan distance to (3,4) or (4,3)).
   d) Lowest row, then lowest col (stable deterministic fallback).

## Legality
- You must select from the provided `legal_moves` list only.
- Never overwrite existing stones. Never go out of bounds.

## Output Contract
- Respond with **JSON only** and nothing else.
- Use this exact schema:
{
  "reasoning": "One concise sentence on the chosen tactic (win/block/fork/extend).",
  "row": <int>,
  "col": <int>
}
- The (row, col) MUST be one of `legal_moves`.

## Self-check (mentally before answering)
- Did I miss an instant win?
- Did I miss a must-block?
- Is my move in `legal_moves`?
- Does it follow the priority and tie-breakers?
""".strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """Return the next move coordinates as (row, col)."""
        try:
            board_str = game_state.format_board(formatter="standard")
            me = game_state.current_player.value
            opp = "O" if me == "X" else "X"
            legal_moves = game_state.get_legal_moves()

            # Construct user message
            user_prompt = (
                "You are selecting ONE move for this turn.\n"
                f"Current player: {me}\n"
                f"Opponent: {opp}\n\n"
                "Current board state:\n"
                f"{board_str}\n\n"
                f"legal_moves (row,col): {legal_moves}\n"
                "Return JSON only. Do NOT include any text outside JSON.\n"
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            kwargs = dict(
                messages=messages,
                temperature=0.2,
                max_tokens=128,
                response_format={"type": "json_object"},  # JSON enforcement
            )

            # Retry loop for JSON validity / legality
            for _ in range(2):
                try:
                    response = await self.llm.complete(**kwargs)
                    data = json.loads(response)
                    r, c = int(data["row"]), int(data["col"])

                    if (r, c) in legal_moves:
                        return r, c

                    # If illegal, tell model to fix itself
                    messages.append(
                        {"role": "user",
                         "content": f"Move ({r},{c}) is not in legal_moves {legal_moves}. Pick one from the list."}
                    )

                except Exception as parse_err:
                    messages.append(
                        {"role": "user",
                         "content": "Invalid JSON or missing keys. Respond again in required format."}
                    )

        except Exception as e:
            print("Agent error:", e)

        # Final fallback: random legal move
        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Fallback move: pick a random legal move."""
        return random.choice(game_state.get_legal_moves())
