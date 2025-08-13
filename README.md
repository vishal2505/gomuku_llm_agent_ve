# Gomoku LLM Agent

A tournament-grade LLM-powered Gomoku agent designed for 8×8 five-in-a-row competition.

## Overview

This agent uses an LLM (Large Language Model) to play Gomoku strategically. It's designed to compete in a tournament environment where agents play head-to-head matches.

## Game Rules

- **Board Size**: 8×8 grid
- **Objective**: Get five pieces in a row (horizontally, vertically, or diagonally)
- **Players**: X and O take turns
- **Winning**: First to achieve five-in-a-row wins
- **Draw**: Board fills up without a winner

## Agent Strategy

The `VishalGomokuLLMAgent` implements a strategic priority system:

1. **INSTANT WIN**: Play winning moves immediately
2. **MUST BLOCK**: Block opponent's winning threats
3. **STRONG THREATS**: Create open-four, open-three, and double-threat patterns
4. **SHAPE & CONTROL**: Prefer center positions and extend longest groups
5. **TIE-BREAKERS**: Use deterministic fallbacks for move selection

## Technical Details

- **Model**: Uses `google/gemma-2-9b-it` LLM
- **Framework**: Built on the competition framework from https://github.com/sitfoxfly/gomoku-ai
- **Response Format**: JSON-enforced responses for reliability
- **Fallback**: Random legal move if LLM fails

## Files

- `vishal_gomuku_agent.py`: Main agent implementation

## Usage

The agent is designed to be integrated into the competition framework. It implements the required `Agent` interface and provides the `get_move()` method for turn-based gameplay.

## Competition

This agent is designed for the SMU MITB Gomoku tournament where agents compete in a leaderboard-style competition using various LLM models up to 10B parameters.
