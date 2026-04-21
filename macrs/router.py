from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from macrs.llm import generate_structured_output
from macrs.models import ActType
from macrs.state import ConversationState


class RouterDecision:
    def __init__(self, selected_act: ActType, rationale: str):
        self.selected_act = selected_act
        self.rationale = rationale


class RouterLLMOutput(BaseModel):
    selected_act: Literal["ask", "recommend", "chitchat"]
    rationale: str = Field(description="Brief explanation for the routing decision")


class AgentRouter:
    """
    Router that decides which agent (ask/recommend/chitchat) should handle
    the current user message based on conversation state.
    
    This runs BEFORE any agent execution to avoid wasting computation
    on agents that won't be used.
    """

    def route(self, user_message: str, state: ConversationState) -> RouterDecision:
        """
        Determine which agent should handle the current turn.
        
        Args:
            user_message: The current user message
            state: Current conversation state including profile, history, etc.
            
        Returns:
            RouterDecision with the selected act and rationale
        """
        sufficient = self._has_sufficient_preferences(state.user_profile)
        
        prompt = (
            "You are the Router for a multi-agent conversational recommender.\n"
            "Goal: select exactly ONE agent (ask / recommend / chitchat) to handle the user's message.\n\n"
            "Decision guidelines:\n"
            "1) If user preferences are insufficient (missing category/type), choose 'ask' to gather information.\n"
            "2) If preferences are sufficient and user is requesting recommendations, choose 'recommend'.\n"
            "3) If user message is casual/social or doesn't fit above cases, choose 'chitchat'.\n"
            "4) Avoid repeating the same act across multiple consecutive turns when possible.\n"
            "5) Consider engagement: keep conversation natural and forward-moving.\n\n"
            f"User message: {user_message}\n"
            f"User profile: {state.user_profile}\n"
            f"Browsing history: {state.browsing_history}\n"
            f"Dialogue history: {state.dialogue_history[-5:]}\n"
            f"Act history: {state.act_history}\n"
            f"Preference sufficiency: {sufficient}\n"
            "Return selected_act and a brief rationale."
        )
        
        llm_output = generate_structured_output(prompt, RouterLLMOutput)
        if not llm_output:
            # Fallback logic if LLM fails
            return self._fallback_route(user_message, state, sufficient)
        
        # Validate and potentially override decision
        selected_act = llm_output.selected_act
        
        # Override: if preferences are sufficient but router didn't select recommend,
        # and user seems to want recommendations, force recommend
        if sufficient and selected_act != "recommend":
            if self._looks_like_recommendation_request(user_message):
                selected_act = "recommend"
        
        return RouterDecision(selected_act=selected_act, rationale=llm_output.rationale)

    def _has_sufficient_preferences(self, profile: Dict[str, Any]) -> bool:
        """Check if user profile has enough information for recommendations."""
        if not profile:
            return False
        key_match = any(k in profile for k in ["category", "type", "product", "item"])
        return key_match and len(profile.keys()) >= 2

    def _looks_like_recommendation_request(self, user_message: str) -> bool:
        """Heuristic to detect if user is asking for recommendations."""
        text = user_message.lower()
        keywords = ["recommend", "suggest", "show me", "find", "looking for", "need"]
        return any(kw in text for kw in keywords)

    def _fallback_route(self, user_message: str, state: ConversationState, sufficient: bool) -> RouterDecision:
        """Fallback routing logic when LLM fails."""
        if not sufficient:
            return RouterDecision(selected_act="ask", rationale="Fallback: insufficient preferences")
        elif self._looks_like_recommendation_request(user_message):
            return RouterDecision(selected_act="recommend", rationale="Fallback: recommendation request detected")
        else:
            return RouterDecision(selected_act="chitchat", rationale="Fallback: default to chitchat")
