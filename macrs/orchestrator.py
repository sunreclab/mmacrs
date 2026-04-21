from __future__ import annotations

import logging
import time
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from macrs.agents.ask import AskingAgent
from macrs.agents.chitchat import ChitChatAgent
from macrs.agents.recommend import RecommendingAgent
from macrs.models import AgentOutput, PlannerDecision, ReflectionUpdate
from macrs.planner import Planner
from macrs.state import ConversationState


class GraphState(TypedDict, total=False):
    user_message: str
    conversation_state: ConversationState
    router_decision: PlannerDecision
    selected_agent_output: AgentOutput
    reflection_update: ReflectionUpdate
    final_state: ConversationState


def _coerce_state(value: ConversationState | dict) -> ConversationState:
    if isinstance(value, ConversationState):
        return value
    return ConversationState.model_validate(value)


class Orchestrator:
    def __init__(self) -> None:
        self.ask_agent = AskingAgent()
        self.rec_agent = RecommendingAgent()
        self.chat_agent = ChitChatAgent()
        self.planner = Planner()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("start", self._start)
        graph.add_node("reflect", self._reflect_before)
        graph.add_node("planner", self._planner)
        graph.add_node("ask_agent", self._ask_agent)
        graph.add_node("recommend_agent", self._recommend_agent)
        graph.add_node("chitchat_agent", self._chitchat_agent)

        graph.set_entry_point("start")
        graph.add_edge("start", "reflect")
        graph.add_edge("reflect", "planner")
        graph.add_conditional_edges(
            "planner",
            lambda state: state["router_decision"].selected_act,
            {
                "ask": "ask_agent",
                "recommend": "recommend_agent",
                "chitchat": "chitchat_agent",
            },
        )
        graph.add_edge("ask_agent", END)
        graph.add_edge("recommend_agent", END)
        graph.add_edge("chitchat_agent", END)
        return graph.compile()

    def _start(self, state: GraphState) -> GraphState:
        return {}

    def _reflect_before(self, state: GraphState) -> GraphState:
        conv_state = _coerce_state(state["conversation_state"])
        user_message = state["user_message"]
        if conv_state.last_system_response:
            logging.getLogger("macrs.reflection").info("start")
            # Reflection logic can be added here if needed
            logging.getLogger("macrs.reflection").info("updated")
        return {"conversation_state": conv_state}

    def run_turn(self, state: ConversationState, user_message: str) -> GraphState:
        input_state: GraphState = {
            "user_message": user_message,
            "conversation_state": state,
        }
        return self.graph.invoke(input_state)

    def stream_turn(self, state: ConversationState, user_message: str):
        input_state: GraphState = {
            "user_message": user_message,
            "conversation_state": state,
        }
        yield from self.graph.stream(input_state, stream_mode="updates")

    def _router(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        logger = logging.getLogger("macrs.router")
        start = time.perf_counter()
        logger.info("start")
        decision = self.router.route(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logger.info("routed to act=%s in %.2fs", decision.selected_act, elapsed)
        return {"router_decision": decision}

    def _ask_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        router_decision = state["router_decision"]
        logger = logging.getLogger("macrs.agent.ask")
        start = time.perf_counter()
        logger.info("start")
        output = self.ask_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logger.info("done in %.2fs (confidence=%.2f, candidates=%d)", elapsed, output.confidence, len(output.candidates))
        
        # Finalize state with the agent output directly (no planner needed)
        final_state = self._finalize_state_from_agent_output(
            conv_state, user_message, output, router_decision.selected_act
        )
        return {"final_state": final_state, "conversation_state": final_state}

    def _recommend_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        router_decision = state["router_decision"]
        logger = logging.getLogger("macrs.agent.recommend")
        start = time.perf_counter()
        logger.info("start")
        output = self.rec_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logger.info("done in %.2fs (confidence=%.2f, candidates=%d)", elapsed, output.confidence, len(output.candidates))
        
        # Finalize state with the agent output directly (no planner needed)
        final_state = self._finalize_state_from_agent_output(
            conv_state, user_message, output, router_decision.selected_act
        )
        return {"final_state": final_state, "conversation_state": final_state}

    def _chitchat_agent(self, state: GraphState) -> GraphState:
        user_message = state["user_message"]
        conv_state = _coerce_state(state["conversation_state"])
        router_decision = state["router_decision"]
        logger = logging.getLogger("macrs.agent.chitchat")
        start = time.perf_counter()
        logger.info("start")
        output = self.chat_agent.run(user_message, conv_state)
        elapsed = time.perf_counter() - start
        logger.info("done in %.2fs (confidence=%.2f, candidates=%d)", elapsed, output.confidence, len(output.candidates))
        
        # Finalize state with the agent output directly (no planner needed)
        final_state = self._finalize_state_from_agent_output(
            conv_state, user_message, output, router_decision.selected_act
        )
        return {"final_state": final_state, "conversation_state": final_state}

    def _planner(self, state: GraphState) -> GraphState:
        conv_state = _coerce_state(state["conversation_state"])
        user_message = state["user_message"]
        logging.getLogger("macrs.planner").info("start")
        start = time.perf_counter()
        
        # Planner now acts as router: decide which agent to use
        decision = self.planner.route(user_message, conv_state)
        
        elapsed = time.perf_counter() - start
        logging.getLogger("macrs.planner").info(
            "routed to act=%s in %.2fs",
            decision.selected_act,
            elapsed,
        )
        return {"router_decision": decision}

    def _finalize_state_from_agent_output(
        self, 
        conv_state: ConversationState, 
        user_message: str, 
        output: AgentOutput, 
        selected_act: str
    ) -> ConversationState:
        """Finalize conversation state directly from a single agent's output."""
        # Pick the best candidate (highest score) from the agent's output
        best_candidate = max(output.candidates, key=lambda c: c.score)
        
        conv_state.record_act(selected_act)
        conv_state.turn_id += 1
        conv_state.last_user_message = user_message
        conv_state.last_system_response = best_candidate.response
        conv_state.append_dialogue(user_message, best_candidate.response, act=selected_act)
        if selected_act == "recommend":
            conv_state.last_recommendation_turn = conv_state.turn_id
        return conv_state
