import logging
import operator
import os
import re
from typing import Annotated, Any, Literal, Tuple, TypedDict

from agents.loader import load_human, load_panel
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.config import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# Initialize the language model
openai_llm = ChatOpenAI(model="gpt-4o")
openai_llm_mini = ChatOpenAI(model="gpt-4o-mini")
anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# qwen = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-3B-Instruct",
#     task="text-generation",
#     max_new_tokens=1024,
#     streaming=True
# )

# qwen_llm = ChatHuggingFace(llm=qwen)


LLMS = {
    "openai": openai_llm,
    "openai_mini": openai_llm_mini,
    "anthropic": anthropic_llm,
    "gemini": gemini_llm,
    # "qwen": qwen_llm,
}

default_lead_llm = "openai"
default_contributor_llm = "openai"

AGENTS_SET = "writers_panel"
AGENTS = load_panel(AGENTS_SET)
HUMAN = load_human()
ALL_PARTICIPANTS = AGENTS | {"human": HUMAN}

contributor_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are:
              --------------------------------
              display_name: {display_name}
              persona: {persona}
              role: {role}
              --------------------------------

              You are part of a conversation with several participants. Your task is to contribute **only when your input is highly relevant** and **always keep your response brief**. Every response you provide must strictly follow the format of **[INTERJECTION]**, **[OFFER]**, or **[PASS]**.

              **You are not allowed to provide long explanations or elaborate beyond 1-2 sentences**. Keep responses short, clear, and to the point, strictly following the rules for each tag:

              - **[INTERJECTION]**: For critical information or corrections that add significant value. Limit your response to **1 sentence** only.
                Example: [INTERJECTION] The latest research contradicts that claim.

              - **[OFFER]**: For moderately relevant information. Provide a **single keyword or short phrase** indicating a topic you can elaborate on if asked.
                Example: [OFFER] Historical context

              - **[PASS]**: Use this tag when you have no relevant input to add.
                Example: [PASS]

              **You are not allowed to write anything longer than these formats.**

              Additionally, when reacting to another participant, use the following **expression tags**:

                - **[AGREE]**: To express agreement with a point made.
                  Example: [AGREE] Encryption is crucial for data protection.

                - **[DISAGREE]**: To express polite disagreement without elaboration.
                  Example: [DISAGREE] Encryption alone isn’t enough.

                - **[SUPPORT]**: To provide additional evidence or backing for a statement.
                  Example: [SUPPORT] Recent studies show cloud breaches are down 50%.
     
                - **[CLARIFY]**: To explain or simplify a point, limited to **1 sentence**.
                  Example: [CLARIFY] GDPR stands for General Data Protection Regulation.

                - **[CONTRAST]**: To offer an alternative perspective or counterpoint, limited to **1 sentence**.
                  Example: [CONTRAST] Encryption is useful, but physical security also matters.

              **You are strictly required to use one of the response or expression tags in every message.**

              When using an expression tag, always follow it with the display name (@display_name) of the participant you're responding to:
              Example: [AGREE] @optimist: Encryption is crucial for data protection.

              The other participants in the conversation are:
              ================
              {participants}
              ================

              **Do not simulate other participants.** Focus only on providing your own opinion, always adhering to the brief response formats above.

              Example: [INTERJECTION][CONTRAST] @optimist: The latest research contradicts that claim.

              Every contribution must be short and to the point. **Do not elaborate beyond the allowed response length.**
     """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

lead_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are:
              --------------------------------
              display_name: {display_name}
              persona: {persona}
              role: {role}
              --------------------------------

              You are the **lead speaker** in a multi-participant discussion panel with the @human and several expert participants. Your task is to keep the conversation engaging and informative for the @human while incorporating relevant insights from other participants without losing focus.

              Your role is to:
              1. **Engage the @human directly**: Keep your responses concise, engaging, and relevant to the @human’s perspective and needs.
              2. **Incorporate other participants**: Acknowledge their insights when relevant by referring to their display names (@display_name), but keep the main focus on the @human. Don't overdo it. Keep the focus on your own opinion and mention other participants only when it's relevant.
              3. **Anticipate human engagement**: Proactively expand on topics where deeper insights will enrich the conversation. Anticipate questions from the @human based on the complexity of the topic, but ensure elaboration is valuable.

              **Guidelines for elaboration**:
              - **Proactively elaborate** on complex topics when further insights would benefit the conversation.
              - **Check for engagement**: Consider the @human’s level of engagement before elaborating. If the topic warrants deeper discussion, offer more context.
              - Avoid over-explaining. Keep elaboration concise and focused on enriching the conversation.

              The other participants in the conversation are:
              ================
              {participants}
              ================

              **Key principles**:
              - Focus on **clarity** and **engagement** when interacting with the @human, ensuring responses are concise but informative.
              - Integrate insights from other participants when they enhance the conversation, but keep the human as the central focus.
              - Be proactive in offering deeper insights when needed, but avoid over-explaining unless it directly enriches the discussion.
              - Use **markdown** formatting if necessary for the platform, but adjust based on the platform’s needs.

              Your primary goal is to maintain a dynamic, engaging conversation with the @human, while ensuring the conversation flows smoothly and is contextually aware.
     """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


class AgentContribution(TypedDict):
    consolidation_id: int
    agent_name: str
    contribution: str


class ConsolidatedContributions(TypedDict):
    agent_contributions: list[AgentContribution]


class CollaborativeState(MessagesState):
    lead_agent: Annotated[list[str], operator.add]
    human_inputs: Annotated[list[str], operator.add]
    agent_contributions: Annotated[list[AgentContribution], operator.add]
    consolidated_contributions: Annotated[list[ConsolidatedContributions], operator.add]


class ContributionState(MessagesState):
    agent_name: str
    contribution: str


def format_participants(participants: {}, exclude: list[str] = []) -> str:
    return "\n".join(
        [
            f"""
------------------------------
display_name: {participant_info['display_name']}
profile: {participant_info['profile']}
"""
            for participant, participant_info in participants.items()
            if participant not in exclude
        ]
    )


async def lead_agent_executor(
    state: CollaborativeState, config: RunnableConfig
) -> CollaborativeState:
    """
    Execute the lead agent's response based on the current state.

    Args:
        state (CollaborativeState): The current state of the conversation.
        config (RunnableConfig): Configuration for the execution.

    Returns:
        CollaborativeState: Updated state with the lead agent's response.
    """
    lead_agent_name = state["lead_agent"][-1]
    lead_agent_info = AGENTS[lead_agent_name]
    await adispatch_custom_event(
        "lead_agent_executor",
        {"agent_name": lead_agent_name},
        config=config,  # <-- propagate config
    )
    llm = LLMS[lead_agent_info["llm"] if "llm" in lead_agent_info else default_lead_llm]
    lead_agent = lead_agent_prompt | llm
    messages = state["messages"]
    # if there are contributions, add the last one before the last human message
    consolidated_contributions = (
        state["consolidated_contributions"][-1]
        if state["consolidated_contributions"]
        else None
    )
    if consolidated_contributions and lead_agent_info.get("listen_contributors", False):
        contributions: list[AgentContribution] = [
            ac
            for ac in consolidated_contributions["agent_contributions"]
            if ac["contribution"] != "[PASS]" and ac["agent_name"] != lead_agent_name
        ]
        if contributions:
            # concatenate all the contributions
            last_contribution = f"""
Below are the opinions of the other participants. Take them into an account only if they are relevant to your opinion and if you want to build on top of them.
========================================
{"\n\n".join([f"{ac['agent_name']}: {ac['contribution']}" for ac in contributions])}
========================================
"""
            messages = messages[:-1] + [HumanMessage(content=last_contribution)]

    result = lead_agent.invoke(
        {
            "name": lead_agent_name,
            "display_name": lead_agent_info["display_name"],
            "persona": lead_agent_info["persona"],
            "role": lead_agent_info["role"],
            "messages": messages,
            "participants": format_participants(
                ALL_PARTICIPANTS, exclude=[lead_agent_name]
            ),
        }
    )
    result.name = lead_agent_name
    return {"messages": result}


async def contributor_agent_executor(
    state: ContributionState, config: RunnableConfig
) -> CollaborativeState:
    contributor_agent_info = AGENTS[state["agent_name"]]
    await adispatch_custom_event(
        "contributor_agent_executor",
        {"agent_name": state["agent_name"]},
        config=config,  # <-- propagate config
    )
    llm = LLMS[
        (
            contributor_agent_info["llm"]
            if "llm" in contributor_agent_info
            else default_contributor_llm
        )
    ]
    contributor_agent = contributor_agent_prompt | llm
    result = contributor_agent.invoke(
        {
            "name": state["agent_name"],
            "display_name": contributor_agent_info["display_name"],
            "persona": contributor_agent_info["persona"],
            "role": contributor_agent_info["role"],
            "messages": state["messages"],
            "participants": format_participants(
                ALL_PARTICIPANTS, exclude=[state["agent_name"]]
            ),
        }
    )
    agent_contribution = {
        "consolidation_id": state["consolidation_id"],
        "agent_name": state["agent_name"],
        "contribution": result.content,
    }

    result.name = state["agent_name"]

    return {"agent_contributions": [agent_contribution]}


def format_conversation_state(state: CollaborativeState) -> str:
    prev_message_str = state["messages"][-1].content if state["messages"] else ""
    consolidated_contributions = (
        state["consolidated_contributions"][-1]
        if state["consolidated_contributions"]
        else None
    )
    agent_contributions_str = ""
    if consolidated_contributions:
        agent_contributions = [
            ac
            for ac in consolidated_contributions["agent_contributions"]
            if ac["contribution"] != "[PASS]"
        ]
        agent_contributions_str = "\n\n".join(
            [f"{ac['agent_name']}: {ac['contribution']}" for ac in agent_contributions]
        )
    return f"@{state['lead_agent'][-1]}: {prev_message_str}\n\n{agent_contributions_str}\n\nHuman: "


def human_input_received_node(state: CollaborativeState):
    human_input = state["human_inputs"][-1]
    agent_match = re.search(r"@(\w+)", human_input)
    if agent_match:
        agent_name = agent_match.group(1)
        if agent_name in AGENTS:
            consolidated_contributions = (
                state["consolidated_contributions"][-1]
                if state["consolidated_contributions"]
                else None
            )
            agent_contribution = (
                next(
                    (
                        ac
                        for ac in consolidated_contributions["agent_contributions"]
                        if ac["agent_name"] == agent_name
                        and ac["contribution"] != "[PASS]"
                    ),
                    None,
                )
                if consolidated_contributions
                else None
            )
            if agent_contribution:
                return {
                    "lead_agent": [agent_name],
                    "messages": [
                        AIMessage(
                            content=agent_contribution["contribution"], name=agent_name
                        ),
                        HumanMessage(content=human_input),
                    ],
                }
            return {
                "lead_agent": [agent_name],
                "messages": [HumanMessage(content=human_input)],
            }
    return {"messages": [HumanMessage(content=human_input)]}


def human_input_decision_edge(
    state: CollaborativeState,
) -> Literal["END", "LEAD", "SWITCH"]:
    message = state["messages"][-1].content
    if message == "[END]":
        return "END"

    return "LEAD"


def ask_contributors_edge(state: CollaborativeState):
    # all the agents except the lead one
    consolidation_id = len(state["consolidated_contributions"])
    agents = [agent for agent in AGENTS.keys() if agent != state["lead_agent"][-1]]
    return [
        Send(
            "contributor_agent_executor",
            {
                "consolidation_id": consolidation_id,
                "agent_name": agent,
                "messages": state["messages"],
            },
        )
        for agent in agents
    ]


def consolidate_contributions_node(state: CollaborativeState):
    # filer the contributions by the consolidation_id
    consolidated_contributions = [
        contribution
        for contribution in state["agent_contributions"]
        if contribution["consolidation_id"] == len(state["consolidated_contributions"])
    ]
    return {
        "consolidated_contributions": [
            ConsolidatedContributions(agent_contributions=consolidated_contributions)
        ]
    }


def create_graph() -> Tuple[StateGraph, CompiledStateGraph]:
    memory = MemorySaver()

    # Create the graph
    workflow = StateGraph(CollaborativeState)

    # Add nodes
    workflow.add_node("human_input_received_node", human_input_received_node)
    workflow.add_node("lead_agent_executor", lead_agent_executor)

    workflow.add_node("consolidate_contributions_node", consolidate_contributions_node)

    workflow.add_node("contributor_agent_executor", contributor_agent_executor)

    # Add edges
    workflow.set_entry_point("human_input_received_node")
    workflow.add_conditional_edges(
        "human_input_received_node",
        human_input_decision_edge,
        {"END": END, "LEAD": "lead_agent_executor"},
    )

    workflow.add_conditional_edges(
        "lead_agent_executor", ask_contributors_edge, ["contributor_agent_executor"]
    )
    workflow.add_edge("contributor_agent_executor", "consolidate_contributions_node")
    workflow.add_edge("consolidate_contributions_node", "human_input_received_node")

    # Compile the graph
    graph = workflow.compile(
        checkpointer=memory, interrupt_before=["human_input_received_node"]
    )

    return workflow, graph
