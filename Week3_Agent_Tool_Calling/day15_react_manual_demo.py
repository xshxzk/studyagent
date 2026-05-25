from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolResult:
    success: bool
    content: str


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], ToolResult]


def calculator(expression: str) -> ToolResult:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return ToolResult(
            success=False,
            content="Calculator only accepts numbers and basic operators.",
        )

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return ToolResult(
            success=False,
            content=f"Failed to calculate expression: {exc}",
        )

    return ToolResult(
        success=True,
        content=f"{expression} = {result}",
    )


def paper_search_stub(query: str) -> ToolResult:
    knowledge = {
        "open-ice": (
            "OPEN-ICE classifies Landsat 7 ETM+, Landsat 8 OLI, and Sentinel-2 MSI "
            "imagery into ice, water, and clouds, combines observations into a denser "
            "time series, applies a temporal filter, and estimates spring breakup dates."
        ),
        "sensors": (
            "The study uses Landsat 7 ETM+, Landsat 8 OLI, and Sentinel-2 MSI scenes "
            "over lakes across northern Canada."
        ),
        "bias": (
            "Compared with Canadian Ice Service observations, OPEN-ICE reported mean "
            "bias errors of -1.10 days for breakup start and -0.69 days for breakup end."
        ),
    }

    normalized_query = query.lower()
    for keyword, answer in knowledge.items():
        if keyword in normalized_query:
            return ToolResult(success=True, content=answer)

    return ToolResult(
        success=False,
        content="No matching paper snippet found in the stub knowledge base.",
    )


TOOLS = {
    "calculator": Tool(
        name="calculator",
        description="Use this tool for arithmetic expressions.",
        run=calculator,
    ),
    "paper_search": Tool(
        name="paper_search",
        description="Use this tool to search a small stub of the Week2 lake ice paper.",
        run=paper_search_stub,
    ),
}


def choose_tool(question: str) -> tuple[str, str, str]:
    normalized_question = question.lower()

    if any(keyword in normalized_question for keyword in ["open-ice", "sensor", "bias"]):
        return (
            "The question asks about the Week2 paper, so I should search paper context.",
            "paper_search",
            question,
        )

    if looks_like_math_expression(question):
        return (
            "I need an exact arithmetic result, so I should use the calculator tool.",
            "calculator",
            question,
        )

    return (
        "I do not need a tool because this is a general conceptual question.",
        "none",
        "",
    )


def looks_like_math_expression(question: str) -> bool:
    has_digit = any(char.isdigit() for char in question)
    has_operator = any(operator in question for operator in ["+", "-", "*", "/"])
    allowed_chars = set("0123456789+-*/(). ")
    return has_digit and has_operator and set(question) <= allowed_chars


def run_manual_react(question: str) -> None:
    print("=" * 80)
    print(f"User Question: {question}")

    thought, tool_name, tool_input = choose_tool(question)
    print(f"Thought: {thought}")

    if tool_name == "none":
        print("Action: None")
        print("Observation: No external tool result.")
        print(
            "Final Answer: An Agent is an LLM-driven controller that can decide when "
            "to use tools, observe their outputs, and then produce an answer."
        )
        return

    tool = TOOLS[tool_name]
    print(f"Action: {tool.name}")
    print(f"Action Input: {tool_input}")

    observation = tool.run(tool_input)
    print(f"Observation: {observation.content}")

    if observation.success:
        print(f"Thought: The tool returned useful evidence, so I can answer now.")
        print(f"Final Answer: {observation.content}")
    else:
        print("Thought: The tool failed, so I should explain the limitation.")
        print(f"Final Answer: I could not complete the tool call. {observation.content}")


def main() -> None:
    questions = [
        "12 * (8 + 4)",
        "What are the main steps of the OPEN-ICE algorithm?",
        "What is an Agent?",
    ]

    for question in questions:
        run_manual_react(question)


if __name__ == "__main__":
    main()
