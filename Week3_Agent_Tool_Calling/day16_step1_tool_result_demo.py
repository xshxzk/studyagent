from tools.tool_types import ToolResult, print_tool_result
from pydantic import BaseModel, Field, ValidationError


class SimpleTool:
    def __init__(self, name: str, description: str, args_schema: type[BaseModel], run):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.run = run


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A basic arithmetic expression, such as '12 * 3'.",
        min_length=1,
    )


def calculator(expression: str) -> ToolResult:
    allowed_chars = set("0123456789+-*/(). ")

    if not set(expression) <= allowed_chars:
        return ToolResult(
            success=False,
            content="Calculator failed.",
            error="Expression contains unsupported characters.",
        )

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return ToolResult(
            success=False,
            content="Calculator failed.",
            error=str(exc),
        )

    return ToolResult(
        success=True,
        content="Calculator finished successfully.",
        data={
            "expression": expression,
            "result": result,
        },
    )


calculator_tool = SimpleTool(
    name="calculator",
    description="Use this tool when you need to calculate a basic arithmetic expression.",
    args_schema=CalculatorInput,
    run=calculator,
)


def main() -> None:
    print("工具信息:")
    print(f"name: {calculator_tool.name}")
    print(f"description: {calculator_tool.description}")
    print(f"args_schema: {calculator_tool.args_schema.model_json_schema()}")

    print("成功工具调用:")
    valid_input = calculator_tool.args_schema(expression="12 * 3")
    print_tool_result(calculator_tool.run(valid_input.expression))

    print("\n失败工具调用:")
    try:
        invalid_input = calculator_tool.args_schema(expression="")
        print_tool_result(calculator_tool.run(invalid_input.expression))
    except ValidationError as exc:
        print_tool_result(
            ToolResult(
                success=False,
                content="Input validation failed.",
                error=str(exc),
            )
        )

    print("\n非法表达式工具调用:")
    invalid_expression = calculator_tool.args_schema(expression="open file")
    print_tool_result(calculator_tool.run(invalid_expression.expression))


if __name__ == "__main__":
    main()
