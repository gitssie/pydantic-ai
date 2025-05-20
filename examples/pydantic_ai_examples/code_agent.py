from typing import Any, TypeVar, Dict, List
from pathlib import Path
import asyncio
import re
from jinja2 import StrictUndefined, Template

from pydantic_ai import exceptions, models
from pydantic_ai.mcp import MCPServer
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, Tool, RunContext
from pydantic_ai.tools import ToolDefinition

from pydantic_ai.agent import Agent
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai_examples.local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor
from pydantic_ai_examples.xml_parser_model import XMLParserModel


OutputDataT = TypeVar("OutputDataT")
StopSequences = ["<end_code/>", "<tool_result>"]

class PythonTool(Tool[AgentDepsT]):
    code_pattern: str = r'```(?:[a-zA-Z_]+)?\s*([\s\S]*?)```'
    function_tools: Dict[str, Tool[AgentDepsT]]
    authorized_imports:list[str]
    additional_authorized_imports:list[str]

    def __init__(
        self,
        name: str = "run_python_code",
        description: str = "Execute Python code in a secure environment and return the results",
        additional_authorized_imports: list[str] | None = None,
        max_print_outputs_length: int = 10000,
        max_retries: int | None = 3,
        function_tools: Dict[str, Tool[AgentDepsT]] = {},
    ):
        self.function_tools = function_tools
        self.additional_authorized_imports = additional_authorized_imports or []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))

        async def run_python_code(ctx:RunContext[AgentDepsT], python_code: str) -> str:
            match = re.search(self.code_pattern, python_code)
            if match:
                python_code = match.group(1).strip()
            else:
                python_code = python_code.strip()
            
            executor:LocalPythonExecutor | None = getattr(ctx,'executor',None)
            if executor is None:
                executor = LocalPythonExecutor(
                    additional_authorized_imports=self.additional_authorized_imports,
                    max_print_outputs_length=max_print_outputs_length,
                )
                executor.send_tools(self.function_tools)
                setattr(ctx,'executor',executor)
            
            state = {'__ctx__': ctx}
            try:
                executor.send_variables(state) # type: ignore
                _, logs, _ = executor(python_code)
                return logs
            except BaseException as e:
                return f"Error: {str(e)}"
            
        super().__init__(
            function=run_python_code,
            name=name,
            description=description,
            max_retries=max_retries,
            takes_ctx=True
        )

class CodeAgent(Agent[AgentDepsT, OutputDataT]): # type: ignore
    _python_tool:PythonTool[AgentDepsT]

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str,
        model_settings: ModelSettings | None = None,
        additional_authorized_imports: List[str] |None = None,
        **kwargs: Any
    ):
        
        if model_settings is None:
            model_settings = ModelSettings(
                temperature=0,
                stop_sequences=StopSequences
            )
        else:
            if "stop_sequences" not in model_settings:
                model_settings["stop_sequences"] = []
            model_settings["stop_sequences"].extend(StopSequences)
            if "temperature" not in model_settings:
                model_settings["temperature"] = 0

        super().__init__(XMLParserModel(model),model_settings=model_settings, **kwargs)

        self._system_prompt_functions.append(
            SystemPromptRunner(self._load_code_agent_prompt, dynamic=True)
        )
        self._python_tool = PythonTool(function_tools=self._function_tools, additional_authorized_imports=additional_authorized_imports)
        #self._register_tool(self._python_tool)
            

    async def _get_tool_definitions(self, run_context: RunContext[AgentDepsT]) -> List[ToolDefinition]:
        tool_definitions: List[ToolDefinition] = []
        
        async def add_tool(tool: Tool[AgentDepsT]) -> None:
            if tool == self._python_tool:
                return None
            ctx = run_context.replace_with(retry=tool.current_retry, tool_name=tool.name)
            if tool_def := await tool.prepare_tool_def(ctx):
                tool_definitions.append(tool_def)

        await asyncio.gather(*map(add_tool, self._function_tools.values()))
        return tool_definitions

    async def _get_mcp_tool_definitions(self, run_context: RunContext[AgentDepsT]) -> List[ToolDefinition]:
        tool_definitions: List[ToolDefinition] = []
    
        async def add_mcp_server_tools(server: MCPServer) -> None:
            if not server.is_running:
                raise exceptions.UserError(f'MCP server is not running: {server}')
            tool_defs = await server.list_tools()
            # TODO(Marcelo): We should check if the tool names are unique. If not, we should raise an error.
            tool_definitions.extend(tool_defs)

        
        await asyncio.gather(*map(add_mcp_server_tools, self._mcp_servers))
        return tool_definitions


    async def _load_code_agent_prompt(self, run_context: RunContext[AgentDepsT]) -> str:
        current_dir = Path(__file__).parent
        prompt_file = current_dir / "prompts" / "code_agent_xml.md"
        
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                template_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Prompt file not found {prompt_file}")
        
        tools = await self._get_tool_definitions(run_context)
        mcp_tools = await self._get_mcp_tool_definitions(run_context)

        authorized_imports = self._python_tool.authorized_imports
        template = Template(template_content, undefined=StrictUndefined)
        rendered_prompt = template.render(tools=tools,mcp_tools=mcp_tools, authorized_imports=authorized_imports)
        
        return rendered_prompt