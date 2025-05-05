from typing import Any, TypeVar, Dict, List
from pathlib import Path
import asyncio
from jinja2 import StrictUndefined, Template

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, Tool, RunContext
from pydantic_ai.tools import ToolDefinition

from pydantic_ai.agent import Agent
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai_examples.local_python_executor import LocalPythonExecutor
from pydantic_ai_examples.xml_parser_model import XMLParserModel

OutputDataT = TypeVar("OutputDataT")
RunOutputDataT = TypeVar("RunOutputDataT")
StopSequences = ["<end_code>", "<observation>"]

class PythonTool(Tool[AgentDepsT]):
    executor:LocalPythonExecutor | None = None
    function_tools: Dict[str, Tool[AgentDepsT]]
    def __init__(
        self,
        name: str = "execute_code",
        description: str = "在安全的环境中执行Python代码并返回结果",
        additional_authorized_imports: list[str] = [],
        max_print_outputs_length: int = 10000,
        max_retries: int | None = 3,
        function_tools: Dict[str, Tool[AgentDepsT]] = {},
    ):
        self.function_tools = function_tools
        
        async def execute_code(code: str) -> str:
            if self.executor is None:
                self.executor = LocalPythonExecutor(
                    additional_authorized_imports=additional_authorized_imports,
                    max_print_outputs_length=max_print_outputs_length,
                )
                self.executor.send_tools(self.function_tools)
            try:
                _, logs, _ = self.executor(code)
                print(f"执行结2222果: {logs}")
                return logs
            except Exception as e:
                return f"执行错误: {str(e)}"
            
        super().__init__(
            function=execute_code,
            name=name,
            description=description,
            max_retries=max_retries,
        )

class CodeAgent(Agent[AgentDepsT, OutputDataT]): # type: ignore
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str,
        model_settings: ModelSettings | None = None,
        **kwargs: Any
    ):
        
        if model_settings is None:
            model_settings = ModelSettings(
                stop_sequences=StopSequences
            )
        else:
            if "stop_sequences" not in model_settings:
                model_settings["stop_sequences"] = []
            model_settings["stop_sequences"].extend(StopSequences)

        super().__init__(XMLParserModel(model),model_settings=model_settings, **kwargs)

        self.my_system_prompts = []
        self._system_prompt_functions.append(
            SystemPromptRunner(self._load_code_agent_prompt, dynamic=True)
        )
        self._register_tool(PythonTool(function_tools=self._function_tools))
            

    async def _get_tool_definitions(self, run_context: RunContext[AgentDepsT]) -> List[ToolDefinition]:
        """获取所有工具的定义，使用prepare_tool_def方法"""
        function_tools: Dict[str, Tool[AgentDepsT]] = self._function_tools
        tool_definitions: List[ToolDefinition] = []
        
        async def add_tool(tool: Tool[AgentDepsT]) -> None:
            if isinstance(tool, PythonTool):
                return None
            ctx = run_context.replace_with(retry=tool.current_retry, tool_name=tool.name)
            if tool_def := await tool.prepare_tool_def(ctx):
                tool_definitions.append(tool_def)
                
        await asyncio.gather(*map(add_tool, function_tools.values()))
        return tool_definitions

    async def _load_code_agent_prompt(self, run_context: RunContext[AgentDepsT]) -> str:
        """加载code_agent.md文件内容，并使用Jinja2渲染模板"""
        # 读取code_agent.md文件
        current_dir = Path(__file__).parent
        prompt_file = current_dir / "prompts" / "code_agent.md"
        
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                template_content = f.read()
        except FileNotFoundError:
            # 如果文件不存在，抛出错误
            raise FileNotFoundError(f"错误: 未找到提示词文件 {prompt_file}")
        
        # 获取工具定义
        tool_definitions = await self._get_tool_definitions(run_context)
        
        # 使用Jinja2渲染模板
        # StrictUndefined确保未定义的变量会抛出错误而不是默默地返回空字符串
        template = Template(template_content, undefined=StrictUndefined)
        rendered_prompt = template.render(tool_definitions=tool_definitions)
        
        return rendered_prompt
    
    
