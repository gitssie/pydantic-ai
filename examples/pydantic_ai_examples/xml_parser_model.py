from __future__ import annotations

from collections.abc import AsyncIterator, AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union, cast, Type
from abc import ABC, abstractmethod

import pydantic
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ModelResponse,
    ModelResponsePart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage
from pydantic_ai.models import KnownModelName, Model, ModelRequestParameters, StreamedResponse
import re


class XMLNode:
    tags: list[str] = []
    type: Optional[str]
    content: str
    delta: str
    buffer: str
    tag_name: Optional[str]
    tool_name: Optional[str]
    pos: int
    next_pos: int
    end: bool
    emit: bool
    part_id: str

    def __init__(self, tags: list[str], buffer: str, pos: int, type: Optional[str] = None,
                 tag_name: Optional[str] = None):
        self.tags = tags
        self.buffer = buffer
        self.pos = pos
        self.next_pos = pos
        self.type = type or "text"
        self.tag_name = tag_name
        self.tool_name = ""
        self.content = ""
        self.delta = ""
        self.end = False
        self.emit = False
        self.part_id = f"node_{id(self)}"

    def has_delta(self) -> bool:
        return not self.delta.isspace()

    def feed(self, data: str, last: bool = False) -> XMLNode:
        self.buffer += data
        if self.end:
            node = XMLNode(
                tags=self.tags,
                type="text",
                buffer=self.buffer,
                pos=self.pos,
            )
            node = node._process_node(last)
            node.content = self._skip_prefix_whitespace(node.content)
            node.delta = self._skip_prefix_whitespace(node.delta)
        else:
            is_start = len(self.content) == 0
            node = self._process_node(last)
            if is_start:
                node.content = self._skip_prefix_whitespace(node.content)
                node.delta = self._skip_prefix_whitespace(node.delta)

        node.end = node.end or last
        if node.end:
            node.content = self._skip_suffix_whitespace(node.content)
            node.delta = self._skip_suffix_whitespace(node.delta)

        return node

    def complete(self) -> Optional[XMLNode]:
        if self.pos >= len(self.buffer):
            return None
        return self.feed("", last=True)

    def _find_node(self, last: bool) -> tuple[int, int] | XMLNode:
        tag_start = self.buffer.find('<', self.pos)
        if tag_start == -1:
            delta = self.buffer[self.pos:]
            self.delta += delta
            self.content += delta
            self.pos = len(self.buffer)
            self.end = last
            return self

        tag_start += 1
        tag_end = self.buffer.find('>', tag_start)

        if tag_end == -1:
            if last or not self._is_valid_tag(tag_start, len(self.buffer)) or tag_start + 16 < len(self.buffer):
                delta = self.buffer[self.pos:]
                self.delta += delta
                self.content += delta
                self.pos = len(self.buffer)
                self.end = last
            return self

        return tag_start, tag_end

    def _process_continue(self, last: bool, next_pos: int) -> XMLNode:
        delta = self.buffer[self.pos:next_pos]
        self.delta += delta
        self.content += delta
        self.pos = next_pos
        self.end = last
        return self._process_node(last)
    
    def _process_node(self, last: bool = False) -> XMLNode:
        if self.pos >= len(self.buffer):
            return self

        if self.type == "text":
            node = self._find_node(last)
            if isinstance(node, XMLNode):
                return node

            tag_start, tag_end = node
            tag_name = self.buffer[tag_start:tag_end]
            next_pos = tag_end + 1
            if not tag_name in self.tags:
                return self._process_continue(last, next_pos)
            
            tag_start -= 1
            if tag_start > self.pos:
                delta = self.buffer[self.pos:tag_start]
                self.delta += delta
                self.content += delta
                self.pos = tag_start
                self.end = True
                return self

            next_pos = self._skip_whitespace(next_pos)

            return XMLNode(
                tags=self.tags,
                type="xml",
                tag_name=tag_name,
                buffer=self.buffer,
                pos=next_pos,
            )._process_node(last)
        else:
            node = self._find_node(last)
            if isinstance(node, XMLNode):
                return node

            tag_start, tag_end = node
            tag_name = self.buffer[tag_start + 1:tag_end]
            next_pos = tag_end + 1
            if tag_name != self.tag_name:
                return self._process_continue(last, next_pos)

            next_pos = self._skip_whitespace(next_pos)

            delta = self.buffer[self.pos:tag_start - 1]
            self.delta += delta
            self.content += delta
            self.pos = next_pos
            self.end = True
            return self

    def __str__(self) -> str:
        return f"XMLNode(type={self.type}, name={self.tag_name}, pos={self.pos}, content={self.content})"

    def __repr__(self) -> str:
        return self.__str__()

    def _skip_whitespace(self, pos: int) -> int:
        i = pos
        length = len(self.buffer)
        while i < length and (self.buffer[i].isspace() or self.buffer[i] == '\n'):
            i += 1
        return i

    def _skip_prefix_whitespace(self, content: str) -> str:
        i = 0
        length = len(content)
        while i < length and (content[i].isspace() or content[i] == '\n'):
            i += 1
        return content[i:]

    def _skip_suffix_whitespace(self, content: str) -> str:
        i = len(content) - 1
        while i >= 0 and (content[i].isspace() or content[i] == '\n'):
            i -= 1
        return content[:i + 1]

    def _is_valid_tag(self, start_pos: int, end_pos: int) -> bool:
        while start_pos < end_pos:
            char = self.buffer[start_pos]
            if not (char.isalpha() or char == '_' or char == '/'):
                return False
            start_pos += 1
        return True


class XMLHandler(ABC):
    @abstractmethod
    def can_handle(self, chunk: XMLNode) -> bool:
        pass

    @abstractmethod
    def get_parsed_tags(self) -> list[str]:
        pass

    @abstractmethod
    def handle(self, parts_manager: Any, chunk: XMLNode, delta: str) -> Optional[ModelResponseStreamEvent]:
        pass

    @abstractmethod
    def part(self, node: XMLNode) -> Optional[ModelResponsePart]:
        pass


class TextHandler(XMLHandler):
    tags: list[str] = ["thought"]
    markdown_start = r'```.*$'
    markdown = r'(```.+$)|(^```$)'
    pattern = r"</(" + "|".join(tag for tag in tags) + r")>"

    def get_parsed_tags(self) -> list[str]:
        return self.tags

    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.type == "text" or chunk.tag_name in self.tags or chunk.tag_name is None

    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta: str) -> Optional[ModelResponseStreamEvent]:
        delta = self.filter(chunk,delta)
        if not delta:
            return None
        return parts_manager.handle_text_delta(vendor_part_id=chunk.part_id, content=delta)

    def filter(self, chunk: XMLNode, delta: str) -> str:
        delta = re.sub(self.pattern, "", delta)
        block_start = re.search(self.markdown_start, delta)
        if block_start:
            start_pos = block_start.start()
            if chunk.content.find('```',0,len(chunk.content) - len(delta)) == -1:
                chunk.delta = delta[start_pos:]
                return delta[:start_pos]
        return delta

    def filter_part(self, node: XMLNode) -> bool:
        node.content = re.sub(self.pattern, "", node.content )
        node.content = re.sub(self.markdown, "", node.content )
        return bool(node.content)

    def part(self, node: XMLNode) -> Optional[ModelResponsePart]:
        if self.filter_part(node):
            return None
        return TextPart(content=node.content)


class ExecuteCodeToolHandler(XMLHandler):
    code_arg: str = "python_code"
    tags: list[str] = ["run_python_code"]

    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.tag_name in self.tags

    def get_parsed_tags(self) -> list[str]:
        return self.tags

    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta: str) -> Optional[ModelResponseStreamEvent]:
        if chunk.emit:
            return None
        tool_name = ""
        if not chunk.tool_name:
            tool_name = chunk.tag_name
            chunk.tool_name = tool_name

        args = None
        if chunk.end:
            args = {self.code_arg: chunk.content}

        return parts_manager.handle_tool_call_delta(vendor_part_id=chunk.part_id, tool_name=tool_name, args=args,
                                                    tool_call_id=chunk.part_id)

    def part(self, node: XMLNode) -> Optional[ModelResponsePart]:
        tool_name = cast(str, node.tag_name)
        args = {self.code_arg: node.content}
        return ToolCallPart(tool_name=tool_name, args=args, tool_call_id=node.part_id)


class MCPToolHandler(XMLHandler):
    tags: list[str] = ["tool_code", "use_mcp_tool", "use_tool"]

    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.tag_name in self.tags

    def get_parsed_tags(self) -> list[str]:
        return self.tags

    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta: str) -> Optional[
        ModelResponseStreamEvent]:
        tool_name = ""
        if not chunk.tool_name:
            tool_name = self._extract_tag_content(chunk.content, "tool_name")
            chunk.tool_name = tool_name

        args = None
        if chunk.end:
            _, args = self._extract_data(chunk.content)

        return parts_manager.handle_tool_call_delta(
            vendor_part_id=chunk.part_id,
            tool_name=tool_name,
            args=args,
            tool_call_id=chunk.part_id
        )

    def _extract_data(self, content: str) -> tuple[str, str]:
        tool_name = self._extract_tag_content(content, "tool_name") or ""
        arguments_str = self._extract_tag_content(content, "arguments") or ""
        return tool_name, arguments_str

    def _extract_tag_content(self, content: str, tag_name: str) -> Optional[str]:
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def part(self, node: XMLNode) -> Optional[ModelResponsePart]:
        tool_name, arguments_str = self._extract_data(node.content)

        return ToolCallPart(tool_name=tool_name,
                            args=arguments_str,
                            tool_call_id=node.part_id)


@dataclass
class HandlersFactory(XMLHandler):
    _tool_handlers: list[XMLHandler] = field(default_factory=lambda: [
        ExecuteCodeToolHandler(),
        MCPToolHandler(),
        TextHandler(),
    ])

    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.tags = [tag for handler in self._tool_handlers for tag in handler.get_parsed_tags()]

    def _get_tool_handler(self, chunk: XMLNode) -> XMLHandler:
        for handler in self._tool_handlers:
            if handler.can_handle(chunk):
                return handler
        return self._tool_handlers[-1]

    def can_handle(self, chunk: XMLNode) -> bool:
        raise NotImplementedError("HandlersFactory is not meant to be used as a handler")

    def get_parsed_tags(self) -> list[str]:
        return self.tags

    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta: str) -> Optional[ModelResponseStreamEvent]:
        if chunk.emit:
            return None

        chunk.delta = ""
        event = self._get_tool_handler(chunk).handle(parts_manager, chunk, delta)
        chunk.emit = chunk.end
        return event

    def part(self, node: XMLNode) -> Optional[ModelResponsePart]:
        return self._get_tool_handler(node).part(node)


@dataclass
class XMLStreamedResponse(StreamedResponse):
    _model_name: str
    _xml_stream: AsyncIterator[XMLNode]
    _usage_getter: Callable[[], Usage]
    _handlers: HandlersFactory
    _timestamp: datetime = field(default_factory=lambda: datetime.now())
    _text_part: list[ModelResponsePart] = field(default_factory=list)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        buffer: str = ""
        async for chunk in self._xml_stream:
            if not isinstance(chunk, XMLNode):
                yield chunk
                continue

            buffer = chunk.buffer
            event = self._handlers.handle(self._parts_manager, chunk, chunk.delta)
            if event is not None:
                yield event

        self._text_part.append(TextPart(content=buffer))

    def get(self) -> ModelResponse:
        response = XMLParsedModelResponse(
            raw_parts=self._text_part,
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp
        )
        return response

    def usage(self) -> Usage:
        return self._usage_getter()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class XMLParsedModelResponse(ModelResponse):
    raw_parts: list[ModelResponsePart] = field(default_factory=list)
    kind: Literal['xml_response'] = 'xml_response'  # type: ignore


# 定义一个装饰器函数，用于在类定义后自动调用指定方法
def auto_install(cls: Type[Any]) -> Type[Any]:
    if hasattr(cls, 'install_xml_parser_support'):
        getattr(cls, 'install_xml_parser_support')()
    return cls


@auto_install
@dataclass(init=False)
class XMLParserModel(Model):
    wrapped: Model
    _handlers: HandlersFactory
    _support_installed: bool = False

    @classmethod
    def install_xml_parser_support(cls) -> bool:
        if cls._support_installed:
            return True

        setattr(
            ModelMessagesTypeAdapter,
            '_type',
            list[Annotated[Union[ModelRequest, ModelResponse, XMLParsedModelResponse], pydantic.Discriminator('kind')]]
        )
        cls._support_installed = True
        return True

    def __init__(self, wrapped: Model | KnownModelName | str):
        from pydantic_ai.models import infer_model
        self.wrapped = infer_model(wrapped)
        self._handlers = HandlersFactory()

    async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        messages = self._build_message(messages)
        model_response, usage = await self.wrapped.request(messages, model_settings, model_request_parameters)

        text_content = ""
        non_text_parts: list[ModelResponsePart] = []
        parts: list[ModelResponsePart] = model_response.parts

        for part in parts:
            if isinstance(part, TextPart):
                text_content += part.content
            else:
                non_text_parts.append(part)

        if not text_content:
            return model_response, usage

        parsed_parts: list[ModelResponsePart] = []

        node = XMLNode(tags=self._handlers.tags, buffer=text_content, pos=0)
        while node:
            node = node.complete()
            if node is None:
                break

            part = self._handlers.part(node)
            if part is not None:
                parsed_parts.append(part)

        parsed_parts.extend(non_text_parts)

        parsed_response = XMLParsedModelResponse(
            parts=parsed_parts,
            model_name=model_response.model_name,
            timestamp=model_response.timestamp,
            raw_parts=parts
        )
        return parsed_response, usage

    @asynccontextmanager
    async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        messages = self._build_message(messages)
        async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters
        ) as response_stream:
            xml_stream = self._wrap_response_stream(response_stream, self._handlers.tags)
            yield XMLStreamedResponse(
                _model_name=self.model_name,
                _xml_stream=xml_stream,
                _handlers=self._handlers,
                _usage_getter=lambda: response_stream.usage(),
            )

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        model_request_parameters.function_tools = []
        return self.wrapped.customize_request_parameters(model_request_parameters)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    async def _wrap_response_stream(self, response_stream: StreamedResponse, tags: list[str]) -> AsyncGenerator[
        Any, None]:
        node = XMLNode(tags=tags, buffer="", pos=0)
        async for event in response_stream:
            text_chunk = None

            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                text_chunk = event.part.content
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                text_chunk = event.delta.content_delta

            if text_chunk:
                node = node.feed(text_chunk)
                yield node
            else:
                yield event

        while node:
            yield node
            node = node.complete()

    def _build_message(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        new_messages: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelResponse) and hasattr(message, "raw_parts"):
                new_messages.append(ModelResponse(
                    parts=cast(list[ModelResponsePart], getattr(message, "raw_parts")),
                    model_name=message.model_name,
                    timestamp=message.timestamp
                ))
            elif isinstance(message, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in message.parts:
                    if (isinstance(part, RetryPromptPart)):
                        new_parts.append(
                            UserPromptPart(content=f"<tool_result>\n{part.model_response()}\n</tool_result>"))
                    elif isinstance(part, ToolReturnPart):
                        new_parts.append(UserPromptPart(
                            content=f"<tool_result>\nLogs:\n{part.model_response_str()}\n</tool_result>"))
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(
                    parts=new_parts,
                    instructions=message.instructions,
                ))
            else:
                new_messages.append(message)
        return new_messages
