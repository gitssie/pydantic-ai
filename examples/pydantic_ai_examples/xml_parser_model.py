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
    type: Optional[str]
    content: str
    delta: str
    buffer:str
    tag_name:Optional[str]
    tool_name:Optional[str]
    pos:int
    end:bool
    emit:bool
    part_id:str

    def __init__(self,buffer:str, pos:int,type:Optional[str]=None, tag_name:Optional[str]=None):
        self.buffer = buffer
        self.pos = pos
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

    def feed(self, data: str,last:bool=False) -> XMLNode:
        self.buffer += data
        if self.end:
            node = XMLNode(
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
        return self.feed("",last=True)

    def _process_node(self,last:bool=False) -> XMLNode:
        
        if self.type == "text":
            tag_start = self.buffer.find('<', self.pos)
            if tag_start == -1:
                delta = self.buffer[self.pos:]
                self.delta += delta
                self.content += delta
                self.pos = len(self.buffer)
                self.end = last
                return self
            
            if tag_start > self.pos:
                self.delta = self.buffer[self.pos:tag_start]
                self.content += self.delta
                self.pos = tag_start
                self.end = True
                return self

            tag_end = self.buffer.find('>', tag_start + 1)
            if tag_end == -1:
                if last or not self._is_valid_tag(tag_start+1,len(self.buffer)):
                    delta = self.buffer[self.pos:]
                    self.delta += delta
                    self.content += delta
                    self.pos = len(self.buffer)
                    self.end = last
                    return self
                elif self.pos + 64 < len(self.buffer):
                    delta = self.buffer[self.pos:]
                    self.delta += delta
                    self.content += delta
                    self.pos = len(self.buffer)
                    self.end = last
                return self
            elif not self._is_valid_tag(tag_start+1,tag_end):
                delta = self.buffer[self.pos:tag_end + 1]
                self.delta += delta
                self.content += delta
                self.pos = tag_end + 1
                self.end = last
                return self
           
            tag_name = self.buffer[tag_start+1:tag_end]

            return XMLNode(
                type="xml",
                tag_name=tag_name,
                buffer=self.buffer,
                pos=tag_end+1,
            )._process_node(last)
        else:
            end_tag = f"</{self.tag_name}>"
            end_pos = self.buffer.find(end_tag, self.pos)
            
            if end_pos == -1:
                end_pos = self.buffer.find("<", self.pos)
                if last:
                    delta = self.buffer[self.pos:]
                    self.delta += delta
                    self.content += delta
                    self.pos = len(self.buffer)
                    self.end = last
                    return self
                elif end_pos == -1:
                    delta = self.buffer[self.pos:]
                    self.delta += delta
                    self.content += delta
                    self.pos = len(self.buffer)
                    self.end = last
                    return self
                else:
                    if end_pos == self.pos and end_pos + len(end_tag) > len(self.buffer):
                        return self
                    next_pos = self.buffer.find("<", end_pos)
                    if next_pos == -1:
                        end_pos = len(self.buffer)
                    else:
                        end_pos = next_pos  
                    delta = self.buffer[self.pos:end_pos]
                    self.delta += delta
                    self.content += delta
                    self.pos = end_pos
                    return self 
    
            delta = self.buffer[self.pos:end_pos]
            self.delta += delta
            self.content += delta
            self.pos = end_pos + len(end_tag)
            self.end = True
            return self

    
    def __str__(self) -> str:
        return f"XMLNode(type={self.type}, name={self.tag_name}, pos={self.pos}, content={self.content})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def _skip_prefix_whitespace(self, content:str) -> str:
        i = 0
        length = len(content)
        while i < length and (content[i].isspace() or content[i] == '\n'):
            i += 1
        return content[i:]
    def _skip_suffix_whitespace(self, content:str) -> str:
        i = len(content) - 1
        while i >= 0 and (content[i].isspace() or content[i] == '\n'):
            i -= 1
        return content[:i+1]

    def _is_valid_tag(self,start_pos:int,end_pos:int) -> bool:
        while start_pos < end_pos:
            char = self.buffer[start_pos]
            if not (char.isalpha() or char == '_'):
                return False
            start_pos += 1
        return True

class XMLHandler(ABC):
    @abstractmethod
    def can_handle(self,chunk: XMLNode) -> bool:
        pass
    
    @abstractmethod
    def handle(self, parts_manager: Any, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
        pass

    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        pass

class TextHandler(XMLHandler):
    text_tags = ["answer", "thought"]
    
    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.type == "text" or chunk.tag_name in self.text_tags or chunk.tag_name is None
    
    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
        if not delta or self.filter(chunk, delta):
            return None
        
        return parts_manager.handle_text_delta(vendor_part_id=chunk.part_id,content=delta)

    def filter(self, chunk:XMLNode, delta:str) -> bool:
        if chunk.end and chunk.content == '```':
            return True
        if delta.startswith('</') and delta.endswith('>'):
            tag_name = delta[2:-1]
            if tag_name in self.text_tags:
                chunk.content = chunk.content[:-len(delta)]
                return True

        i = len(chunk.content)
        if i < 16 and chunk.content.startswith('```'):
           if chunk.end and re.match(r'^```[a-zA-Z_]+$', chunk.content):
               return True
           elif chunk.end:
               return False
           else:
               chunk.delta = delta
               return True
        return False

    def filter_part(self, node: XMLNode) -> bool:
        if not node.content or node.content == '```':
            return True
       
        for tag in self.text_tags:
            end_tag = f"</{tag}>"
            if node.content.endswith(end_tag):
                node.content = node.content[:-len(end_tag)]
                return False


        i = len(node.content)
        if i < 16 and re.match(r'^```[a-zA-Z_]+$', node.content):
            return True
        return False

    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        if self.filter_part(node):
            return None
        return TextPart(content=node.content)

class ExecuteCodeToolHandler(XMLHandler):
    code_arg:str = "python_code"
    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.tag_name == 'run_python_code'
    
    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
        if chunk.emit:
            return None
        tool_name = ""
        if not chunk.tool_name:
            tool_name = chunk.tag_name
            chunk.tool_name = tool_name
        
        args = None
        if chunk.end:
            args = {self.code_arg: chunk.content}
        
        return parts_manager.handle_tool_call_delta(vendor_part_id=chunk.part_id,tool_name=tool_name, args=args,tool_call_id=chunk.part_id )

    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        tool_name = cast(str,node.tag_name)
        args = {self.code_arg: node.content}
        return ToolCallPart(tool_name=tool_name,args=args, tool_call_id=node.part_id)

class MCPToolHandler(XMLHandler):
    def can_handle(self, chunk: XMLNode) -> bool:
        return chunk.tag_name == 'use_mcp_tool' or chunk.tag_name == 'use_tool'
    
    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
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
    
    def _extract_data(self,content:str) -> tuple[str,str]:
        tool_name = self._extract_tag_content(content, "tool_name") or ""
        arguments_str = self._extract_tag_content(content, "arguments") or ""
        return tool_name, arguments_str

    def _extract_tag_content(self, content: str, tag_name: str) -> Optional[str]:
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        tool_name, arguments_str = self._extract_data(node.content)

        return ToolCallPart(tool_name=tool_name,
                            args=arguments_str,
                            tool_call_id=node.part_id)
    
class DefaultToolHandler(XMLHandler):
    def can_handle(self, chunk: XMLNode) -> bool:
        return True
    
    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
        tool_name = ""
        if not chunk.tool_name:
            tool_name = chunk.tag_name
            chunk.tool_name = tool_name
        
        return parts_manager.handle_tool_call_delta(
            vendor_part_id=chunk.part_id,
            tool_name=tool_name,
            args=chunk.content if chunk.end else None,
            tool_call_id=chunk.part_id
        )
    
    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        tool_name = cast(str,node.tag_name)
        return ToolCallPart(tool_name=tool_name, args=node.content, tool_call_id=node.part_id)

@dataclass
class HandlersFactory(XMLHandler):    
    _tool_handlers: list[XMLHandler] = field(default_factory=lambda: [
        TextHandler(),
        ExecuteCodeToolHandler(),
        MCPToolHandler(),
        DefaultToolHandler()
    ])

    def _get_tool_handler(self, chunk: XMLNode) -> XMLHandler:
        for handler in self._tool_handlers:
            if handler.can_handle(chunk):
                return handler
        return self._tool_handlers[-1]
    
    def can_handle(self, chunk: XMLNode) -> bool:
        raise NotImplementedError("HandlersFactory is not meant to be used as a handler")
    
    def handle(self, parts_manager: ModelResponsePartsManager, chunk: XMLNode, delta:str) -> Optional[ModelResponseStreamEvent]:
        if chunk.emit:
            return None
        
        chunk.delta = ""
        event = self._get_tool_handler(chunk).handle(parts_manager, chunk, delta)
        chunk.emit = chunk.end
        return event
    
    def part(self,node: XMLNode) -> Optional[ModelResponsePart]:
        return self._get_tool_handler(node).part(node)


@dataclass
class XMLStreamedResponse(StreamedResponse):
    _model_name: str
    _xml_stream: AsyncIterator[XMLNode]
    _usage_getter: Callable[[], Usage]
    _timestamp: datetime = field(default_factory=lambda: datetime.now())
    _text_part:list[ModelResponsePart] = field(default_factory=list)
    _handlers: HandlersFactory = field(default_factory=HandlersFactory)
    
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        buffer:str = ""
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
    kind: Literal['xml_response'] = 'xml_response' # type: ignore



# 定义一个装饰器函数，用于在类定义后自动调用指定方法
def auto_install(cls: Type[Any]) -> Type[Any]:
    if hasattr(cls, 'install_xml_parser_support'):
        getattr(cls, 'install_xml_parser_support')()
    return cls
    
@auto_install
@dataclass(init=False)
class XMLParserModel(Model):
    wrapped: Model
    _handlers: HandlersFactory = field(default_factory=HandlersFactory)
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
        parts: list[ModelResponsePart] =  model_response.parts
        
        for part in parts:
            if isinstance(part, TextPart):
                text_content += part.content
            else:
                non_text_parts.append(part)
        
        if not text_content:
            return model_response, usage
            
        parsed_parts: list[ModelResponsePart] = []
        
        node = XMLNode(buffer=text_content, pos=0)
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
            xml_stream = self._wrap_response_stream(response_stream)
            yield XMLStreamedResponse(
                _model_name=self.model_name,
                _xml_stream=xml_stream,
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
            
    async def _wrap_response_stream(self, response_stream: StreamedResponse) -> AsyncGenerator[Any, None]:
        node = XMLNode(buffer="", pos=0)
        async for event in response_stream:
            text_chunk = None
            
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                text_chunk = event.part.content
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                text_chunk = event.delta.content_delta
                
            if text_chunk:
                node = node.feed(text_chunk)
                if node.has_delta() or node.end:
                    yield node
            else:
                yield event

  
        while node:
            yield node
            node = node.complete()

    def _build_message(self,messages:list[ModelMessage]) -> list[ModelMessage]:
        new_messages:list[ModelMessage] = []
        for message in messages:
            if isinstance(message,ModelResponse) and hasattr(message,"raw_parts"):
                new_messages.append(ModelResponse(
                    parts= cast(list[ModelResponsePart],getattr(message,"raw_parts")),
                    model_name=message.model_name,
                    timestamp=message.timestamp
                ))
            elif isinstance(message,ModelRequest):
                new_parts:list[ModelRequestPart] = []
                for part in message.parts:
                    if(isinstance(part,RetryPromptPart)):
                        new_parts.append(UserPromptPart(content=f"<tool_result>\n{part.model_response()}\n</tool_result>"))
                    elif isinstance(part,ToolReturnPart):
                        new_parts.append(UserPromptPart(content=f"<tool_result>\nLogs:\n{part.model_response_str()}\n</tool_result>"))
                    else:
                        new_parts.append(part)
                new_messages.append(ModelRequest(
                    parts=new_parts,
                    instructions=message.instructions,
                ))
            else:
                new_messages.append(message)
        return new_messages