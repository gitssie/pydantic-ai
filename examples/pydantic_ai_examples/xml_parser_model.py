from __future__ import annotations

from collections.abc import AsyncIterator, AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest, 
    ModelResponseStreamEvent, 
    PartDeltaEvent,
    PartStartEvent,
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



text_tags = ["answer", "thought", "final_answer"]

class XMLNode:
    type: Optional[str]
    content: str
    delta: str
    buffer:str
    tag_name:Optional[str]
    tag_name_delta:Optional[str]
    pos:int
    end:bool
    part_id:Optional[str]

    def __init__(self,buffer:str, pos:int,type:Optional[str]=None, tag_name:Optional[str]=None):
        self.buffer = buffer
        self.pos = pos
        self.type = type or "text"
        self.tag_name = tag_name
        self.tag_name_delta = tag_name
        self.content = ""
        self.delta = ""
        self.end = False
        self.part_id = f"node_{id(self)}"

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

        if node.end:
            node.content = self._skip_suffix_whitespace(node.content)
            node.delta = self._skip_suffix_whitespace(node.delta)
        
        return node

    def complete(self) -> Optional[XMLNode]:
        if self.pos >= len(self.buffer):
            return None
        return self.feed("",last=True)

    def _process_node(self,last:bool=False) -> XMLNode:
        self.delta = ""
        if self.type == "text":
            tag_start = self.buffer.find('<', self.pos)
            if tag_start == -1:
                self.delta = self.buffer[self.pos:]
                self.content += self.delta
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
                    self.delta = self.buffer[self.pos:]
                    self.content += self.delta
                    self.pos = len(self.buffer)
                    self.end = last
                    return self
                elif self.pos + 64 < len(self.buffer):
                    self.delta = self.buffer[self.pos:]
                    self.content += self.delta
                    self.pos = len(self.buffer)
                    self.end = last
                return self
            elif not self._is_valid_tag(tag_start+1,tag_end):
                self.delta = self.buffer[self.pos:]
                self.content += self.delta
                self.pos = len(self.buffer)
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
                    self.delta = self.buffer[self.pos:]
                    self.content += self.delta
                    self.pos = len(self.buffer)
                    self.end = last
                    return self
                elif end_pos == -1:
                    self.delta = self.buffer[self.pos:]
                    self.content += self.delta
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
                    self.delta = self.buffer[self.pos:end_pos]
                    self.content += self.delta
                    self.pos = end_pos
                    return self 
    
            self.delta = self.buffer[self.pos:end_pos]
            self.content += self.delta
            self.pos = end_pos + len(end_tag)
            self.end = True
            return self

    
    def __str__(self) -> str:
        return f"XMLNode(type={self.type}, tag={self.tag_name}, pos={self.pos}, content={self.content})"
    
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

@dataclass
class XMLStreamedResponse(StreamedResponse):
    _model_name: str
    _xml_stream: AsyncIterator[XMLNode]
    _usage_getter: Callable[[], Usage]
    _timestamp: datetime = field(default_factory=lambda: datetime.now())
    _text_part:list[ModelResponsePart] = field(default_factory=list)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        buffer:str = ""
        async for chunk in self._xml_stream:
            if not isinstance(chunk, XMLNode):
                yield chunk
                continue
            buffer = chunk.buffer
            try:
                if chunk.type == "text" or chunk.tag_name in text_tags or chunk.tag_name is None:
                   yield self._parts_manager.handle_text_delta(
                            vendor_part_id=chunk.part_id,
                            content=chunk.delta)
                else:
                    event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=chunk.part_id,
                            tool_name=chunk.tag_name_delta,
                            args={"code":chunk.content},
                            tool_call_id=chunk.part_id)
                    chunk.tag_name_delta = ""
                    if event is not None:
                        yield event
            except Exception as e:
                error_msg = f"parse xml error: {str(e)}"
                yield self._parts_manager.handle_text_delta(vendor_part_id='error', content=error_msg)
        
        self._text_part.append(TextPart(content=buffer))

    def get(self) -> ModelResponse:
        return XMLParsedModelResponse(
            raw_parts=self._text_part, parts=self._parts_manager.get_parts(), model_name=self.model_name, timestamp=self.timestamp
        )
    
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
    raw_parts: list[ModelResponsePart]|None = field(default_factory=list)

@dataclass(init=False)
class XMLParserModel(Model):
    wrapped: Model
    
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
            if node.type == "text" or node.tag_name in text_tags or node.tag_name is None:
                parsed_parts.append(TextPart(content=node.content.strip()))
            else:
                parsed_parts.append(ToolCallPart(
                                tool_name=node.tag_name,
                                args={"code": node.content.strip()},
                                tool_call_id=f"{node.tag_name}_{id(node)}"
                            ))
    
        parsed_parts.extend(non_text_parts)
        
        parsed_response = XMLParsedModelResponse(
            parts=parsed_parts,
            model_name=model_response.model_name,
            timestamp=model_response.timestamp,
            raw_parts = parts
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
                if node.delta:
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
                    parts= cast(list[ModelResponsePart],message.__getattribute__("raw_parts")),
                    model_name=message.model_name,
                    timestamp=message.timestamp
                ))
                continue
            elif isinstance(message,ModelRequest):
                new_messages.append(message)
                for i,part in enumerate(message.parts):
                    if isinstance(part,ToolReturnPart):
                        message.parts[i] = UserPromptPart(content=f"<observation>\nExecution logs:\n{part.content}\n</observation>")
            else:
                new_messages.append(message)
        return new_messages
