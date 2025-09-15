# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.tools import ToolParameter
from llama_stack.providers.utils.tools.mcp import (
    _parse_parameter_from_schema,
    _resolve_refs,
    list_mcp_tools,
)


class TestMCPSchemaResolvers:
    """Test cases for MCP JSON Schema $ref resolution functionality."""

    def test_resolve_refs_simple_reference(self):
        """Test resolving a simple $ref to a definition."""
        schema = {"$ref": "#/$defs/FlightInfo"}
        defs = {
            "FlightInfo": {
                "type": "object",
                "properties": {"flight_number": {"type": "string"}, "date": {"type": "string"}},
                "required": ["flight_number", "date"],
            }
        }

        result = _resolve_refs(schema, defs)

        assert result == defs["FlightInfo"]
        assert "$ref" not in result
        assert result["properties"]["flight_number"]["type"] == "string"

    def test_resolve_refs_nested_object(self):
        """Test resolving $ref in nested objects."""
        schema = {
            "type": "object",
            "properties": {"flight": {"$ref": "#/$defs/FlightInfo"}, "passenger": {"$ref": "#/$defs/Passenger"}},
        }
        defs = {
            "FlightInfo": {"type": "object", "properties": {"flight_number": {"type": "string"}}},
            "Passenger": {"type": "object", "properties": {"name": {"type": "string"}}},
        }

        result = _resolve_refs(schema, defs)

        assert result["type"] == "object"
        assert "$ref" not in result["properties"]["flight"]
        assert "$ref" not in result["properties"]["passenger"]
        assert result["properties"]["flight"]["properties"]["flight_number"]["type"] == "string"
        assert result["properties"]["passenger"]["properties"]["name"]["type"] == "string"

    def test_resolve_refs_array_items(self):
        """Test resolving $ref in array items."""
        schema = {
            "type": "array",
            "items": {"anyOf": [{"$ref": "#/$defs/FlightInfo"}, {"type": "object", "additionalProperties": True}]},
        }
        defs = {
            "FlightInfo": {
                "type": "object",
                "properties": {
                    "flight_number": {"type": "string", "description": "Flight number"},
                    "date": {"type": "string", "description": "Flight date"},
                },
                "required": ["flight_number", "date"],
                "title": "FlightInfo",
            }
        }

        result = _resolve_refs(schema, defs)

        flight_info = result["items"]["anyOf"][0]
        assert "$ref" not in flight_info
        assert flight_info["type"] == "object"
        assert flight_info["title"] == "FlightInfo"
        assert flight_info["properties"]["flight_number"]["description"] == "Flight number"
        assert "flight_number" in flight_info["required"]

    def test_resolve_refs_recursive_resolution(self):
        """Test resolving $ref that references another $ref."""
        schema = {"$ref": "#/$defs/MainType"}
        defs = {
            "MainType": {"$ref": "#/$defs/NestedType"},
            "NestedType": {"type": "object", "properties": {"value": {"type": "string"}}},
        }

        result = _resolve_refs(schema, defs)

        assert result == defs["NestedType"]
        assert result["properties"]["value"]["type"] == "string"

    def test_resolve_refs_unresolvable_reference(self):
        """Test handling of unresolvable $ref."""
        schema = {"$ref": "#/$defs/NonExistentType"}
        defs = {"SomeOtherType": {"type": "string"}}

        result = _resolve_refs(schema, defs)

        # Should return the original schema if ref cannot be resolved
        assert result == schema
        assert result["$ref"] == "#/$defs/NonExistentType"

    def test_resolve_refs_non_defs_reference(self):
        """Test handling of $ref that doesn't follow #/$defs/ pattern."""
        schema = {"$ref": "#/properties/someProperty"}
        defs = {"FlightInfo": {"type": "object"}}

        result = _resolve_refs(schema, defs)

        # Should return original schema for non-defs references
        assert result == schema

    def test_resolve_refs_no_reference(self):
        """Test that schema without $ref passes through unchanged."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        defs = {"FlightInfo": {"type": "object"}}

        result = _resolve_refs(schema, defs)

        assert result == schema

    def test_resolve_refs_empty_defs(self):
        """Test resolving refs with empty defs."""
        schema = {"$ref": "#/$defs/FlightInfo"}
        defs = {}

        result = _resolve_refs(schema, defs)

        # Should return original schema if def doesn't exist
        assert result == schema

    def test_parse_parameter_from_schema_basic(self):
        """Test parsing basic parameter from schema."""
        schema = {"type": "string", "description": "User identifier", "title": "User ID"}
        required_params = ["user_id"]

        result = _parse_parameter_from_schema("user_id", schema, required_params)

        assert isinstance(result, ToolParameter)
        assert result.name == "user_id"
        assert result.parameter_type == "string"
        assert result.description == "User identifier"
        assert result.title == "User ID"
        assert result.required is True
        assert result.items is None
        assert result.default is None

    def test_parse_parameter_from_schema_array_with_items(self):
        """Test parsing array parameter with items schema."""
        schema = {
            "type": "array",
            "description": "List of flights",
            "title": "Flights",
            "items": {
                "type": "object",
                "properties": {"flight_number": {"type": "string"}, "date": {"type": "string"}},
            },
        }
        required_params = ["flights"]

        result = _parse_parameter_from_schema("flights", schema, required_params)

        assert result.name == "flights"
        assert result.parameter_type == "array"
        assert result.description == "List of flights"
        assert result.title == "Flights"
        assert result.required is True
        assert result.items is not None
        assert result.items["type"] == "object"
        assert "flight_number" in result.items["properties"]

    def test_parse_parameter_from_schema_optional(self):
        """Test parsing optional parameter."""
        schema = {"type": "integer", "description": "Age of passenger", "default": 25}
        required_params = ["name"]  # age is not in required params

        result = _parse_parameter_from_schema("age", schema, required_params)

        assert result.name == "age"
        assert result.parameter_type == "integer"
        assert result.required is False
        assert result.default == 25

    def test_parse_parameter_from_schema_missing_type(self):
        """Test parsing parameter with missing type defaults to string."""
        schema = {"description": "Some description", "title": "Some Title"}
        required_params = []

        result = _parse_parameter_from_schema("param", schema, required_params)

        assert result.parameter_type == "string"  # Default type
        assert result.description == "Some description"

    def test_parse_parameter_from_schema_complex_items(self):
        """Test parsing parameter with complex items containing anyOf."""
        schema = {
            "type": "array",
            "title": "Passengers",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"first_name": {"type": "string"}, "last_name": {"type": "string"}},
                        "required": ["first_name", "last_name"],
                        "title": "Passenger",
                    },
                    {"type": "object", "additionalProperties": True},
                ]
            },
        }
        required_params = ["passengers"]

        result = _parse_parameter_from_schema("passengers", schema, required_params)

        assert result.parameter_type == "array"
        assert result.title == "Passengers"
        assert "anyOf" in result.items
        assert len(result.items["anyOf"]) == 2
        assert result.items["anyOf"][0]["title"] == "Passenger"


class TestMCPListTools:
    """Test cases for the list_mcp_tools function with real-world schema examples."""

    @pytest.fixture
    def mock_client_session(self):
        """Create a mock MCP client session."""
        session = AsyncMock()

        # Mock tool with $ref schemas
        mock_tool = MagicMock()
        mock_tool.name = "book_reservation"
        mock_tool.description = "Book a flight reservation"
        mock_tool.inputSchema = {
            "$defs": {
                "FlightInfo": {
                    "type": "object",
                    "properties": {
                        "flight_number": {
                            "type": "string",
                            "description": "Flight number such as 'HAT001'",
                            "title": "Flight Number",
                        },
                        "date": {"type": "string", "description": "Flight date in YYYY-MM-DD format", "title": "Date"},
                    },
                    "required": ["flight_number", "date"],
                    "title": "FlightInfo",
                },
                "Passenger": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "Passenger's first name",
                            "title": "First Name",
                        },
                        "last_name": {"type": "string", "description": "Passenger's last name", "title": "Last Name"},
                    },
                    "required": ["first_name", "last_name"],
                    "title": "Passenger",
                },
            },
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "title": "User ID"},
                "flights": {
                    "type": "array",
                    "title": "Flights",
                    "items": {
                        "anyOf": [{"$ref": "#/$defs/FlightInfo"}, {"type": "object", "additionalProperties": True}]
                    },
                },
                "passengers": {
                    "type": "array",
                    "title": "Passengers",
                    "items": {
                        "anyOf": [{"$ref": "#/$defs/Passenger"}, {"type": "object", "additionalProperties": True}]
                    },
                },
            },
            "required": ["user_id", "flights", "passengers"],
        }

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        session.list_tools.return_value = mock_tools_result

        return session

    @patch("llama_stack.providers.utils.tools.mcp.client_wrapper")
    async def test_list_mcp_tools_resolves_refs(self, mock_client_wrapper, mock_client_session):
        """Test that list_mcp_tools properly resolves $ref references."""
        mock_client_wrapper.return_value.__aenter__.return_value = mock_client_session

        result = await list_mcp_tools("http://localhost:8765", {})

        assert len(result.data) == 1
        tool = result.data[0]
        assert tool.name == "book_reservation"
        assert tool.description == "Book a flight reservation"
        assert len(tool.parameters) == 3

        # Check user_id parameter (no refs)
        user_id_param = next(p for p in tool.parameters if p.name == "user_id")
        assert user_id_param.parameter_type == "string"
        assert user_id_param.required is True

        # Check flights parameter (contains resolved $ref)
        flights_param = next(p for p in tool.parameters if p.name == "flights")
        assert flights_param.parameter_type == "array"
        assert flights_param.title == "Flights"
        assert flights_param.required is True
        assert "anyOf" in flights_param.items

        # Verify that the $ref was resolved
        flight_info = flights_param.items["anyOf"][0]
        assert "$ref" not in flight_info
        assert flight_info["type"] == "object"
        assert flight_info["title"] == "FlightInfo"
        assert "flight_number" in flight_info["properties"]
        assert flight_info["properties"]["flight_number"]["description"] == "Flight number such as 'HAT001'"

        # Check passengers parameter (contains resolved $ref)
        passengers_param = next(p for p in tool.parameters if p.name == "passengers")
        assert passengers_param.parameter_type == "array"
        assert passengers_param.title == "Passengers"
        assert passengers_param.required is True

        # Verify that the $ref was resolved for passengers too
        passenger_info = passengers_param.items["anyOf"][0]
        assert "$ref" not in passenger_info
        assert passenger_info["type"] == "object"
        assert passenger_info["title"] == "Passenger"
        assert "first_name" in passenger_info["properties"]
        assert "last_name" in passenger_info["properties"]

    @patch("llama_stack.providers.utils.tools.mcp.client_wrapper")
    async def test_list_mcp_tools_with_empty_defs(self, mock_client_wrapper):
        """Test list_mcp_tools with schema containing no $defs."""
        session = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "simple_tool"
        mock_tool.description = "A simple tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "A message", "title": "Message"}},
            "required": ["message"],
        }

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        session.list_tools.return_value = mock_tools_result

        mock_client_wrapper.return_value.__aenter__.return_value = session

        result = await list_mcp_tools("http://localhost:8765", {})

        assert len(result.data) == 1
        tool = result.data[0]
        assert tool.name == "simple_tool"
        assert len(tool.parameters) == 1

        message_param = tool.parameters[0]
        assert message_param.name == "message"
        assert message_param.parameter_type == "string"
        assert message_param.description == "A message"
        assert message_param.title == "Message"
        assert message_param.required is True

    @patch("llama_stack.providers.utils.tools.mcp.client_wrapper")
    async def test_list_mcp_tools_with_unresolvable_refs(self, mock_client_wrapper):
        """Test list_mcp_tools gracefully handles unresolvable $refs."""
        session = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "broken_tool"
        mock_tool.description = "Tool with broken refs"
        mock_tool.inputSchema = {
            "$defs": {"ValidType": {"type": "string"}},
            "type": "object",
            "properties": {
                "valid_param": {"$ref": "#/$defs/ValidType"},
                "broken_param": {"$ref": "#/$defs/NonExistentType"},
            },
            "required": ["valid_param", "broken_param"],
        }

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        session.list_tools.return_value = mock_tools_result

        mock_client_wrapper.return_value.__aenter__.return_value = session

        result = await list_mcp_tools("http://localhost:8765", {})

        assert len(result.data) == 1
        tool = result.data[0]
        assert len(tool.parameters) == 2

        # Valid param should be resolved
        valid_param = next(p for p in tool.parameters if p.name == "valid_param")
        assert valid_param.parameter_type == "string"

        # Broken param should keep the $ref (graceful degradation)
        broken_param = next(p for p in tool.parameters if p.name == "broken_param")
        # The parameter will be created but items might contain the unresolved ref
        assert broken_param.name == "broken_param"

    @patch("llama_stack.providers.utils.tools.mcp.client_wrapper")
    async def test_list_mcp_tools_multiple_tools(self, mock_client_wrapper):
        """Test list_mcp_tools with multiple tools having different schema patterns."""
        session = AsyncMock()

        # Tool 1: Has $refs
        tool1 = MagicMock()
        tool1.name = "complex_tool"
        tool1.description = "Complex tool with refs"
        tool1.inputSchema = {
            "$defs": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}, "title": "Person"}},
            "type": "object",
            "properties": {"person": {"$ref": "#/$defs/Person"}},
            "required": ["person"],
        }

        # Tool 2: No $refs
        tool2 = MagicMock()
        tool2.name = "simple_tool"
        tool2.description = "Simple tool without refs"
        tool2.inputSchema = {
            "type": "object",
            "properties": {"text": {"type": "string", "title": "Text"}},
            "required": ["text"],
        }

        mock_tools_result = MagicMock()
        mock_tools_result.tools = [tool1, tool2]
        session.list_tools.return_value = mock_tools_result

        mock_client_wrapper.return_value.__aenter__.return_value = session

        result = await list_mcp_tools("http://localhost:8765", {})

        assert len(result.data) == 2

        # Check complex tool
        complex_tool = next(t for t in result.data if t.name == "complex_tool")
        person_param = complex_tool.parameters[0]
        assert person_param.name == "person"
        assert "$ref" not in str(person_param.items)  # Should be resolved

        # Check simple tool
        simple_tool = next(t for t in result.data if t.name == "simple_tool")
        text_param = simple_tool.parameters[0]
        assert text_param.name == "text"
        assert text_param.parameter_type == "string"
        assert text_param.title == "Text"


class TestMCPEdgeCases:
    """Test edge cases and error conditions for MCP tools."""

    def test_resolve_refs_deeply_nested(self):
        """Test resolving deeply nested $refs."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {"level2": {"type": "array", "items": {"$ref": "#/$defs/DeepType"}}},
                }
            },
        }
        defs = {"DeepType": {"type": "object", "properties": {"nested_value": {"type": "string"}}}}

        result = _resolve_refs(schema, defs)

        deep_type = result["properties"]["level1"]["properties"]["level2"]["items"]
        assert "$ref" not in deep_type
        assert deep_type["properties"]["nested_value"]["type"] == "string"

    def test_resolve_refs_array_primitives(self):
        """Test that primitive arrays without $refs pass through."""
        schema = {"type": "array", "items": {"type": "string"}}
        defs = {}

        result = _resolve_refs(schema, defs)

        assert result == schema
        assert result["items"]["type"] == "string"

    def test_resolve_refs_mixed_anyof(self):
        """Test resolving $refs in anyOf with mixed types."""
        schema = {
            "type": "array",
            "items": {
                "anyOf": [{"$ref": "#/$defs/TypeA"}, {"type": "string"}, {"$ref": "#/$defs/TypeB"}, {"type": "number"}]
            },
        }
        defs = {
            "TypeA": {"type": "object", "properties": {"a": {"type": "string"}}},
            "TypeB": {"type": "object", "properties": {"b": {"type": "integer"}}},
        }

        result = _resolve_refs(schema, defs)

        any_of = result["items"]["anyOf"]
        assert len(any_of) == 4

        # First should be resolved TypeA
        assert "$ref" not in any_of[0]
        assert any_of[0]["properties"]["a"]["type"] == "string"

        # Second should be unchanged string
        assert any_of[1]["type"] == "string"

        # Third should be resolved TypeB
        assert "$ref" not in any_of[2]
        assert any_of[2]["properties"]["b"]["type"] == "integer"

        # Fourth should be unchanged number
        assert any_of[3]["type"] == "number"

    def test_parse_parameter_from_schema_all_fields(self):
        """Test parsing parameter with all possible fields."""
        schema = {
            "type": "array",
            "description": "A comprehensive parameter",
            "title": "Comprehensive Parameter",
            "default": [],
            "items": {"type": "object", "properties": {"field1": {"type": "string"}, "field2": {"type": "integer"}}},
        }
        required_params = ["comprehensive_param"]

        result = _parse_parameter_from_schema("comprehensive_param", schema, required_params)

        assert result.name == "comprehensive_param"
        assert result.parameter_type == "array"
        assert result.description == "A comprehensive parameter"
        assert result.title == "Comprehensive Parameter"
        assert result.required is True
        assert result.default == []
        assert result.items["type"] == "object"
        assert "field1" in result.items["properties"]
        assert "field2" in result.items["properties"]
